"""Pure metric computations for timestamped audio captioning (TAC).

Re-implementation of TER, GER and tcpCapSim from LongCap (arXiv:2611.11125),
Sec. 2.2 and App. A.2. No LLM calls here: grouping and value scoring are
supplied by the caller, so these functions are deterministic and testable.

Interval convention: [start, end) in seconds. An event with end < start is
treated as zero-length (counted in diagnostics, never silently repaired in
the input files).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json

# Keys excluded from tcpCapSim (paper App. A.2: free-text caption and the
# structural speaker_id are excluded; `type` is included).
EXCLUDED_KEYS = {"caption", "speaker_id"}

TIME_TOLERANCE = 5.0  # seconds, tcpCapSim group time constraint


# --------------------------------------------------------------------------
# Interval helpers
# --------------------------------------------------------------------------

def dur(iv):
    return max(0.0, iv[1] - iv[0])


def merge(intervals):
    """Union of intervals as a sorted, disjoint list."""
    ivs = sorted((s, e) for s, e in intervals if e > s)
    out = []
    for s, e in ivs:
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def total(intervals):
    return sum(e - s for s, e in intervals)


def intersect(a, b):
    """Intersection of two merged (sorted, disjoint) interval lists."""
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def gap(a, b):
    """Distance between two intervals (0 if they overlap or touch)."""
    return max(0.0, max(a[0], b[0]) - min(a[1], b[1]))


def interval_of(ev):
    s, e = float(ev["start"]), float(ev["end"])
    return (s, e) if e >= s else (s, s)


# --------------------------------------------------------------------------
# Groups
# --------------------------------------------------------------------------

@dataclass
class Group:
    ref: list = field(default_factory=list)   # indices into reference events
    hyp: list = field(default_factory=list)   # indices into predicted events


def complete_groups(groups, n_ref, n_hyp):
    """Ensure every event belongs to exactly one group (singletons for the rest)."""
    seen_r = {i for g in groups for i in g.ref}
    seen_h = {j for g in groups for j in g.hyp}
    out = [g for g in groups if g.ref or g.hyp]
    out += [Group(ref=[i]) for i in range(n_ref) if i not in seen_r]
    out += [Group(hyp=[j]) for j in range(n_hyp) if j not in seen_h]
    return out


# --------------------------------------------------------------------------
# TER: DER accounting applied to semantic groups
# --------------------------------------------------------------------------

def ter_components(groups, ref_iv, hyp_iv):
    """Return dict with miss, fa, conf, ref (seconds)."""
    deltas = []  # (time, d_ref, d_hyp, d_corr)
    t_ref = 0.0
    for g in groups:
        r = merge(ref_iv[i] for i in g.ref)
        h = merge(hyp_iv[j] for j in g.hyp)
        c = intersect(r, h)
        t_ref += total(r)
        for s, e in r:
            deltas += [(s, 1, 0, 0), (e, -1, 0, 0)]
        for s, e in h:
            deltas += [(s, 0, 1, 0), (e, 0, -1, 0)]
        for s, e in c:
            deltas += [(s, 0, 0, 1), (e, 0, 0, -1)]
    deltas.sort(key=lambda x: x[0])
    miss = fa = conf = 0.0
    nr = nh = nc = 0
    prev = None
    k = 0
    while k < len(deltas):
        t = deltas[k][0]
        if prev is not None and t > prev:
            span = t - prev
            miss += span * max(0, nr - nh)
            fa += span * max(0, nh - nr)
            conf += span * (min(nr, nh) - nc)
        while k < len(deltas) and deltas[k][0] == t:
            _, a, b, c = deltas[k]
            nr += a
            nh += b
            nc += c
            k += 1
        prev = t
    return {"miss": miss, "fa": fa, "conf": conf, "ref": t_ref}


# --------------------------------------------------------------------------
# GER: (T_overlap - T_match + T_dup) / sum of reference event durations
# --------------------------------------------------------------------------

def _hungarian_max(weights):
    """Maximum-weight assignment for a rectangular matrix (list of lists).

    Uses scipy when available, otherwise an O(n^3) Hungarian algorithm.
    Returns the total weight of the optimal one-to-one matching.
    """
    n_r = len(weights)
    n_c = len(weights[0]) if n_r else 0
    if n_r == 0 or n_c == 0:
        return 0.0
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        import numpy as np  # type: ignore
        w = np.asarray(weights, dtype=float)
        ri, ci = linear_sum_assignment(w, maximize=True)
        return float(w[ri, ci].sum())
    except ImportError:
        pass
    n = max(n_r, n_c)
    big = max(max(row) for row in weights)
    # cost matrix for minimisation, padded to square
    cost = [[(big - weights[i][j]) if i < n_r and j < n_c else big
             for j in range(n)] for i in range(n)]
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    tot = 0.0
    for j in range(1, n + 1):
        i = p[j]
        if i and i <= n_r and j <= n_c:
            tot += weights[i - 1][j - 1]
    return tot


def max_matched_overlap(r_ivs, h_ivs):
    """Maximum total overlap under one-to-one matching of two event lists.

    The bipartite graph is split into connected components of positive
    overlap first, so the assignment problems stay small.
    """
    edges = {}
    for a, ri in enumerate(r_ivs):
        for b, hi in enumerate(h_ivs):
            w = overlap(ri, hi)
            if w > 0:
                edges[(a, b)] = w
    if not edges:
        return 0.0
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(("r", a)), find(("h", b))
        if ra != rb:
            parent[ra] = rb
    comps = {}
    for (a, b), w in edges.items():
        comps.setdefault(find(("r", a)), []).append((a, b, w))
    best = 0.0
    for es in comps.values():
        if len(es) == 1:
            best += es[0][2]
            continue
        rs = sorted({a for a, _, _ in es})
        hs = sorted({b for _, b, _ in es})
        ri = {a: k for k, a in enumerate(rs)}
        hi = {b: k for k, b in enumerate(hs)}
        mat = [[0.0] * len(hs) for _ in rs]
        for a, b, w in es:
            mat[ri[a]][hi[b]] = w
        best += _hungarian_max(mat)
    return best


def ger_components(groups, ref_iv, hyp_iv):
    """Return dict with overlap, match, dup, ref_events (seconds)."""
    t_ov = t_match = t_dup = 0.0
    for g in groups:
        r_list = [ref_iv[i] for i in g.ref]
        h_list = [hyp_iv[j] for j in g.hyp]
        r = merge(r_list)
        h = merge(h_list)
        t_ov += total(intersect(r, h))
        t_dup += sum(dur(x) for x in h_list) - total(h)
        if r_list and h_list:
            t_match += max_matched_overlap(r_list, h_list)
    t_ref_ev = sum(dur(x) for x in ref_iv)
    return {"overlap": t_ov, "match": t_match, "dup": t_dup, "ref_events": t_ref_ev}


# --------------------------------------------------------------------------
# tcpCapSim
# --------------------------------------------------------------------------

def _is_empty(v):
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == "" or v.strip().lower() in {"null", "none"}
    if isinstance(v, (list, dict)):
        return len(v) == 0
    return False


def _render(v):
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return "、".join(_render(x) for x in v if not _is_empty(x))
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


def pool_values(events):
    """Pool non-empty attribute values by key over a list of events.

    Returns {key: pooled_string}; each key counts once per side per group.
    """
    pooled = {}
    for ev in events:
        for k, v in (ev.get("attributes") or {}).items():
            if k in EXCLUDED_KEYS or _is_empty(v):
                continue
            s = _render(v)
            if not s:
                continue
            lst = pooled.setdefault(k, [])
            if s not in lst:
                lst.append(s)
    return {k: " | ".join(v) for k, v in pooled.items()}


def time_ok(r_ivs, h_ivs, tol=TIME_TOLERANCE):
    return any(gap(a, b) <= tol for a in r_ivs for b in h_ivs)


def capsim_items(groups, ref_events, hyp_events, ref_iv, hyp_iv):
    """Collect what tcpCapSim needs from one grouping.

    Returns (pairs, n_ref_keys, n_hyp_keys) where pairs is a list of
    (key, ref_value, hyp_value) for shared keys of time-valid groups.
    """
    pairs = []
    n_ref = n_hyp = 0
    for g in groups:
        rp = pool_values([ref_events[i] for i in g.ref])
        hp = pool_values([hyp_events[j] for j in g.hyp])
        n_ref += len(rp)
        n_hyp += len(hp)
        if not rp or not hp:
            continue
        if not time_ok([ref_iv[i] for i in g.ref], [hyp_iv[j] for j in g.hyp]):
            continue
        for k in rp:
            if k in hp:
                pairs.append((k, rp[k], hp[k]))
    return pairs, n_ref, n_hyp


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def rates(ter, ger, cap_credit, n_ref_keys, n_hyp_keys):
    out = {}
    out["TER"] = (ter["miss"] + ter["fa"] + ter["conf"]) / ter["ref"] if ter["ref"] else float("nan")
    out["GER"] = ((ger["overlap"] - ger["match"] + ger["dup"]) / ger["ref_events"]
                  if ger["ref_events"] else float("nan"))
    p = cap_credit / n_hyp_keys if n_hyp_keys else 0.0
    r = cap_credit / n_ref_keys if n_ref_keys else 0.0
    out["capsim_precision"] = p
    out["capsim_recall"] = r
    out["tcpCapSim"] = (p + r) / 2
    return out
