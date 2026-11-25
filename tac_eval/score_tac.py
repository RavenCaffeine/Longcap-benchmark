#!/usr/bin/env python3
"""Score TAC predictions against LongCapBench-TAC-Long style references.

Re-implements the LongCap paper's scoring (arXiv:2611.11125, Sec. 2.2, App. A.2):

  1. Event grouping: an LLM proposes groups of reference/predicted events that
     describe the same occurrence, in 60 s steps with a 15 s look-back; groups
     sharing events across windows are merged. M (=3) independent proposals.
  2. TER, GER, tcpCapSim per proposal; each metric keeps its best proposal.
  3. tcpCapSim value agreement is judged by an LLM on {0, .25, .5, .75, 1}.

The paper used DeepSeek-V4-Flash (thinking) for both LLM roles. Any
OpenAI-compatible chat endpoint works here; put LLM_BASE_URL, LLM_API_KEY,
LLM_MODEL (and optionally LLM_EXTRA) in a .env file (see .env.example), or
export them, or pass the matching flags. `--grouper heuristic --judge exact` runs
fully offline (NOT the paper's protocol; for smoke tests only).

All LLM replies are cached in <workdir>/llm_cache.jsonl, so reruns are cheap
and an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tac_metrics as tm

WINDOW_STEP = 60.0
WINDOW_LOOKBACK = 15.0
SCORE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)

# Attribute keys shown to the grouping LLM, per event type.
GROUPING_KEYS = {
    "speech": ["文字转写", "说话人身份", "说话人性别"],
    "music": ["乐器列表", "歌词转写", "声部类型", "音乐情绪"],
    "sound_event": ["声音事件识别"],
}


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------

def load_jsonl(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r
    return out


# --------------------------------------------------------------------------
# LLM client with on-disk cache
# --------------------------------------------------------------------------

class LLM:
    def __init__(self, base_url, api_key, model, cache_path, extra=None, timeout=900):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.extra = extra or {}
        self.timeout = timeout
        self.cache_path = Path(cache_path)
        self.lock = threading.Lock()
        self.cache = {}
        self.calls = 0
        if self.cache_path.exists():
            with self.cache_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.cache[r["k"]] = r["v"]
                    except (json.JSONDecodeError, KeyError):
                        pass

    def chat(self, messages, temperature=0.0, tag=""):
        key = hashlib.sha256(json.dumps(
            [self.model, messages, temperature, tag, self.extra],
            ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        body = {"model": self.model, "messages": messages, "temperature": temperature}
        body.update(self.extra)
        data = json.dumps(body).encode()
        last = None
        for attempt in range(6):
            req = urllib.request.Request(
                self.base_url + "/chat/completions", data=data,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read())
                text = out["choices"][0]["message"]["content"] or ""
                break
            except (OSError, http.client.HTTPException, KeyError, IndexError, json.JSONDecodeError) as e:
                last = e
                time.sleep(min(60, 2 ** attempt * 2))
        else:
            raise RuntimeError(f"LLM request failed after retries: {last}")
        with self.lock:
            self.calls += 1
            self.cache[key] = text
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"k": key, "v": text}, ensure_ascii=False) + "\n")
        return text


def extract_json(text):
    text = re.sub(r"(?s)<think>.*?</think>", "", text)
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        raise ValueError("no JSON object in reply")
    return json.loads(text[s:e + 1])


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------

GROUP_SYSTEM = """You align audio events from a human REFERENCE annotation (ids R*) with a model PREDICTION (ids P*) of the same recording.
Group together events that describe the same occurrence: the same utterance, the same music passage, or the same occurrence of a sound.
Rules:
- Each listed event belongs to at most one group. Leave out events with no counterpart; they become singleton groups.
- A group may hold several R and/or several P events when one side split or merged the occurrence.
- Repeated occurrences from the same speaker or sound source are different occurrences; keep them in separate groups.
- Decide from timing and content (type, transcript, speaker, sound source, instruments). Times are seconds from the start of the recording.
Reply with JSON only: {"groups": [{"R": ["R3"], "P": ["P1", "P2"]}]}"""


def describe(ev, iv, tag):
    a = ev.get("attributes") or {}
    t = a.get("type", "?")
    parts = []
    for k in GROUPING_KEYS.get(t, []):
        v = a.get(k)
        if not tm._is_empty(v):
            s = tm._render(v)
            parts.append(f"{k}={s[:100]}")
    return f"{tag} [{iv[0]:.2f}-{iv[1]:.2f}] {t} | " + "; ".join(parts)


def windows(duration):
    n = max(1, math.ceil(duration / WINDOW_STEP))
    for k in range(n):
        yield (k * WINDOW_STEP - WINDOW_LOOKBACK, (k + 1) * WINDOW_STEP)


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def groups_from_uf(uf, n_ref, n_hyp):
    comps = {}
    for i in range(n_ref):
        if ("r", i) in uf.p:
            comps.setdefault(uf.find(("r", i)), tm.Group()).ref.append(i)
    for j in range(n_hyp):
        if ("h", j) in uf.p:
            comps.setdefault(uf.find(("h", j)), tm.Group()).hyp.append(j)
    return tm.complete_groups(list(comps.values()), n_ref, n_hyp)


def heuristic_window_groups(r_idx, h_idx, ref_ev, hyp_ev, ref_iv, hyp_iv):
    """Offline stand-in: each prediction joins the same-type reference it overlaps most."""
    pairs = []
    for j in h_idx:
        t = (hyp_ev[j].get("attributes") or {}).get("type")
        best, bw = None, 0.0
        for i in r_idx:
            if (ref_ev[i].get("attributes") or {}).get("type") != t:
                continue
            w = tm.overlap(ref_iv[i], hyp_iv[j])
            if w > bw:
                best, bw = i, w
        if best is not None:
            pairs.append(([best], [j]))
    return pairs


def llm_window_groups(llm, r_idx, h_idx, ref_ev, hyp_ev, ref_iv, hyp_iv, proposal, temperature, key, w):
    if not r_idx or not h_idx:
        return [], False
    lines = [describe(ref_ev[i], ref_iv[i], f"R{n}") for n, i in enumerate(r_idx)]
    lines += [describe(hyp_ev[j], hyp_iv[j], f"P{n}") for n, j in enumerate(h_idx)]
    user = (f"Window {w[0]:.0f}s to {w[1]:.0f}s. Events overlapping this window "
            f"(full intervals):\n" + "\n".join(lines))
    msgs = [{"role": "system", "content": GROUP_SYSTEM}, {"role": "user", "content": user}]
    for attempt in range(3):
        tag = f"group|{key}|{w[0]}|m{proposal}|try{attempt}"
        try:
            obj = extract_json(llm.chat(msgs, temperature=temperature, tag=tag))
            pairs = []
            for g in obj.get("groups", []):
                rs = [r_idx[int(x[1:])] for x in g.get("R", []) if re.fullmatch(r"R\d+", str(x)) and int(x[1:]) < len(r_idx)]
                hs = [h_idx[int(x[1:])] for x in g.get("P", []) if re.fullmatch(r"P\d+", str(x)) and int(x[1:]) < len(h_idx)]
                if rs or hs:
                    pairs.append((rs, hs))
            return pairs, False
        except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
            continue
    # three unparseable replies: fall back to the heuristic for this window
    return heuristic_window_groups(r_idx, h_idx, ref_ev, hyp_ev, ref_iv, hyp_iv), True


def group_recording(key, ref_ev, hyp_ev, ref_iv, hyp_iv, duration, args, llm, pool, proposal):
    span_end = max([duration] + [iv[1] for iv in ref_iv] + [iv[1] for iv in hyp_iv])
    jobs = []
    for w in windows(span_end):
        r_idx = [i for i, iv in enumerate(ref_iv) if iv[1] > w[0] and iv[0] < w[1] or (iv[0] == iv[1] and w[0] <= iv[0] < w[1])]
        h_idx = [j for j, iv in enumerate(hyp_iv) if iv[1] > w[0] and iv[0] < w[1] or (iv[0] == iv[1] and w[0] <= iv[0] < w[1])]
        if args.grouper == "heuristic":
            jobs.append(((heuristic_window_groups(r_idx, h_idx, ref_ev, hyp_ev, ref_iv, hyp_iv), False), None))
        else:
            jobs.append((None, pool.submit(llm_window_groups, llm, r_idx, h_idx, ref_ev, hyp_ev,
                                           ref_iv, hyp_iv, proposal, args.group_temperature, key, w)))
    uf = UnionFind()
    fallbacks = 0
    for res, fut in jobs:
        pairs, fb = res if fut is None else fut.result()
        fallbacks += fb
        for rs, hs in pairs:
            nodes = [("r", i) for i in rs] + [("h", j) for j in hs]
            for n in nodes:
                uf.find(n)
            for n in nodes[1:]:
                uf.union(nodes[0], n)
    return groups_from_uf(uf, len(ref_ev), len(hyp_ev)), fallbacks


# --------------------------------------------------------------------------
# Value judging for tcpCapSim
# --------------------------------------------------------------------------

JUDGE_SYSTEM = """You score the semantic agreement between a reference value and a predicted value of one attribute of an audio event.
Attribute names and values are mostly Chinese; transcripts keep the spoken language. Pooled values are separated by " | ".
Scale: 1 = same meaning; 0.75 = mostly consistent, minor differences in detail; 0.5 = partly consistent; 0.25 = slightly related; 0 = contradictory or unrelated.
For transcripts and lyrics, judge how much of the content agrees.
Reply with JSON only: {"scores": [s0, s1, ...]}, one score per item, in order."""


def norm(s):
    return re.sub(r"\s+", "", s).lower()


def snap(x):
    x = min(1.0, max(0.0, float(x)))
    return min(SCORE_LEVELS, key=lambda v: abs(v - x))


def judge_batch(llm, batch, max_chars):
    items = [json.dumps({"i": n, "key": k, "ref": r[:max_chars], "pred": h[:max_chars]}, ensure_ascii=False)
             for n, (k, r, h) in enumerate(batch)]
    msgs = [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"{len(batch)} items:\n" + "\n".join(items)}]
    for attempt in range(3):
        try:
            obj = extract_json(llm.chat(msgs, temperature=0.0, tag=f"judge|try{attempt}"))
            scores = obj["scores"]
            if len(scores) != len(batch):
                raise ValueError("length mismatch")
            return [snap(s) for s in scores], False
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    if len(batch) > 1:  # split and retry smaller batches
        mid = len(batch) // 2
        a, fa = judge_batch(llm, batch[:mid], max_chars)
        b, fb = judge_batch(llm, batch[mid:], max_chars)
        return a + b, fa or fb
    return [0.0], True


def judge_all(pairs, args, llm, pool):
    """Return {(key, ref, hyp): score} for all unique pairs."""
    scores = {}
    todo = []
    for p in set(pairs):
        if norm(p[1]) == norm(p[2]):
            scores[p] = 1.0
        elif args.judge == "exact":
            scores[p] = 0.0
        else:
            todo.append(p)
    todo.sort()
    failures = 0
    if todo:
        bs = args.judge_batch
        batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]
        futs = [pool.submit(judge_batch, llm, b, args.max_value_chars) for b in batches]
        for b, f in zip(batches, futs):
            s, failed = f.result()
            failures += failed
            for p, v in zip(b, s):
                scores[p] = v
    return scores, failures


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def prepare(rec, max_seconds):
    evs = rec.get("events", []) if rec else []
    if max_seconds is not None:
        evs = [e for e in evs if float(e["start"]) < max_seconds]
    ivs = [tm.interval_of(e) for e in evs]
    reversed_n = sum(1 for e in evs if float(e["end"]) < float(e["start"]))
    return evs, ivs, reversed_n


def load_env_file(path):
    """Load KEY=VALUE lines into os.environ; variables already set win.

    Supports comments (#), blank lines, an optional `export ` prefix and
    single or double quotes around the value. Returns the keys it set.
    """
    p = Path(path)
    if not p.is_file():
        return []
    loaded = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        elif " #" in v:  # inline comment on an unquoted value
            v = v.split(" #", 1)[0].rstrip()
        if k and k not in os.environ:
            os.environ[k] = v
            loaded.append(k)
    return loaded


def main(argv=None):
    # Read --env-file before building the parser, since flag defaults come
    # from the environment. Default: .env in the current directory, then
    # next to this script.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file")
    known, _ = pre.parse_known_args(argv)
    if known.env_file:
        if not Path(known.env_file).is_file():
            sys.exit(f"error: env file not found: {known.env_file}")
        env_path = known.env_file
    else:
        env_path = next((p for p in (Path(".env"), Path(__file__).with_name(".env")) if p.is_file()), None)
    if env_path:
        loaded = load_env_file(env_path)
        print(f"[env] {env_path}: loaded {', '.join(loaded) or 'nothing new'}", file=sys.stderr)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-file", help="KEY=VALUE file with LLM_* settings (default: ./.env, then next to this script)")
    ap.add_argument("--ref", required=True, help="references.jsonl")
    ap.add_argument("--hyp", required=True, help="predictions.jsonl (LongCap output format)")
    ap.add_argument("--workdir", required=True, help="cache, groupings and report go here")
    ap.add_argument("--keys", nargs="*", help="only score these recording keys")
    ap.add_argument("--max-seconds", type=float, help="only events starting before this time (cheap trial runs)")
    ap.add_argument("--grouper", choices=["llm", "heuristic"], default="llm")
    ap.add_argument("--judge", choices=["llm", "exact"], default="llm")
    ap.add_argument("--proposals", type=int, default=3, help="number of grouping proposals (paper: 3)")
    ap.add_argument("--group-temperature", type=float, default=0.7)
    ap.add_argument("--judge-batch", type=int, default=40)
    ap.add_argument("--max-value-chars", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    ap.add_argument("--llm-extra", default=os.environ.get("LLM_EXTRA", "{}"),
                    help="extra JSON merged into every request body, e.g. thinking options")
    args = ap.parse_args(argv)

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    if args.grouper == "heuristic":
        args.proposals = 1
    llm = None
    if args.grouper == "llm" or args.judge == "llm":
        key = os.environ.get("LLM_API_KEY")
        if not key or not args.model:
            ap.error("LLM mode needs LLM_API_KEY and --model/LLM_MODEL (or use --grouper heuristic --judge exact)")
        llm = LLM(args.base_url, key, args.model, work / "llm_cache.jsonl", json.loads(args.llm_extra))

    refs = load_jsonl(args.ref)
    hyps = load_jsonl(args.hyp)
    keys = args.keys or sorted(refs)
    missing = [k for k in keys if k not in hyps]
    if missing:
        print(f"warning: no prediction for {missing}; scored as empty", file=sys.stderr)

    pool = ThreadPoolExecutor(args.workers)
    data = {}
    all_pairs = []
    diag = {"group_fallback_windows": 0, "reversed_ref": 0, "reversed_hyp": 0}
    for k in keys:
        ref_ev, ref_iv, rr = prepare(refs[k], args.max_seconds)
        hyp_ev, hyp_iv, rh = prepare(hyps.get(k), args.max_seconds)
        diag["reversed_ref"] += rr
        diag["reversed_hyp"] += rh
        duration = float(refs[k]["duration"])
        if args.max_seconds is not None:
            duration = min(duration, args.max_seconds)
        per_m = []
        for m in range(args.proposals):
            print(f"[group] {k} proposal {m + 1}/{args.proposals}", file=sys.stderr, flush=True)
            groups, fb = group_recording(k, ref_ev, hyp_ev, ref_iv, hyp_iv, duration, args, llm, pool, m)
            diag["group_fallback_windows"] += fb
            pairs, n_rk, n_hk = tm.capsim_items(groups, ref_ev, hyp_ev, ref_iv, hyp_iv)
            all_pairs += pairs
            per_m.append({"groups": groups, "pairs": pairs, "n_ref_keys": n_rk, "n_hyp_keys": n_hk})
            with (work / f"groups_{k}_m{m}.json").open("w", encoding="utf-8") as f:
                json.dump([{"ref": g.ref, "hyp": g.hyp} for g in groups], f)
        data[k] = (ref_iv, hyp_iv, per_m)

    print(f"[judge] {len(set(all_pairs))} unique value pairs", file=sys.stderr, flush=True)
    scores, judge_fail = judge_all(all_pairs, args, llm, pool)
    diag["judge_failed_items"] = judge_fail
    pool.shutdown()

    total = {"ter": {"miss": 0.0, "fa": 0.0, "conf": 0.0, "ref": 0.0},
             "ger": {"overlap": 0.0, "match": 0.0, "dup": 0.0, "ref_events": 0.0},
             "credit": 0.0, "n_ref_keys": 0, "n_hyp_keys": 0}
    per_proposal_tot = [json.loads(json.dumps(total)) for _ in range(args.proposals)]
    report = {"recordings": {}}
    for k, (ref_iv, hyp_iv, per_m) in data.items():
        cands = []
        for m, d in enumerate(per_m):
            ter = tm.ter_components(d["groups"], ref_iv, hyp_iv)
            ger = tm.ger_components(d["groups"], ref_iv, hyp_iv)
            credit = sum(scores[p] for p in d["pairs"])
            r = tm.rates(ter, ger, credit, d["n_ref_keys"], d["n_hyp_keys"])
            cands.append((ter, ger, credit, d["n_ref_keys"], d["n_hyp_keys"], r))
            pt = per_proposal_tot[m]
            for a in ter:
                pt["ter"][a] += ter[a]
            for a in ger:
                pt["ger"][a] += ger[a]
            pt["credit"] += credit
            pt["n_ref_keys"] += d["n_ref_keys"]
            pt["n_hyp_keys"] += d["n_hyp_keys"]
        bt = min(cands, key=lambda c: c[5]["TER"])
        bg = min(cands, key=lambda c: c[5]["GER"])
        bc = max(cands, key=lambda c: c[5]["tcpCapSim"])
        for a in bt[0]:
            total["ter"][a] += bt[0][a]
        for a in bg[1]:
            total["ger"][a] += bg[1][a]
        total["credit"] += bc[2]
        total["n_ref_keys"] += bc[3]
        total["n_hyp_keys"] += bc[4]
        report["recordings"][k] = {
            "TER": bt[5]["TER"], "GER": bg[5]["GER"], "tcpCapSim": bc[5]["tcpCapSim"],
            "capsim_precision": bc[5]["capsim_precision"], "capsim_recall": bc[5]["capsim_recall"],
            "ter_components": bt[0], "ger_components": bg[1],
            "per_proposal": [c[5] for c in cands]}

    report["micro"] = tm.rates(total["ter"], total["ger"], total["credit"], total["n_ref_keys"], total["n_hyp_keys"])
    report["micro_components"] = total
    report["per_proposal_micro"] = [tm.rates(p["ter"], p["ger"], p["credit"], p["n_ref_keys"], p["n_hyp_keys"])
                                    for p in per_proposal_tot]
    report["diagnostics"] = diag | {"llm_calls_this_run": llm.calls if llm else 0}
    report["settings"] = {k: v for k, v in vars(args).items() if k not in {"llm_extra", "env_file"}}
    with (work / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'recording':28s} {'TER%':>8s} {'GER%':>8s} {'tcpCapSim%':>11s}")
    for k, r in report["recordings"].items():
        print(f"{k:28s} {100 * r['TER']:8.2f} {100 * r['GER']:8.2f} {100 * r['tcpCapSim']:11.2f}")
    mi = report["micro"]
    print(f"{'MICRO':28s} {100 * mi['TER']:8.2f} {100 * mi['GER']:8.2f} {100 * mi['tcpCapSim']:11.2f}")
    print(f"\nreport: {work / 'report.json'}  diagnostics: {report['diagnostics']}")


if __name__ == "__main__":
    main()
