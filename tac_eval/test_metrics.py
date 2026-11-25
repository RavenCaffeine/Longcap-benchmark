"""Unit tests for tac_metrics, including the worked examples in LongCap App. A.2.

Run: python -m unittest test_metrics -v
"""
import unittest

import tac_metrics as tm
from tac_metrics import Group


def ev(s, e, **attrs):
    return {"start": s, "end": e, "attributes": attrs}


class TestGER(unittest.TestCase):
    def rate(self, groups, ref, hyp):
        ri = [tm.interval_of(e) for e in ref]
        hi = [tm.interval_of(e) for e in hyp]
        t = tm.ter_components(groups, ri, hi)
        g = tm.ger_components(groups, ri, hi)
        r = tm.rates(t, g, 0.0, 1, 1)
        return r["TER"], r["GER"]

    def test_split(self):
        # Paper: ref [0,10) split into preds [0,5) and [5,10) -> TER 0, GER 0.5
        ref = [ev(0, 10)]
        hyp = [ev(0, 5), ev(5, 10)]
        ter, ger = self.rate([Group([0], [0, 1])], ref, hyp)
        self.assertAlmostEqual(ter, 0.0)
        self.assertAlmostEqual(ger, 0.5)

    def test_merge(self):
        # Merging two ref events into one prediction gives the same penalty.
        ref = [ev(0, 5), ev(5, 10)]
        hyp = [ev(0, 10)]
        ter, ger = self.rate([Group([0, 1], [0])], ref, hyp)
        self.assertAlmostEqual(ter, 0.0)
        self.assertAlmostEqual(ger, 0.5)

    def test_duplicate(self):
        # Predicting twice for one ref event adds 10 s through T_dup -> GER 1.
        ref = [ev(0, 10)]
        hyp = [ev(0, 10), ev(0, 10)]
        ter, ger = self.rate([Group([0], [0, 1])], ref, hyp)
        self.assertAlmostEqual(ter, 0.0)
        self.assertAlmostEqual(ger, 1.0)

    def test_perfect(self):
        ref = [ev(0, 4), ev(2, 8)]
        hyp = [ev(0, 4), ev(2, 8)]
        ter, ger = self.rate([Group([0], [0]), Group([1], [1])], ref, hyp)
        self.assertAlmostEqual(ter, 0.0)
        self.assertAlmostEqual(ger, 0.0)


class TestTER(unittest.TestCase):
    def test_miss_fa_conf(self):
        ri = [(0, 10), (20, 30)]
        hi = [(5, 10), (20, 30), (40, 45)]
        # group 0: ref0 + hyp0 (miss 5 s); group 1: ref1 alone; group 2: hyp1 alone
        # over [20,30): 1 ref group active, 1 hyp group active, different -> conf 10
        # hyp2 alone in its own group -> fa 5
        groups = [Group([0], [0]), Group([1], []), Group([], [1]), Group([], [2])]
        t = tm.ter_components(groups, ri, hi)
        self.assertAlmostEqual(t["miss"], 5)
        self.assertAlmostEqual(t["conf"], 10)
        self.assertAlmostEqual(t["fa"], 5)
        self.assertAlmostEqual(t["ref"], 20)

    def test_overlapping_refs_in_group_merge(self):
        # Overlapping refs inside one group are merged for T_ref.
        t = tm.ter_components([Group([0, 1], [])], [(0, 6), (4, 10)], [])
        self.assertAlmostEqual(t["ref"], 10)
        self.assertAlmostEqual(t["miss"], 10)


class TestMatching(unittest.TestCase):
    def test_hungarian_prefers_global_optimum(self):
        # Greedy would take the largest edge r0-h1 (5.5) and then r1-h0 (1);
        # here that is also optimal, and must beat r0-h0 + r1-h1 (5 + 0).
        r = [(0, 6), (6, 11)]
        h = [(1, 7), (0, 5.5)]
        # r0-h0:5, r0-h1:5.5, r1-h0:1, r1-h1:0 -> best 5.5+1=6.5 vs 5+0=5
        self.assertAlmostEqual(tm.max_matched_overlap(r, h), 6.5)

    def test_pure_python_matches_bruteforce(self):
        import itertools, random
        rnd = random.Random(0)
        for _ in range(50):
            n, m = rnd.randint(1, 5), rnd.randint(1, 5)
            w = [[rnd.choice([0, rnd.random()]) for _ in range(m)] for _ in range(n)]
            brute = 0.0
            k = min(n, m)
            for rows in itertools.permutations(range(n), k):
                for cols in itertools.permutations(range(m), k):
                    brute = max(brute, sum(w[a][b] for a, b in zip(rows, cols)))
            self.assertAlmostEqual(tm._hungarian_max(w), brute, places=9)


class TestCapSim(unittest.TestCase):
    def test_pooling_and_time_constraint(self):
        ref = [ev(0, 2, type="speech", 文字转写="hi", caption="x"),
               ev(100, 101, type="sound_event", 声音事件识别="狗叫")]
        hyp = [ev(6, 7, type="speech", 文字转写="hi", 说话人性别="男"),
               ev(110, 111, type="sound_event", 声音事件识别="狗叫")]
        ri = [tm.interval_of(e) for e in ref]
        hi = [tm.interval_of(e) for e in hyp]
        groups = [Group([0], [0]), Group([1], [1])]
        pairs, n_r, n_h = tm.capsim_items(groups, ref, hyp, ri, hi)
        # caption excluded; group 0 gap 4 s ok; group 1 gap 9 s fails
        self.assertEqual(n_r, 4)
        self.assertEqual(n_h, 5)
        self.assertEqual(sorted(p[0] for p in pairs), ["type", "文字转写"])
        r = tm.rates({"miss": 0, "fa": 0, "conf": 0, "ref": 1},
                     {"overlap": 0, "match": 0, "dup": 0, "ref_events": 1}, 2.0, n_r, n_h)
        self.assertAlmostEqual(r["tcpCapSim"], (2 / 5 + 2 / 4) / 2)

    def test_key_counted_once_per_group(self):
        ref = [ev(0, 1, type="speech"), ev(1, 2, type="speech")]
        pooled = tm.pool_values(ref)
        self.assertEqual(pooled, {"type": "speech"})


if __name__ == "__main__":
    unittest.main()
