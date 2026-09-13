import unittest
import numpy as np
from nanodiff.sampling_statistics import counts, js_counts, bootstrap_fixed_checkpoints, paired_sign_test


class SamplingStatisticsTests(unittest.TestCase):
    def test_no_cross_sequence_pairs(self):
        np.testing.assert_array_equal(counts([[0,0],[1,1]],2,2),[1,0,0,1])

    def test_lagged_pairs(self):
        np.testing.assert_array_equal(counts([[0,1,0,1]],2,2,2),[1,0,0,1])

    def test_triples_and_count(self):
        actual=counts([[0,1,0,1],[1,0,1,0]],2,3)
        self.assertEqual(actual.sum(),4)
        self.assertEqual(actual[2],2)
        self.assertEqual(actual[5],2)

    def test_js_extremes(self):
        self.assertEqual(js_counts([2,0],[0,3]),1.)
        self.assertEqual(js_counts([2,1],[4,2]),0.)

    def test_matches_independent_counter_implementation(self):
        from collections import Counter
        from nanodiff.metrics import js_divergence
        rng=np.random.default_rng(19)
        a,b=rng.integers(0,4,(7,20)),rng.integers(0,4,(11,20))
        for order,lag in [(1,1),(3,1),(2,4)]:
            def counter(x):
                return Counter(tuple(row[i+j*lag] for j in range(order))
                               for row in x for i in range(len(row)-(order-1)*lag))
            self.assertAlmostEqual(js_counts(counts(a,4,order,lag),counts(b,4,order,lag)),
                                   js_divergence(counter(a),counter(b)),places=13)

    def test_bootstrap_fixed_checkpoints(self):
        result=bootstrap_fixed_checkpoints([[1,1,1],[3,3,3]],100)
        self.assertEqual(result['mean'],2.)
        self.assertEqual(result['ci95'],[2.,2.])

    def test_bootstrap_repeatable(self):
        self.assertEqual(bootstrap_fixed_checkpoints([[1,2,4]],100),bootstrap_fixed_checkpoints([[1,2,4]],100))

    def test_sign_test_exact(self):
        self.assertEqual(paired_sign_test([-1]*5+[0])['p_two_sided'],.0625)
        self.assertEqual(paired_sign_test([-1,1])['p_two_sided'],1.)
        self.assertEqual(paired_sign_test([0])['ties'],1)

    def test_reject_invalid(self):
        with self.assertRaises(ValueError): counts([[0,1]],2,3)
        with self.assertRaises(ValueError): counts([[2,0]],2)
        with self.assertRaises(ValueError): js_counts([0,0],[1,0])
        with self.assertRaises(ValueError): bootstrap_fixed_checkpoints([[np.nan]])
