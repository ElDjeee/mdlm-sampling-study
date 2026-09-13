import unittest
import numpy as np
from nanodiff.exact_copy import ExactCopy,digits


class ExactCopyTests(unittest.TestCase):
    def test_state_encoding(self):
        np.testing.assert_array_equal(digits(np.array([0,1,2,3]),3,2),[[0,0],[1,0],[2,0],[0,1]])

    def test_independent_baseline(self):
        ex=ExactCopy(2)
        probs=np.full((len(ex.states),ex.length,2),.5)
        for steps in (1,4,16):
            p=ex.output_distribution(probs,steps)
            np.testing.assert_allclose(p,1/16,atol=1e-12)
            result=ex.metrics(p)
            self.assertAlmostEqual(result['valid_probability'],.25)
            self.assertAlmostEqual(result['total_variation'],.75)
            self.assertAlmostEqual(result['exact_nll_bits_per_character'],1.)

    def test_oracle_matches_closed_form(self):
        for pairs in (1,2):
            ex=ExactCopy(pairs)
            for steps in (1,2,4,16,64):
                result=ex.metrics(ex.output_distribution(ex.oracle_probabilities(),steps))
                valid=(1-1/(2*steps))**pairs
                self.assertAlmostEqual(result['valid_probability'],valid,places=11)
                self.assertAlmostEqual(result['total_variation'],1-valid,places=11)

    def test_exact_target_entropy(self):
        ex=ExactCopy(2)
        result=ex.metrics(ex.target)
        self.assertEqual(result['total_variation'],0.)
        self.assertEqual(result['exact_nll_bits_per_character'],.5)

    def test_mode_collapse_detected(self):
        ex=ExactCopy(2)
        p=np.zeros(16);p[0]=1.
        result=ex.metrics(p)
        self.assertEqual(result['valid_probability'],1.)
        self.assertEqual(result['total_variation'],.75)
        self.assertEqual(result['min_valid_mode_probability'],0.)

    def test_empirical_encoding(self):
        ex=ExactCopy(1)
        p=ex.sample_distribution(np.array([[0,0],[1,1],[0,0],[0,1]]))
        np.testing.assert_array_equal(p,[.5,0,.25,.25])

    def test_rejects_invalid_probabilities(self):
        ex=ExactCopy(1)
        with self.assertRaises(ValueError):ex.output_distribution(np.ones((9,2,2)))

    def test_oracle_conditionals(self):
        ex=ExactCopy(2)
        result=ex.conditional_metrics(ex.oracle_probabilities())
        self.assertEqual(result['determined_mask_accuracy'],1.)
        self.assertEqual(result['ambiguous_mean_absolute_error_from_half'],0.)
