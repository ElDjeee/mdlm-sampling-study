import unittest
import numpy as np
from nanodiff.metrics import js_divergence,sample_metrics,NGram


class MetricTests(unittest.TestCase):
    def test_js_identical(self):
        self.assertAlmostEqual(js_divergence({'a':2,'b':1},{'a':4,'b':2}),0.)

    def test_js_disjoint(self):
        self.assertAlmostEqual(js_divergence({'a':1},{'b':1}),1.)

    def test_js_symmetric(self):
        a,b={'a':3,'b':1},{'b':4,'c':8}
        self.assertAlmostEqual(js_divergence(a,b),js_divergence(b,a))

    def test_copy_detection(self):
        result=sample_metrics(['abcdefgh','xxxxxxxx'],'abcdefghzz','00abcdefgh00',4)
        self.assertEqual(result['exact_copy_fraction'],.5)
        self.assertEqual(result['copied_4gram_fraction'],.5)

    def test_character_distinct_not_semantics(self):
        result=sample_metrics(['abcdabcd'],'abcd','abcdabcd')
        self.assertEqual(result['distinct_1'],.5)
        self.assertGreaterEqual(result['repetition_4'],0.)

    def test_ngram_probabilities_normalized(self):
        model=NGram(np.array([0,1]*64),3,16)
        np.testing.assert_allclose(model.probs.sum(-1),1.)
        self.assertAlmostEqual(model.unigram.sum(),1.)

    def test_ngram_learns_alternating_pattern(self):
        model=NGram(np.array([0,1]*1000),2,16)
        windows=np.array([[0,1]*8])
        self.assertLess(float(model.bpc(windows).mean()),.1)
        self.assertGreater(float(model.bpc(windows,True).mean()),.9)

    def test_ngram_seed_and_range(self):
        model=NGram(np.array([0,1]*64),3,16)
        a=model.sample(0,4,32)
        np.testing.assert_array_equal(a,model.sample(0,4,32))
        self.assertTrue(np.all((a>=0)&(a<3)))


if __name__=='__main__': unittest.main()
