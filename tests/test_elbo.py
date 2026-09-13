"""Independent, exactly enumerable two-token reference model."""
import itertools
import unittest
import jax.numpy as jnp
import numpy as np
from nanodiff.diffusion import weighted_loss
from nanodiff.model import ModelConfig


class ExactReferenceTests(unittest.TestCase):
    def test_context_dependent_nelbo_bounds_exact_nll(self):
        # Reverse process has T=2, mask masses 1 -> 1/2 -> 0.
        # Enumerate every state and compare against a normalized exact model.
        mask=2
        cfg=ModelConfig(2,8,1,2)
        states=list(itertools.product(range(3),repeat=2))
        def prediction(state):
            rows=[]
            for i in range(2):
                neighbor=state[1-i]
                p0=.8 if neighbor==0 else (.2 if neighbor==1 else .6)
                rows.append([p0,1-p0])
            return np.array(rows)
        def transition(source,dest,reveal):
            p=prediction(source);probability=1.
            for i in range(2):
                if source[i]!=mask:
                    probability*=float(dest[i]==source[i])
                elif dest[i]==mask:
                    probability*=1-reveal
                else:
                    probability*=reveal*p[i,dest[i]]
            return probability
        all_probabilities=[]
        for target in itertools.product(range(2),repeat=2):
            exact_p=sum(transition((mask,mask),z,.5)*transition(z,target,1.) for z in states)
            all_probabilities.append(exact_p)
            nelbo=0.
            for previous,m in [(0.,.5),(.5,1.)]:
                for masked in itertools.product([False,True],repeat=2):
                    count=sum(masked)
                    q=m**count*(1-m)**(2-count)
                    z=tuple(mask if masked[i] else target[i] for i in range(2))
                    logits=jnp.array(np.log(prediction(z)))[None]
                    per_char=weighted_loss(logits,jnp.array([target]),jnp.array([masked]),jnp.array([(m-previous)/m]),cfg)
                    nelbo+=q*float(per_char)*2
            self.assertGreaterEqual(nelbo+1e-6,-np.log(exact_p))
            self.assertTrue(np.isfinite(nelbo))
        self.assertAlmostEqual(sum(all_probabilities),1.,places=12)


if __name__=='__main__':unittest.main()
