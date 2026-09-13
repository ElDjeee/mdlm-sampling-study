import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import itertools
import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from nanodiff.model import ModelConfig,init_params,forward,empty_cache,decode_step
from nanodiff.diffusion import corruption,mask_probability,weighted_loss,objective,diffusion_sample,ar_sample,draw_tokens
from nanodiff.training import init_state,make_train_step,batch_at,save_checkpoint,load_checkpoint
from nanodiff.data import load_data,evaluation_windows


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = ModelConfig(5,16,1,2)
        cls.key = jax.random.PRNGKey(13)
        cls.params = init_params(cls.key,cls.cfg)
        cls.x = jnp.array([[0,1,2,3],[1,2,0,4]],jnp.int32)

    def test_zero_noise(self):
        noisy,mask=corruption(self.key,self.x,0.,self.cfg)
        np.testing.assert_array_equal(noisy,self.x)
        self.assertFalse(np.any(mask))

    def test_complete_noise(self):
        noisy,mask=corruption(self.key,self.x,1.,self.cfg)
        self.assertTrue(np.all(mask)); self.assertTrue(np.all(noisy==self.cfg.mask_id))

    def test_special_tokens_never_corrupted(self):
        x=jnp.array([[0,self.cfg.pad_id,self.cfg.bos_id]])
        noisy,mask=corruption(self.key,x,1.,self.cfg)
        np.testing.assert_array_equal(mask,[[True,False,False]])
        np.testing.assert_array_equal(noisy[0,1:],x[0,1:])

    def test_statistical_mask_rate(self):
        _,mask=corruption(self.key,jnp.zeros((1000,100),jnp.int32),.37,self.cfg)
        self.assertLess(abs(float(mask.mean())-.37),.006)

    def test_schedule_endpoints_monotonicity(self):
        for schedule in ['linear','cosine','log']:
            m=np.array(mask_probability(jnp.linspace(0,1,101),schedule))
            self.assertAlmostEqual(float(m[0]),0.,places=6)
            self.assertAlmostEqual(float(m[-1]),1.,places=6)
            self.assertTrue(np.all(np.diff(m)>0))

    def test_loss_only_masked_valid(self):
        x=jnp.array([[0,1,self.cfg.pad_id]])
        mask=jnp.array([[True,False,True]])
        logits=jnp.zeros((1,3,5))
        base=weighted_loss(logits,x,mask,jnp.ones(1),self.cfg)
        changed=logits.at[0,1:,0].set(100.)
        self.assertAlmostEqual(float(base),float(weighted_loss(changed,x,mask,jnp.ones(1),self.cfg)))
        self.assertAlmostEqual(float(base),np.log(5)/2,places=6)

    def test_empty_mask_has_zero_loss(self):
        value=weighted_loss(jnp.zeros((2,4,5)),self.x,jnp.zeros_like(self.x,bool),jnp.ones(2),self.cfg)
        self.assertEqual(float(value),0.)

    def test_exact_expected_uniform_nelbo(self):
        # Enumerate every Bernoulli mask on two tokens. E[NELBO]=log(V),
        # independent of schedule and T. This catches masked-count normalization.
        cfg=ModelConfig(2,8,1,2)
        x=jnp.zeros((1,2),jnp.int32)
        for schedule in ['linear','cosine','log']:
            total=0.
            for k in range(1,5):
                m=float(mask_probability(jnp.array(k/4),schedule))
                prev=float(mask_probability(jnp.array((k-1)/4),schedule))
                for pattern in itertools.product([False,True],repeat=2):
                    count=sum(pattern)
                    prob=m**count*(1-m)**(2-count)
                    loss=weighted_loss(jnp.zeros((1,2,2)),x,jnp.array([pattern]),jnp.array([(m-prev)/m]),cfg)
                    total+=prob*float(loss)
            self.assertAlmostEqual(total,np.log(2),places=6)

    def test_causal_has_no_future_leak(self):
        a=forward(self.params,self.x,self.cfg,True)
        b=forward(self.params,self.x.at[:,-1].set(0),self.cfg,True)
        np.testing.assert_allclose(a[:,:-1],b[:,:-1],atol=1e-6)

    def test_bidirectional_uses_future(self):
        a=forward(self.params,self.x,self.cfg)
        b=forward(self.params,self.x.at[:,-1].set(0),self.cfg)
        self.assertGreater(float(jnp.max(jnp.abs(a[:,0]-b[:,0]))),1e-5)

    def test_padding_keys_do_not_affect_valid_logits(self):
        x=jnp.array([[0,1,2,self.cfg.pad_id]])
        a=forward(self.params,x,self.cfg)
        params=dict(self.params,embedding=self.params['embedding'].at[self.cfg.pad_id].set(100.))
        b=forward(params,x,self.cfg)
        np.testing.assert_allclose(a[:,:3],b[:,:3],atol=1e-6)

    def test_cache_matches_full_forward(self):
        full=forward(self.params,self.x,self.cfg,True)
        cache=empty_cache(self.cfg,2,4)
        outputs=[]
        for i in range(4):
            logits,cache=decode_step(self.params,self.x[:,i],jnp.array(i),cache,self.cfg)
            outputs.append(logits)
        np.testing.assert_allclose(jnp.stack(outputs,1),full,atol=2e-6,rtol=2e-5)

    def test_cache_with_local_distance_prior(self):
        cfg=ModelConfig(5,16,1,2,.1,True)
        full=forward(self.params,self.x,cfg,True)
        cache=empty_cache(cfg,2,4)
        outputs=[]
        for i in range(4):
            logits,cache=decode_step(self.params,self.x[:,i],jnp.array(i),cache,cfg)
            outputs.append(logits)
        np.testing.assert_allclose(jnp.stack(outputs,1),full,atol=2e-6,rtol=2e-5)

    def test_sampler_invariants_all_strategies(self):
        initial=jnp.full_like(self.x,self.cfg.mask_id).at[:,0].set(self.x[:,0])
        for strategy in ['ancestral','random','confidence']:
            for schedule in ['linear','cosine','log']:
                final,trace=diffusion_sample(self.params,self.key,initial,self.cfg,4,strategy,schedule)
                self.assertTrue(np.all(np.asarray(final)<self.cfg.vocab_size))
                np.testing.assert_array_equal(final[:,0],self.x[:,0])
                counts=np.sum(np.asarray(trace)==self.cfg.mask_id,axis=-1)
                self.assertTrue(np.all(np.diff(counts,axis=0)<=0))
                prior=np.concatenate([np.asarray(initial)[None],np.asarray(trace)[:-1]],0)
                unmasked=prior!=self.cfg.mask_id
                np.testing.assert_array_equal(np.asarray(trace)[unmasked],prior[unmasked])

    def test_sampler_seed_reproducible(self):
        x=jnp.full_like(self.x,self.cfg.mask_id)
        a,_=diffusion_sample(self.params,self.key,x,self.cfg,4)
        b,_=diffusion_sample(self.params,self.key,x,self.cfg,4)
        np.testing.assert_array_equal(a,b)

    def test_all_observed_context_preserved(self):
        result,_=diffusion_sample(self.params,self.key,self.x,self.cfg,4)
        np.testing.assert_array_equal(result,self.x)

    def test_ar_sampling_valid_and_reproducible(self):
        a=ar_sample(self.params,self.key,self.cfg,2,8)
        b=ar_sample(self.params,self.key,self.cfg,2,8)
        np.testing.assert_array_equal(a,b)
        self.assertTrue(np.all((np.asarray(a)>=0)&(np.asarray(a)<self.cfg.vocab_size)))

    def test_sampling_greedy_and_top_k(self):
        logits=jnp.array([[[1.,2.,4.,3.,0.]]])
        self.assertEqual(int(draw_tokens(self.key,logits,0.)[0,0]),2)
        self.assertEqual(int(draw_tokens(self.key,logits,1.,1)[0,0]),2)

    def test_all_objectives_finite_gradients(self):
        for kind in ['ar','mdlm','mlm','unweighted']:
            value,grads=jax.value_and_grad(objective)(self.params,self.x,self.key,self.cfg,kind)
            self.assertTrue(np.isfinite(value))
            self.assertTrue(all(np.all(np.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)))

    def test_checkpoint_resume_exact(self):
        cfg=self.cfg
        state=init_state(self.key,cfg)
        step=make_train_step(cfg,'mdlm','linear',8,.001,10)
        state,_=step(state,self.x)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ckpt.npz'
            save_checkpoint(path,state,cfg,{'example':1})
            restored,other_cfg,meta=load_checkpoint(path)
        self.assertEqual(cfg,other_cfg); self.assertEqual(meta,{'example':1})
        np.testing.assert_array_equal(forward(state['params'],self.x,cfg),forward(restored['params'],self.x,cfg))
        a,_=step(state,self.x); b,_=step(restored,self.x)
        for left,right in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
            np.testing.assert_array_equal(left,right)

    def test_fixed_batch_overfit(self):
        cfg=ModelConfig(2,16,1,2)
        state=init_state(self.key,cfg)
        x=jnp.tile(jnp.array([[0,1,0,1,0,1,0,1]],jnp.int32),(4,1))
        # Force all-mask reconstruction: even hardest input must learn this
        # deterministic position-dependent sequence, not just copy visible data.
        noisy=jnp.full_like(x,cfg.mask_id)
        def loss_fn(p):
            return weighted_loss(forward(p,noisy,cfg),x,jnp.ones_like(x,bool),jnp.ones(4),cfg)
        from nanodiff.training import update
        @jax.jit
        def step(s):
            loss,grads=jax.value_and_grad(loss_fn)(s['params'])
            s,_=update(s,grads,.01,0.)
            return s,loss
        before=float(loss_fn(state['params']))
        for _ in range(100): state,loss=step(state)
        after=float(loss_fn(state['params']))
        self.assertLess(after,.08); self.assertLess(after,before/5)

    def test_deterministic_batches(self):
        data=np.arange(100,dtype=np.int32)
        np.testing.assert_array_equal(batch_at(data,1,2,4,8),batch_at(data,1,2,4,8))
        self.assertFalse(np.array_equal(batch_at(data,1,2,4,8),batch_at(data,1,3,4,8)))

    def test_no_split_overlap_and_train_only_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'text.txt'
            path.write_text('a'*900+'b'*50+'c'*50)
            data,chars,meta=load_data(path,100)
        self.assertEqual(chars,['a'])
        self.assertEqual(meta['used_characters'],dict(train=100,val=50,test=50))
        self.assertEqual(meta['unknown_counts'],dict(train=0,val=50,test=50))
        np.testing.assert_array_equal(data['test'],np.ones(50))

    def test_evaluation_windows_disjoint(self):
        windows=evaluation_windows(np.arange(1000),10,20)
        self.assertEqual(len(set(windows.reshape(-1).tolist())),windows.size)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError): ModelConfig(5,15,1,2)
        with self.assertRaises(ValueError): mask_probability(.5,'invalid')


if __name__=='__main__': unittest.main()
