"""Independent exact enumeration of a finite absorbing-mask generative chain.

Tiny binary benchmark x=(b0,b1,b2,b3,b0,b1,b2,b3), with iid fair bits b.
Enumerating 3^8 noisy states and 5^8 allowed source/destination edges gives
the entire learned output distribution, without estimating it by sampling.
No oracle is used to train the neural denoiser or to generate its outputs.
"""
import numpy as np


def digits(indices,base,length):
    return (np.asarray(indices)[:,None]//(base**np.arange(length))[None,:])%base


class ExactCopy:
    def __init__(self,pairs=4):
        if pairs not in (1,2,3,4):
            raise ValueError('Exact enumeration is limited to 1..4 pairs')
        self.pairs=pairs
        self.length=2*pairs
        self.states=digits(np.arange(3**self.length),3,self.length).astype(np.int32)
        choices=digits(np.arange(5**self.length),5,self.length)
        source_digits=np.array([0,1,2,2,2])[choices]
        dest_digits=np.array([0,1,0,1,2])[choices]
        self.source=(source_digits*3**np.arange(self.length)).sum(-1)
        self.dest=(dest_digits*3**np.arange(self.length)).sum(-1)
        self.revealed=(source_digits==2)&(dest_digits!=2)
        self.dest_bits=np.minimum(dest_digits,1)
        self.num_revealed=self.revealed.sum(-1)
        self.num_remaining=(dest_digits==2).sum(-1)
        self.binary=digits(np.arange(2**self.length),2,self.length).astype(np.int32)
        self.final_ids=(self.binary*3**np.arange(self.length)).sum(-1)
        self.valid=(self.binary[:,:pairs]==self.binary[:,pairs:]).all(-1)
        self.target=self.valid.astype(float)/2**pairs

    def oracle_probabilities(self):
        partner=np.roll(self.states,self.pairs,axis=1)
        p0=np.where(partner==2,.5,(partner==0).astype(float))
        return np.stack([p0,1-p0],axis=-1)

    def output_distribution(self,probabilities,steps=64):
        probabilities=np.asarray(probabilities,np.float64)
        if probabilities.shape!=(len(self.states),self.length,2):
            raise ValueError('Invalid denoiser probability shape')
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities<0):
            raise ValueError('Invalid probability values')
        if not np.allclose(probabilities.sum(-1),1.,atol=1e-6):
            raise ValueError('Unnormalized denoiser')
        probabilities=probabilities/probabilities.sum(-1,keepdims=True)
        edge=probabilities[self.source[:,None],np.arange(self.length)[None,:],self.dest_bits]
        clean_factor=np.where(self.revealed,edge,1.).prod(-1)
        mass=np.zeros(len(self.states),np.float64)
        mass[-1]=1.  # Fully masked state: (2,...,2).
        max_mass_error=0.
        for k in range(steps,0,-1):
            reveal=1./k
            transition=clean_factor*(reveal**self.num_revealed)*((1-reveal)**self.num_remaining)
            mass=np.bincount(self.dest,weights=mass[self.source]*transition,minlength=len(self.states))
            max_mass_error=max(max_mass_error,abs(mass.sum()-1.))
        if max_mass_error>1e-8:
            raise ArithmeticError(f'Probability mass not conserved: {max_mass_error}')
        return mass[self.final_ids]

    def metrics(self,distribution):
        p=np.asarray(distribution)
        if not np.isclose(p.sum(),1.,atol=1e-8):raise ValueError('Output mass not 1')
        valid_probs=p[self.valid]
        return dict(valid_probability=float(p[self.valid].sum()),
                    total_variation=float(.5*np.abs(p-self.target).sum()),
                    exact_nll_bits_per_character=float(-np.log2(np.maximum(valid_probs,1e-300)).mean()/self.length),
                    min_valid_mode_probability=float(valid_probs.min()),
                    max_valid_mode_probability=float(valid_probs.max()))

    def sample_distribution(self,samples):
        samples=np.asarray(samples)
        if samples.ndim!=2 or samples.shape[1]!=self.length or not np.all((samples==0)|(samples==1)):
            raise ValueError('Invalid generated binary sequences')
        ids=(samples*2**np.arange(self.length)).sum(-1)
        return np.bincount(ids,minlength=2**self.length)/len(samples)

    def conditional_metrics(self,probabilities):
        partner=np.roll(self.states,self.pairs,axis=1)
        consistent=((self.states[:,:self.pairs]==self.states[:,self.pairs:])|
                    (self.states[:,:self.pairs]==2)|(self.states[:,self.pairs:]==2)).all(-1)
        identifiable=(self.states==2)&(partner!=2)&consistent[:,None]
        uncertain=(self.states==2)&(partner==2)&consistent[:,None]
        accuracy=(np.argmax(probabilities,-1)==partner)[identifiable].mean()
        return dict(determined_mask_accuracy=float(accuracy),
                    ambiguous_mean_absolute_error_from_half=float(np.abs(probabilities[:,:,0][uncertain]-.5).mean()),
                    note='Exhaustive consistent noisy states, uniformly weighted; not unseen modes or language generalization')
