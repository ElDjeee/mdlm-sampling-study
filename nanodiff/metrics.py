"""Transparent character-level diagnostics; none is a semantic-quality metric."""
from collections import Counter
import numpy as np


def ngrams(text,n):
    return Counter(text[i:i+n] for i in range(max(0,len(text)-n+1)))


def js_divergence(a,b):
    keys=sorted(set(a)|set(b))
    if not keys: return 0.
    p=np.array([a.get(k,0) for k in keys],float)
    q=np.array([b.get(k,0) for k in keys],float)
    p=p/max(p.sum(),1);q=q/max(q.sum(),1);m=(p+q)/2
    def kl(x):
        keep=x>0
        return float(np.sum(x[keep]*np.log2(x[keep]/m[keep])))
    return .5*(kl(p)+kl(q))


def sample_metrics(samples,reference,training,copy_length=32):
    result={}
    for n in (1,2,3):
        counts=Counter()
        for s in samples: counts.update(ngrams(s,n))
        result[f'distinct_{n}']=len(counts)/max(sum(counts.values()),1)
        result[f'js_{n}']=js_divergence(counts,ngrams(reference,n))
    result['repetition_4']=float(np.mean([1-len(ngrams(s,4))/max(len(s)-3,1) for s in samples]))
    result['exact_copy_fraction']=float(np.mean([s in training for s in samples]))
    spans=[s[i:i+copy_length] for s in samples for i in range(max(0,len(s)-copy_length+1))]
    result[f'copied_{copy_length}gram_fraction']=sum(s in training for s in spans)/max(len(spans),1)
    result['sample_count']=len(samples)
    result['generated_characters']=sum(map(len,samples))
    return result


class NGram:
    """Additive-smoothed order-2 Markov model (character trigrams)."""
    def __init__(self,training,vocab_size,length=128,smoothing=.1):
        self.vocab_size=vocab_size
        self.bos=vocab_size
        counts=np.full((vocab_size+1,vocab_size+1,vocab_size),smoothing,np.float64)
        # Train using the same BOS/chunk convention as the neural likelihood.
        chunks=training[:len(training)//length*length].reshape(-1,length)
        prior=np.concatenate([np.full((len(chunks),2),self.bos),chunks],-1)
        np.add.at(counts,(prior[:,:-2].reshape(-1),prior[:,1:-1].reshape(-1),chunks.reshape(-1)),1.)
        self.probs=counts/counts.sum(-1,keepdims=True)
        unigram=np.bincount(training,minlength=vocab_size)+smoothing
        self.unigram=unigram/unigram.sum()

    def bpc(self,windows,unigram=False):
        if unigram: p=self.unigram[windows]
        else:
            prior=np.concatenate([np.full((len(windows),2),self.bos),windows],-1)
            p=self.probs[prior[:,:-2],prior[:,1:-1],windows]
        return -np.log2(p).mean(-1)

    def sample(self,seed,count,length,unigram=False):
        rng=np.random.default_rng(seed)
        result=np.empty((count,length),np.int32)
        a=b=np.full(count,self.bos,np.int32)
        for i in range(length):
            probs=np.broadcast_to(self.unigram,(count,self.vocab_size)) if unigram else self.probs[a,b]
            draw=rng.random(count)
            token=(draw[:,None]>np.cumsum(probs,axis=-1)).sum(-1)
            token=np.minimum(token,self.vocab_size-1).astype(np.int32)
            result[:,i]=token
            a,b=b,token
        return result
