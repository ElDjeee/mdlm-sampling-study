import hashlib
from pathlib import Path
import numpy as np

SOURCE = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def load_data(path,train_chars=200000):
    raw = Path(path).read_bytes()
    text = raw.decode('utf-8')
    n = len(text)
    cuts = (int(.9*n),int(.95*n))
    train = text[:min(train_chars,cuts[0])] if train_chars else text[:cuts[0]]
    val,test = text[cuts[0]:cuts[1]],text[cuts[1]:]
    chars = sorted(set(train))
    lookup = {c:i for i,c in enumerate(chars)}
    unk = len(chars)
    def encode(s): return np.array([lookup.get(c,unk) for c in s],np.int32)
    encoded = {name:encode(s) for name,s in [('train',train),('val',val),('test',test)]}
    metadata = dict(source=SOURCE,sha256=hashlib.sha256(raw).hexdigest(),
                    total_characters=n,split_boundaries=[0,cuts[0],cuts[1],n],
                    used_characters={k:len(v) for k,v in encoded.items()},
                    unknown_counts={k:int((v==unk).sum()) for k,v in encoded.items()},
                    characters=chars,unk_id=unk,split_policy='contiguous 90/5/5; training prefix capped; vocabulary fitted on used training text only')
    return encoded,chars,metadata


def decode(tokens,chars):
    vocab = chars+['�','█','⟨BOS⟩','⟨PAD⟩']
    return ''.join(vocab[int(t)] for t in tokens)


def evaluation_windows(data,count,length,seed=731):
    starts = np.arange(0,len(data)-length+1,length)
    rng = np.random.default_rng(seed)
    starts = rng.permutation(starts)[:count]
    return data[starts[:,None]+np.arange(length)[None]]
