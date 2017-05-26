#python example to infer document vectors from trained doc2vec model
import gensim.models as g
import codecs
import sys

#parameters
model       = "apnews_data/doc2vec.bin"

#inference hyper-parameters
start_alpha = 0.01
infer_epoch = 1000

#load model
m = g.Doc2Vec.load(model)

def clean(t):
    return t.read().strip().split()

def infer(t):
    print( m.infer_vector(clean(t), alpha=start_alpha, steps=infer_epoch) )

if len(sys.argv) > 1:
    for f in sys.argv[1:]:
        infer(open(f))
else:
    print("Reading on stdin...")
    infer(sys.stdin)

