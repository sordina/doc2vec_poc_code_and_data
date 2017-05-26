#python example to infer document vectors from trained doc2vec model
import gensim.models as g
import codecs
import sys
import os
import scipy.spatial.distance

NOISY = not os.environ.get("QUIET",False)

if len(sys.argv) != 3:
    if NOISY:
        print("Error")
        print("Please provide two files as arguments")
    sys.exit()

def clean(t):
    return t.read().strip().split()

def infer(t):
    return m.infer_vector(clean(t), alpha=start_alpha, steps=infer_epoch)

#parameters
model       = "apnews_data/doc2vec.bin"

#inference hyper-parameters
start_alpha = 0.01
infer_epoch = 1000

#load model
m = g.Doc2Vec.load(model)

v1 = infer(open(sys.argv[1]))
if NOISY:
    print("Vector for {} = {}".format(sys.argv[1], v1))

v2 = infer(open(sys.argv[2]))
if NOISY:
    print("Vector for {} = {}".format(sys.argv[2], v2))

d = scipy.spatial.distance.cosine(v1,v2)
if NOISY:
    print("Distance between documents {}".format(d))
else:
    print(d)
