# -*- coding: utf-8 -*-

import os
import sys
import time
import pickle
import numpy as np

from scipy.stats import spearmanr
from numpy.linalg import norm

from ppr import *
from pprtf import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def filepath2name(filepath):
    return '.'.join(filepath.split('/')[-1].split('.')[:-1])


def pr2ranks(r):
    n = len(r)
    pr = {i: r[i] for i in range(n)}
    tops = sorted(pr, key=pr.get, reverse=True)
    ranks = [0 for _ in range(n)]
    for i in range(n):
        ranks[tops[i]] = i
    return ranks


def solve(filepath, n, seed=None, verbose=False):
    if not seed:
        np.random.seed(int(time.time()))
    else:
        np.random.seed(seed)
    q = np.random.rand(n)
    q = q / q.sum()
    model = PPRIterative(epsilon=1e-8, verbose=verbose)
    model.preprocess(filepath)
    r = model.query(q)
    ranks = pr2ranks(r)
    pickle.dump((q, r, ranks), open('data/{}_sol.dat'.format(filepath2name(filepath)), 'wb'))


def profile(filepath, n, exact=True, save=False, verbose=True, use_gpu=False, report=open('temp.txt', 'w')):
    if exact:
        tol = 0
    else:
        tol = None
    solpath = 'data/{}_sol.dat'.format(filepath2name(filepath))
    if not os.path.isfile(solpath):
        solve(filepath, n, seed=0, verbose=verbose)
    q, r, ranks = pickle.load(open(solpath, 'rb'))
    if use_gpu:
        model_classes = [PPRIterativeTF, PPRLUDecompositionTF, PPRBearTF]
    else:
        model_classes = [PPRIterative, PPRLUDecomposition, PPRBear]
    for model_class in model_classes:
        with tf.Session() as sess:
            start = time.time()
            if use_gpu:
                model = model_class(sess, n, filepath, drop_tol=tol, verbose=verbose)
            else:
                model = model_class(drop_tol=tol, verbose=verbose)
                model.preprocess(filepath)
            end = time.time()
            if use_gpu:
                sess.run(tf.global_variables_initializer())
            elapsed = end - start
            if save:
                model.save('models/{}.ppr'.format(model.alias))
            print("[{}]({},{},n={})".format(model.alias, 'gpu' if use_gpu else 'cpu', 'exact' if exact else 'apprx', n), file=report)
            print("preprocess\t{}".format(elapsed), file=report)
            start = time.time()
            r_ = model.query(q)
            end = time.time()
            elapsed = end - start
            print("query time\t{}".format(elapsed), file=report)
            ranks_ = pr2ranks(r_)
            spearman = spearmanr(ranks, ranks_)
            r_ = r_ / r_.sum()
            print("diff norm\t{}".format(norm(r - r_)), file=report)
            print("cosine sim\t{}".format(r.dot(r_) / norm(r) / norm(r_)), file=report)
            print("spearman corr\t{}".format(spearman.correlation), file=report)
            print("", file=report)


if __name__ == '__main__':
    use_gpu = True
    data = 'small'
    report = open('temp.txt', 'w')
    data2n = {
        'patent': 3774768,  # Patent citation
        'stan': 281903,  # Stanford hyperlink
        'email': 36692,  # Email-enron
        'small': 15,  # My custom data
    }
    profile('data/{}.csv'.format(data), data2n[data], exact=True, save=False, verbose=True, use_gpu=use_gpu, report=report)
