# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import tensorflow as tf

from scipy.stats import spearmanr
from numpy.linalg import norm

from ppr import PPRIterative
from ppr_tf import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def pr2ranks(r):
    n = len(r)
    pr = {i: r[i] for i in range(n)}
    tops = sorted(pr, key=pr.get, reverse=True)
    ranks = [0 for _ in range(n)]
    for i in range(n):
        ranks[tops[i]] = i
    return ranks


def profile(filename, n, seed=None, exact=True):
    if seed:
        np.random.seed(seed)
    else:
        np.random.seed(int(time.time()))
    if exact:
        tolerance = 0
    else:
        tolerance = None
    q = np.random.rand(n)
    q = q / q.sum()
    baseline = PPRIterative()
    baseline.preprocess(filename)
    r = baseline.query(q)
    ranks = pr2ranks(r)
    model_classes = [PPRIterativeTF, PPRLUDecompositionTF]
    for model_class in model_classes:
        with tf.Session() as sess:
            start = time.time()
            model = model_class(sess, n, filename, tolerance=tolerance)
            end = time.time()
            sess.run(tf.global_variables_initializer())
            elapsed = end - start
            print("[{}](gpu,{},n={})".format(model.alias, "exact" if exact else "apprx", n))
            print("preprocess\t{}".format(elapsed))
            start = time.time()
            r_ = model.query(q)
            end = time.time()
            elapsed = end - start
            print("query time\t{}".format(elapsed))
            ranks_ = pr2ranks(r_)
            spearman = spearmanr(ranks, ranks_)
            r_ = r_ / r_.sum()
            print("diff norm\t{}".format(norm(r - r_)))
            print("cosine sim\t{}".format(r.dot(r_) / norm(r) / norm(r_)))
            print("spearman corr\t{}".format(spearman.correlation))
            print("")


if __name__ == '__main__':
    filename, n, exact = 'data/email.csv', 36692, True
    # filename, n, exact = 'data/email.csv', 36692, False
    # filename, n, exact = 'data/small.csv', 15, True
    # filename, n, exact = 'data/small.csv', 15, False
    profile(filename, n, exact=exact, seed=999)
