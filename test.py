# -*- coding: utf-8 -*-

import csv
import time
import numpy as np

from ops import *
from ppr import *
from scipy.stats import spearmanr


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
    models = [PPRIterative(), PPRLUDecomposition(tolerance=tolerance), PPRBear(tolerance=tolerance)]
    q = np.random.rand(n)
    q = q / q.sum()
    for model in models:
        start = time.time()
        model.preprocess(filename)
        end = time.time()
        elapsed = end - start
        print("preprocessing\t{}\t{}".format(model.alias, elapsed))
        start = time.time()
        model.query(q)
        end = time.time()
        elapsed = end - start
        print("query time\t{}\t{}".format(model.alias, elapsed))


def sanity(filename, n, seed=None, exact=True):
    norm = np.linalg.norm
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
    baseline = PPRNetworkx()
    baseline.preprocess(filename)
    r = baseline.query(q)
    ranks = pr2ranks(r)
    models = [PPRIterative(), PPRLUDecomposition(tolerance=tolerance), PPRBear(tolerance=tolerance)]
    for model in models:
        model.preprocess(filename)
        model.save('models/' + model.alias + '.ppr')
        r_ = model.query(q)
        ranks_ = pr2ranks(r_)
        spearman = spearmanr(ranks, ranks_)
        r_ = r_ / r_.sum()
        print(model.alias, norm(r - r_), r.dot(r_) / norm(r) / norm(r_), spearman.correlation)


if __name__ == '__main__':
    # filename, n, exact = 'data/email.csv', 36692, True
    # filename, n, exact = 'data/email.csv', 36692, False
    # filename, n, exact = 'data/small.csv', 15, True
    filename, n, exact = 'data/small.csv', 15, False
    sanity(filename, n, exact=exact, seed=999)
    profile(filename, n, exact=exact, seed=999)
