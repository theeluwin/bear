# -*- coding: utf-8 -*-

import csv
import pickle
import numpy as np

from collections import defaultdict
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu, spilu

from utils import *


class PPRBase(object):

    def __init__(self, jump_prob=0.05):
        raise NotImplementedError

    def preprocess(self, filename):
        raise NotImplementedError

    def query(self, q):
        raise NotImplementedError

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError


class PPRNetworkx(PPRBase):

    def __init__(self, jump_prob=0.05, epsilon=1e-8, max_iteration=100):
        """
        Computes PPR using networkx library. `epsilon` denotes convergence threshold.

        Args:
            jump_prob (float): Jumping probability of PPR.
            epsilon (float): Convergence threshold (uses l2-norm of difference).
            max_iteration (int): Maximum number of allowed iterations.
        """
        try:
            import networkx
        except ModuleNotFoundError:
            raise Exception("To use this class, install networkx via `pip install networkx`")
        self.networkx = networkx
        self.alias = 'ntkx'
        self.d = 1 - jump_prob
        self.e = epsilon
        self.max_iteration = max_iteration

    def preprocess(self, filename):
        """Remember: row-first ordered csv file only!"""
        nodes = {}
        graph = self.networkx.Graph()
        with open(filename) as file:
            reader = csv.reader(file)
            for line in reader:
                i = int(line[0])
                j = int(line[1])
                if i not in nodes:
                    graph.add_node(i)
                    nodes[i] = True
                if j not in nodes:
                    graph.add_node(j)
                    nodes[j] = True
                graph.add_edge(i, j, weight=1)
        self.n = len(list(nodes.keys()))
        self.graph = graph

    def query(self, q):
        q = {i: q[i] for i in range(self.n)}
        pagerank = self.networkx.pagerank(self.graph, alpha=self.d, personalization=q, tol=self.e, weight='weight', max_iter=self.max_iteration)
        return np.array([pagerank[i] for i in range(self.n)])

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump((self.d, self.e, self.max_iteration, self.n, self.graph), file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            self.d, self.e, self.max_iteration, self.n, self.graph = pickle.load(file)


class PPRIterative(PPRBase):

    def __init__(self, jump_prob=0.05, epsilon=1e-4, max_iteration=100):
        """
        Computes PPR using iterative method. `epsilon` denotes convergence threshold.

        Args:
            jump_prob (float): Jumping probability of PPR.
            epsilon (float): Convergence threshold (uses l2-norm of difference).
            max_iteration (int): Maximum number of allowed iterations.
        """
        self.alias = 'iter'
        self.c = jump_prob
        self.d = 1 - self.c
        self.e = epsilon
        self.max_iteration = max_iteration

    def preprocess(self, filename):
        """Remember: row-first ordered csv file only!"""
        self.A = read_matrix(filename, d=self.d)
        self.n, _ = self.A.shape

    def query(self, q):
        q = q / q.sum()
        old_r = np.ones(self.n) / self.n
        iteration = 0
        while True:
            new_r = self.A @ old_r + self.c * q
            if np.linalg.norm(new_r - old_r) < self.e:
                break
            old_r = new_r
            iteration += 1
            if iteration > self.max_iteration:
                break
        return new_r

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump((self.c, self.e, self.max_iteration, self.n, self.A), file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            self.c, self.e, self.max_iteration, self.n, self.A = pickle.load(file)


class PPRLUDecomposition(PPRBase):

    def __init__(self, jump_prob=0.05, tolerance=1e-8):
        """
        Computes PPR using LU decomposition. `tolerance` denotes approximation (set 0 for exact solution). Note that the approximation might return non-stochastic pagerank values.

        Args:
            jump_prob (float): Jumping probability of PPR.
            tolerance (float): Drops entries with absolute value lower than this value when computing inverse of LU. I hear some rhyme.
        """
        self.alias = 'ludc'
        self.c = jump_prob
        self.d = 1 - self.c
        self.t = tolerance
        self.exact = False

    def preprocess(self, filename):
        """Remember: row-first ordered csv file only!"""
        H = read_matrix(filename, d=-self.d, add_identity=True)
        n, _ = H.shape
        if self.t is None:
            self.t = np.power(n, -0.5)
        elif self.t == 0:
            self.exact = True
        self.perm = degree_reverse_rank_perm(H)
        H = reorder_matrix(H, self.perm).tocsc()
        if self.exact:
            self.LU = splu(H)
        else:
            self.LU = spilu(H, drop_tol=self.t)

    def query(self, q):
        q = q / q.sum()
        q = reorder_vector(q, self.perm, reverse=True)
        r = self.LU.solve(self.c * q)
        return reorder_vector(r, self.perm)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump((self.c, self.perm, serialize_slu(self.LU)), file)

    def load(self, filename):
        # todo: recover SuperLU from serialized slu
        """
        with open(filename, 'rb') as file:
            self.c, self.perm, slu = pickle.load(file)
        """
        raise NotImplementedError


class PPRBear(PPRBase):

    def __init__(self, jump_prob=0.05, tolerance=1e-8, k=None, greedy=True):
        """
        Computes PPR using BEAR with SlashBurn. `tolerance` denotes approximation (set 0 for exact solution). Note that the approximation might return non-stochastic pagerank values. `k` and `greedy` are options for SlashBurn.

        Args:
            jump_prob (float): Jumping probability of PPR.
            tolerance (float): Drops entries with absolute value lower than this value when computing inverse of LUs (H11, S from the paper).
            k (int): SlashBurn finds top-k hubs. There is a rule of thumb, so if you're not familiar with SlashBurn, then leave it to None.
            greedy (bool): Hub selection on SlashBurn. See SlashBurn for more details.
        """
        self.alias = 'bear'
        self.c = jump_prob
        self.d = 1 - self.c
        self.t = tolerance
        self.k = k
        self.greedy = greedy
        self.exact = False

    def preprocess(self, filename):
        """Remember: row-first ordered csv file only!"""
        H = read_matrix(filename, d=-self.d, add_identity=True)
        self.n, _ = H.shape
        if self.t is None:
            self.t = np.power(self.n, -0.5)
        elif self.t == 0:
            self.exact = True
        if self.k is None:
            self.k = max(1, int(0.001 * self.n))
        self.perm_H, wing = slashburn(H, self.k, self.greedy)
        self.body = self.n - wing
        H = reorder_matrix(H, self.perm_H)
        H11, H12, H21, H22 = matrix_partition(H, self.body)
        del H
        H11 = H11.tocsc()
        if self.exact:
            self.LU1 = splu(H11)
        else:
            self.LU1 = spilu(H11, drop_tol=self.t)
        del H11
        S = H22 - H21 @ self.LU1.solve(H12.toarray())
        S = coo_matrix(S)
        self.perm_S = degree_reverse_rank_perm(S)
        S = reorder_matrix(S, self.perm_S)
        self.H12 = reorder_matrix(H12, self.perm_S, fix_row=True)
        self.H21 = reorder_matrix(H21, self.perm_S, fix_col=True)
        S = S.tocsc()
        del H12, H21, H22
        if self.exact:
            self.LU2 = splu(S)
        else:
            self.LU2 = spilu(S, drop_tol=self.t)
        # issue: this approximation drops accuracy way too much! why?
        """
        if not self.exact:
            H12 = drop_tolerance(self.H12, self.t)
            del self.H12
            self.H12 = H12
            H21 = drop_tolerance(self.H21, self.t)
            del self.H21
            self.H21 = H21
        """
        del S

    def query(self, q):
        q = q / q.sum()
        q = reorder_vector(q, self.perm_H, reverse=True)
        q1, q2 = q[:self.body], q[self.body:]
        q2 = reorder_vector(q2, self.perm_S, reverse=True)
        r2 = self.c * self.LU2.solve(q2 - self.H21 @ self.LU1.solve(q1))
        r1 = self.LU1.solve(self.c * q1 - self.H12 @ r2)
        r2 = reorder_vector(r2, self.perm_S)
        r = np.concatenate((r1, r2))
        return reorder_vector(r, self.perm_H)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump((self.c, self.body, self.perm_H, self.perm_S, self.H12, self.H21, serialize_slu(self.LU1), serialize_slu(self.LU2)), file)

    def load(self, filename):
        # todo: recover SuperLU from serialized slu
        """
        with open(filename, 'rb') as file:
            self.c, self.body, self.perm_H, self.perm_S, self.H12, self.H21, slu1, slu2 = pickle.load(file)
        """
        raise NotImplementedError
