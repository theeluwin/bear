# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu, spilu, inv

from utils import *


class PPRBaseTF(object):

    def __init__(self, sess, n, filename, jump_prob=0.05):
        raise NotImplementedError

    def query(self, q):
        raise NotImplementedError


class PPRIterativeTF(PPRBaseTF):

    def __init__(self, sess, n, filename, jump_prob=0.05, epsilon=1e-4, max_iteration=100, **kwargs):
        """
        Computes PPR using iterative method. `epsilon` denotes convergence threshold.

        Args:
            sess (Session): tensorflow session.
            n (int): Number of nodes.
            filename (str): A csv file denoting the graph.
            jump_prob (float): Jumping probability of PPR.
            epsilon (float): Convergence threshold (uses l2-norm of difference).
            max_iteration (int): Maximum number of allowed iterations.
        """
        self.alias = 'iter'
        self.sess = sess
        self.n = n
        self.c = jump_prob
        self.e = epsilon
        self.max_iteration = max_iteration
        d = 1 - self.c
        self.node2index, A = read_matrix(filename, d=d)
        with tf.variable_scope('ppr_iterative_tf'):
            t_A = tf.SparseTensorValue(list(zip(A.row, A.col)), A.data, dense_shape=[n, n])
            t_old_r = tf.Variable((np.ones(n) / n)[:, np.newaxis])
            self.t_q = tf.placeholder(tf.float64, shape=[n, 1])
            self.t_new_r = tf.Variable((np.ones(n) / n)[:, np.newaxis])
            self.t_new_r_assign = tf.assign(self.t_new_r, tf.sparse_tensor_dense_matmul(t_A, t_old_r) + self.c * self.t_q)
            self.t_old_r_assign = tf.assign(t_old_r, self.t_new_r)
            self.t_loss = tf.norm(self.t_new_r - t_old_r)
        del A

    def query(self, q):
        q = q / q.sum()
        q = q[:, np.newaxis]
        feed = {self.t_q: q}
        iteration = 0
        while True:
            self.sess.run(self.t_new_r_assign, feed_dict=feed)
            loss = self.sess.run(self.t_loss, feed_dict=feed)
            if loss < self.e:
                break
            self.sess.run(self.t_old_r_assign, feed_dict=feed)
            iteration += 1
            if iteration > self.max_iteration:
                break
        return self.sess.run(self.t_new_r, feed_dict=feed).flatten()


class PPRLUDecompositionTF(PPRBaseTF):

    def __init__(self, sess, n, filename, jump_prob=0.05, tolerance=1e-8):
        """
        Computes PPR using LU decomposition.

        Args:
            sess (Session): tensorflow session.
            n (int): Number of nodes.
            filename (str): A csv file denoting the graph.
            jump_prob (float): Jumping probability of PPR.
            tolerance (float): Drops entries with absolute value lower than this value when computing inverse of LU.
        """
        self.alias = 'ludc'
        self.sess = sess
        self.n = n
        self.c = jump_prob
        d = 1 - self.c
        t = tolerance
        exact = False
        if t is None:
            t = np.power(n, -0.5)
        elif t == 0:
            exact = True
        self.node2index, H = read_matrix(filename, d=-d, add_identity=True)
        self.perm = degree_reverse_rank_perm(H)
        H = reorder_matrix(H, self.perm).tocsc()
        if exact:
            self.LU = splu(H)
        else:
            self.LU = spilu(H, drop_tol=t)
        Linv = inv(self.LU.L).tocoo()
        Uinv = inv(self.LU.U).tocoo()
        with tf.variable_scope('ppr_lu_decomposition_tf'):
            t_Linv = tf.SparseTensorValue(list(zip(Linv.row, Linv.col)), Linv.data, dense_shape=[n, n])
            t_Uinv = tf.SparseTensorValue(list(zip(Uinv.row, Uinv.col)), Uinv.data, dense_shape=[n, n])
            self.t_q = tf.placeholder(tf.float64, shape=[self.n, 1])
            self.t_r = tf.sparse_tensor_dense_matmul(t_Uinv, tf.sparse_tensor_dense_matmul(t_Linv, self.c * self.t_q))

    def query(self, q):
        """
        Here's how calculations are done:

            H = I - dA
            H' = permT @ H @ perm
            H' = PrT @ L @ U @ PcT
            r = c @ perm @ Pc @ Uinv @ Linv @ Pr @ permT @ q

        Not bad, huh.
        """
        q = q / q.sum()
        q = reorder_vector(q, self.perm, reverse=True)
        q = reorder_vector(q, self.LU.perm_r)[:, np.newaxis]
        r = self.sess.run(self.t_r, feed_dict={self.t_q: q}).flatten()
        r = reorder_vector(r, self.LU.perm_c)
        return reorder_vector(r, self.perm)


class PPRBear(PPRBaseTF):

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
        self.node2index, H = read_matrix(filename, d=-self.d, add_identity=True)
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
