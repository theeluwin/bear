# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu, spilu, inv

from utils import *


_sdmm = tf.sparse_tensor_dense_matmul


class PPRBaseTF(object):

    def __init__(self, sess, n, filename, jump_prob=0.05):
        raise NotImplementedError

    def pp(self, message, *args, **kwargs):
        if self.verbose:
            print('[{}] {}'.format(self.alias, message), *args, **kwargs)

    def query(self, q):
        raise NotImplementedError


class PPRIterativeTF(PPRBaseTF):

    def __init__(self, sess, n, filename, jump_prob=0.05, epsilon=1e-4, max_iteration=100, drop_tol=1e-8, verbose=False):
        """
        Computes PPR using iterative method. `epsilon` denotes convergence threshold.

        Args:
            sess (Session): tensorflow session.
            n (int): Number of nodes.
            filename (str): A csv file denoting the graph.
            jump_prob (float): Jumping probability of PPR.
            epsilon (float): Convergence threshold (uses l2-norm of difference).
            max_iteration (int): Maximum number of allowed iterations.
            drop_tol (float): No effect.
            verbose (bool): Prints step messages if True.
        """
        self.alias = 'iter'
        self.verbose = verbose
        self.pp("initializing")
        self.sess = sess
        self.n = n
        self.c = jump_prob
        self.e = epsilon
        self.max_iteration = max_iteration
        d = 1 - self.c
        self.pp("preprocessing")
        self.node2index, A = read_matrix(filename, d=d)
        self.pp("tf init")
        with tf.variable_scope('ppr_iterative_tf'):
            t_A = tf.SparseTensorValue(list(zip(A.row, A.col)), A.data, dense_shape=[n, n])
            t_old_r = tf.Variable((np.ones(n) / n)[:, np.newaxis])
            self.t_cq = tf.placeholder(tf.float64, shape=[n, 1])
            self.t_new_r = tf.Variable((np.ones(n) / n)[:, np.newaxis])
            self.t_new_r_assign = tf.assign(self.t_new_r, _sdmm(t_A, t_old_r) + self.t_cq)
            self.t_old_r_assign = tf.assign(t_old_r, self.t_new_r)
            self.t_loss = tf.norm(self.t_new_r - t_old_r)
        del A

    def query(self, q):
        q = q / q.sum()
        q = q[:, np.newaxis]
        feed = {self.t_cq: self.c * q}
        iteration = 0
        self.pp("querying")
        while True:
            self.sess.run(self.t_new_r_assign, feed_dict=feed)
            loss = self.sess.run(self.t_loss, feed_dict=feed)
            if loss < self.e:
                break
            self.sess.run(self.t_old_r_assign, feed_dict=feed)
            iteration += 1
            self.pp("iteration {}".format(iteration), end='\r')
            if iteration > self.max_iteration:
                break
        self.pp("converged")
        return self.sess.run(self.t_new_r, feed_dict=feed).flatten()


class PPRLUDecompositionTF(PPRBaseTF):

    def __init__(self, sess, n, filename, jump_prob=0.05, drop_tol=1e-8, verbose=False):
        """
        Computes PPR using LU decomposition.

        Args:
            sess (Session): tensorflow session.
            n (int): Number of nodes.
            filename (str): A csv file denoting the graph.
            jump_prob (float): Jumping probability of PPR.
            drop_tol (float): Drops entries with absolute value lower than this value when computing inverse of LU.
            verbose (bool): Prints step messages if True.
        """
        self.alias = 'ludc'
        self.verbose = verbose
        self.pp("initializing")
        self.sess = sess
        self.n = n
        self.c = jump_prob
        d = 1 - self.c
        t = drop_tol
        exact = False
        if t is None:
            t = np.power(n, -0.5)
        elif t == 0:
            exact = True
        self.pp("reading")
        self.node2index, H = read_matrix(filename, d=-d, add_identity=True)
        self.pp("sorting H")
        self.perm = degree_reverse_rank_perm(H)
        H = reorder_matrix(H, self.perm).tocsc()
        self.pp("computing LU decomposition")
        if exact:
            self.LU = splu(H)
        else:
            self.LU = spilu(H, drop_tol=t)
        Linv = inv(self.LU.L).tocoo()
        Uinv = inv(self.LU.U).tocoo()
        self.pp("tf init")
        with tf.variable_scope('ppr_lu_decomposition_tf'):
            t_Linv = tf.SparseTensorValue(list(zip(Linv.row, Linv.col)), Linv.data, dense_shape=self.LU.L.shape)
            t_Uinv = tf.SparseTensorValue(list(zip(Uinv.row, Uinv.col)), Uinv.data, dense_shape=self.LU.U.shape)
            self.t_q = tf.placeholder(tf.float64, shape=[self.n, 1])
            self.t_r = _sdmm(t_Uinv, _sdmm(t_Linv, self.c * self.t_q))

    def query(self, q):
        """
        Here's how calculations are done:

            H = I - dA
            H' = permT @ H @ perm
            H' = PrT @ L @ U @ PcT
            r = c @ perm @ Pc @ Uinv @ Linv @ Pr @ permT @ q

        Not bad, huh.
        """
        self.pp("sorting q")
        q = q / q.sum()
        q = reorder_vector(q, self.perm, reverse=True)
        q = reorder_vector(q, self.LU.perm_r)[:, np.newaxis]
        self.pp("computing LU solve")
        r = self.sess.run(self.t_r, feed_dict={self.t_q: q}).flatten()
        r = reorder_vector(r, self.LU.perm_c)
        return reorder_vector(r, self.perm)


class PPRBearTF(PPRBaseTF):

    def __init__(self, sess, n, filename, jump_prob=0.05, drop_tol=1e-8, k=None, greedy=True, verbose=False):
        """
        Computes PPR using BEAR with SlashBurn. `tolerance` denotes approximation (set 0 for exact solution). Note that the approximation might return non-stochastic pagerank values. `k` and `greedy` are options for SlashBurn.

        Args:
            sess (Session): tensorflow session.
            n (int): Number of nodes.
            filename (str): A csv file denoting the graph.
            jump_prob (float): Jumping probability of PPR.
            tolerance (float): Drops entries with absolute value lower than this value when computing inverse of LUs (H11, S from the paper).
            k (int): SlashBurn finds top-k hubs. There is a rule of thumb, so if you're not familiar with SlashBurn, then leave it to None.
            greedy (bool): Hub selection on SlashBurn. See SlashBurn for more details.
            verbose (bool): Prints step messages if True.
        """
        self.alias = 'bear'
        self.verbose = verbose
        self.pp("initializing")
        self.sess = sess
        self.n = n
        self.c = jump_prob
        self.d = 1 - self.c
        d = 1 - self.c
        t = drop_tol
        exact = False
        if t is None:
            t = np.power(n, -0.5)
        elif t == 0:
            exact = True
        self.pp("reading")
        self.node2index, H = read_matrix(filename, d=-d, add_identity=True)
        if k is None:
            k = max(1, int(0.001 * self.n))
        self.pp("running slashburn")
        self.perm_H, self.wing = slashburn(H, k, greedy)
        self.body = self.n - self.wing
        self.pp("sorting H")
        H = reorder_matrix(H, self.perm_H)
        self.pp("partitioning H")
        H11, H12, H21, H22 = matrix_partition(H, self.body)
        del H
        H11 = H11.tocsc()
        self.pp("computing LU decomposition on H11")
        if exact:
            self.LU1 = splu(H11)
        else:
            self.LU1 = spilu(H11, drop_tol=t)
        del H11
        self.pp("computing LU1 solve")
        L1inv = inv(self.LU1.L).tocoo()
        U1inv = inv(self.LU1.U).tocoo()
        S = H22 - H21 @ reorder_matrix(U1inv @ L1inv @ reorder_matrix(H12, self.LU1.perm_r), self.LU1.perm_c)
        S = coo_matrix(S)
        self.pp("sorting S")
        self.perm_S = degree_reverse_rank_perm(S)
        S = reorder_matrix(S, self.perm_S)
        H12 = reorder_matrix(H12, self.perm_S, fix_row=True)
        H21 = reorder_matrix(H21, self.perm_S, fix_col=True)
        S = S.tocsc()
        del H22
        self.pp("computing LU decomposition on S")
        if exact:
            self.LU2 = splu(S)
        else:
            self.LU2 = spilu(S, drop_tol=t)
        del S
        L2inv = inv(self.LU2.L).tocoo()
        U2inv = inv(self.LU2.U).tocoo()
        self.pp("tf init")
        with tf.variable_scope('ppr_bear_tf'):
            t_L1inv = tf.SparseTensorValue(list(zip(L1inv.row, L1inv.col)), L1inv.data, dense_shape=self.LU1.L.shape)
            t_U1inv = tf.SparseTensorValue(list(zip(U1inv.row, U1inv.col)), U1inv.data, dense_shape=self.LU1.U.shape)
            t_L2inv = tf.SparseTensorValue(list(zip(L2inv.row, L2inv.col)), L2inv.data, dense_shape=self.LU2.L.shape)
            t_U2inv = tf.SparseTensorValue(list(zip(U2inv.row, U2inv.col)), U2inv.data, dense_shape=self.LU2.U.shape)
            t_H12 = tf.SparseTensorValue(list(zip(H12.row, H12.col)), H12.data, dense_shape=H12.shape)
            t_H21 = tf.SparseTensorValue(list(zip(H21.row, H21.col)), H21.data, dense_shape=H21.shape)
            self.t_q1 = tf.placeholder(tf.float64, shape=[self.body, 1])
            self.t_q2 = tf.placeholder(tf.float64, shape=[self.wing, 1])
            self.t_z1 = _sdmm(t_U1inv, _sdmm(t_L1inv, self.t_q1))
            self.t_z1p = tf.placeholder(tf.float64, shape=[self.body, 1])
            self.t_z2 = self.t_q2 - _sdmm(t_H21, self.t_z1p)
            self.t_z2p = tf.placeholder(tf.float64, shape=[self.wing, 1])
            self.t_r2 = self.c * _sdmm(t_U2inv, _sdmm(t_L2inv, self.t_z2p))
            self.t_r2p = tf.placeholder(tf.float64, shape=[self.wing, 1])
            self.t_z3 = self.c * self.t_q1 - _sdmm(t_H12, self.t_r2p)
            self.t_z3p = tf.placeholder(tf.float64, shape=[self.body, 1])
            self.t_r1 = _sdmm(t_U1inv, _sdmm(t_L1inv, self.t_z3p))

    def query(self, q):
        self.pp("sorting q")
        q = q / q.sum()
        q = reorder_vector(q, self.perm_H, reverse=True)
        q1, q2 = q[:self.body], q[self.body:]
        q2 = reorder_vector(q2, self.perm_S, reverse=True)[:, np.newaxis]
        q1 = reorder_vector(q1, self.LU1.perm_r)[:, np.newaxis]
        z1 = self.sess.run(self.t_z1, feed_dict={self.t_q1: q1})
        z1 = reorder_vector(z1, self.LU1.perm_c)[:, np.newaxis]
        z2 = self.sess.run(self.t_z2, feed_dict={self.t_q2: q2, self.t_z1p: z1})
        z2 = reorder_vector(z2, self.LU2.perm_r)[:, np.newaxis]
        self.pp("computing r2 with LU2 solve")
        r2 = self.sess.run(self.t_r2, feed_dict={self.t_z2p: z2})
        r2 = reorder_vector(r2, self.LU2.perm_c)[:, np.newaxis]
        z3 = self.sess.run(self.t_z3, feed_dict={self.t_q1: q1, self.t_r2p: r2})
        z3 = reorder_vector(z3, self.LU1.perm_r)[:, np.newaxis]
        self.pp("computing r1 with LU1 solve")
        r1 = self.sess.run(self.t_r1, feed_dict={self.t_z3p: z3})
        r1 = reorder_vector(r1, self.LU1.perm_c)
        r2 = reorder_vector(r2, self.perm_S)
        r = np.concatenate((r1, r2))
        return reorder_vector(r, self.perm_H)
