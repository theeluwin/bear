# -*- coding: utf-8 -*-

import csv
import numpy as np

from collections import defaultdict
from scipy.sparse import coo_matrix


def serialize_slu(slu):
    """Serializes scipy SLU object. Should be deprecated."""
    return (slu.perm_c, slu.perm_r, slu.L, slu.U)


def scc(nodes=[], edges=[]):
    """
    A function that computes strongly connected components using Kosaraju's algorithm. Returns list of components, which are list of nodes.

    Args:
        nodes (list): List of nodes of any type.
        edges (list): List of edges, where each edge is a pair of nodes (ex: (0, 1)).

    Returns:
        List of strongly connected components. Each component is just a list of nodes.
    """
    alists = defaultdict(lambda: [])
    rlists = defaultdict(lambda: [])
    avisited = defaultdict(lambda: False)
    rvisited = defaultdict(lambda: False)
    leader = defaultdict(lambda: None)
    components = defaultdict(lambda: [])
    f = {}
    r = {}
    nodes = nodes
    edges = edges
    n = len(nodes)
    for u, v in edges:
        alists[u].append(v)
        rlists[v].append(u)
    t = 0
    for s in nodes:
        if rvisited[s]:
            continue
        rvisited[s] = True
        stack = [s]
        while len(stack):
            i = stack[-1]
            sink = True
            for j in rlists[i]:
                if not rvisited[j]:
                    rvisited[j] = True
                    stack.append(j)
                    sink = False
                    break
            if sink:
                t += 1
                f[i] = t
                stack.pop()
    for key in f:
        r[f[key]] = key
    for t in range(n, 0, -1):
        s = r[t]
        if avisited[s]:
            continue
        avisited[s] = True
        stack = [s]
        while len(stack):
            i = stack.pop()
            leader[i] = s
            components[s].append(i)
            for j in alists[i]:
                if not avisited[j]:
                    avisited[j] = True
                    stack.append(j)
    return [components[leader] for leader in components]


def read_matrix(filename, normalize=True, transpose=True, d=1, add_identity=False):
    """
    Reads coo matrix from csv file. Each line should denote an edge pair (like (0, 1)), and **must** be row-first ordered. `d` denote scalar multiplication, and `add_identity` adds an identity matrix after the scalar multiplication.

    Args:
        filename (str): Path to the csv file.
        normalize (bool): If True, applies row-normalization.
        transpose (bool): If True, transpose it **after** normalization.
        d (float): Scalar multiplication.
        add_identity (bool): Adds an identity matrix after all above things are done.

    Returns:
        A scipy coo_matrix. For example, for `d` = -0.85 and all the other options being True, it will produces a matrix of `(I - dM)` where `M` is a (stochastic) google matrix.
    """
    row = []
    col = []
    data = []
    max_i = 0
    max_j = 0
    with open(filename) as file:
        reader = csv.reader(file)
        current_i = -1
        current_norm = 0
        for line in reader:
            i = int(line[0])
            j = int(line[1])
            if i > max_i:
                max_i = i
            if j > max_j:
                max_j = j
            if current_i != i:
                if current_norm:
                    if normalize:
                        value = d / current_norm
                    else:
                        value = d
                    row.extend(current_row)
                    col.extend(current_col)
                    data.extend([value for _ in range(current_norm)])
                current_i = i
                current_norm = 0
                current_row = []
                current_col = []
            if transpose:
                current_row.append(j)
                current_col.append(i)
            else:
                current_row.append(i)
                current_col.append(j)
            current_norm += 1
    if current_norm:
        if normalize:
            value = d / current_norm
        else:
            value = d
        row.extend(current_row)
        col.extend(current_col)
        data.extend([value for _ in range(current_norm)])
    n = (max_i if max_i > max_j else max_j) + 1
    if add_identity:
        mono = list(range(n))
        row.extend(mono)
        col.extend(mono)
        data.extend([1 for _ in range(n)])
    return coo_matrix((data, (row, col)), shape=(n, n), dtype=np.float64)


def verbose_matrix(A):
    """Stringfies matrix, considering only non-zero entries. Used for debugging."""
    n, m = A.shape
    A = A.toarray()
    return '\n'.join([' '.join(['1' if A[i][j] else ' ' for j in range(m)]) for i in range(n)])


def matrix_partition(A, n, m=None):
    """
    Partitions a coo matrix `A` into four block matrix, with upper-left block having size of `n` by `m`.

    Args:
        A (coo_matrix): A matrix to be partitioned.
        n (int): Row-size for the upper-left block.
        m (int): Column-size for the upper-left block. If None, then m is set to n.

    Returns:
        A tuple of 4 coo matrices.
    """
    A = A.tocoo()
    N, M = A.shape
    if m is None:
        m = n
    row = (([], []), ([], []))
    col = (([], []), ([], []))
    data = (([], []), ([], []))
    for idx, (i, j) in enumerate(zip(A.row, A.col)):
        if i < n:
            if j < m:
                row[0][0].append(i)
                col[0][0].append(j)
                data[0][0].append(A.data[idx])
            else:
                row[0][1].append(i)
                col[0][1].append(j - m)
                data[0][1].append(A.data[idx])
        else:
            if j < m:
                row[1][0].append(i - n)
                col[1][0].append(j)
                data[1][0].append(A.data[idx])
            else:
                row[1][1].append(i - n)
                col[1][1].append(j - m)
                data[1][1].append(A.data[idx])
    A11 = coo_matrix((data[0][0], (row[0][0], col[0][0])), shape=(n, m), dtype=A.dtype)
    A12 = coo_matrix((data[0][1], (row[0][1], col[0][1])), shape=(n, M - m), dtype=A.dtype)
    A21 = coo_matrix((data[1][0], (row[1][0], col[1][0])), shape=(N - n, m), dtype=A.dtype)
    A22 = coo_matrix((data[1][1], (row[1][1], col[1][1])), shape=(N - n, M - m), dtype=A.dtype)
    del row, col, data
    return A11, A12, A21, A22


def drop_tolerance(A, t):
    """
    Drops entry of `A` having absolute value lower than `t`.

    Args:
        A (coo_matrix): Given coo matrix.
        t (float): Tolerance threshld.

    Returns:
        A coo matrix.
    """
    A = A.tocoo()
    row = []
    col = []
    data = []
    for idx, (i, j) in enumerate(zip(A.row, A.col)):
        value = A.data[idx]
        if value < t or value > -t:
            continue
        row.append(i)
        col.append(j)
        data.append(value)
    A = coo_matrix((data, (row, col)), shape=A.shape, dtype=A.dtype)
    del row, col, data
    return A


def degree_reverse_rank_perm(A, reverse=False):
    """
    Computes permutation that sorts nodes by degree.

    Args:
        A (coo_matrix): Given coo matrix.
        reverse (bool): If True, sorts with descending order.

    Returns:
        A permutation of node indices. Like `(i -> j)` is denoted as `perm[i] = j`.
    """
    n, _ = A.shape
    degree = {i: 0 for i in range(n)}
    for i, j in zip(A.row, A.col):
        degree[j] += 1
    bottoms = sorted(degree, key=degree.get, reverse=reverse)
    perm = [0 for _ in range(n)]
    for i in range(n):
        perm[bottoms[i]] = i
    return perm


def reorder_matrix(A, perm, fix_row=False, fix_col=False):
    """
    Reorders given coo matrix with given permutation. You can fix either row or column.

    Args:
        A (coo_matrix): Given coo matrix.
        perm (list): List of node indicies denoting permutation.
        fix_row (bool): If True, reorders column only.
        fix_col (bool): If True, reorders row only.

    Returns:
        A coo matrix.
    """
    A = A.tocoo()
    if not fix_row:
        row = [perm[i] for i in A.row]
    else:
        row = A.row
    if not fix_col:
        col = [perm[j] for j in A.col]
    else:
        col = A.col
    A = coo_matrix((A.data, (row, col)), shape=A.shape, dtype=A.dtype)
    del row, col
    return A


def reorder_vector(v, perm, reverse=False):
    """
    Reorders given vector with given permutation.

    Args:
        v (list or numpy array): Given vector.
        perm (list): List of node indicies denoting permutation.
        reverse (bool): If True, reorders with inverse permutation.

    Returns:
        numpy array of 1d.
    """
    n = len(v)
    w = np.zeros(n)
    for i in range(n):
        if reverse:
            w[perm[i]] = v[i]
        else:
            w[i] = v[perm[i]]
    return np.array(w)


def slashburn(A, k=None, greedy=True):
    """
    Computes SlashBurn of given coo matrix. Currently, only size-ordering works for CCs.

    Args:
        A (coo_matrix): Given coo matrix. It should be an valid adjacency matrix. It considers non-zero entries as edges, and ignores self-loops.
        k (int): For hub selection. There is known rule of thumb, though. `k` = 1 produces (perfectly) optimal solution, but it will be slow.
        greedy (bool): If True, it uses greedy algorithm for hub selection. Slightly slow but slightly more accurate.

    Returns:
        (perm, wing): Permutation of node indicies and size of wing (int).
    """
    n, _ = A.shape
    if k is None:
        k = max(1, int(0.001 * n))
    head = []
    tail = []
    degree = {i: 0 for i in range(n)}
    alists = {i: [] for i in range(n)}
    for i, j in zip(A.row, A.col):
        if i == j:
            continue
        degree[j] += 1
        alists[i].append(j)
    iteration = 0
    while True:
        iteration += 1
        if greedy:
            for _ in range(k):
                if not len(degree):
                    break
                top = max(degree, key=degree.get)
                head.append(top)
                alist = alists[top]
                del degree[top]
                del alists[top]
                for target in alist:
                    if target in degree:
                        degree[target] -= 1
        else:
            tops = sorted(degree, key=degree.get, reverse=True)[:k]
            head.extend(tops)
            for top in tops:
                alist = alists[top]
                del degree[top]
                del alists[top]
                for target in alist:
                    if target in degree:
                        degree[target] -= 1
        if not len(degree):
            break
        nodes = list(degree.keys())
        edges = []
        for source in alists:
            for target in alists[source]:
                if target in alists:
                    edges.append((source, target))
        ccs = scc(nodes, edges)
        m = len(ccs)
        sizes = {i: len(ccs[i]) for i in range(m)}
        ordering = sorted(sizes, key=sizes.get)
        ccs = [ccs[ordering[i]] for i in range(m)]
        # todo: implement hub-ordering
        for cc in ccs:
            size = len(cc)
            if size == 1 or size < k:
                tail.extend(cc)
                for bottom in cc:
                    alist = alists[bottom]
                    del degree[bottom]
                    del alists[bottom]
                    for target in alist:
                        if target in degree:
                            degree[target] -= 1
        assert len(head) + len(tail) + len(degree) == n
        if not len(degree):
            break
    tops = tail + head[::-1]
    perm = [0 for _ in range(n)]
    for i in range(n):
        perm[tops[i]] = i
    return perm, iteration * k
