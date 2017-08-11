# Bear

Python implementation of [BEAR][1] and [SlashBurn][2].
This is an experimental repo for PPR (personalized pagerank), but well, it does work in blazing speed.

---

### Requirements

* `scipy`
* `numpy`
* `networkx` (optional)
* `tensorflow` (optional)

I used `tensorflow` for GPU acceleration of matrix computations.
Why not `pytorch` or `theano`? PR it.

---

### Documentation

All PPR implementation assumes that given graph has no sink (irreducible), which means normalized-tranposed google matrix should be a valid stochastic matrix.
Note that any of approximation scheme can produce a non-stochastic vectors. You should normalize it if necessary.

Any of PPR class defined in `ppr.py` has 4(5) methods:
* `__init__(self, jump_prob=0.05, *args, **kwargs)`: `jump_prob` is a common option for all algorithms denoting a jumping probability for PPR, while other options are specified for each algorithm.
* `preprocess(self, filename)`: Reads matrix directly from csv file, then proceed to preprocessing step defined in each algorithm. The csv file **must** be csv format of graph edges, **sorted in row-first order** (see `data/small.csv` for sample). Note that we can compute the H matrix while reading!
* `query(self, q)`: Computes PPR for given query vector `q`. It should be a 1-d `numpy` array of dimension `n`. Returns pagerank vector.
* `save(self, filename)`: Just pickle dumping.
* `load(self, filename)`: Just pickle loading. Not fully implemented (PR it).

Four implemented PPR algorithms are:
* `PPRNetworkx`: Computes PPR using `networkx`. Requires installation of the library. Note that this is the slowest iterative method (since `networkx` is implemented for general purpose).
* `PPRIterative`: Computes PPR using iterative method. Simplest baseline. No preprocessing, but slow in query-time.
* `PPRLUDecomposition`: Computes PPR using LU decomposition after ordering nodes with degree of nodes.
* `PPRBear`: Computes PPR using BEAR with SlashBurn. I do know that this `PPR` prefix is lame.

---

### Usage

Assume graph is store in some csv file named `small.csv` with row-first order (**must** be):
```
0,1
1,0
1,2
2,1
2,3
...
```
(This example data can be found in `data/small.csv`.)
You can compute a simple personalized pagerank via following code.
```python
import numpy as np
from ppr import PPRBear as Bear
bear = Bear()
bear.preprocess('data/small.csv')
r = bear.query(np.ones(15)/15)
print(r.sum())  # 1.0
```
Simple benchmark code is located in `benchmark.py`. But well, you should try these algorithms on graphs with millions of nodes.

Note that there are also many useful functions in `utils.py`! Specially, full implementation of Strongly Connected Components (Kosaraju's algorithm) and SlashBurn (Kang's algorithm) is in there.
```python
from utils import scc
nodes = [1, 2, 3, 4]
edges = [(1, 2), (2, 1), (3, 4), (4, 3)]
ccs = scc(nodes, edges)
print(ccs)  # [[3, 4], [1, 2]]
```
```python
import numpy as np
from scipy.sparse import coo_matrix
from utils import verbose_matrix, reorder_matrix, slashburn
A = np.array([[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1]], dtype=np.int32)
A = coo_matrix(A)
print(verbose_matrix(A))
"""
1 1
1 1 1 1
  1 1
  1   1
"""
perm, wing = slashburn(A)
print(wing)  # 1
A = reorder_matrix(A, perm)
print(verbose_matrix(A))
"""
1     1
  1   1
    1 1
1 1 1 1
"""
```

---

### Using with Tensorflow

They're located in `ppr_tf.py`:
* `PPRIterativeTF`
* `PPRLUDecompositionTF`
* `PPRBearTF`

Due to the initialization of tensorflow variables, preprocessing steps are merged into `__init__`.
See `benchmark.py` for more detailed usage.
Currently, since tensorflow does not support various sparse matrix manipulations like LU decomposition solver, only sparse-dense multiplication is used.
Note that my GPU is GeForce GTX 1080.
Results of some experiments are located in `profile.txt`.
You can download some large datasets in [here][3].

[1]: http://dl.acm.org/citation.cfm?id=2723716
[2]: http://ieeexplore.ieee.org/abstract/document/6807798/
[3]: https://datalab.snu.ac.kr/bear/
