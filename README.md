# TensorTree

Vectorized computation of tree likelihoods.

### Features
- Supports multiple models
- Highly parallel, processes trees in layers, models and input examples are treated in parallel
- Multiple compute backends (TensorFlow, pytorch)

### Minimal example

```
import tensortree
from io import StringIO
import numpy as np

# parse a tree 
handle = StringIO("(A:0.1,B:0.2,(C:0.3,D:0.4):0.5);")
tree = tensortree.TreeHandler.read(handle)

# leaves with shape (n, L, 1, d)
leaves = np.array([[0,2,3], [1,1,0], [2,1,0], [3,1,2]])
leaves = np.eye(4)[leaves]
leaves = leaves[:,np.newaxis]
leaf_names = ['A', 'B', 'C', 'D']

# use a rate matrix, here Jukes-Cantor
rate_matrix = np.array([[[-1, 1./3, 1./3, 1./3], 
                         [1./3, -1, 1./3, 1./3], 
                         [1./3, 1./3, -1, 1./3], 
                         [1./3, 1./3, 1./3, -1]]])


L = tensortree.model.loglik(leaves, leaf_names, tree, rate_matrix, 
                equilibrium_logits=np.log([[1./4, 1./4, 1./4, 1./4]]),
                leaves_are_probabilities=True)
```
`[[-7.6051, -5.7626, -7.2497]]`
