# evoten

Vectorized computation of tree likelihoods.

### Features
- Supports multiple models
- Highly parallel, processes trees in layers, models and input examples are treated in parallel
- Multiple compute backends (TensorFlow, pytorch)

### Installation

For users:

```bash
git clone https://github.com/felbecker/evoten
cd evoten
pip install -e .[tensorflow]  # or [torch] for pytorch backend
```

For developers:

```bash
pip install -e .[tensorflow,torch,docs]
```


### Minimal example

See [here](https://github.com/felbecker/TensorTree/blob/main/test/example.ipynb).


