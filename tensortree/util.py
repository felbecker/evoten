import os
import numpy as np


####################################################################################################
# Backend
####################################################################################################

supported_backends = ["tensorflow", "pytorch"]
    
""" Wraps a backend module.
"""
class Backend(object):
    def __init__(self, backend_module):
        self.backend_module = backend_module

    def __get__(self, instance, owner):
        return self.backend_module
    
    def __getattr__(self, name):
        return getattr(self.backend_module, name)


# Placeholder for lazy backend loading. The user has to decide which backend to use.
# Errors if no backend is loaded.
class NullModule(object):
    def _raise(self):
        raise ValueError("No backend loaded. Please call tensortree.load_backend(<name>) first." 
                         + f"Supported backends are: {supported_backends}.")

    def __get__(self, instance, owner):
        self._raise()
        return None
    

    def __getattr__(self, name):
        self._raise()
        return None


"""load the backend, e.g. tensorflow
    Must be called before any other function in the library.
"""
def set_backend(backend_name = "tensorflow"):
    _validate_backend(backend_name)
    if backend_name == "tensorflow":
        import tensortree.backend_tf as backend_module
    elif backend_name == "tensorflow_uncompiled": #only for testing
        import tensortree.backend_tf as backend_module
        backend_module.decorator = lambda x: x
    elif backend_name == "pytorch":
        import tensortree.backend_pytorch as backend_module
    backend.backend_module = backend_module


def _validate_backend(backend):
    if backend not in supported_backends + ["tensorflow_uncompiled"]:
        raise ValueError(f"Backend must be one of {supported_backends}")


# initial case: load nothing and wait for user to call set_backend
backend = Backend(NullModule())



####################################################################################################
# Input/output utility
####################################################################################################

""" One-hot encodes a list of strings over the given alphabet.
"""
def encode_one_hot(sequences, alphabet):
    ids = np.array([[alphabet.index(c) for c in seq] for seq in sequences])
    return np.eye(len(alphabet))[ids]