import os

supported_backends = ["tensorflow", "pytorch"]


"""load the backend, e.g. tensorflow
    depends on the environment variable TENSORTREE_BACKEND
    that can be set before importing tensortree
"""
def load_backend(default = "pytorch"):
    backend_name = os.environ.get("TENSORTREE_BACKEND", default)
    _validate_backend(backend_name)
    if backend_name == "tensorflow":
        import tensortree.backend_tf as backend
    elif backend_name == "pytorch":
        import tensortree.backend_pytorch as backend
    return backend


def _validate_backend(backend):
    if backend not in supported_backends:
        raise ValueError(f"Backend must be one of {supported_backends}")