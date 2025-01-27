import os
import numpy as np



default_dtype = np.float32
supported_backends = ["tensorflow", "pytorch"]
    
  

class Backend():
    def __init__(self):
        # the wrapped backend that contains the actual implementation
        # the global backend will always be the same object to avoid importing issues
        # but the wrapped backend can change at will
        self.wrapped_backend : Backend = None
    
    
    def make_rate_matrix(self, exchangeabilities, equilibrium, epsilon=1e-16, normalized=True):
        """Constructs a stack of normalized rate matrices, i.e. 1 time unit = 1 expected mutation per site.

        Args:
            exchangeabilities: Symmetric, positive-semidefinite exchangeability matrices with zero diagonal. Shape: (k, d, d) 
            equilibrium: A vector of relative frequencies. Shape: (k, d) 
            epsilon: Small value to avoid division by zero.
            normalized: If True, the rate matrices are normalized.

        Returns:
            Normalized rate matrices. Output shape: (k, d, d) 
        """
        return self.wrapped_backend.make_rate_matrix(exchangeabilities, equilibrium, epsilon, normalized)
    
    
    def make_transition_probs(self, rate_matrix, distances):
        """Constructs a probability matrix of mutating or copying one an input to another over a given amount of evolutionary time.
        
        Args:
            rate_matrix: Rate matrix of shape (k, d, d).
            distances: Evolutionary times of shape (n, k) 
        
        Returns:
            Stack of probability matrices of shape (n, k, d, d)
        """
        return self.wrapped_backend.make_transition_probs(rate_matrix, distances)
    

    def make_branch_lengths(self, kernel):
        """
        Converts a kernel of parameters to positive branch lengths.
        """
        return self.wrapped_backend.make_branch_lengths(kernel)
    

    def make_symmetric_pos_semidefinite(self, kernel):
        """
        Constructs a stack of symmetric, positive-semidefinite matrices with zero diagonal from a parameter kernel.
        Note: Uses an overparameterized d x d kernel for speed of computation.

        Args:
            kernel: Tensor of shape (k, d, d).
        """
        return self.wrapped_backend.make_symmetric_pos_semidefinite(kernel)
    

    def make_equilibrium(kernel):
        """ Constructs a stack of equilibrium distributions from a parameter kernel.
        """
        return self.wrapped_backend.make_equilibrium(kernel)
    

    def traverse_branch(self, X, branch_probabilities, transposed=False, logarithmic=True):
        """
        Computes P(X | X') for a branch {X, X'} given the transition matrix.

        Args:
            X: tensor with logits of shape (n, k, L, d). 
            branch_probabilities: The transition matrices along each branch. Tensor of shape (n, k, d, d). 
            transposed: If True, computes P(X' | X) instead.
            
        Returns:
            logits of shape (n, k, L, d)
        """
        return self.wrapped_backend.traverse_branch(X, branch_probabilities, transposed, logarithmic)
    

    def aggregate_children_log_probs(self, X, parent_map, num_ancestral):
        r"""
        Aggregates the partial log-likelihoods of child nodes.

        Args:
            X: tensor of shape (n, ...)
            parent_map: A list-like object of shape (n) that contains the index of parent nodes.
            num_ancestral: Total number of ancestral nodes.

        Returns:
            tensor of shape (num_ancestral, ...).
        """
        return self.wrapped_backend.aggregate_children_log_probs(X, parent_map, num_ancestral)
    
    
    def loglik_from_root_logits(self, root_logits, equilibrium_logits):
        """
        Computes log likelihoods given root logits and equilibrium distributions.

        Args:
            root_logits: Logits at the root node of shape (k, L, d)
            equilibrium_logits: Equilibrium distribution logits of shape (k, d)

        Returns:
            Log-likelihoods of shape (k, L)
        """
        return self.wrapped_backend.loglik_from_root_logits(root_logits, equilibrium_logits)
    

    def marginals_from_beliefs(self, beliefs):
        """
        Computes marginal distributions log P(u) from beliefs log P(u, data).

        Args:
            beliefs: Logits of shape (n, k, L, d)

        Returns:
            Marginal distributions of shape (n, k, L, d)
        """
        return self.wrapped_backend.marginals_from_beliefs(beliefs)
    

    def logits_from_probs(self, probs, log_zero_val=-1e3):
        """ Computes element-wise logarithm with output_i=log_zero_val where x_i=0.
        """
        return self.wrapped_backend.logits_from_probs(probs, log_zero_val)
    

    def probs_from_logits(self, logits):
        """ 
        Computes element-wise exp
        """
        return self.wrapped_backend.probs_from_logits(logits)


    def reorder(self, tensor, permutation, axis=0):
        """ 
        Reorders a tensor along an axis.
        """
        return self.wrapped_backend.reorder(tensor, permutation, axis)


    def concat(self, tensors, axis=0):
        """ 
        Concatenates tensors along an axis.
        """
        return self.wrapped_backend.concat(tensors, axis)


    def make_zeros(self, leaves, models, num_nodes):
        """
        Initializes the ancestral logits tensor with zeros.
        """
        return self.wrapped_backend.make_zeros(leaves, models, num_nodes)
    


def set_backend(backend_name = "tensorflow"):
    """ Loads one of the following backends: ["tensorflow", "pytorch"]
        Must be called before any other function in the library.
    """
    _validate_backend(backend_name)
    if backend_name == "tensorflow":
        from tensortree.backend_tf import BackendTF 
        backend.wrapped_backend = BackendTF()
    elif backend_name == "pytorch":
        from tensortree.backend_pytorch import BackendTorch 
        backend.wrapped_backend = BackendTorch()
    


def _validate_backend(backend):
    if backend not in supported_backends:
        raise ValueError(f"Backend must be one of {supported_backends}")



# null object that raises errors upon usage that request to call set_backend
class NullBackend(Backend):
    def __getattr__(self, name):
        raise ValueError("No backend loaded. Please call tensortree.set_backend(<name>) first." 
                         + f"Supported backends are: {supported_backends}.")


# initial case: load dummy and wait for user to call set_backend
backend : Backend = Backend()
backend.wrapped_backend = NullBackend()



####################################################################################################
# Input/output utility
####################################################################################################

def encode_one_hot(sequences, alphabet):
    """ One-hot encodes a list of strings over the given alphabet.
    """
    ids = np.array([[alphabet.index(c) for c in seq] for seq in sequences])
    return np.eye(len(alphabet), dtype=default_dtype)[ids]