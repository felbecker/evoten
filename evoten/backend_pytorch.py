import numpy as np
import torch

from evoten import Backend, util
from evoten.expm_gtr_torch import expm_gtr, expm_ntr


def _ensure_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, list):
        return torch.tensor(x)
    return x


# Documentation in the base class. Hover over the method name to see the docstring.
class BackendTorch(Backend):

    def make_rate_matrix(
            self,
            exchangeabilities,
            equilibrium,
            epsilon=1e-16,
            normalized=True
        ):
        exchangeabilities = _ensure_tensor(exchangeabilities)
        equilibrium = _ensure_tensor(equilibrium)
        Q = exchangeabilities * equilibrium.unsqueeze(-2)
        diag = torch.sum(Q, -1, keepdim=True)
        eye = torch.eye(Q.shape[-1], dtype=Q.dtype, device=Q.device)
        Q = Q - diag * eye
        # normalize
        if normalized:
            mue = equilibrium.unsqueeze(-1) * diag
            mue = torch.sum(mue, dim=-2, keepdim=True)
            Q = Q / mue.clamp(min=epsilon)
        return Q


    def make_transition_probs(self, rate_matrix, distances, equilibrium):
        # P[b,m,i,j] = P(X(tau_b) = j | X(0) = i; model m))
        return expm_gtr(
            _ensure_tensor(rate_matrix),
            _ensure_tensor(distances),
            _ensure_tensor(equilibrium),
        )


    def make_transition_probs_ntr(self, rate_matrix, distances):
        return expm_ntr(_ensure_tensor(rate_matrix), _ensure_tensor(distances))


    def make_branch_lengths(self, kernel):
        kernel = _ensure_tensor(kernel)
        return torch.nn.functional.softplus(kernel)


    def inverse_softplus(self, branch_lengths):
        epsilon=1e-16
        branch_lengths = _ensure_tensor(branch_lengths)
        # Cast to float64 to prevent overflow of large entries
        branch_lengths64 = branch_lengths.double()
        result = torch.log(torch.expm1(branch_lengths64) + epsilon)
        # Cast back to the original data type of `features`
        return result.to(branch_lengths.dtype)


    def make_symmetric_pos_semidefinite(self, kernel):
        kernel = _ensure_tensor(kernel)
        R = 0.5 * (kernel + kernel.transpose(-1, -2)) #make symmetric
        R = torch.nn.functional.softplus(R)
        R = R - torch.diag_embed(torch.diagonal(R, dim1=-2, dim2=-1)) #zero diagonal
        return R


    def make_equilibrium(self, kernel):
        kernel = _ensure_tensor(kernel)
        return torch.nn.functional.softmax(kernel, dim=-1)


    def traverse_branch(
            self,
            X,
            transition_probs,
            transposed=False,
            logarithmic=True,
            stable=True,
        ):
        X = _ensure_tensor(X)
        transition_probs = _ensure_tensor(transition_probs)
        if stable and logarithmic:
            # Numerically stable log-space matmul using the subtract-max trick
            # (mirrors BackendTF.traverse_branch): exponentiating the raw log
            # partials underflows to zero for deep subtrees, which the plain
            # path below then maps to the log_zero sentinel.
            P = transition_probs.to(X.dtype)
            # Handle broadcasting in the L dimension
            if P.ndim == X.ndim + 1 and P.shape[-3] == 1:
                P = P[..., 0, :, :]
            P_T = P if transposed else P.transpose(-1, -2)
            max_X = torch.amax(X, dim=-1, keepdim=True)       # (..., L, 1)
            X_exp = torch.exp(X - max_X)                      # (..., L, d), values in (0, 1]
            if P.ndim == X.ndim:
                # P: (..., d, d), X_exp: (..., L, d) -> result: (..., L, d)
                # transposed=False (upward):   X_exp @ P^T  [result[i] = sum_j P[i,j]*X_exp[j]]
                # transposed=True  (downward): X_exp @ P    [result[j] = sum_i X_exp[i]*P[i,j]]
                result_exp = torch.matmul(X_exp, P_T)
            else:
                # P: (..., L, d, d) -- position-specific matrices
                result_exp = torch.matmul(X_exp.unsqueeze(-2), P_T)[..., 0, :]
            epsilon = float(np.finfo(util.default_dtype).tiny)
            return torch.log(result_exp.clamp(min=epsilon)) + max_X

        # fast matmul version, but requires conversion, might be numerically
        # unstable
        if logarithmic:
            X = self.probs_from_logits(X)
        if transition_probs.ndim == X.ndim + 1 and transition_probs.shape[-3] == 1:
            transition_probs = transition_probs[..., 0, :, :]
        if transposed:
            X = torch.einsum("...td,...dz->...tz", X, transition_probs)
        else:
            X = torch.einsum("...td,...zd->...tz", X, transition_probs)
        if logarithmic:
            X = self.logits_from_probs(X)
        return X


    def aggregate_children_log_probs(self, X, parent_map, num_ancestral):
        parent_map = _ensure_tensor(parent_map).to(device=X.device, dtype=torch.int64)
        Y = torch.zeros(
            num_ancestral, *X.shape[1:], dtype=X.dtype, device=X.device
        )
        # out of place (autograd-safe), index vector only (no expanded index)
        return Y.index_add(0, parent_map, X)


    def loglik_from_root_logits(self, root_logits, equilibrium_logits):
        equilibrium_logits = _ensure_tensor(equilibrium_logits)
        return torch.logsumexp(root_logits + equilibrium_logits, dim=-1)


    def marginals_from_beliefs(self, beliefs, same_loglik=True):
        loglik = torch.logsumexp(beliefs[-1:] if same_loglik else beliefs,
                                 dim=-1,
                                 keepdim=True)
        return beliefs - loglik


    def logits_from_probs(self, probs, log_zero_val=-1e3):
        probs = _ensure_tensor(probs)
        epsilon = float(np.finfo(util.default_dtype).tiny)
        logits = torch.log(probs.clamp(min=epsilon))
        zero_mask = (probs == 0).to(logits.dtype)
        logits = (1-zero_mask) * logits + zero_mask * log_zero_val
        return logits


    def probs_from_logits(self, logits):
        return torch.exp(logits)


    def gather(self, tensor, indices, axis=0):
        tensor = _ensure_tensor(tensor)
        indices = _ensure_tensor(indices).to(device=tensor.device, dtype=torch.int64)
        return torch.index_select(tensor, axis, indices.reshape(-1)).reshape(
            *tensor.shape[:axis], *indices.shape, *tensor.shape[axis + 1:]
        )


    def concat(self, tensors, axis=0):
        tensors = [_ensure_tensor(t) for t in tensors]
        return torch.cat(tensors, axis=axis)


    def expand(self, X, axis):
        X = _ensure_tensor(X)
        return torch.unsqueeze(X, axis)


    def make_zeros(self, leaves, models, num_nodes):
        return torch.zeros(
            (num_nodes, models, leaves.shape[-2], leaves.shape[-1]),
            dtype=leaves.dtype, device=leaves.device
        )


    def ensure_tensor(self, x):
        return _ensure_tensor(x)


    def ensure_index(self, values, like):
        device = like.device if isinstance(like, torch.Tensor) else None
        return torch.as_tensor(np.asarray(values, dtype=np.int64), device=device)


    def device_key(self, like):
        return str(like.device) if isinstance(like, torch.Tensor) else "cpu"
