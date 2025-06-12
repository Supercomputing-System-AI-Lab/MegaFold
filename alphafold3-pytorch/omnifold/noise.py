import numpy as np
import torch
from beartype.typing import List, Tuple


class DiffusionSDE:
    """Diffusion SDE class.

    Adapted from: https://github.com/HannesStark/FlowSite
    and https://github.com/BioinfoMachineLearning/FlowDock
    """

    def __init__(self, sigma: torch.Tensor, tau_factor: float = 5.0):
        """Initialize the Diffusion SDE class."""
        self.lamb = 1 / sigma**2
        self.tau_factor = tau_factor

    def var(self, t: torch.Tensor) -> torch.Tensor:
        """Calculate the variance of the diffusion SDE."""
        return (1 - torch.exp(-self.lamb * t)) / self.lamb

    def max_t(self) -> float:
        """Calculate the maximum time of the diffusion SDE."""
        return self.tau_factor / self.lamb

    def mu_factor(self, t: torch.Tensor) -> torch.Tensor:
        """Calculate the mu factor of the diffusion SDE."""
        return torch.exp(-self.lamb * t / 2)


class HarmonicSDE:
    """Harmonic SDE class.

    Adapted from: https://github.com/HannesStark/FlowSite
    and https://github.com/BioinfoMachineLearning/FlowDock
    """

    def __init__(self, J: torch.Tensor | None = None, diagonalize: bool = True):
        """Initialize the Harmonic SDE class."""
        self.l_index = 1
        self.use_cuda = False
        if not diagonalize:
            return
        if J is not None:
            self.D, self.P = np.linalg.eigh(J)
            self.N = self.D.size

    @staticmethod
    def diagonalize(
        N,
        ptr: torch.Tensor,
        edges: List[Tuple[int, int]] | None = None,
        antiedges: List[Tuple[int, int]] | None = None,
        a=1,
        b=0.3,
        lamb: torch.Tensor | None = None,
        device: str | torch.device | None = None,
    ):
        """Diagonalize using the Harmonic SDE."""
        device = device or ptr.device
        J = torch.zeros((N, N), device=device)
        if edges is None:
            for i, j in zip(np.arange(N - 1), np.arange(1, N)):
                J[i, i] += a
                J[j, j] += a
                J[i, j] = J[j, i] = -a
        else:
            for i, j in edges:
                J[i, i] += a
                J[j, j] += a
                J[i, j] = J[j, i] = -a
        if antiedges is not None:
            for i, j in antiedges:
                J[i, i] -= b
                J[j, j] -= b
                J[i, j] = J[j, i] = b
        if edges is not None:
            J += torch.diag(lamb)

        Ds, Ps = [], []
        for start, end in zip(ptr[:-1], ptr[1:]):
            D, P = torch.linalg.eigh(J[start:end, start:end])
            D_ = D
            if edges is None:
                D_inv = 1 / D
                D_inv[0] = 0
                D_ = D_inv
            Ds.append(D_)
            Ps.append(P)
        return torch.cat(Ds), torch.block_diag(*Ps)

    def eigens(self, t):
        """Calculate the eigenvalues of `sigma_t` using the Harmonic SDE."""
        np_ = torch if self.use_cuda else np
        D = 1 / self.D * (1 - np_.exp(-t * self.D))
        t = torch.tensor(t, device="cuda").float() if self.use_cuda else t
        return np_.where(D != 0, D, t)

    def conditional(self, mask, x2):
        """Calculate the conditional distribution using the Harmonic SDE."""
        J_11 = self.J[~mask][:, ~mask]
        J_12 = self.J[~mask][:, mask]
        h = -J_12 @ x2
        mu = np.linalg.inv(J_11) @ h
        D, P = np.linalg.eigh(J_11)
        z = np.random.randn(*mu.shape)
        return (P / D**0.5) @ z + mu

    def A(self, t, invT=False):
        """Calculate the matrix `A` using the Harmonic SDE."""
        D = self.eigens(t)
        A = self.P * (D**0.5)
        if not invT:
            return A
        AinvT = self.P / (D**0.5)
        return A, AinvT

    def Sigma_inv(self, t):
        """Calculate the inverse of the covariance matrix `Sigma` using the Harmonic SDE."""
        D = 1 / self.eigens(t)
        return (self.P * D) @ self.P.T

    def Sigma(self, t):
        """Calculate the covariance matrix `Sigma` using the Harmonic SDE."""
        D = self.eigens(t)
        return (self.P * D) @ self.P.T

    @property
    def J(self):
        """Return the matrix `J`."""
        return (self.P * self.D) @ self.P.T

    def rmsd(self, t):
        """Calculate the root mean square deviation using the Harmonic SDE."""
        l_index = self.l_index
        D = 1 / self.D * (1 - np.exp(-t * self.D))
        return np.sqrt(3 * D[l_index:].mean())

    def sample(self, t, x=None, score=False, k=None, center=True, adj=False):
        """Sample from the Harmonic SDE."""
        l_index = self.l_index
        np_ = torch if self.use_cuda else np
        if x is None:
            if self.use_cuda:
                x = torch.zeros((self.N, 3), device="cuda").float()
            else:
                x = np.zeros((self.N, 3))
        if t == 0:
            return x
        z = (
            np.random.randn(self.N, 3)
            if not self.use_cuda
            else torch.randn(self.N, 3, device="cuda").float()
        )
        D = self.eigens(t)
        xx = self.P.T @ x
        if center:
            z[0] = 0
            xx[0] = 0
        if k:
            z[k + l_index :] = 0
            xx[k + l_index :] = 0

        out = np_.exp(-t * self.D / 2)[:, None] * xx + np_.sqrt(D)[:, None] * z

        if score:
            score = -(1 / np_.sqrt(D))[:, None] * z
            if adj:
                score = score + self.D[:, None] * out
            return self.P @ out, self.P @ score
        return self.P @ out

    def score_norm(self, t, k=None, adj=False):
        """Calculate the score norm using the Harmonic SDE."""
        if k == 0:
            return 0
        l_index = self.l_index
        np_ = torch if self.use_cuda else np
        k = k or self.N - 1
        D = 1 / self.eigens(t)
        if adj:
            D = D * np_.exp(-self.D * t)
        return (D[l_index : k + l_index].sum() / self.N) ** 0.5

    def inject(self, t, modes):
        """Inject noise along the given modes using the Harmonic SDE."""
        z = (
            np.random.randn(self.N, 3)
            if not self.use_cuda
            else torch.randn(self.N, 3, device="cuda").float()
        )
        z[~modes] = 0
        A = self.A(t, invT=False)
        return A @ z

    def score(self, x0, xt, t):
        """Calculate the score of the diffusion kernel using the Harmonic SDE."""
        Sigma_inv = self.Sigma_inv(t)
        mu_t = (self.P * np.exp(-t * self.D / 2)) @ (self.P.T @ x0)
        return Sigma_inv @ (mu_t - xt)

    def project(self, X, k, center=False):
        """Project onto the first `k` nonzero modes using the Harmonic SDE."""
        l_index = self.l_index
        D = self.P.T @ X
        D[k + l_index :] = 0
        if center:
            D[0] = 0
        return self.P @ D

    def unproject(self, X, mask, k, return_Pinv=False):
        """Find the vector along the first k nonzero modes whose mask is closest to `X`"""
        l_index = self.l_index
        PP = self.P[mask, : k + l_index]
        Pinv = np.linalg.pinv(PP)
        out = self.P[:, : k + l_index] @ Pinv @ X
        if return_Pinv:
            return out, Pinv
        return out

    def energy(self, X):
        """Calculate the energy using the Harmonic SDE."""
        l_index = self.l_index
        return (self.D[:, None] * (self.P.T @ X) ** 2).sum(-1)[l_index:] / 2

    @property
    def free_energy(self):
        """Calculate the free energy using the Harmonic SDE."""
        l_index = self.l_index
        return 3 * np.log(self.D[l_index:]).sum() / 2

    def KL_H(self, t):
        """Calculate the Kullback-Leibler divergence using the Harmonic SDE."""
        l_index = self.l_index
        D = self.D[l_index:]
        return -3 * 0.5 * (np.log(1 - np.exp(-D * t)) + np.exp(-D * t)).sum(0)
