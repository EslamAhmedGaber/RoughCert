"""Gaussian representation interface (the shared object of all modules).

A representation of a field Y_s = W(e_s) is a map s -> g(s) in R^k such that
    W(Pi e_s) = g(s) . xi,        xi ~ N(0, I_k),
for an ORTHOGONAL projection Pi of the Gaussian Hilbert space.  Then
    R_Pi(s, u) = g(s) . g(u),     D(s, u) = R(s, u) - R_Pi(s, u)
is the exact residual covariance: positive semidefinite, and the residual
W((I - Pi) e_s) is independent of xi.  Every downstream statement (ledgers,
residual closure, coherent reconstruction) requires exactly this property.

``is_projection`` records whether a backend satisfies it *by construction*;
``check`` tests it numerically on a grid (it can refute, not prove).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import cholesky, solve_triangular


class Representation:
    name = "abstract"
    is_projection = True
    nested = True

    def __init__(self, field):
        self.field = field

    @property
    def rank(self):  # pragma: no cover - interface
        raise NotImplementedError

    def loadings(self, s):  # pragma: no cover - interface
        raise NotImplementedError

    def RPi(self, s, u):
        gs = self.loadings(np.atleast_1d(np.asarray(s, float)).ravel())
        gu = self.loadings(np.atleast_1d(np.asarray(u, float)).ravel())
        return np.einsum("ik,ik->i", gs, gu).reshape(np.broadcast(np.asarray(s), np.asarray(u)).shape)

    def RPi_pairs(self, S, U):
        """R_Pi on arrays of pairs (S, U) of equal shape."""
        S = np.asarray(S, float)
        U = np.asarray(U, float)
        gs = self.loadings(S.ravel())
        gu = self.loadings(U.ravel())
        return np.einsum("ik,ik->i", gs, gu).reshape(S.shape)

    def D(self, S, U):
        return self.field.R(S, U) - self.RPi_pairs(S, U)

    def breakpoints(self):
        """Points where s -> g(s) may lose smoothness (anchors, grid nodes)."""
        return list(self.field.singular_points())

    def check(self, grid=None, tol=1e-10):
        """Numerical PSD/consistency check of the residual kernel on a grid."""
        T = self.field.T
        grid = np.linspace(T / 40, T, 40) if grid is None else np.asarray(grid, float)
        S, U = np.meshgrid(grid, grid, indexing="ij")
        Dm = self.D(S, U)
        Dm = 0.5 * (Dm + Dm.T)
        ev = np.linalg.eigvalsh(Dm)
        scale = max(np.max(np.abs(self.field.R(S, U))), 1e-300)
        return {"backend": self.name, "rank": self.rank, "min_eig_D_rel": float(ev.min() / scale),
                "max_diag_D_rel": float(np.max(np.diag(Dm)) / scale),
                "psd_ok": bool(ev.min() >= -tol * scale),
                "declared_projection": self.is_projection}

    def describe(self):
        return {"backend": self.name, "rank": self.rank, "projection": self.is_projection,
                "nested": self.nested, **self.field.describe()}


class NullRepresentation(Representation):
    """Pi = 0 (no retained coordinates)."""
    name = "null"

    @property
    def rank(self):
        return 0

    def loadings(self, s):
        return np.zeros((np.size(s), 0))

    def RPi_pairs(self, S, U):
        return np.zeros(np.broadcast(np.asarray(S), np.asarray(U)).shape)


class IdentityRepresentation(Representation):
    """Pi = Id (formal; only R_Pi = R is used)."""
    name = "identity"

    @property
    def rank(self):
        return math.inf

    def loadings(self, s):  # pragma: no cover
        raise NotImplementedError("identity has no finite loading; use RPi_pairs")

    def RPi_pairs(self, S, U):
        return self.field.R(S, U)

    def breakpoints(self):
        return list(self.field.singular_points())


class AnchorProjection(Representation):
    """Conditioning on the field values at anchor times (any covariance, any H).

    W(Pi e_s) = R(s, A) R(A, A)^{-1} Y_A,  so  g(s) = L^{-1} R(A, s) with R(A,A) = L L^T.
    This is the projection onto span{e_a : a in A}; it is exact in the continuum.
    """
    name = "anchors"

    def __init__(self, field, anchors):
        super().__init__(field)
        a = np.asarray(sorted(set(float(x) for x in anchors)), float)
        if np.any(a <= 0) or np.any(a > field.T):
            raise ValueError("anchors must lie in (0, T]")
        self.anchors = a
        RA = field.R(a[:, None], a[None, :])
        self.L = cholesky(RA + 0.0 * np.eye(a.size), lower=True)

    @property
    def rank(self):
        return self.anchors.size

    def loadings(self, s):
        s = np.atleast_1d(np.asarray(s, float)).ravel()
        RAs = self.field.R(self.anchors[:, None], s[None, :])          # (k, n)
        return solve_triangular(self.L, RAs, lower=True).T             # (n, k)

    def breakpoints(self):
        return sorted(set(list(self.field.singular_points()) + list(self.anchors)))

    def refine(self, new_anchor):
        return AnchorProjection(self.field, list(self.anchors) + [float(new_anchor)])


class EigenProjection(Representation):
    """Continuum Karhunen--Loeve projection via a Nystrom discretisation of R on a fine grid.

    The retained coordinates are the leading Nystrom eigenfunctions, which are linear
    functionals of the field values on the grid; the projection is therefore onto the span of
    those functionals (an exact orthogonal projection of the Gaussian space).  The loading of an
    off-grid point is computed exactly by conditioning: g(s) = Cov(Y_s, xi).
    """
    name = "nystrom-KL"

    def __init__(self, field, k, n_grid=600):
        super().__init__(field)
        x, w = np.polynomial.legendre.leggauss(n_grid)
        T = field.T
        t = 0.5 * T * (x + 1.0)
        w = 0.5 * T * w
        C = field.R(t[:, None], t[None, :])
        Wh = np.sqrt(w)
        A = Wh[:, None] * C * Wh[None, :]
        lam, V = np.linalg.eigh(A)
        idx = np.argsort(lam)[::-1][:k]
        lam, V = lam[idx], V[:, idx]
        # xi_j = lam_j^{-1/2} sum_i sqrt(w_i) V_ij Y_{t_i}  (orthonormal functionals of the grid values)
        self.t, self.coef = t, (Wh[:, None] * V) / np.sqrt(lam)[None, :]   # (n_grid, k)
        Cov = self.coef.T @ C @ self.coef
        Lc = cholesky(0.5 * (Cov + Cov.T), lower=True)                    # re-orthonormalise
        self.coef = self.coef @ np.linalg.inv(Lc).T
        self.k = k

    @property
    def rank(self):
        return self.k

    def loadings(self, s):
        s = np.atleast_1d(np.asarray(s, float)).ravel()
        Rst = self.field.R(s[:, None], self.t[None, :])                 # (n, n_grid)
        return Rst @ self.coef


class GridFactor(Representation):
    """Paper IX finite factors on a grid (spectral, endpoint-conditioned, dyadic hierarchy).

    The factor is a projection of the GRID Gaussian vector; between grid nodes the rows are
    interpolated linearly, which defines the interpolated driver used by the solver.  For
    functionals of the original continuum field the loading of an off-grid point is obtained by
    exact conditioning on the retained functionals (as in EigenProjection).
    """
    name = "grid-factor"

    def __init__(self, field, n_grid, M, kind="endpoint"):
        super().__init__(field)
        T = field.T
        self.grid = T * np.arange(1, n_grid + 1) / n_grid
        C = field.R(self.grid[:, None], self.grid[None, :])
        if kind == "spectral":
            lam, V = np.linalg.eigh(C)
            idx = np.argsort(lam)[::-1][:M]
            G = V[:, idx] * np.sqrt(lam[idx])[None, :]
        elif kind == "endpoint":
            g0 = C[:, -1] / math.sqrt(C[-1, -1])
            Cp = C - np.outer(g0, g0)
            lam, V = np.linalg.eigh(Cp)
            idx = np.argsort(lam)[::-1][:M - 1]
            G = np.column_stack([g0, V[:, idx] * np.sqrt(np.maximum(lam[idx], 0.0))[None, :]])
        elif kind == "dyadic":
            order = [n_grid - 1]
            L = int(round(math.log2(n_grid)))
            for lev in range(1, L + 1):
                for j in range(1, 2 ** (lev - 1) + 1):
                    order.append((2 * j - 1) * n_grid // 2 ** lev - 1)
            order = order[:M]
            sel = np.asarray(order)
            CA = C[np.ix_(sel, sel)]
            La = cholesky(CA, lower=True)
            G = solve_triangular(La, C[sel, :], lower=True).T
        else:
            raise ValueError(kind)
        self.G, self.kind, self.M = G, kind, M
        # retained coordinates as functionals of grid values: xi = A Y_grid with Cov(xi)=I, G = C A^T
        self.A = np.linalg.lstsq(C, G, rcond=None)[0].T                  # (M, n_grid)
        self.name = f"grid-factor[{kind}]"

    @property
    def rank(self):
        return self.M

    def loadings(self, s):
        s = np.atleast_1d(np.asarray(s, float)).ravel()
        Rsg = self.field.R(s[:, None], self.grid[None, :])
        return Rsg @ self.A.T

    def residual_trace_fraction(self):
        C = self.field.R(self.grid[:, None], self.grid[None, :])
        return float(np.trace(C - self.G @ self.G.T) / np.trace(C))


class CovFourierProjection(Representation):
    """CovFourier covariance-orthogonal projection (Paper 2), fBm with 1/2 <= H < 1.

    e_s = 1_[0,s] in the integrand space; g(s) = white CovFourier coordinates of 1_[0,s].
    Requires the covfourier package (v1.2.0) on sys.path.
    """
    name = "CovFourier"

    def __init__(self, field, n, Naux=1 << 14, enrich=True):
        super().__init__(field)
        if field.name not in ("fBm", "Brownian") or not (0.5 <= field.H < 1.0):
            raise ValueError("CovFourier backend supports fBm with 1/2 <= H < 1 only (unsupported input)")
        from covfourier.engine import CovarianceFourier
        self.cf = CovarianceFourier(field.H, n, T=field.T, Naux=Naux, enrich=enrich)
        self.Naux, self.n = Naux, n
        self._cache = {}

    @property
    def rank(self):
        return self.cf.D

    def _indicator(self, t):
        T, N = self.field.T, self.Naux
        k = np.arange(N, dtype=float)
        a = np.empty(N)
        a[0] = t / math.sqrt(T)
        w = k[1:] * math.pi / T
        a[1:] = math.sqrt(2.0 / T) * np.sin(w * t) / w
        return a

    def loadings(self, s):
        s = np.atleast_1d(np.asarray(s, float)).ravel()
        out = np.empty((s.size, self.rank))
        for i, t in enumerate(s):
            key = round(float(t), 15)
            if key not in self._cache:
                r = self.cf.project(self._indicator(float(t)))
                self._cache[key] = r["coords"] * self.field.scale
            out[i] = self._cache[key]
        return out


class L2CosineTruncation(Representation):
    """Paper 1 noise truncation B^J(t) = sum_{k<J} m_k(t) Z_k (NOT a covariance projection for H != 1/2).

    Provided as a negative control: it yields a valid finite Gaussian approximation, but
    R - g g^T is not its residual covariance and the residual is correlated with xi.
    The loading below is expressed in white coordinates xi = L^{-1} Z (G_J = L L^T).
    """
    name = "L2-cosine (Paper 1)"
    is_projection = False

    def __init__(self, field, J):
        super().__init__(field)
        from covfourier.gram import CosineGram
        self.Gm = CosineGram(field.H, field.T)
        GJ = self.Gm.dense(J)
        self.L = cholesky(GJ, lower=True)
        self.J = J

    @property
    def rank(self):
        return self.J

    def loadings(self, s):
        s = np.atleast_1d(np.asarray(s, float)).ravel()
        T = self.field.T
        k = np.arange(self.J, dtype=float)
        M = np.empty((s.size, self.J))
        M[:, 0] = s / math.sqrt(T)
        w = k[1:] * math.pi / T
        M[:, 1:] = math.sqrt(2.0 / T) * np.sin(np.outer(s, w)) / w[None, :]
        return (M @ self.L) * self.field.scale     # B^J(s) = m(s)^T Z = (L^T m(s)) . xi
