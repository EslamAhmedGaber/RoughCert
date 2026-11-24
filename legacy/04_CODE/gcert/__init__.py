"""gcert -- certified expansion of Gaussian-driven outputs (research integration lab, 2026-10-31).

Layers: fields (covariances) -> representation (orthogonal projection interface) ->
functionals (subordinated outputs, chaos data, kernels) -> ledger (exact joint error) ->
transfer (Wasserstein quantile / ES certificates) -> solve (adaptive procedure).
"""
from .fields import FBM, RiemannLiouville, Brownian, CallableField, FractionalOU
from .quadrature import Measure
from .representation import (AnchorProjection, EigenProjection, GridFactor, CovFourierProjection,
                             L2CosineTruncation, NullRepresentation, IdentityRepresentation)
from .functionals import (WickExponential, WickJet, PathwiseExponential, PositivePart, Indicator,
                          Polynomial, Generic)
from .ledger import compute_ledger, first_moment, second_moment
