"""certflow -- formal verification of neural-network controllers through implicit (power-flow) models.

Core (application-agnostic):
    certflow.fp        rigorous floating-point error bounds
    certflow.tm        degree-2 Taylor models / affine zonotopes
    certflow.nn        architecture-agnostic NN reachability (NNReachable), DeepZ ReLU/Tanh/Sigmoid,
                       torch / SB3 translators, masked-argmax reachable-action analysis
    certflow.krawczyk  validated Krawczyk existence/uniqueness for F(z) = S(e) with affine Jacobian
    certflow.sets      input zonotopes and e-space sub-boxes
    certflow.split     adaptive splitting + Windows-safe process pool
Application:
    certflow.power     AC power flow: grid, controllers, verifier, pluggable constraints, restoration
"""
from . import fp, krawczyk, nn, policy, robustness, sets, split, tm
from .tm import TM

__version__ = "0.1.0"
__all__ = ["fp", "tm", "nn", "krawczyk", "policy", "robustness", "sets", "split", "TM"]
