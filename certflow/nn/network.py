"""Architecture-agnostic NN reachability.

The verifier only talks to :class:`NNReachable`:

    reach(x: TM) -> TM      sound enclosure of the network output over the input set x(e)
    forward(x)   -> y       concrete evaluation (float64), used for operating points / Monte-Carlo checks

:class:`Sequential` composes the layer transformers of :mod:`certflow.nn.layers`; :func:`from_torch` and
:func:`from_state_dict` translate trained models into it (torch is NOT required by the core).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..fp import U, U32
from ..tm import TM
from .layers import (ACTIVATIONS, Layer, Linear, MinMaxDenormalize, MinMaxNormalize, ReachContext, ReLU, Sigmoid,
                     Tanh, enter_identity, exit_identity)


class NNReachable(ABC):
    """Interface consumed by every verifier: an input->output map with a sound set-valued extension."""

    in_dim: int
    out_dim: int

    @abstractmethod
    def reach(self, x: TM) -> TM:
        ...

    @abstractmethod
    def forward(self, x) -> np.ndarray:
        ...

    def __call__(self, x):
        return self.forward(x)


class Sequential(NNReachable):
    """A chain of layers.  Optional fused entry (e.g. MinMaxNormalize) first and fused exit
    (e.g. MinMaxDenormalize) last.

    deploy_u : unit roundoff of the deployed network's arithmetic (float64 -> 2^-53, float32 -> 2^-24).
               The output enclosure covers what the DEPLOYED float network outputs, not only the exact-real one.
    fold_below : smooth-activation relaxation errors at or below this size are folded into the interval
               error instead of creating a new symbol (0 = never fold; keeps DeepZ semantics).
    """

    def __init__(self, layers, deploy_u=U, fold_below=0.0, in_dim=None):
        self.layers = list(layers)
        self.ctx = ReachContext(deploy_u=float(deploy_u), fold_below=float(fold_below))
        lin = [L for L in self.layers if isinstance(L, Linear)]
        self.in_dim = in_dim if in_dim is not None else (lin[0].W.shape[1] if lin else None)
        self.out_dim = lin[-1].W.shape[0] if lin else None

    def __repr__(self):
        return "Sequential(" + ", ".join(map(repr, self.layers)) + f"; deploy_u=2^{np.log2(self.ctx.deploy_u):.0f})"

    def forward(self, x):
        y = np.asarray(x, float)
        for L in self.layers:
            y = L.forward(y)
        return y

    def reach(self, x: TM) -> TM:
        layers = self.layers
        if layers and getattr(layers[0], "is_entry", False):
            s = layers[0].enter(x, self.ctx)
            layers = layers[1:]
        else:
            s = enter_identity(x, self.ctx)
        exit_layer = None
        if layers and getattr(layers[-1], "is_exit", False):
            exit_layer = layers[-1]
            layers = layers[:-1]
        for L in layers:
            s = L.reach(s, self.ctx)
        return exit_layer.exit(s, self.ctx) if exit_layer is not None else exit_identity(s, self.ctx)

    def reach_interval(self, x: TM):
        """Convenience: rigorous output box (lo, hi) over the input set."""
        return self.reach(x).bounds()


# ==================================================================================================
# translators
# ==================================================================================================

def _np(t):
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.asarray(t, dtype=float)


def _deploy_u_of(t):
    dt = str(getattr(t, "dtype", "float64"))
    return U32 if ("32" in dt or "16" in dt) else U


def _parse_torch_seq(mod):
    """Translate a torch Sequential (or any module whose children run in order) into layers.  Matching is by
    class name, so torch itself never needs importing here."""
    out = []
    children = list(mod.children()) if hasattr(mod, "children") else list(mod)
    for m in children:
        nm = type(m).__name__
        if nm == "Linear":
            out.append(Linear(_np(m.weight), None if m.bias is None else _np(m.bias)))
        elif nm == "ReLU":
            out.append(ReLU())
        elif nm == "Tanh":
            out.append(Tanh())
        elif nm == "Sigmoid":
            out.append(Sigmoid())
        elif nm in ("Identity", "Flatten", "Dropout"):
            continue
        elif nm == "Sequential":
            out.extend(_parse_torch_seq(m))
        else:
            raise NotImplementedError(f"no abstract transformer for torch layer {nm!r}; add a Layer subclass")
    return out


def from_torch(model, deploy_u=None, fold_below=0.0) -> Sequential:
    """Translate a torch model into a :class:`Sequential`.

    Supported: nn.Sequential chains of Linear / ReLU / Tanh / Sigmoid (Identity/Flatten/Dropout skipped), and
    modules with ``mlp`` + min-max buffers ``xmn,xmx,ymn,ymx`` (a min-max normalised dispatch network:
    ``mlp((x-xmn)/(xmx-xmn+1e-8))*(ymx-ymn)+ymn``).  ``deploy_u`` defaults to the parameter dtype's unit roundoff."""
    params = list(model.parameters()) if hasattr(model, "parameters") else []
    if deploy_u is None:
        deploy_u = _deploy_u_of(params[0]) if params else U
    if all(hasattr(model, k) for k in ("mlp", "xmn", "xmx", "ymn", "ymx")):
        layers = ([MinMaxNormalize(_np(model.xmn), _np(model.xmx), 1e-8)] + _parse_torch_seq(model.mlp)
                  + [MinMaxDenormalize(_np(model.ymn), _np(model.ymx))])
    else:
        layers = _parse_torch_seq(model)
    return Sequential(layers, deploy_u=deploy_u, fold_below=fold_below)


def from_state_dict(sd, linear_keys, activation="tanh", deploy_u=U32, final_activation=None, fold_below=0.0):
    """Build an MLP from a state dict / npz of numpy arrays.

    linear_keys : ordered list of key prefixes, one per Linear layer, e.g.
                  ["mlp_extractor.policy_net.0", "mlp_extractor.policy_net.2", "action_net"]
                  (``<prefix>.weight`` of shape (out, in) and ``<prefix>.bias``).
    activation  : "relu" | "tanh" | "sigmoid" between linear layers (not after the last one, unless
                  final_activation is given)."""
    act = ACTIVATIONS[activation]
    layers: list[Layer] = []
    for i, k in enumerate(linear_keys):
        b = sd[k + ".bias"] if (k + ".bias") in sd else None
        layers.append(Linear(np.asarray(sd[k + ".weight"], float), None if b is None else np.asarray(b, float)))
        if i < len(linear_keys) - 1:
            layers.append(act())
    if final_activation:
        layers.append(ACTIVATIONS[final_activation]())
    return Sequential(layers, deploy_u=deploy_u, fold_below=fold_below)


def sb3_actor(sd, fold_below=0.0) -> Sequential:
    """Actor (logits) of a stable-baselines3 on-policy MlpPolicy (PPO/A2C/MaskablePPO, Tanh by default),
    from its state dict / the extracted npz.  Deployed in float32 (SB3 default)."""
    idx = sorted({int(k.split(".")[2]) for k in sd if k.startswith("mlp_extractor.policy_net.") and k.endswith(".weight")})
    keys = [f"mlp_extractor.policy_net.{i}" for i in idx] + ["action_net"]
    return from_state_dict(sd, keys, activation="tanh", deploy_u=U32, fold_below=fold_below)


def sb3_critic(sd, fold_below=0.0) -> Sequential:
    idx = sorted({int(k.split(".")[2]) for k in sd if k.startswith("mlp_extractor.value_net.") and k.endswith(".weight")})
    keys = [f"mlp_extractor.value_net.{i}" for i in idx] + ["value_net"]
    return from_state_dict(sd, keys, activation="tanh", deploy_u=U32, fold_below=fold_below)
