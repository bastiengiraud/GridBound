"""certflow.audit -- the user-facing layer: describe your grid, your agent / controller and your limits in
pandapower terms, and get certified answers with concrete counterexamples.

RL agents (discrete actions):
    model   = GridModel(net, observation=[obs.bus_vm(), obs.line_loading(), ...],
                        actions=[act.idle(), act.shift_tap(0, +1), ...], limits=Limits(vm_pu=(0.95, 1.05)))
    auditor = RLAuditor(policy, model)
    auditor.next_step(net, load_width=0.05)      # are all actions the agent can take safe?
    auditor.noise_robustness(net)                # how much measurement noise can it take?
    auditor.multi_step(net, steps=3).plot("f.png")  # certified reachable states over several steps

NN dispatch controllers (loads -> generator setpoints, e.g. an OPF proxy):
    verify_dispatch(net, model, load_width=0.1)  # are all operational limits kept for every load in the set?
"""
from .grid_model import (ActionSpace, GridModel, GridState, InjectionSet, Limits, LoadUncertainty, ObservationSpec,
                         Screen, act, obs)
from .rl import ActionResult, NextStepReport, NoiseReport, RLAuditor, as_policy
from ..nn.network import normalizer_layers

__all__ = ["GridModel", "GridState", "ObservationSpec", "ActionSpace", "Limits", "LoadUncertainty", "InjectionSet",
           "Screen", "obs", "act", "RLAuditor", "NextStepReport", "NoiseReport", "ActionResult", "as_policy",
           "normalizer_layers"]
from .dispatch import DispatchReport, load_box, verify_dispatch
from .multistep import Envelope, MultiStepReport, compare_3d, width_table
from .plots import compare_multistep

__all__ += ["verify_dispatch", "DispatchReport", "load_box", "MultiStepReport", "Envelope", "compare_multistep", "compare_3d", "width_table"]
