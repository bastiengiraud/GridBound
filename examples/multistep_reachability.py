"""Mode 3 on a small feeder: the states an RL agent can drive the grid into over 3 steps, with a 3D plot.

    python examples/multistep_reachability.py            # writes examples/multistep_reachability.png

Grid and agent: see feeder_demo.py (OLTC transformer, 4-bus MV line, loads and two PV parks; a demo agent trained
on a voltage rule).  Load model: all loads move together (one symbol), anywhere within +-10 % of the snapshot at
step 0, and then by up to +-3 % before each further action.  The audit certifies every action sequence the agent can
take and bounds every bus voltage and line loading after each action; the plot shows these bounds step by step.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feeder_demo import auditor  # noqa: E402

from certflow.audit import LoadUncertainty  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

aud, model, net = auditor()
model.check_against_pandapower()
rep = aud.multi_step(net, uncertainty=LoadUncertainty(rel=0.10, directions="common"), steps=3, step_change=0.03,
                     max_depth=3, n_check=200)
print()
print(rep.summary())
out = os.path.join(HERE, "multistep_reachability.png")
rep.plot(out)
print(f"\n3D plot written to {os.path.relpath(out)}")
