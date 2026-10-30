"""Audit an NN that was trained on OPF data to output generator setpoints.

    python examples/opf_dispatch_audit.py

The network below is trained in a few seconds on a simple proportional dispatch rule so that the example runs on its
own; use your own trained model instead.  Its layout is deliberately not the "standard" one, to show how the
layout is declared: inputs are the active loads in per unit, outputs are [Vg; Pg] with Pg in per unit.
"""
import warnings

import numpy as np
import torch

from certflow.audit import verify_dispatch
from certflow.power import case9, from_matpower

warnings.filterwarnings("ignore")

mpc = case9()                                            # MATPOWER data; a pandapower net works the same way
g = from_matpower(mpc)
Pd0 = g.names["load_pq"][0]
pg0, vg0 = g.names["gen_pg"], g.names["gen_vg"]

# a stand-in for your trained network ---------------------------------------------------------------------------
torch.manual_seed(0)
X = torch.tensor(Pd0 / 100 * np.random.default_rng(0).uniform(0.8, 1.2, (2000, 1)), dtype=torch.float32)
ratio = X.sum(1, keepdim=True) / float(Pd0.sum() / 100)
Y = torch.cat([torch.tensor(vg0, dtype=torch.float32) + 0.02 * (ratio - 1),
               ratio * torch.tensor(pg0 / 100, dtype=torch.float32)], 1)
model = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16), torch.nn.ReLU(),
                            torch.nn.Linear(16, 4))
opt = torch.optim.Adam(model.parameters(), 1e-2)
for _ in range(1500):
    opt.zero_grad()
    loss = ((model(X) - Y) ** 2).mean()
    loss.backward()
    opt.step()

layout = dict(inputs="pd", input_units="pu", outputs=("vg", "pg"), pg_units="pu")

# 1) every operational limit, for every load within +-10 % of nominal ---------------------------------------------
print(verify_dispatch(mpc, model, load_width=0.10, **layout).summary())

# 2) the same network on a grid with tight reactive limits: violations, and a certified correction ---------------
tight = case9()
tight["gen"][1:, 3] = [8.0, -2.0]                        # Qmax of the two PV generators
print()
print(verify_dispatch(tight, model, load_width=0.10, restore=True, **layout).summary())

# other load sets: verify_dispatch(..., load_set=certflow.sets.low_rank_load_set(x0, Gdirs, frac))
