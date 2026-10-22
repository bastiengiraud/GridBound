"""The WSCC-9 RL example's hard-coded network data matches the PSS/E file of the training environment."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples"))


def test_wscc9_data_matches_raw():
    import wscc9_env as env
    path = os.environ.get("WSCC9_RAW", env.RAW_PATH)
    if not os.path.exists(path):
        print(f"  skipped: {path} not found")
        return
    raw = env.read_raw(path)
    assert set(raw["branch"]) == set(env.LINES + env.XFMRS)          # same branches, same orientation
    for k, v in raw["branch"].items():
        assert np.allclose(v, env._BR[k], atol=1e-12), (k, v, env._BR[k])
    assert np.allclose([raw["load"][b] for b in env.LOAD_BUSES], env.S_NOM)
    assert np.isclose(raw["pg"][2], 163.0) and np.isclose(raw["pg"][3], 85.0)   # == OperatingState() defaults
    assert np.allclose([raw["vm"][1], raw["vm"][2], raw["vm"][3]], [1.04, 1.025, 1.025])
