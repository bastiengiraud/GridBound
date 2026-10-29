"""Minimal runner (works without pytest):  python -m tests.run_tests"""
import importlib
import time
import traceback

MODULES = ["tests.test_core", "tests.test_power", "tests.test_shunt_pf", "tests.test_policy", "tests.test_wscc9_data", "tests.test_audit"]

if __name__ == "__main__":
    fails = 0
    for m in MODULES:
        mod = importlib.import_module(m)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            t = time.time()
            try:
                getattr(mod, name)()
                print(f"PASS {m}.{name}  ({time.time() - t:.1f}s)")
            except Exception:
                fails += 1
                print(f"FAIL {m}.{name}")
                traceback.print_exc()
    print("all passed" if not fails else f"{fails} failed")
    raise SystemExit(1 if fails else 0)
