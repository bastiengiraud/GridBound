#!/bin/bash
# One-time setup on the cluster (run on a login node, from the repository root):
#
#   git clone https://github.com/bastiengiraud/GridBound.git && cd GridBound
#   bash shell/scripts/setup_env.sh
#
# Creates .venv/ with certflow installed in editable mode (numpy + scipy only; add EXTRAS=all for
# pandapower + torch). Re-run after changing dependencies; `git pull` alone is enough for code changes.
set -e

PY_MODULE="${PY_MODULE:-python3/3.11.9}"
VENV_DIR="${VENV_DIR:-$(pwd)/.venv}"
EXTRAS="${EXTRAS:-test}"

if [ ! -f pyproject.toml ] || [ ! -d certflow ]; then
    echo "error: run this from the repository root"; exit 1
fi
if command -v module >/dev/null 2>&1; then
    module load "$PY_MODULE" || { echo "module $PY_MODULE not found; pick one from 'module avail python3' and set PY_MODULE"; exit 1; }
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[$EXTRAS]"

python -c "import certflow, numpy, scipy; print('certflow', certflow.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__)"
echo "done: environment in $VENV_DIR (module $PY_MODULE). Submit jobs from $(pwd) with:  bsub < shell/scripts/<job>.sh"
