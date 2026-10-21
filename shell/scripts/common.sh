#!/bin/bash
# Sourced by the job scripts: go to the repository root, load Python, activate the venv, pin threads.
#
#   PY_MODULE   environment module providing Python >= 3.9   (default below; see `module avail python3`)
#   VENV_DIR    virtual environment created by setup_env.sh  (default: <repo>/.venv)

PY_MODULE="${PY_MODULE:-python3/3.11.9}"

# LSF starts jobs in the directory they were submitted from ($LS_SUBCWD); submit from the repo root.
REPO_DIR="${REPO_DIR:-${LS_SUBCWD:-$(pwd)}}"
cd "$REPO_DIR" || { echo "cannot cd to $REPO_DIR"; exit 1; }
if [ ! -f pyproject.toml ] || [ ! -d certflow ]; then
    echo "error: run/submit from the repository root (got $REPO_DIR)"; exit 1
fi
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"

if command -v module >/dev/null 2>&1; then
    module load "$PY_MODULE" 2>/dev/null || echo "warning: 'module load $PY_MODULE' failed; using $(command -v python3)"
fi
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "error: no virtual environment at $VENV_DIR -- run 'bash shell/scripts/setup_env.sh' once first"; exit 1
fi
source "$VENV_DIR/bin/activate"

# one BLAS thread per job slot (the verification is numpy-heavy but small-matrix; oversubscription only hurts)
NTHREADS="${LSB_DJOB_NUMPROC:-1}"
export OMP_NUM_THREADS="$NTHREADS" OPENBLAS_NUM_THREADS="$NTHREADS" MKL_NUM_THREADS="$NTHREADS"
export PYTHONUNBUFFERED=1          # stream progress into the LSF log

echo "host $(hostname)   job ${LSB_JOBID:-local}${LSB_JOBINDEX:+[$LSB_JOBINDEX]}   $(date)"
echo "repo $REPO_DIR @ $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')   $(python --version)"
