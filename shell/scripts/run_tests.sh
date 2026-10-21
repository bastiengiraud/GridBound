#!/bin/bash
# Test suite (soundness by sampling, case9 end-to-end, policy/robustness).  Submit from the repo root:
#   bsub < shell/scripts/run_tests.sh
#BSUB -q hpc
#BSUB -J certflow_tests
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -M 3GB
#BSUB -W 00:30
#BSUB -o shell/outputs/certflow_tests_%J.out
#BSUB -e shell/outputs/certflow_tests_%J.err

cd "${LS_SUBCWD:-.}" && source shell/scripts/common.sh || { echo "error: submit from the repository root"; exit 1; }
python -m tests.run_tests
