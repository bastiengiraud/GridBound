#!/bin/bash
# All WSCC-9 RL case studies in ONE job (one-step reachability, two-step, measurement noise; all operating
# points), run one after the other.  Submit from the repo root:
#   bsub < shell/scripts/wscc9_rl_case_studies.sh
# Extra options go through ARGS, e.g. enforce N-1 or a fast smoke run:
#   ARGS="--require-n1" bsub < shell/scripts/wscc9_rl_case_studies.sh
#   ARGS="--quick"      bsub < shell/scripts/wscc9_rl_case_studies.sh
#   ARGS="--study two-step --track both --step-change common" bsub < shell/scripts/wscc9_rl_case_studies.sh
# (bsub passes your current environment, so ARGS reaches the job.)  For the same work in parallel, use
# wscc9_rl_array.sh instead.
#BSUB -q hpc
#BSUB -J wscc9_rl
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 6GB
#BSUB -W 04:00
#BSUB -o shell/outputs/wscc9_rl_%J.out
#BSUB -e shell/outputs/wscc9_rl_%J.err

cd "${LS_SUBCWD:-.}" && source shell/scripts/common.sh || { echo "error: submit from the repository root"; exit 1; }
python examples/wscc9_actor_reachability.py ${ARGS}
