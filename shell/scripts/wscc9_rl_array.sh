#!/bin/bash
# The WSCC-9 RL case studies as an LSF job ARRAY: one (study, operating point) pair per element, all in
# parallel.  Submit from the repo root:
#   bsub < shell/scripts/wscc9_rl_array.sh
#   ARGS="--require-n1" bsub < shell/scripts/wscc9_rl_array.sh      # extra options for every element
# Re-run a single element, e.g. element 9:   bsub -J "wscc9_rl_arr[9]" < shell/scripts/wscc9_rl_array.sh
# Re-boxing vs history tracking in the multi-step study (elements 7-8):
#   ARGS="--track both --step-change common" bsub -J "wscc9_rl_arr[7-8]" < shell/scripts/wscc9_rl_array.sh
#
# Elements (operating points: python examples/wscc9_actor_reachability.py --list-ops):
#    1-6   one-step reachability   (--study reach    --ops 0..5)
#    7-8   two-step reachability   (--study two-step --ops 0 / 5)
#    9-14  measurement noise       (--study noise    --ops 0..5)
# Logs: shell/outputs/wscc9_rl_arr_<jobid>_<element>.out
#BSUB -q hpc
#BSUB -J "wscc9_rl_arr[1-14]"
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=3GB]"
#BSUB -M 4GB
#BSUB -W 02:00
#BSUB -o shell/outputs/wscc9_rl_arr_%J_%I.out
#BSUB -e shell/outputs/wscc9_rl_arr_%J_%I.err

TASKS=(
    ""                                                    # LSF indices start at 1
    "--study reach --ops 0"    "--study reach --ops 1"    "--study reach --ops 2"
    "--study reach --ops 3"    "--study reach --ops 4"    "--study reach --ops 5"
    "--study two-step --ops 0" "--study two-step --ops 5"
    "--study noise --ops 0"    "--study noise --ops 1"    "--study noise --ops 2"
    "--study noise --ops 3"    "--study noise --ops 4"    "--study noise --ops 5"
)

cd "${LS_SUBCWD:-.}" && source shell/scripts/common.sh || { echo "error: submit from the repository root"; exit 1; }
IDX="${LSB_JOBINDEX:-${1:-1}}"            # outside LSF: bash shell/scripts/wscc9_rl_array.sh <element>
TASK="${TASKS[$IDX]}"
if [ -z "$TASK" ]; then echo "no task for element $IDX (valid: 1..$(( ${#TASKS[@]} - 1 )))"; exit 1; fi
echo "element $IDX: $TASK ${ARGS}"
python examples/wscc9_actor_reachability.py $TASK ${ARGS}
