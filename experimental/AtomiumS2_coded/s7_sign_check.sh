#!/bin/bash
#PBS -A APS_IRI
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -N s7sign
#PBS -j oe
# ===========================================================================
# A/B the step-7 theta sign against the EXISTING pass-1 checkpoint.
#
#     qsub s7_sign_check.sh
#
# Runs step7.py twice on rec/nopos/checkpoints/checkpoint_0768.h5 -- the
# volume that is already on disk -- changing NOTHING but --theta-sign.  No
# reconstruction, no writes into the pipeline: both runs land in
# ~/s7_check/{new,old} and neither touches the correct_correct3D_extra.txt
# sitting next to the configs.
#
# WHY.  autofocus.scan_theta negates theta for the multi-distance
# convention, but this pipeline reconstructs at +theta everywhere (steps15.py
# stage 5 and CodedReader both use +deg2rad(theta)), so every step-7 result
# so far searched the mirrored geometry.  -1 reproduces that; +1 is the fix.
#
# WHAT TO COMPARE.  Two lines from each run:
#     entropy <x>, <delta> against no correction      (negative is better)
#     found    y ptp <..>  x ptp <..> search px
# A much larger improvement at +1 means the sign was the bug.  Both about
# the same means it was not, and the next thing to check is the rotation
# axis (check_rot_axis.py) rather than the drift.
#
# The +1 run goes FIRST on purpose: if the job hits walltime, that is the
# one worth having.
#
# COST.  ~18 min per run at --ntheta 7200 --nslice 64 on 4 GPUs.  If that is
# too tight, add --nslice 32 to BOTH lines (halves it; the advisor measured
# 16 -> 512 slices moving the answer by 0.004 px) -- but keep the two runs
# identical or the comparison is worthless.
# ===========================================================================
set -u

SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"

source "${SCRIPT_DIR}/../polaris_env.sh"
echo "python: $(which python)"

OUT="${HOME}/s7_check"
mkdir -p "${OUT}"

# step7 reads the checkpoint from cfg.path_out, and the checkpoint this check
# wants is the PRE-CONVENTION one in rec/nopos -- a flat directory that is not
# any run's rec/<run>/pass1. The configs now DERIVE path_out from run=/rec=
# (see runs/README.md), so there is no literal `rec/pass1` left to substitute;
# override the whole key in a throwaway copy instead. Set PASS1_OUT to point
# this at some other checkpoint directory.
#
# Configs live in a run folder now. Pass a different run folder as $1.
RUN_DIR="${1:-${SCRIPT_DIR}/runs/baseline}"
SRC="${RUN_DIR}/config_step6_bin1_pass1.conf"
[ -f "${SRC}" ] || { echo "ERROR: no such config: ${SRC}"; exit 1; }
PASS1_OUT="${PASS1_OUT:-/eagle/APS_IRI/cekmekci/AtomiumS2_coded/rec/nopos}"
sed "s|^path_out=.*|path_out=${PASS1_OUT}|" "${SRC}" > "${OUT}/s7.conf"
echo "config: ${OUT}/s7.conf"
grep -E '^path_out=' "${OUT}/s7.conf"

MPI="mpiexec -n 4 --ppn 4 --depth=8 --cpu-bind depth \
     --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=1 \
     ${SCRIPT_DIR}/set_affinity_gpu_polaris.sh python"

echo
echo "################ theta-sign +1  (THE FIX) $(date) ################"
${MPI} "${SCRIPT_DIR}/step7.py" "${OUT}/s7.conf" \
    --iter 768 --bin 0 --ntheta 7200 --nslice 64 \
    --theta-sign 1 --out "${OUT}/new"

echo
echo "################ theta-sign -1  (OLD BEHAVIOUR) $(date) ################"
${MPI} "${SCRIPT_DIR}/step7.py" "${OUT}/s7.conf" \
    --iter 768 --bin 0 --ntheta 7200 --nslice 64 \
    --theta-sign -1 --out "${OUT}/old"

echo
echo "=== DONE $(date) ==="
echo "compare the 'entropy ... against no correction' and 'found y ptp / x ptp'"
echo "lines from the two blocks above."
