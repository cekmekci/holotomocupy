#!/bin/bash
#PBS -A APS_IRI
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -N s7trim
#PBS -j oe
# ===========================================================================
# A/B the step-7 SCORED REGION against the EXISTING pass-1 checkpoint.
#
#     qsub s7_trim_check.sh
#
# Runs step7.py twice on rec/pass1/checkpoints/checkpoint_0768.h5 -- the
# volume already on disk -- changing NOTHING but --score-radius.  No
# reconstruction, no writes into the pipeline: both runs land in
# ~/s7_trim/{r075,r100} and neither touches the configs next to this script.
#
# WHY.  The 2026-11-12 step-7 run ran away to 6071 search px on a 1280 grid.
# The cause is not the theta sign -- it is that the seed rebuilt on
# 2026-11-11 carries two strong ROTATIONALLY-SYMMETRIC features that the old
# seed did not, both inside the scored region:
#
#   r = 0.80   a dark bowl bottoming at -1.82, exactly n/nobj = 2048/2560,
#              i.e. the edge of the measured data.  Past it every value is
#              stage 5's pad_taper, not the detector.
#   r = 0.949  a bright ring at +0.47, exactly sqrt(mask_r) for mask_r=0.9 --
#              tomo.py:62 tests `x**2 + y**2 < mask_r`, a SQUARED radius --
#              from the FBP apodisation stage 5 gained the same day.
#
# Background peak-to-peak inside r<0.9 is 1.67 against a sample contrast of
# about 2.7.  Neither feature can be SHARPENED by a per-angle drift (they are
# identical at every angle), but a runaway drift can SMEAR THEM FLAT, and
# that is worth ~1.58 nats where genuinely focusing the sample is worth
# ~0.004.  The optimiser takes the 400:1 trade, every time.
#
# WHAT TO COMPARE.  Per run:
#     the per-degree ladder -- entropy should DECREASE and the rungs should
#         CONVERGE, instead of "maximum number of function evaluations
#         exceeded" with the shift still growing
#     found    y ptp <..>  x ptp <..> search px
# r075 converging to tens of px while r100 runs away to thousands confirms
# the diagnosis, and r075's shifts are then the ones worth trusting.  BOTH
# running away means the scored region was not the problem and the next
# suspect is the Paganin low-frequency artefact itself -- the bowl is real
# in the seed either way, and pass 2 inherits it.
#
# SEPARATE --out PER ARM IS REQUIRED, not tidiness: step7.py names its
# outputs step7_it{iter}_n{n}_th{ntheta}_nz{nz}, which does NOT include the
# score radius, so a shared --out would have the second run silently
# overwrite the first one's csv/png.
#
# The r075 run goes FIRST on purpose: if the job hits walltime, that is the
# one worth having.  r100 only reproduces the failure already on record.
#
# COST.  ~12 min per run at --ntheta 7200 --nslice 64 on 4 GPUs (the
# 2026-11-12 run's search alone was 468 s).  If that is too tight, add
# --nslice 32 to BOTH lines -- but keep the two identical or the comparison
# is worthless.
# ===========================================================================
set -u

SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"

source "${SCRIPT_DIR}/../polaris_env.sh"
echo "python: $(which python)"

OUT="${HOME}/s7_trim"
mkdir -p "${OUT}"

# No sed fixup needed, unlike s7_sign_check.sh: config_step6_bin1_pass1.conf
# already points path_out at rec/pass1, which is where the finished pass-1
# checkpoint actually is.
CONF="${SCRIPT_DIR}/config_step6_bin1_pass1.conf"
echo "config: ${CONF}"
grep -E '^path_out=' "${CONF}"

MPI="mpiexec -n 4 --ppn 4 --depth=8 --cpu-bind depth \
     --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=1 \
     ${SCRIPT_DIR}/set_affinity_gpu_polaris.sh python"

echo
echo "########### score-radius 0.75  (REAL DATA ONLY) $(date) ###########"
${MPI} "${SCRIPT_DIR}/step7.py" "${CONF}" \
    --iter 768 --bin 0 --ntheta 7200 --nslice 64 \
    --score-radius 0.75 --out "${OUT}/r075"

echo
echo "########### score-radius 1.0  (CURRENT, THE CONTROL) $(date) ###########"
${MPI} "${SCRIPT_DIR}/step7.py" "${CONF}" \
    --iter 768 --bin 0 --ntheta 7200 --nslice 64 \
    --score-radius 1.0 --out "${OUT}/r100"

echo
echo "=== DONE $(date) ==="
echo "compare the per-degree ladder and the 'found y ptp / x ptp' lines."
echo "outputs: ${OUT}/r075  and  ${OUT}/r100"
