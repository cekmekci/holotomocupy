#!/bin/bash
#PBS -A APS_IRI
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=01:00:00
#PBS -N AtomiumS2_coded_polaris_test_run
#PBS -q debug
#PBS -j oe
#PBS -m be
#PBS -M cekmekci@anl.gov

# ===========================================================================
# AtomiumS2, coded aperture -- POLARIS SANITY-CHECK LADDER, NOT THE REAL
# RECONSTRUCTION.  Same toy data as ../tomo5_test/tomo5_run.sh (400 of 7200
# angles, 16x16/8x8 binning instead of the real 4x4/2x2/1x1), run through the
# ACTUAL Polaris job mechanics -- PBS, mpiexec, set_affinity_gpu_polaris.sh,
# the GPU healthcheck -- so that is validated cheaply (minutes, 1 hr
# walltime) before committing a multi-day allocation to ../polaris_run.sh.
#
#     qsub polaris_test_run.sh
#
# REDESIGNED 2026-11-10 (twice -- see ../config_step6_bin0.conf's header for
# the history) to mirror ../polaris_run.sh's final shape. Pre-redesign state
# of this whole folder is archived in ../old_two_pass_pipeline/. Unlike
# ../tomo5_test/tomo5_run.sh (a demo -- step 7 is called with a deliberately
# tuned --deg since it's known to diverge at this toy scale), this script
# otherwise mirrors ../polaris_run.sh exactly: no step7 flag overrides beyond
# --ntheta, the advisor's own defaults throughout. The only differences from
# ../polaris_run.sh are the toy bin levels, toy ntheta/stride, and the
# separate rec/test_pass1 + rec/test_refined output dirs. If step7 diverges
# here the way it did before, step7.py's own guard will refuse to write a bad
# correction and PASS 2 will not run -- useful signal about step7 at toy
# scale, not a reason to loosen the guard.
#
# NOTE ON COST: the "rebuild the seed" stage below (steps15.py stage 5) has
# NO toy-scale subsetting -- it reads ntheta/nz/n straight off the raw scan's
# own shape and ignores cfg.ntheta/cfg.n entirely, confirmed from the code.
# So that ONE stage costs exactly what it costs in the real ladder (building
# the whole obj_init.npy did); everything else in this script is genuinely
# cheap. Run deliberately at full cost anyway, so this toy ladder still
# exercises the EXACT mechanism end to end, not an approximation of it.
#
# Requires codedHT.h5, code.npy and obj_init.npy already transferred to the
# eagle tree (see ../polaris_run.sh's header for the transfer list and
# sizes) -- same files the real ladder uses, read read-only here.
#
# PASS 1, POSITIONS FROZEN (config_step6_bin{4,3}_pass1.conf, path_out
# rec/test_pass1):
#
#     step6   config_step6_bin4_pass1.conf   iter    0 -> 32
#     step6   config_step6_bin3_pass1.conf   iter   32 -> 48
#
# STEP 7 reads the pass-1 bin3_pass1 checkpoint, fits the per-angle drift,
# and writes rec/test_pass1/correct_correct3D_extra.txt (via --out, so it
# can never collide with the real ladder's correct_correct3D_extra.txt,
# which sits next to ../polaris_run.sh's configs with no directory prefix).
#
# REBUILD THE SEED (config_steps15_test_corrected.conf).  Reruns stage 5 with
# the TOY pass 1's correction folded into pos_obj, producing
# obj_init_test_corrected.npy -- separate from both obj_init.npy and the real
# ladder's obj_init_corrected.npy, so no ladder can collide with another.
#
# PASS 2 (config_step6_bin{4,3}.conf, path_out rec/test_refined) -- an
# ordinary restart, same shape as a single-pass run: `obj_init_file` points
# at the corrected seed above (bin 4 only), `pos` is nominal + extra via
# correct3d_extra_file (bin 4 only), `prb`/`code`/`pos_code` read fresh from
# step0's own files:
#
#     step6   config_step6_bin4.conf   iter   0 -> 32
#     step6   config_step6_bin3.conf   iter  32 -> 48
#
# RE-SUBMISSION HAZARD, STRIPING -- identical to ../polaris_run.sh; see that
# file's header. Stripe both rec/test_pass1 and rec/test_refined before the
# first run:
#     lfs setstripe -c 8 -S 16M <path_out>    # rec/test_refined
#     lfs setstripe -c 8 -S 16M <path_out>    # rec/test_pass1
# ===========================================================================

# --- user configuration ---
# Software environment (modules + conda env). Two levels up from this script
# (polaris_test/ -> AtomiumS2_coded/ -> experimental/), same file
# ../polaris_run.sh uses.
HTC_ENV=${HTC_ENV:-"${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/../../polaris_env.sh"}
# HEALTHCHECK=0  skips the ~30 s GPU probe;  RUN_NODES=N  uses N healthy nodes.
# PREFLIGHT=0    skips the input check (it is seconds; leave it on).
# --------------------------

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4          # one rank per Polaris A100
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

# Directory the job was submitted from (PBS_O_WORKDIR when submitted via qsub;
# falls back to the script's own directory for local ./polaris_test_run.sh
# testing). Plain $(pwd) does NOT work: PBS starts the job in $HOME, not
# where you qsub'd.
SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"   # polaris_test/: the toy configs + this script
parent="$(dirname "${SCRIPT_DIR}")"                                            # AtomiumS2_coded/: step6.py, step7.py, preflight.py, set_affinity_gpu_polaris.sh
rec_dir="$(dirname "${parent}")"                                               # experimental/: gpu_healthcheck.sh, polaris_env.sh

cd "${rec_dir}"
exec > >(tee "${SCRIPT_DIR}/pbs-${PBS_JOBID}.out") 2>&1

echo "Sample dir:  ${SCRIPT_DIR}"
echo "Parent dir:  ${parent}"
echo "Rec dir:     ${rec_dir}"
echo "Jobid: $PBS_JOBID"
echo "Running on host: $(hostname)"
echo "Running on nodes: $(cat $PBS_NODEFILE)"
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}"

# Modules + conda env. polaris_env.sh loads PrgEnv-gnu, cray-mpich,
# cudatoolkit, cray-hdf5-parallel and activates the holotomocupy env; it must
# be sourced inside the job, not just at install time, or the cray-mpich-linked
# mpi4py and h5py will not find their libraries.
[ -r "${HTC_ENV}" ] || { echo "ERROR: HTC_ENV not readable: ${HTC_ENV}"; exit 1; }
source "${HTC_ENV}"
echo "python: $(which python)"

# Preflight each config right before IT runs (see preflight() below), not all
# four upfront: bin3_pass1's checkpoint dependency (checkpoint_0032.h5) does
# not exist until bin4_pass1 has actually run, so checking it before
# bin4_pass1 even starts would abort a from-scratch submission at the very
# first checkpoint handoff, every time -- same reasoning as ../polaris_run.sh.
preflight () {
    [ "${PREFLIGHT:-1}" = "1" ] || return 0
    python "${parent}/preflight.py" "${SCRIPT_DIR}/$1" || {
        echo "ERROR: preflight failed for $1; aborting."; exit 1; }
}

# Where step 7 writes correct_correct3D_extra.txt and where its checkpoint
# input lives -- MUST match path_out in config_step6_bin{4,3}_pass1.conf, or
# the "apply the correction" step below reads/writes the wrong place.
OUT="/eagle/APS_IRI/cekmekci/AtomiumS2_coded/rec/test_pass1"

# See the re-submission hazard in the header above.
EXTRA="${OUT}/correct_correct3D_extra.txt"
[ -f "${EXTRA}" ] && { echo "ERROR: ${EXTRA} already exists. If pass 1 + step 7 already ran and this is a pass-2-only resume, comment out PASS 1 and STEP 7 below and resubmit; otherwise step 7 will overwrite it against whatever pass-1 checkpoint happens to be on disk."; exit 1; }

# Drop nodes whose GPUs cannot take a CUDA context. PBS has no Slurm-style
# --exclude -- `-l select=` can pin a host but cannot negate one -- so a node
# that comes up with cudaErrorDevicesUnavailable can only be filtered from
# inside the job. Must run AFTER polaris_env.sh: the probe needs cupy.
HOSTOPT=""
if [ "${HEALTHCHECK:-1}" = "1" ]; then
    GOOD="${SCRIPT_DIR}/nodes.good.${PBS_JOBID}"
    bash "${rec_dir}/gpu_healthcheck.sh" "${GOOD}" "${NRANKS}" "${RUN_NODES:-1}" || { echo "ERROR: too few healthy nodes in this allocation; aborting."; exit 1; }
    head -n "${RUN_NODES:-$(wc -l < "${GOOD}")}" "${GOOD}" > "${GOOD}.run"
    NNODES=$(wc -l < "${GOOD}.run")
    export NTOTRANKS=$(( NNODES * NRANKS ))
    HOSTOPT="--hostfile ${GOOD}.run"
    echo "Running on ${NNODES} healthy nodes  TOTAL_NUM_RANKS=${NTOTRANKS}"
fi

MPI="mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} ${parent}/set_affinity_gpu_polaris.sh python"

# --- the pipeline; comment out a line to skip that stage ---------------------

# PASS 1, POSITIONS FROZEN -------------------------------------------------
preflight config_step6_bin4_pass1.conf
echo "=== pass1 bin4 START $(date) ==="
${MPI} "${parent}/step6.py" "${SCRIPT_DIR}/config_step6_bin4_pass1.conf" || exit $?

preflight config_step6_bin3_pass1.conf
echo "=== pass1 bin3 START $(date) ==="
${MPI} "${parent}/step6.py" "${SCRIPT_DIR}/config_step6_bin3_pass1.conf" || exit $?

# STEP 7 -----------------------------------------------------------------
# No --deg override here -- unlike ../tomo5_test/tomo5_run.sh (a demo), this
# script stands in for ../polaris_run.sh and uses the advisor's own step7
# defaults throughout. --iter 48: the pass-1 bin3_pass1 checkpoint
# (start_iter=32, niter=48 in config_step6_bin3_pass1.conf). --bin 0: search
# on the native bin-3 grid. --out: see OUT above.
#
# --ntheta MUST match pass 1's own angle count, not the raw scan's: step7's
# scan_theta picks its search angles by an even stride over the FULL scan
# (ntheta_scan, read from in_file), independent of which angles the
# checkpoint was actually built from. The real ladder's pass 1
# (config_step6_bin1_pass1.conf) uses ntheta=7200, stride=1 -- ALL angles --
# so --ntheta 7200 there searches exactly what the object knows about. This
# toy ladder's pass 1 uses ntheta=400, stride=18 -- the object has no real
# information at the other 6800 -- so --ntheta 7200 here was forward-
# projecting the checkpoint at thousands of angles it was never built from,
# which is almost certainly why this diverged far harder (ptp 1772 on a
# 320-px grid) than the real run (ptp 3.8) ever did. Match pass 1's count.
echo "=== step7 START $(date) ==="
STEP7RANKS=$(( NTOTRANKS < 8 ? NTOTRANKS : 8 ))
mpiexec ${HOSTOPT} -n ${STEP7RANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${parent}/set_affinity_gpu_polaris.sh" python "${parent}/step7.py" "${SCRIPT_DIR}/config_step6_bin3_pass1.conf" --iter 48 --bin 0 --ntheta 400 --score-radius 0.75 --out "${OUT}" || exit $?

# REBUILD THE SEED ----------------------------------------------------------
# steps15.py stage 5 only, same MPI launcher as step6/step7. Reads
# rec/test_pass1/correct_correct3D_extra.txt that STEP 7 just wrote; writes
# obj_init_test_corrected.npy, which config_step6_bin4.conf below reads as
# its obj_init_file. See the header's NOTE ON COST -- this one stage is NOT
# cheap, unlike everything else in this script.
echo "=== rebuild seed START $(date) ==="
${MPI} "${parent}/steps15.py" "${SCRIPT_DIR}/config_steps15_test_corrected.conf" || exit $?

# PASS 2, REFINED -------------------------------------------------------
preflight config_step6_bin4.conf
echo "=== pass2 bin4 START $(date) ==="
${MPI} "${parent}/step6.py" "${SCRIPT_DIR}/config_step6_bin4.conf" || exit $?

preflight config_step6_bin3.conf
echo "=== pass2 bin3 START $(date) ==="
${MPI} "${parent}/step6.py" "${SCRIPT_DIR}/config_step6_bin3.conf" || exit $?

echo "=== DONE $(date) ==="
