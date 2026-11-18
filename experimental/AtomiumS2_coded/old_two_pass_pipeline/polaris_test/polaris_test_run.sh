#!/bin/bash
#PBS -A APSDataAnalysis
#PBS -l select=1:system=polaris
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
# Unlike ../tomo5_test/tomo5_run.sh (a demo -- step 7 is called with a
# deliberately tuned --deg since it's known to diverge at this toy scale and
# the point there is just to show the shape of the pipeline), this script
# mirrors ../polaris_run.sh exactly: no step7 flag overrides, the advisor's
# own defaults throughout. The only differences from ../polaris_run.sh are
# the toy bin levels (4/3 instead of 2/1/0), the toy ntheta/stride, and the
# separate rec/test_pass1 + rec/test_refined output dirs. If step7 diverges
# here the way it did under tomo5_run.sh, step7.py's own guard will refuse to
# write a bad correction and PASS 2 will not run -- that is useful signal
# about step7 at toy scale, not a reason to loosen the guard.
#
# Requires codedHT.h5, code.npy and obj_init.npy already transferred to the
# eagle tree (see ../polaris_run.sh's header for the transfer list and
# sizes) -- same files the real ladder uses, read read-only here.
#
# PASS 1, EVERYTHING REFINES (config_step6_bin{4,3}_pass1.conf, path_out
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
# PASS 2, REFINED (config_step6_bin{4,3}.conf, path_out rec/test_refined):
#
#     step6   config_step6_bin4.conf   iter   0 -> 32
#     step6   config_step6_bin3.conf   iter  32 -> 48
#
# RE-SUBMISSION HAZARD, STRIPING -- identical to ../polaris_run.sh; see that
# file's header. Strike both rec/test_pass1 and rec/test_refined before the
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

# MUST match correct3d_extra_file in config_step6_bin{4,3}.conf, or step 7
# writes somewhere pass 2 never looks and pass 2 silently runs uncorrected.
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

# PASS 1, EVERYTHING REFINES ----------------------------------------------------
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
echo "=== step7 START $(date) ==="
STEP7RANKS=$(( NTOTRANKS < 8 ? NTOTRANKS : 8 ))
mpiexec ${HOSTOPT} -n ${STEP7RANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${parent}/set_affinity_gpu_polaris.sh" python "${parent}/step7.py" "${SCRIPT_DIR}/config_step6_bin3_pass1.conf" --iter 48 --bin 0 --ntheta 7200 --out "${OUT}" || exit $?

# PASS 2, REFINED --------------------------------------------------------
preflight config_step6_bin4.conf
echo "=== pass2 bin4 START $(date) ==="
${MPI} "${parent}/step6.py" "${SCRIPT_DIR}/config_step6_bin4.conf" || exit $?

preflight config_step6_bin3.conf
echo "=== pass2 bin3 START $(date) ==="
${MPI} "${parent}/step6.py" "${SCRIPT_DIR}/config_step6_bin3.conf" || exit $?

echo "=== DONE $(date) ==="
