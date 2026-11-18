#!/bin/bash
#PBS -A APS_IRI
#PBS -l select=24:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=2:00:00
#PBS -q prod
#PBS -N AtomiumS2_coded_polaris_run
#PBS -j oe
#PBS -m be
#PBS -M cekmekci@anl.gov


# ===========================================================================
# AtomiumS2, coded aperture -- single-distance HT through a gold mask,
# ESRF ID16A, 33.35 keV.
# THE RECONSTRUCTION LADDER, AS ONE JOB, TWO PASSES:
#
#     qsub polaris_run.sh
#
# To run only part of it -- resuming after a preemption, or redoing one rung
# -- COMMENT OUT the mpiexec lines at the bottom that you do not want.  Each
# ends in `|| exit $?` so a failed stage stops the job instead of letting the
# next rung seed itself from a checkpoint that was never written.
#
# ONLY STEP 6 (AND NOW STEP 7) RUN HERE.  Everything upstream -- step0a/0b,
# step0.py, and steps15.py (or the equivalent notebooks) -- is done on tomo5,
# and its three products are transferred to eagle before this is submitted:
#
#     codedHT.h5     170 GB
#     code.npy       513 MB
#     obj_init.npy    65 GB
#
# (data_nfp.h5 and prb.npy stay on tomo5; step 6 never reads them.)  Edit
# `path` and `path_out` in every config to the eagle tree first -- the block
# at the top of each has both pairs.
#
# PASS 1, EVERYTHING REFINES (config_step6_bin{2,1}_pass1.conf, separate
# path_out rec/pass1).  bin 2 starts fresh from the Paganin seed; bin 1
# resumes through the checkpoint bin 2 wrote.  Every variable moves, pos
# included -- there is NO frozen `nopos` pass.
#
#     step6   config_step6_bin2_pass1.conf   iter    0 -> 512
#     step6   config_step6_bin1_pass1.conf   iter  512 -> 768
#
# This is where the coded pipeline departs from the multi-distance one it was
# copied from, deliberately.  There, pass 1 freezes pos so the volume step 7
# measures is not biased by a drift correction already baked into it.  With a
# coded aperture that backfires: rho[pos]=0 freezes the OBJECT positions but
# leaves `code` and `pos_code` free, so the aperture simply drifts to absorb
# the object's misalignment and the drift hides in pos_code.
# See config_step6_bin2_pass1.conf's header for the full argument.
#
# STEP 7's theta sign was fixed on 2026-11: autofocus.scan_theta negates for
# the multi-distance convention, but this pipeline reconstructs at +theta
# everywhere, so --theta-sign now defaults to +1 (as find_shifts.py in
# holotomo-coded-atomiums2 already did).  Results from before that date
# searched the mirrored geometry.  See step7.py's header, which also records
# what the search can and cannot measure.
#
# STEP 7 reads the pass-1 bin-1 checkpoint, fits the per-angle drift by
# entropy autofocus, and writes correct_correct3D_extra.txt next to the
# configs.
#
# PASS 2, REFINED (config_step6_bin{2,1,0}.conf, path_out rec/refined).  The
# same ladder again from the seed, now reading correct_correct3D_extra.txt via
# correct3d_extra_file and refining pos on top of it.  rho is IDENTICAL to
# pass 1; the only differences are the extra file, the bin-0 rung, and
# path_out:
#
#     step6   config_step6_bin2.conf   iter   0 -> 512
#     step6   config_step6_bin1.conf   iter 512 -> 768
#     step6   config_step6_bin0.conf   iter 768 -> 896
#
# RESUME IS MANUAL, per pass.  find_latest_checkpoint globs
# checkpoint_*{start_iter:04}.h5 and returns None when start_iter == 0, so it
# finds the checkpoint you NAME, not the newest on disk.  After a preemption:
# `ls {path_out}/checkpoints/`, set start_iter in the config to the highest
# checkpoint_NNNN, comment out the finished mpiexec lines, resubmit.  The same
# keying is what makes the bin2->bin1(->bin0) handoff work within each pass,
# so it is not a bug to fix.
#
# RE-SUBMISSION HAZARD.  If correct_correct3D_extra.txt already exists when
# this is resubmitted, pass 1 config_step6_bin*_pass1.conf has no
# correct3d_extra_file key so it is unaffected -- but step 7 OVERWRITES the
# file.  If pass 1 has already finished and you are only resuming pass 2,
# comment out PASS 1 and STEP 7 below before resubmitting, or step 7 will
# refit against whatever pass-1 checkpoint happens to be on disk and silently
# replace a good correction.
#
# CREATE AND STRIPE BOTH OUTPUT DIRECTORIES before the first run -- eagle
# defaults to one OST and a bin-0 checkpoint is large.  lfs setstripe does
# NOT create missing parents, so mkdir first:
#     mkdir -p /eagle/APS_IRI/cekmekci/AtomiumS2_coded/rec
#     lfs setstripe -c 8 -S 16M <path_out>    # rec/pass1
#     lfs setstripe -c 8 -S 16M <path_out>    # rec/refined
# ===========================================================================

# --- user configuration ---
# Software environment (modules + conda env). See the Polaris setup notes.
HTC_ENV=${HTC_ENV:-"${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/../polaris_env.sh"}
# HEALTHCHECK=0  skips the ~30 s GPU probe;  RUN_NODES=N  uses N healthy nodes.
# PREFLIGHT=0    skips the input check (it is seconds; leave it on).
# --------------------------

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4          # one rank per Polaris A100
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

# Directory the job was submitted from (PBS_O_WORKDIR when submitted via qsub;
# falls back to the script's own directory for local ./polaris_run.sh testing).
# Plain $(pwd) does NOT work: PBS starts the job in $HOME, not where you qsub'd.
SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
rec_dir="$(dirname "${SCRIPT_DIR}")"

cd "${rec_dir}"
exec > >(tee "${SCRIPT_DIR}/pbs-${PBS_JOBID}.out") 2>&1

echo "Sample dir:  ${SCRIPT_DIR}"
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
# five upfront: bin1_pass1's checkpoint dependency (checkpoint_0512.h5) does
# not exist until bin2_pass1 has actually run, so checking it before bin2_pass1
# even starts would abort a from-scratch submission at the very first
# checkpoint handoff, every time -- a missing code.npy or a config whose ncode
# disagrees with it still costs seconds here instead of a node-hour inside the
# windowed shift, just one stage at a time.
preflight () {
    [ "${PREFLIGHT:-1}" = "1" ] || return 0
    python "${SCRIPT_DIR}/preflight.py" "${SCRIPT_DIR}/$1" || {
        echo "ERROR: preflight failed for $1; aborting."; exit 1; }
}

# See the re-submission hazard at the top of this file.
EXTRA="${SCRIPT_DIR}/correct_correct3D_extra.txt"
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

MPI="mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} ${SCRIPT_DIR}/set_affinity_gpu_polaris.sh python"

# --- the pipeline; comment out a line to skip that stage ---------------------

# PASS 1, EVERYTHING REFINES --------------------------------------------------
# bin 2: the only rung with start_iter=0, so the only one that reads
# obj_init.npy. bin 1 inherits the object, probe, aperture and both position
# sets through the checkpoint. pos refines in BOTH -- no frozen pass; see the
# header and config_step6_bin2_pass1.conf for why.
preflight config_step6_bin2_pass1.conf
echo "=== pass1 bin2 START $(date) ==="
${MPI} "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin2_pass1.conf" || exit $?

preflight config_step6_bin1_pass1.conf
echo "=== pass1 bin1 START $(date) ==="
${MPI} "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin1_pass1.conf" || exit $?

# STEP 7 -----------------------------------------------------------------
# Fits the per-angle drift and writes correct_correct3D_extra.txt next to the
# configs -- picked up by every pass-2 config below via correct3d_extra_file.
# Splits z slices over the ranks and allreduces the 256-bin histogram, so the
# answer does not depend on -n.  -n is capped at 8: a vertical shift mixes z,
# so every rank repeats it, and that cost does not amortize past a handful of
# ranks. --iter 768: the pass-1 bin-1 checkpoint (start_iter=512, niter=768
# in config_step6_bin1_pass1.conf). --bin 0: search on the native bin-1 grid.
# --ntheta 7200: all acquired angles, explicit rather than the default quarter
# of the scan -- a subsample doesn't reduce cost here the way it does for
# step6 (step7 forward-projects a checkpoint that's already built; the cost
# is dominated by the FBP/entropy evaluations, not reading more frames), and
# there is no reason to throw away 3/4 of the angles for free.
echo "=== step7 START $(date) ==="
STEP7RANKS=$(( NTOTRANKS < 8 ? NTOTRANKS : 8 ))
mpiexec ${HOSTOPT} -n ${STEP7RANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" python "${SCRIPT_DIR}/step7.py" "${SCRIPT_DIR}/config_step6_bin1_pass1.conf" --iter 768 --bin 0 --ntheta 7200 || exit $?

# PASS 2, REFINED --------------------------------------------------------
preflight config_step6_bin2.conf
echo "=== pass2 bin2 START $(date) ==="
${MPI} "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin2.conf" || exit $?

preflight config_step6_bin1.conf
echo "=== pass2 bin1 START $(date) ==="
${MPI} "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin1.conf" || exit $?

preflight config_step6_bin0.conf
echo "=== pass2 bin0 START $(date) ==="
${MPI} "${SCRIPT_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin0.conf" || exit $?

echo "=== DONE $(date) ==="
