#!/bin/bash
#PBS -A APS_IRI
#PBS -l select=24:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=3:00:00
#PBS -q prod
#PBS -N AtomS2c_amp_psf
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
# -- set the RUN_* stage switches (see "which stages run" below), e.g.
#
#     qsub -v RUN_PASS1=0,RUN_STEP7=0,RUN_SEED=0 polaris_run.sh   # pass 2 only
#
# rather than commenting the mpiexec lines out: the
# correct_correct3D_extra.txt handling has to know whether step 7 will run,
# and a commented-out line is invisible to the shell.  Within a stage, each
# mpiexec still ends in `|| exit $?` so a failure stops the job instead of
# letting the next rung seed itself from a checkpoint that was never written.
#
# STEP 6, STEP 7 AND THE STAGE-5 SEED REBUILD RUN HERE.  Everything further
# upstream -- step0a/0b, step0.py, and stages 1-3 of steps15.py (or the
# equivalent notebooks) -- is done on tomo5, and these products are
# transferred to eagle before this is submitted:
#
#     ht/codedHT.h5     170 GB    step 6, step 7, stage 5
#     ht/obj_init.npy   134 GB    step 6 pass 1   (65 GB before nobj=2560, 2026-11-11)
#     nfp/code.npy      513 MB    step 6, stage 5
#     nfp/prb.npy        ~4 MB    stage 5 ONLY -- steps15.py:435, read
#                                 unconditionally.  Step 6 does not read it
#                                 (no prb_file key in its configs; it takes
#                                 prb from codedHT.h5), so it is easy to leave
#                                 behind -- and then REBUILD THE SEED below
#                                 dies partway through the job.
#
# data_nfp.h5 is NOT needed here: stage 5 only opens it to recover a nominal
# z1_code (steps15.py:397), and config_steps15_corrected.conf sets z1_code
# explicitly, so that branch never runs.
#
# All four are SHARED between runs and are never written to by this job.
#
# ONE FOLDER = ONE RUN.  This script and its six configs define a single run,
# named by the `run=` key at the top of every config in this folder.  To make
# another run:
#
#     cp -r runs/baseline runs/<new>
#     sed -i 's/^run=.*/run=<new>/'          runs/<new>/*.conf
#     sed -i 's/^#PBS -N .*/#PBS -N AtomS2c_<new>/' runs/<new>/polaris_run.sh
#     # then edit ONLY the parameter the run is about (paganin, rho, model, ...)
#
# Everything run-specific is derived from `run=`, so no file on disk carries
# the run name and nothing collides.  See ../README.md for the full layout.
#
# REDESIGNED 2026-11-10 (twice -- see config_step6_bin0.conf's header for
# the history of the version in between).
#
# PASS 1, POSITIONS FROZEN (config_step6_bin{2,1}_pass1.conf, separate
# path_out rec/pass1).  bin 2 starts fresh from the Paganin seed; bin 1
# resumes through the checkpoint bin 2 wrote.  rho[pos]=0 on both rungs --
# briefly unfrozen, reverted: the unfrozen pass-1 object looked worse on
# inspection. See config_step6_bin2_pass1.conf's header for the history.
#
#     step6   config_step6_bin2_pass1.conf   iter    0 -> 256
#     step6   config_step6_bin1_pass1.conf   iter  256 -> 512
#
# STEP 7 reads the pass-1 bin-1 checkpoint, fits the per-angle drift by
# entropy autofocus, and writes correct_correct3D_extra.txt next to the
# configs.  --theta-sign defaults to +1 (this pipeline reconstructs at
# +theta, unlike the multi-distance convention autofocus.scan_theta was
# written for) -- see step7.py's header, which also records what the search
# can and cannot measure.
#
# REBUILD THE SEED (config_steps15_corrected.conf).  Reruns stage 5
# (steps15.py, start_step=5 -- skips stages 1-3, reuses the
# shifts_cor/shifts_code/prb already in codedHT.h5) with step 7's correction
# folded into pos_obj, producing obj_init_corrected.npy. This is the actual
# fix for picking up step 7's correction: patching `pos` alone onto pass 1's
# finished object (an earlier version of this file) doesn't work, because
# that object was reconstructed consistent with the OLD, uncorrected
# positions -- a corrected `pos` and an object built for the old one disagree
# about the geometry. Rebuilding the seed from scratch with the correction
# already in it avoids that disagreement entirely. Same MPI environment as
# step6/step7 (needs mpi4py + h5py driver='mpio'), so it runs right here
# rather than needing tomo5 (steps15.py is tomo5-only for the FULL run,
# stages 1-3 included; a stage-5-only rerun has no such requirement).
#
# PASS 2 (config_step6_bin{2,1,0}.conf, path_out rec/refined) -- an ordinary
# restart, same shape as a single-pass run: `obj_init_file` now points at the
# corrected seed above (bin 2 only), `pos` is nominal + extra via
# correct3d_extra_file (bin 2 only, same mechanism as pass 1's seed used),
# `prb`/`code`/`pos_code` all read fresh from step0's own files -- nothing
# here is inherited from pass 1's checkpoint:
#
#     step6   config_step6_bin2.conf   iter   0 -> 256
#     step6   config_step6_bin1.conf   iter 256 -> 512
#     step6   config_step6_bin0.conf   iter 512 -> 768
#
# RESUME IS MANUAL, per pass.  find_latest_checkpoint globs
# checkpoint_*{start_iter:04}.h5 and returns None when start_iter == 0, so it
# finds the checkpoint you NAME, not the newest on disk.  After a preemption:
# `ls {path_out}/checkpoints/`, set start_iter in the config to the highest
# checkpoint_NNNN, comment out the finished mpiexec lines, resubmit.  The same
# keying is what makes the bin2->bin1(->bin0) handoff work within each pass,
# so it is not a bug to fix.
#
# RE-SUBMISSION.  Handled by the RUN_* switches now, not by hand-editing.
# correct_correct3D_extra.txt is backed up with a timestamp when RUN_STEP7=1
# is about to rewrite it, and left untouched when RUN_STEP7=0 because pass 2
# reads it; RUN_STEP7=0 with RUN_PASS2=1 and no such file on disk is a hard
# error rather than a silent uncorrected run.  Pass 1's
# config_step6_bin*_pass1.conf has no correct3d_extra_file key, so pass 1 is
# unaffected either way.  Still unguarded: obj_init_corrected.npy, which
# REBUILD THE SEED overwrites silently (plain np.save) against whatever is in
# correct_correct3D_extra.txt at the time -- so set RUN_SEED=0 if you want to
# keep the seed you have.
#
# CREATE AND STRIPE BOTH OUTPUT DIRECTORIES before the first run -- eagle
# defaults to one OST and a bin-0 checkpoint is large.  lfs setstripe does
# NOT create missing parents, so mkdir first.  With R=the `rec` key and
# N=the `run` key from the configs:
#     mkdir -p $R/$N/pass1 $R/$N/refined
#     lfs setstripe -c 8 -S 16M $R/$N/pass1
#     lfs setstripe -c 8 -S 16M $R/$N/refined
# ===========================================================================

# --- user configuration ---
# Software environment (modules + conda env). See the Polaris setup notes.
HTC_ENV=${HTC_ENV:-"${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/../../../polaris_env.sh"}
# HEALTHCHECK=0  skips the ~30 s GPU probe;  RUN_NODES=N  uses N healthy nodes.
# PREFLIGHT=0    skips the input check (it is seconds; leave it on).
# --------------------------

# --- which stages run.  1 = run, 0 = skip -----------------------------------
# These REPLACE commenting out the mpiexec lines, because the
# correct_correct3D_extra.txt handling below has to know whether step 7 is
# going to run: if it is, the old file is about to be overwritten and must be
# backed up; if it is not, that same file is pass 2's INPUT and must be left
# exactly where it is.  A commented-out line is invisible to the shell, so
# that distinction cannot be made by commenting.
#
#   full ladder from scratch   qsub polaris_run.sh
#   pass 2 only, reusing an existing correct_correct3D_extra.txt and seed:
#                              qsub -v RUN_PASS1=0,RUN_STEP7=0,RUN_SEED=0 polaris_run.sh
#   redo step 7 only:          qsub -v RUN_PASS1=0,RUN_SEED=0,RUN_PASS2=0 polaris_run.sh
RUN_PASS1=${RUN_PASS1:-1}   # step6 bin2_pass1 -> bin1_pass1, positions frozen
RUN_STEP7=${RUN_STEP7:-1}   # autofocus      -> correct_correct3D_extra.txt
RUN_SEED=${RUN_SEED:-1}     # steps15.py st.5 -> obj_init_corrected.npy
RUN_PASS2=${RUN_PASS2:-1}   # step6 bin2 -> bin1 -> bin0
echo "stages: PASS1=${RUN_PASS1} STEP7=${RUN_STEP7} SEED=${RUN_SEED} PASS2=${RUN_PASS2}"
# --------------------------

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4          # one rank per Polaris A100
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

# Directory the job was submitted from (PBS_O_WORKDIR when submitted via qsub;
# falls back to the script's own directory for local ./polaris_run.sh testing).
# Plain $(pwd) does NOT work: PBS starts the job in $HOME, not where you qsub'd.
# SCRIPT_DIR is THIS RUN's folder: its six configs, its correct_correct3D_extra.txt,
# its PBS log. CODE_DIR is the shared python, two levels up -- step6.py and friends
# are not copied per run. rec_dir holds gpu_healthcheck.sh and polaris_env.sh.
SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"   # runs/<run>/
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"                                  # AtomiumS2_coded/
rec_dir="$(cd "${SCRIPT_DIR}/../../.." && pwd)"                                # experimental/

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
    python "${CODE_DIR}/preflight.py" "${SCRIPT_DIR}/$1" || {
        echo "ERROR: preflight failed for $1; aborting."; exit 1; }
}

# correct_correct3D_extra.txt.  Step 7 WRITES it; pass 2 READS it.  This used
# to be a hard error whenever the file existed, which made a pass-2-only
# resubmission impossible -- the one case where the file legitimately exists
# and must be kept.  Now it depends on RUN_STEP7:
#
#   RUN_STEP7=1  step 7 is about to overwrite it, so move the old one aside
#                with a timestamp rather than losing it.  Never an error.
#   RUN_STEP7=0  the file IS the input to pass 2, so leave it alone -- but
#                fail loudly if pass 2 wants it and it is not there, instead
#                of letting CodedReader silently reconstruct with no
#                correction at all.
EXTRA="${SCRIPT_DIR}/correct_correct3D_extra.txt"
if [ "${RUN_STEP7}" = "1" ]; then
    if [ -f "${EXTRA}" ]; then
        BAK="${EXTRA}.bak.$(date +%Y%m%d-%H%M%S)"
        mv "${EXTRA}" "${BAK}" || { echo "ERROR: could not back up ${EXTRA}"; exit 1; }
        echo "NOTE: step 7 will rewrite correct_correct3D_extra.txt; previous one kept as $(basename "${BAK}")"
    fi
elif [ "${RUN_PASS2}" = "1" ] && [ ! -f "${EXTRA}" ]; then
    echo "ERROR: RUN_STEP7=0 and RUN_PASS2=1, but ${EXTRA} does not exist."
    echo "       Pass 2's configs name it via correct3d_extra_file, and a missing"
    echo "       file means the ladder would run with NO drift correction at all."
    echo "       Either run step 7 (RUN_STEP7=1) or put the file in place first."
    exit 1
fi

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

MPI="mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} ${CODE_DIR}/set_affinity_gpu_polaris.sh python"

# --- the pipeline; comment out a line to skip that stage ---------------------

# PASS 1, POSITIONS FROZEN -----------------------------------------------
# bin 2: the only rung with start_iter=0, so the only one that reads
# obj_init.npy. bin 1 inherits the object, probe, aperture and both position
# sets through the checkpoint. pos does not move in either (rho[pos]=0); see
# config_step6_bin2_pass1.conf's header for why (and for the history of
# briefly trying otherwise).
if [ "${RUN_PASS1}" = "1" ]; then
    preflight config_step6_bin2_pass1.conf
    echo "=== pass1 bin2 START $(date) ==="
    ${MPI} "${CODE_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin2_pass1.conf" || exit $?

    preflight config_step6_bin1_pass1.conf
    echo "=== pass1 bin1 START $(date) ==="
    ${MPI} "${CODE_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin1_pass1.conf" || exit $?
else
    echo "=== pass1 SKIPPED (RUN_PASS1=0) ==="
fi

# STEP 7 -----------------------------------------------------------------
# Fits the per-angle drift and writes correct_correct3D_extra.txt next to the
# configs -- picked up by every pass-2 config below via correct3d_extra_file.
# Splits z slices over the ranks and allreduces the 256-bin histogram, so the
# answer does not depend on -n.  -n is capped at 8: a vertical shift mixes z,
# so every rank repeats it, and that cost does not amortize past a handful of
# ranks. --iter 512: the pass-1 bin-1 checkpoint (start_iter=256, niter=512
# in config_step6_bin1_pass1.conf). --bin 0: search on the native bin-1 grid.
# --ntheta 7200: all acquired angles, explicit rather than the default quarter
# of the scan -- a subsample doesn't reduce cost here the way it does for
# step6 (step7 forward-projects a checkpoint that's already built; the cost
# is dominated by the FBP/entropy evaluations, not reading more frames), and
# there is no reason to throw away 3/4 of the angles for free.
#
# --score-radius 0.75 IS REQUIRED, not a tuning knob (added 2026-11-12).  At
# the default 1.0 this search ran away to 6071 px on a 1280 grid and returned
# a destroyed volume, confirmed by A/B against 0.75 on the same checkpoint
# (s7_trim_check.sh).  The seed carries two rotationally-symmetric features
# inside the full scored disc -- a bowl at r = n/nobj = 0.80, the edge of the
# measured data, and a ring at r = sqrt(mask_r) = 0.949 from stage 5's FBP
# apodisation.  No per-angle drift can sharpen either, but a runaway one
# smears them flat for ~1.58 nats against ~0.004 for actually focusing the
# sample, so the optimiser destroys the image.  0.75 scores real data only.
if [ "${RUN_STEP7}" = "1" ]; then
    echo "=== step7 START $(date) ==="
    STEP7RANKS=$(( NTOTRANKS < 8 ? NTOTRANKS : 8 ))
    mpiexec ${HOSTOPT} -n ${STEP7RANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} "${CODE_DIR}/set_affinity_gpu_polaris.sh" python "${CODE_DIR}/step7.py" "${SCRIPT_DIR}/config_step6_bin1_pass1.conf" --iter 512 --bin 0 --ntheta 7200 --score-radius 0.75 || exit $?
else
    echo "=== step7 SKIPPED (RUN_STEP7=0), reusing $(basename "${EXTRA}") ==="
fi

# REBUILD THE SEED ---------------------------------------------------------
# steps15.py stage 5 only (start_step=5 in config_steps15_corrected.conf),
# same MPI launcher as step6/step7. Reads correct_correct3D_extra.txt that
# STEP 7 just wrote; writes obj_init_corrected.npy, which config_step6_bin2.conf
# below reads as its obj_init_file.
if [ "${RUN_SEED}" = "1" ]; then
    echo "=== rebuild seed START $(date) ==="
    ${MPI} "${CODE_DIR}/steps15.py" "${SCRIPT_DIR}/config_steps15_corrected.conf" || exit $?
else
    echo "=== rebuild seed SKIPPED (RUN_SEED=0) ==="
fi

# PASS 2, REFINED -----------------------------------------------------------
if [ "${RUN_PASS2}" = "1" ]; then
    preflight config_step6_bin2.conf
    echo "=== pass2 bin2 START $(date) ==="
    ${MPI} "${CODE_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin2.conf" || exit $?

    preflight config_step6_bin1.conf
    echo "=== pass2 bin1 START $(date) ==="
    ${MPI} "${CODE_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin1.conf" || exit $?

    preflight config_step6_bin0.conf
    echo "=== pass2 bin0 START $(date) ==="
    ${MPI} "${CODE_DIR}/step6.py" "${SCRIPT_DIR}/config_step6_bin0.conf" || exit $?
else
    echo "=== pass2 SKIPPED (RUN_PASS2=0) ==="
fi

echo "=== DONE $(date) ==="
