#!/bin/bash
#PBS -A APS_IRI
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -N AtomiumS2_coded_steps15
#PBS -j oe
#PBS -m be
#PBS -M cekmekci@anl.gov

# ===========================================================================
# AtomiumS2, coded aperture -- STAGE 5 ONLY, standalone.
#
#     qsub polaris_run_steps15.sh
#
# Regenerates obj_init.npy (steps15.py, config_steps15.conf, start_step=5 --
# stages 1-3 already ran and wrote shifts_cor/shifts_code/prb to codedHT.h5).
# Deliberately its OWN job/script, separate from polaris_run.sh: tuning
# stage 5 itself (Paganin params, the pad-to-nobj-before-FBP taper added
# 2026-11-11) means rerunning THIS repeatedly, and bundling it into the
# 24-node/2-hour polaris_run.sh would mean paying for PASS 1, STEP 7 and
# PASS 2 on every tuning iteration whether or not the seed actually changed.
#
# NODE COUNT / WALLTIME here are independent of polaris_run.sh's -- lower
# `select=` and switch to the debug queue for quick iteration while tuning.
# Stage 5 has NO angle/resolution subsetting (confirmed from its own code:
# it reads ntheta/nz/n straight off the raw scan's shape), so fewer nodes
# buys you cheaper iteration at the cost of wall-clock, not less total work
# -- there is no "toy" version of this stage the way step6/step7 have one.
#
# Requires codedHT.h5 already on eagle (step0 and stages 1-3 already ran).
# ===========================================================================

# --- user configuration ---
HTC_ENV=${HTC_ENV:-"${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/../polaris_env.sh"}
# HEALTHCHECK=0  skips the ~30 s GPU probe;  RUN_NODES=N  uses N healthy nodes.
# --------------------------

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4          # one rank per Polaris A100
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

# Directory the job was submitted from (PBS_O_WORKDIR when submitted via qsub;
# falls back to the script's own directory for local ./polaris_run_steps15.sh
# testing). Plain $(pwd) does NOT work: PBS starts the job in $HOME, not
# where you qsub'd.
SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
rec_dir="$(dirname "${SCRIPT_DIR}")"

cd "${rec_dir}"
exec > >(tee "${SCRIPT_DIR}/pbs-steps15-${PBS_JOBID}.out") 2>&1

echo "Sample dir:  ${SCRIPT_DIR}"
echo "Rec dir:     ${rec_dir}"
echo "Jobid: $PBS_JOBID"
echo "Running on host: $(hostname)"
echo "Running on nodes: $(cat $PBS_NODEFILE)"
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}"

# Modules + conda env -- see polaris_run.sh's header for why this must be
# sourced inside the job.
[ -r "${HTC_ENV}" ] || { echo "ERROR: HTC_ENV not readable: ${HTC_ENV}"; exit 1; }
source "${HTC_ENV}"
echo "python: $(which python)"

# Drop nodes whose GPUs cannot take a CUDA context -- see polaris_run.sh's
# header for why this can only be done from inside the job.
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

echo "=== steps15 (stage 5 only, start_step=5) START $(date) ==="
${MPI} "${SCRIPT_DIR}/steps15.py" "${SCRIPT_DIR}/config_steps15.conf" || exit $?
echo "=== DONE $(date) ==="
