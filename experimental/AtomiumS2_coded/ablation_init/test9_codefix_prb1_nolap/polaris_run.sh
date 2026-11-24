#!/bin/bash
#PBS -A APS_IRI
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -N AtomS2_init_t9
#PBS -j oe

# ===========================================================================
# INITIALISATION ABLATION -- test7 with the Laplacian regularizer OFF
#
#     cd <this folder> && qsub polaris_run.sh
#
# ONE bin-2 step6 run against this folder's own config_step6_bin2.conf.  No
# ladder, no step 7, no seed rebuild: the point is to compare how the solver
# descends from three different starting points, nothing else.  Each test
# writes to its own path_out, so they can run in any order and never collide.
#
# Nothing in ../.. is modified or copied.  step6.py, preflight.py,
# set_affinity_gpu_polaris.sh and the conda env are all referenced in place.
#
# Sizing. The measured 24-node run (bin2, full ntheta=7200, 512 iterations)
# took 12m46s with 75 angles/rank. At stride=6 (1200 angles) on 1 node (4
# ranks, 300 angles/rank -- 4x that baseline's per-rank load) the estimate is
# ~766s * 4 = ~51 minutes, inside the 1-hour debug cap but without much
# margin. If it runs long, drop niter in the config rather than raising the
# walltime -- the comparison only needs all four tests to use the SAME niter
# and the SAME stride, not a converged result.
#
# BEFORE THE FIRST SUBMISSION:
#     mkdir -p /eagle/APS_IRI/cekmekci/AtomiumS2_coded/rec/ablation_init/test9_codefix_prb1_nolap
#     lfs setstripe -c 8 -S 16M /eagle/APS_IRI/cekmekci/AtomiumS2_coded/rec/ablation_init/test9_codefix_prb1_nolap
# ===========================================================================

set -u

NRANKS=4          # one rank per Polaris A100
NTHREADS=4
NDEPTH=8

# This folder (PBS_O_WORKDIR when qsub'd from here), then the sample and
# experimental directories above it.  Plain $(pwd) does NOT work: PBS starts
# the job in $HOME, not where you qsub'd.
SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"     # experimental/AtomiumS2_coded
EXP_DIR="$(cd "${SAMPLE_DIR}/.." && pwd)"           # experimental
CFG="${SCRIPT_DIR}/config_step6_bin2.conf"
HTC_ENV=${HTC_ENV:-"${EXP_DIR}/polaris_env.sh"}

NNODES=$(wc -l < "$PBS_NODEFILE")
export NTOTRANKS=$(( NNODES * NRANKS ))

cd "${EXP_DIR}"
exec > >(tee "${SCRIPT_DIR}/pbs-${PBS_JOBID}.out") 2>&1

echo "Test:        test9_codefix_prb1_nolap"
echo "Config:      ${CFG}"
echo "Sample dir:  ${SAMPLE_DIR}"
echo "Jobid:       ${PBS_JOBID}"
echo "Nodes:       ${NNODES}  ranks ${NTOTRANKS}  per node ${NRANKS}"

[ -r "${CFG}" ] || { echo "ERROR: config not readable: ${CFG}"; exit 1; }
[ -r "${HTC_ENV}" ] || { echo "ERROR: HTC_ENV not readable: ${HTC_ENV}"; exit 1; }
source "${HTC_ENV}"
echo "python: $(which python)"

# Catches a missing code.npy / flat_code.npy, or an ncode that disagrees with
# it, in seconds instead of a node-hour inside the windowed shift.  An absent
# obj_init_file is only a WARNING here, which is exactly what tests 1 and 2
# rely on.
if [ "${PREFLIGHT:-1} " != "0 " ]; then
    python "${SAMPLE_DIR}/preflight.py" "${CFG}" || {
        echo "ERROR: preflight failed; aborting."; exit 1; }
fi

# Drop nodes whose GPUs cannot take a CUDA context. Must run AFTER the env is
# sourced: the probe needs cupy.
HOSTOPT=""
if [ "${HEALTHCHECK:-1}" = "1" ]; then
    GOOD="${SCRIPT_DIR}/nodes.good.${PBS_JOBID}"
    bash "${EXP_DIR}/gpu_healthcheck.sh" "${GOOD}" "${NRANKS}" "${RUN_NODES:-1}" || {
        echo "ERROR: too few healthy nodes; aborting."; exit 1; }
    head -n "${RUN_NODES:-$(wc -l < "${GOOD}")}" "${GOOD}" > "${GOOD}.run"
    NNODES=$(wc -l < "${GOOD}.run")
    export NTOTRANKS=$(( NNODES * NRANKS ))
    HOSTOPT="--hostfile ${GOOD}.run"
    echo "Running on ${NNODES} healthy nodes  TOTAL_NUM_RANKS=${NTOTRANKS}"
fi

echo "=== test9_codefix_prb1_nolap bin2 START $(date) ==="
mpiexec ${HOSTOPT} -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth \
    --env OMP_NUM_THREADS=${NTHREADS} "${SAMPLE_DIR}/set_affinity_gpu_polaris.sh" \
    python "${SAMPLE_DIR}/step6.py" "${CFG}" || exit $?
echo "=== test9_codefix_prb1_nolap bin2 DONE $(date) ==="
echo "=== ALL STAGES DONE $(date) ==="
