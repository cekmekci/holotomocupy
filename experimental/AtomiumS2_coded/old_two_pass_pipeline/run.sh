#!/usr/bin/env bash
# Local coarse-to-fine ladder for the coded-aperture reconstruction.
#
# One step6 invocation per bin level, coarsest first. The three rungs SHARE
# path_out and hand off through start_iter: bin 2 starts fresh from the
# Paganin seed, bin 1 and bin 0 resume from the checkpoint above and
# CodedReader upsamples it x2. Run preflight.py first if anything upstream
# changed.
#
#   ./run.sh                          # bin 2 -> 1 -> 0
#   ./run.sh config_step6_bin2        # one rung
#   NP=4 ./run.sh                     # 4 GPUs
#
# On a cluster use polaris_run.sh (PBS) or esrf.slurm (SLURM) instead; both
# run a single rung, selected by the CONFIG line at the top of the file.
#
# The hpcx module loaded by ~/.bashrc on tomo5 exports OPAL_PREFIX into the
# conda env's own Open MPI, which then cannot find its runtime files and dies
# as "prterun-exec-failed". Clearing it makes this work either way.
#
# UCX also warns on this machine, every run: "UCX is unable to handle
# VM_UNMAP event ... may cause performance degradation or data corruption."
# --mca opal_common_ucx_opal_mem_hooks 1 is UCX's own suggested fix; added
# unconditionally below rather than left to recur per invocation.
set -euo pipefail   # pipefail: without it, `mpirun ... | tee` always exits 0
                    # (bash's -e only looks at tee's own status), so a crashed
                    # step6.py would be silently swallowed and the next rung
                    # would seed itself from a checkpoint that was never
                    # written -- exactly what this script exists to prevent
here="$(cd "$(dirname "$0")" && pwd)"
NP="${NP:-1}"
BIND="${BIND:-${here}/set_affinity_gpu.sh}"
unset OPAL_PREFIX PRTE_PREFIX PMIX_PREFIX OPAL_PKGDATADIR 2>/dev/null || true

run () {
    echo "=== $1"
    mkdir -p "${here}/out"
    mpirun --mca opal_common_ucx_opal_mem_hooks 1 -np "${NP}" ${BIND} \
        python "${here}/step6.py" \
        "${here}/$1.conf" 2>&1 | tee "${here}/out/$1.out"
}

if [ $# -gt 0 ]; then
    run "$1"
else
    run config_step6_bin2
    run config_step6_bin1
    run config_step6_bin0
fi
