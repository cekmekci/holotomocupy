#!/bin/bash
#PBS -A APS_IRI
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -N axis_sweep
#PBS -j oe
# ===========================================================================
# Rotation-axis sweep: direct reconstruction sharpness vs rot_axis_offset,
# on MEASURED holograms (Paganin run once from codedHT.h5, cached; only the
# cheap shift+FBP+entropy step repeats per candidate). See sweep_rot_axis.py's
# own header for why this is a different, more trustworthy test than step 7:
# there is no re-projected/self-consistent-by-construction sinogram here --
# every candidate is a real reconstruction from real data.
#
#     qsub sweep_rot_axis_polaris.sh
#
# Single GPU, single rank, no MPI -- the whole point of running this as its
# own small job rather than tying up an interactive node (tomo5 busy with
# someone else, this gives the same answer on Polaris instead). ~210 MB of
# GPU memory at the defaults below (400 angles x 64 z-rows x 2048 px), an
# order of magnitude under the 2.4 GB step7.py already ran clean on 7200
# angles -- see sweep_rot_axis.py's own memory-estimate comment if you want
# to push ntheta/nz higher.
#
# Needs codedHT.h5, code.npy, prb.npy and data_nfp.h5 already on eagle --
# confirmed transferred. config_steps15.conf's eagle block is active.
# ===========================================================================

SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"

source "${SCRIPT_DIR}/../polaris_env.sh"
echo "python: $(which python)"
echo "Jobid: $PBS_JOBID"
echo "host:  $(hostname)"

OUT="${HOME}/axis_sweep"
mkdir -p "${OUT}"

echo "=== sweep START $(date) ==="
OPENBLAS_NUM_THREADS=1 python "${SCRIPT_DIR}/sweep_rot_axis.py" \
    "${SCRIPT_DIR}/config_steps15.conf" \
    --ntheta 400 --nz 64 --lo -35 --hi -15 --step 0.5 \
    --out "${OUT}"
echo "=== DONE $(date) ==="
echo "results: ${OUT}/rot_axis_sweep_n400_nz64.csv and .png"
