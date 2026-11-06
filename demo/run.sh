#!/usr/bin/env bash
# One GPU.   ./run.sh 02_holotomography.py [options]
#
# If a system MPI module (e.g. nvhpc-hpcx) is loaded in this shell, its
# OPAL_PREFIX / LD_LIBRARY_PATH outrank the conda env's own self-contained
# Open MPI even when its mpirun/python are called by full path -- Open MPI
# resolves its runtime files from OPAL_PREFIX, not from the binary's location.
# mpi4py loads libmpi.so the same way, so this bites even the single-GPU path.
# Clearing it here makes the script work regardless of what the shell loaded.
set -eu
here="$(dirname "$(readlink -f "$0")")"
env="/home/beams/CEKMEKCI/miniforge3/envs/holotomocupy_coded_atomiums2"
export PYTHONPATH="${here}/../src"
unset OPAL_PREFIX PRTE_PREFIX PMIX_PREFIX OPAL_PKGDATADIR 2>/dev/null || true
export LD_LIBRARY_PATH="${env}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "${env}/bin/python" "${here}/${1:-02_holotomography.py}" "${@:2}"
