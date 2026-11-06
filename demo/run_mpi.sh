#!/usr/bin/env bash
# N GPUs.   ./run_mpi.sh 4 02_holotomography.py [options]
#
# If a system MPI module (e.g. nvhpc-hpcx) is loaded in this shell, its
# OPAL_PREFIX / LD_LIBRARY_PATH outrank the conda env's own self-contained
# Open MPI even when mpirun is called by full path -- Open MPI resolves its
# runtime files (help text, MCA plugins) from OPAL_PREFIX, not from the
# binary's location, and a mismatched prefix fails as "prterun-exec-failed"
# with no useful message. Clearing it here makes the script work regardless
# of what the shell loaded.
set -eu
here="$(dirname "$(readlink -f "$0")")"
env="/home/beams/CEKMEKCI/miniforge3/envs/holotomocupy_coded_atomiums2"
export PYTHONPATH="${here}/../src"
unset OPAL_PREFIX PRTE_PREFIX PMIX_PREFIX OPAL_PKGDATADIR 2>/dev/null || true
export LD_LIBRARY_PATH="${env}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "${env}/bin/mpirun" -np "${1:-4}" \
     "${here}/bind.sh" "${env}/bin/python" "${here}/${2:-02_holotomography.py}" "${@:3}"
