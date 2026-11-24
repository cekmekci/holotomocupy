#!/usr/bin/env bash
# The 2-D coded-holography tests.  Serial first, then the MPI check on 2 ranks.
#
#   ./run.sh                     # all of them
#   ./run.sh cascade             # only files whose name contains "cascade"
#   NP=4 ./run.sh                # a different rank count for the MPI part
#
# As in tests/coded/run.sh: a non-interactive shell skips the module lines in
# .bashrc, and the conda env may carry an older holotomocupy than this repo,
# so PYTHONPATH has to point at src/.
set -eu
here="$(dirname "$(readlink -f "$0")")"
repo="$(cd "${here}/../.." && pwd)"
export PYTHONPATH="${repo}/src"
PY="${HTC_PYTHON:-$(command -v python)}"
MPIRUN="${MPIRUN:-$(command -v mpirun)}"
NP="${NP:-2}"
# Never /tmp for the cupy cache: a full /tmp breaks everything else on the host.
[ -n "${HTC_SCRATCH:-}" ] || export HTC_SCRATCH="${HOME}"
[ -n "${CUPY_CACHE_DIR:-}" ] || export CUPY_CACHE_DIR="${HTC_SCRATCH}/.cupy"

filter="${1:-}"
fail=0
for f in test_cascade_holo_coded test_rec_holo_coded test_io_holo_coded; do
    [ -n "${filter}" ] && case "${f}" in *"${filter}"*) ;; *) continue ;; esac
    echo "================ ${f}.py"
    "${PY}" "${here}/${f}.py" || fail=1
done

# The one that has something MPI-specific to say: a missing allreduce on
# grads['obj'] is invisible on a single rank.
if [ -z "${filter}" ]; then
    echo "================ test_io_holo_coded.py   (${NP} ranks)"
    "${MPIRUN}" -np "${NP}" "${PY}" "${here}/test_io_holo_coded.py" || fail=1
    echo "================ test_rec_holo_coded.py mpi   (${NP} ranks)"
    "${PY}" "${here}/test_rec_holo_coded.py" test_mpi || fail=1   # writes the reference
    "${MPIRUN}" -np "${NP}" "${PY}" "${here}/test_rec_holo_coded.py" mpi || fail=1
fi

[ "${fail}" -eq 0 ] && echo "ALL HOLO-CODED TESTS PASSED" || echo "SOME HOLO-CODED TESTS FAILED"
exit "${fail}"
