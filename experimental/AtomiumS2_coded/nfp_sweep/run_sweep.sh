#!/bin/bash
# One step0 NFP reconstruction per trial z1c laid out by prep_sweep.py.
# Sequential: each trial wants every GPU, so they cannot overlap.
#
#   bash run_sweep.sh                 # every trial that has no code.npy yet
#   NP=2 bash run_sweep.sh            # override the rank count
#   bash run_sweep.sh z1c_17750um     # named trials only
#
# Nothing here touches /data3/cekmekci/AtomiumS2_coded/nfp -- the baseline
# result from 2026-11-18 15:21 stays where it is and is the reference the
# sweep is compared against.  Each trial's stdout goes to its own step0.log.
set -u

#   ROOT=/data3/cekmekci/AtomiumS2_coded/nfp_psf bash run_sweep.sh   # the psf sweep
ROOT=${ROOT:-/data3/cekmekci/AtomiumS2_coded/nfp_sweep}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)     # AtomiumS2_coded
NP=${NP:-$(nvidia-smi -L | wc -l)}

if [ $# -gt 0 ]; then
    TRIALS=()
    for t in "$@"; do TRIALS+=("$ROOT/$t"); done
else
    # any subfolder carrying a config -- serves the z1c sweep and the psf sweep alike
    mapfile -t TRIALS < <(ls -d "$ROOT"/*/ 2>/dev/null \
                          | while read -r d; do [ -f "$d/config_step0.conf" ] && echo "${d%/}"; done \
                          | sort)
fi

if [ ${#TRIALS[@]} -eq 0 ]; then
    echo "no trials under $ROOT -- run prep_sweep.py first"; exit 1
fi

echo "$NP ranks, ${#TRIALS[@]} trial(s), from $SRC"
cd "$SRC" || exit 1

for d in "${TRIALS[@]}"; do
    name=$(basename "$d")
    if [ -f "$d/code.npy" ]; then
        echo "== $name: code.npy exists, skipping"
        continue
    fi
    if [ ! -f "$d/config_step0.conf" ]; then
        echo "== $name: no config_step0.conf, skipping"
        continue
    fi
    echo "== $name  $(date '+%F %T')"
    mpirun -np "$NP" -mca opal_common_ucx_opal_mem_hooks 1 \
        ./set_affinity_gpu.sh python step0.py "$d/config_step0.conf" \
        > "$d/step0.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "   FAILED rc=$rc -- see $d/step0.log; stopping"
        tail -20 "$d/step0.log"
        exit $rc
    fi
    echo "   done $(date '+%F %T')  $(tail -1 "$d/conv_nfp.csv")"
done

echo
echo "all done -- next: python collect.py"
