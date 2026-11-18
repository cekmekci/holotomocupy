#!/usr/bin/env bash
# ===========================================================================
# AtomiumS2, coded aperture -- the bin4/bin3 TOY LADDER, as one script.
#
#     ./tomo5_run.sh
#
# Full blueprint of the two-pass ladder, same shape as ../polaris_run.sh, at
# toy scale (bin 4/3, 400 of 7200 angles) so it runs in minutes instead of a
# multi-day allocation.
#
#     PASS 1   bin4_pass1 (iter 0->32)  ->  bin3_pass1 (iter 32->48)
#     STEP 7   drift autofocus on the bin3_pass1 checkpoint
#     PASS 2   bin4 (iter 0->32)  ->  bin3 (iter 32->48), reads step 7's correction
#
# KNOWN ISSUE: step 7's Nelder-Mead search on this toy data reliably wanders
# off (reproduced at both --deg 9 and --deg 2) -- ShiftFourier is periodic, so
# a coefficient that aliases to the right answer scores the same as the
# actual small one, and the optimizer has no reason to prefer it. Not
# established whether this is toy-scale-specific or also shows up on real
# data; step7.py's own guard catches it either way and refuses to write a
# bad result, which is why PASS 2 may not get reached. Kept in as the
# blueprint anyway -- comment out STEP 7 and PASS 2 below if you just want
# PASS 1 (the part that's actually validated: checkpoint handoff, seed
# centering, rot_axis_offset, the writer/reader round trip).
#
# step6.py, step7.py, preflight.py and set_affinity_gpu.sh live one level up,
# in the real pipeline's folder -- only the configs and this script are
# toy-specific. Comment out any `run`/step7/pass-2 line below to skip it.
# ===========================================================================

set -euo pipefail   # pipefail: without it, `... | tee` always exits 0 (bash's
                    # -e only looks at tee's own status), so a crashed step6.py
                    # would be silently swallowed and the next rung would run
                    # anyway, seeding itself from a checkpoint that was never
                    # written
here="$(cd "$(dirname "$0")" && pwd)"       # this folder: the toy configs
parent="$(dirname "${here}")"               # one level up: step6.py, step7.py,
                                             # preflight.py, set_affinity_gpu.sh
NP="${NP:-4}"
BIND="${BIND:-${parent}/set_affinity_gpu.sh}"
# MUST match correct3d_extra_file in config_step6_bin{4,3}.conf, or step 7
# writes somewhere pass 2 never looks and pass 2 silently runs uncorrected.
OUT="/data3/cekmekci/AtomiumS2_coded/rec/test_pass1"

# The hpcx module loaded by ~/.bashrc on tomo5 exports OPAL_PREFIX into the
# conda env's own Open MPI ("prterun-exec-failed" otherwise), and UCX warns
# "unable to handle VM_UNMAP event ... may cause performance degradation or
# data corruption" without the memory-hook flag.
unset OPAL_PREFIX PRTE_PREFIX PMIX_PREFIX OPAL_PKGDATADIR 2>/dev/null || true
MPI="mpirun --mca opal_common_ucx_opal_mem_hooks 1 -np ${NP} ${BIND} python -u"

mkdir -p "${here}/out"
run () {
    # Preflight THIS rung right before it runs, not all four upfront: bin3's
    # checkpoint dependency does not exist until bin4 has actually run, so
    # checking it before bin4 even starts would abort a from-scratch run at
    # the first checkpoint handoff, every time -- caught running this exact
    # script, which is exactly why it exists.
    if [ "${PREFLIGHT:-1}" = "1" ]; then
        python "${parent}/preflight.py" "${here}/$1.conf" || {
            echo "ERROR: preflight failed for $1; aborting."; exit 1; }
    fi
    echo "=== $1 START $(date) ==="
    ${MPI} "${parent}/step6.py" "${here}/$1.conf" 2>&1 | tee "${here}/out/$1.out"
}

# See the re-submission hazard at the top of this file / ../polaris_run.sh.
EXTRA="${OUT}/correct_correct3D_extra.txt"
[ -f "${EXTRA}" ] && { echo "ERROR: ${EXTRA} already exists. If pass 1 + step 7 already ran and this is a pass-2-only rerun, comment out PASS 1 and STEP 7 below and rerun; otherwise step 7 will overwrite it against whatever pass-1 checkpoint happens to be on disk."; exit 1; }

# --- the pipeline; comment out a line to skip that stage ---------------------

# PASS 1, EVERYTHING REFINES ----------------------------------------------------
run config_step6_bin4_pass1
run config_step6_bin3_pass1

# STEP 7 -- --deg 2: measured, degree 2 already captures the real signal here;
# -u: unbuffered, so progress shows up live under `| tee` instead of all at exit.
echo "=== step7 START $(date) ==="
python -u "${parent}/step7.py" "${here}/config_step6_bin3_pass1.conf" \
    --iter 48 --bin 0 --deg 2 --ntheta 7200 --out "${OUT}" 2>&1 | tee "${here}/out/step7.out"

# PASS 2, REFINED --------------------------------------------------------
run config_step6_bin4
run config_step6_bin3

echo "=== DONE $(date) ==="
