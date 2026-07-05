#!/bin/bash
# SIDE EXPLORATION: GOFS 3.1 reanalysis pilot — reduce 72 dates of 2015
# (days 1,6,11,16,21,26 of each month) to daily 2D OHC fields.
# Sequential on purpose: shares download bandwidth with the RTOFS backfill.
set -u
PY=/home/suramya/HHP-Prediction/hhp-env/bin/python
SCRIPT=/home/suramya/HHP-Prediction/OHC/exploration/build_gofs31_daily_ohc_fields.py
OUT=/data/suramya/gofs31_ohc_fields_2015
LOG=${OUT}/run_pilot.log
mkdir -p "${OUT}"

DATES=""
for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    for d in 01 06 11 16 21 26; do
        DATES="${DATES} 2015${m}${d}"
    done
done

echo "=== $(date -u +%FT%TZ) starting GOFS 3.1 pilot ($(echo ${DATES} | wc -w) dates) ===" >> "${LOG}"
"${PY}" "${SCRIPT}" --dates ${DATES} >> "${LOG}" 2>&1
echo "=== $(date -u +%FT%TZ) finished (exit $?) ===" >> "${LOG}"
touch ${OUT}/PILOT_DONE
