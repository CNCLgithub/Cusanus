#!/usr/bin/env bash
set -euo pipefail
########################################
# Goes through every folder in `WRKDIR`
# and uses `ffmpeg` to generate moveies
########################################
WRKDIR="${1:-$PWD}"
#OUTDIR="${WRKDIR}/videos"
#[[ -d "${OUTDIR}" ]] || mkdir -p "${OUTDIR}"

ffmpeg -y -framerate 24 -i "${WRKDIR}/images/%d.png" \
     -hide_banner -crf 5 -preset slow \
     -c:v libx264  -pix_fmt yuv420p "${WRKDIR}/video.mp4"

# find $WRKDIR -type d -exec \
#     ffmpeg -y -framerate 24 -i '{}/images/%d.png' \
#     -hide_banner -crf 5 -preset slow \
#     -c:v libx264  -pix_fmt yuv420p 'videos/{}'.mp4 \;
