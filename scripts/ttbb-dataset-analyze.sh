#!/bin/bash
#
# Plot the tt+bb stitching scale factor k per ttbar channel and era.
#
# Reads the stitched dataset metadata (default
# coffea4bees/metadata/datasets/TT_stitched.yml), where stage 3 recorded, for each
# (channel, era), the factor applied to the genWeight of the tt+B events taken
# from the dedicated 4FS TTbb sample:
#
#     k = sumw_ttB(inclusive) / sumw_ttB(TTbb)
#
# Produces two CMS-style panels, Run 2 and Run 3, with the three channels on the
# x axis and one offset point per era.
#
# NOTE: the metadata records, per era, which region k was measured over
# (picoAOD.stitch.k_selection). Run 3 uses the ANALYSIS preselection; the Run 2
# entries predate that option and use the picoAOD level. These are different
# quantities, so each panel is labelled with its own and the script warns when
# both appear. See skimmer/tools/plot_ttbb_stitch_factors.py.
#
# Usage:
#   bash coffea4bees/scripts/ttbb-dataset-analyze.sh [--output-base DIR] [--input YML]

set -eo pipefail

INPUT="coffea4bees/metadata/datasets/TT_stitched.yml"
OUTPUT_BASE_DIR="output/"
FORMAT="png"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-base) OUTPUT_BASE_DIR="$2"; shift 2 ;;
        --input)       INPUT="$2";           shift 2 ;;
        --format)      FORMAT="$2";          shift 2 ;;
        -h|--help)
            sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ ! -f "$INPUT" ]; then
    echo "ERROR: input metadata not found: $INPUT" >&2
    exit 1
fi

OUTDIR="${OUTPUT_BASE_DIR%/}/ttbb_stitch_plots"

# Match the other scripts: run inside the container when one is available, so
# mplhep/matplotlib come from the analysis environment rather than the host.
RUN_CONTAINER=""
if [ -z "$CI" ] && [ -x "./run_container" ]; then
    RUN_CONTAINER="./run_container"
fi

echo "Reading stitch factors from : $INPUT"
echo "Writing plots to            : $OUTDIR"

$RUN_CONTAINER python coffea4bees/skimmer/tools/plot_ttbb_stitch_factors.py \
    --input "$INPUT" \
    --outdir "$OUTDIR" \
    --format "$FORMAT"
