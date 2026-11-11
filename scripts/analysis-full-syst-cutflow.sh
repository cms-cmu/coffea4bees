#!/bin/bash

# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output/" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

INPUT_DIR="${OUTPUT_BASE_DIR%/}/ttHbb_full_syst/PhaseF_1_analysis"
INPUT_FILE="${INPUT_DIR}/histAll_ttHbb_full_syst_signals_syst.coffea"
OUTPUT_DIR="${INPUT_DIR}"
OUTPUT_YML="${OUTPUT_DIR}/cutflow_systematics.yml"
OUTPUT_TXT="${OUTPUT_DIR}/cutflow_validation_systematics.txt"
OUTPUT_HTML="${OUTPUT_DIR}/cutflow_systematics.html"
KNOWN_COUNTS="coffea4bees/analysis/tests/known_Counts_systematics_ttHbb.yml"

echo "############### Configuration"
echo "Output base:      $OUTPUT_BASE_DIR"
echo "Input file:       $INPUT_FILE"
echo "Output yml:       $OUTPUT_YML"
echo "Known cutflow:    $KNOWN_COUNTS"
echo ""

create_output_directory "$OUTPUT_DIR"

if [ ! -f "$INPUT_FILE" ]; then
    for candidate in \
        "output/ttHbb_full_syst/PhaseF_1_analysis/histAll_ttHbb_full_syst_signals_syst.coffea" \
        "barista_framework/output/ttHbb_full_syst/PhaseF_1_analysis/histAll_ttHbb_full_syst_signals_syst.coffea" \
        "${OUTPUT_BASE_DIR%/}/barista_framework/output/ttHbb_full_syst/PhaseF_1_analysis/histAll_ttHbb_full_syst_signals_syst.coffea" \
        "${OUTPUT_BASE_DIR%/}/output/ttHbb_full_syst/PhaseF_1_analysis/histAll_ttHbb_full_syst_signals_syst.coffea" \
        "${CI_PROJECT_DIR:-.}/barista_framework/output/ttHbb_full_syst/PhaseF_1_analysis/histAll_ttHbb_full_syst_signals_syst.coffea"
    do
        if [ -f "$candidate" ]; then
            INPUT_FILE="$candidate"
            INPUT_DIR="$(dirname "$candidate")"
            OUTPUT_DIR="$INPUT_DIR"
            OUTPUT_YML="${OUTPUT_DIR}/cutflow_systematics.yml"
            OUTPUT_TXT="${OUTPUT_DIR}/cutflow_validation_systematics.txt"
            OUTPUT_HTML="${OUTPUT_DIR}/cutflow_systematics.html"
            echo "Found input coffea file at fallback path: $INPUT_FILE"
            break
        fi
    done
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input coffea file not found: $INPUT_FILE"
    echo "Searched standard paths and fallbacks. Available files under output/:"
    ls -la output/ || true
    ls -la barista_framework/output/ || true
    exit 1
fi

echo "############### Running systematic cutflow analysis"
python3 src/tools/cutflow_systematics.py "$INPUT_FILE" \
    --output-yml "$OUTPUT_YML" \
    --output-html "$OUTPUT_HTML" \
    --output-txt "$OUTPUT_TXT" \
    --known-counts "$KNOWN_COUNTS" \
    --tolerance 0.001

STATUS=$?

echo ""
echo "############### Generated cutflow:"
cat "$OUTPUT_YML"
echo ""

echo "############### Validation summary:"
cat "$OUTPUT_TXT"
echo ""

if [ $STATUS -eq 0 ]; then
    echo "############### SUCCESS: Systematic cutflow analysis completed successfully!"
else
    echo "############### FAILED: Systematic cutflow comparison failed."
    exit 1
fi
