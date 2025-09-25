#!/bin/bash

# Source common functions
source "src/scripts/common.sh"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --output-base DIR         Base output directory (default: output/)"
    echo "  --input-file FILE         Input coffea file (default: analysis_test/test.coffea)"
    echo "  --input-subdir DIR        Input subdirectory (default: analysis_test)"
    echo "  --output-filename FILE    Output cutflow filename (default: test_cutflow.yml)"
    echo "  --output-subdir DIR       Output subdirectory (default: analysis_test_cutflows)"
    echo "  --known-cutflow FILE      Known cutflow file for comparison (default: bbreww/tests/known_cutflow_hh_bbww_processor.yml)"
    echo "  --error-threshold FLOAT  Error threshold for cutflow comparison (default: 0.001)"
    echo "  --cutflow-list LIST      Comma-separated list of cutflows to check (default: ['passJetMult', 'passPreSel', 'passDiJetMass', 'SR', 'SB', 'passSvB', 'failSvB'])"
    echo "  --help                    Show this help message"
    exit 1
}

# Function to display configuration
display_config() {
    echo "############### Configuration"
    echo "Output base:      $OUTPUT_BASE"
    echo "Input file:       $INPUT_FILE"
    echo "Output file:      $OUTPUT_FILE"
    echo "Known cutflow:    $KNOWN_CUTFLOW"
    echo "Error threshold:  $ERROR_THRESHOLD"
    echo "Cutflow list:     ${CUTFLOW_LIST:-All}"
    echo ""
}

# Default values
declare -A DEFAULTS=(
    ["OUTPUT_BASE"]="output/"
    ["INPUT_SUBDIR"]="analysis_test"
    ["INPUT_FILENAME"]="test.coffea"
    ["OUTPUT_FILENAME"]="test_cutflow.yml"
    ["OUTPUT_SUBDIR"]="analysis_test_cutflows"
    ["KNOWN_CUTFLOW"]="coffea4bees/analysis/tests/known_Counts.yml"
    ["ERROR_THRESHOLD"]="0.001"
    ["CUTFLOW_LIST"]="passJetMult,passPreSel,passDiJetMass,SR,SB"
)

# Initialize variables with defaults
OUTPUT_BASE="${DEFAULTS[OUTPUT_BASE]}"
INPUT_SUBDIR="${DEFAULTS[INPUT_SUBDIR]}"
INPUT_FILENAME="${DEFAULTS[INPUT_FILENAME]}"
INPUT_FILE=""
OUTPUT_FILENAME="${DEFAULTS[OUTPUT_FILENAME]}"
OUTPUT_SUBDIR="${DEFAULTS[OUTPUT_SUBDIR]}"
KNOWN_CUTFLOW="${DEFAULTS[KNOWN_CUTFLOW]}"
ERROR_THRESHOLD="${DEFAULTS[ERROR_THRESHOLD]}"
CUTFLOW_LIST="${DEFAULTS[CUTFLOW_LIST]}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --input-file)
            # Allow full path or just filename (will use default subdir)
            if [[ "$2" == *"/"* ]]; then
                INPUT_FILE="$2"
            else
                INPUT_FILENAME="$2"
            fi
            shift 2
            ;;
        --input-subdir)
            INPUT_SUBDIR="$2"
            shift 2
            ;;
        --output-filename)
            OUTPUT_FILENAME="$2"
            shift 2
            ;;
        --output-subdir)
            OUTPUT_SUBDIR="$2"
            shift 2
            ;;
        --known-cutflow)
            KNOWN_CUTFLOW="$2"
            shift 2
            ;;
        --error-threshold)
            ERROR_THRESHOLD="$2"
            shift 2
            ;;
        --cutflow-list)
            CUTFLOW_LIST="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Save our parsed values before sourcing the initial variables script
declare -A SAVED_VARS=(
    ["OUTPUT_BASE"]="$OUTPUT_BASE"
    ["INPUT_SUBDIR"]="$INPUT_SUBDIR"
    ["INPUT_FILENAME"]="$INPUT_FILENAME"
    ["INPUT_FILE"]="$INPUT_FILE"
    ["OUTPUT_FILENAME"]="$OUTPUT_FILENAME"
    ["OUTPUT_SUBDIR"]="$OUTPUT_SUBDIR"
    ["KNOWN_CUTFLOW"]="$KNOWN_CUTFLOW"
    ["ERROR_THRESHOLD"]="$ERROR_THRESHOLD"
    ["CUTFLOW_LIST"]="$CUTFLOW_LIST"
)

# Restore our configuration variables after setup
OUTPUT_BASE="${SAVED_VARS[OUTPUT_BASE]}"
INPUT_SUBDIR="${SAVED_VARS[INPUT_SUBDIR]}"
INPUT_FILENAME="${SAVED_VARS[INPUT_FILENAME]}"
INPUT_FILE="${SAVED_VARS[INPUT_FILE]}"
OUTPUT_FILENAME="${SAVED_VARS[OUTPUT_FILENAME]}"
OUTPUT_SUBDIR="${SAVED_VARS[OUTPUT_SUBDIR]}"
KNOWN_CUTFLOW="${SAVED_VARS[KNOWN_CUTFLOW]}"
ERROR_THRESHOLD="${SAVED_VARS[ERROR_THRESHOLD]}"
CUTFLOW_LIST="${SAVED_VARS[CUTFLOW_LIST]}"

# Set up file paths
if [[ -z "$INPUT_FILE" ]]; then
    INPUT_FILE="${OUTPUT_BASE}/${INPUT_SUBDIR}/${INPUT_FILENAME}"
fi
OUTPUT_DIR="${OUTPUT_BASE}/${OUTPUT_SUBDIR}/"
OUTPUT_FILE="${OUTPUT_DIR}${OUTPUT_FILENAME}"

# Display configuration
display_config

# Create output directory if it doesn't exist
create_output_directory "$OUTPUT_DIR"

echo "############### Running cutflow analysis"
# Run the Python script to dump cutflow
IFS=',' read -ra cutflows <<< "$CUTFLOW_LIST"
cmd=(python coffea4bees/analysis/tests/dumpCutFlow.py --input "$INPUT_FILE" -o "$OUTPUT_FILE" -c )
for cf in "${cutflows[@]}"; do
    cmd+=("$cf")
done
run_command "${cmd[@]}"

if [ $? -ne 0 ]; then
    echo "Failed to dump cutflow to YAML"
    exit 1
fi

ls -lh "$OUTPUT_FILE"
echo "############### Generated cutflow:"
cat "$OUTPUT_FILE"

echo "############### Running cutflow comparison"
# Run the cutflow unit test
cmd=(python coffea4bees/analysis/tests/cutflow_test.py --inputFile "$INPUT_FILE" --knownCounts "$KNOWN_CUTFLOW" --error_threshold "$ERROR_THRESHOLD")
run_command "${cmd[@]}"

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "############### SUCCESS: Cutflow analysis completed successfully!"
else
    echo "############### FAILED: Cutflow comparison failed."
    exit 1
fi

echo "############### Final output"
ls -lh "$OUTPUT_FILE"
