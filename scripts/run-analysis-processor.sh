#!/bin/bash

# Source common functions
source "src/scripts/common.sh"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
  --output-base DIR              Base output directory (default: output/)
  --processor PATH               Path to processor file
  --dataset-metadata PATH        Path to metadata file
  --config PATH                  Path to config file
  --triggers PATH                Path to triggers file
  --luminosities PATH            Path to luminosities file
  --datasets "DATASET1 ..."      Space-separated datasets
  --year YEAR                    Analysis year
  --output-filename FILE         Output filename
  --output-subdir DIR            Output subdirectory
  --additional-flags FLAGS       Additional flags for runner.py
  --no-test                      Disable test mode
  --condor                       Enable condor mode
  -h, --help                     Show this help message
EOF
    exit 1
}

# Function to display configuration
display_config() {
    display_section_header "Configuration"
    echo "Processor:          $PROCESSOR_PATH"
    echo "Datasets Metadata:  $METADATA_PATH"
    echo "Config:             $CONFIG_PATH"
    echo "Triggers:           $TRIGGERS_PATH"
    echo "Luminosities:       $LUMINOSITIES_PATH"
    echo "Datasets:           $DATASETS"
    echo "Year:               $YEAR"
    echo "Output filename:    $OUTPUT_FILENAME"
    echo "Test mode:          $([ -n "$TEST_MODE" ] && echo "enabled" || echo "disabled")"
    echo "Output subdir:      $OUTPUT_SUBDIR"
    echo "Condor mode:        $([ -n "$CONDOR_MODE" ] && echo "enabled" || echo "disabled")"
    echo "Additional flags:   ${ADDITIONAL_FLAGS:-"(none)"}"
    echo ""
}

# Default values
declare -A DEFAULTS=(
    ["OUTPUT_BASE"]="output/"
    ["PROCESSOR_PATH"]="coffea4bees/analysis/processors/processor_HH4b.py"
    ["METADATA_PATH"]="${DATASET:-coffea4bees/metadata/datasets_HH4b_Run2/}"
    ["CONFIG_PATH"]="coffea4bees/analysis/metadata/HH4b.yml"
    ["TRIGGERS_PATH"]="coffea4bees/metadata/triggers_HH4b.yml"
    ["LUMINOSITIES_PATH"]="coffea4bees/metadata/luminosities_HH4b.yml"
    ["DATASETS"]="TTToSemiLeptonic"
    ["YEAR"]="UL18"
    ["OUTPUT_FILENAME"]="test.coffea"
    ["TEST_MODE"]="-t"
    ["OUTPUT_SUBDIR"]="analysis_test"
    ["ADDITIONAL_FLAGS"]=""
    ["CONDOR_MODE"]=""
)

# Initialize variables with defaults
OUTPUT_BASE="${DEFAULTS[OUTPUT_BASE]}"
PROCESSOR_PATH="${DEFAULTS[PROCESSOR_PATH]}"
METADATA_PATH="${DEFAULTS[METADATA_PATH]}"
CONFIG_PATH="${DEFAULTS[CONFIG_PATH]}"
TRIGGERS_PATH="${DEFAULTS[TRIGGERS_PATH]}"
LUMINOSITIES_PATH="${DEFAULTS[LUMINOSITIES_PATH]}"
DATASETS="${DEFAULTS[DATASETS]}"
YEAR="${DEFAULTS[YEAR]}"
OUTPUT_FILENAME="${DEFAULTS[OUTPUT_FILENAME]}"
TEST_MODE="${DEFAULTS[TEST_MODE]}"
OUTPUT_SUBDIR="${DEFAULTS[OUTPUT_SUBDIR]}"
ADDITIONAL_FLAGS="${DEFAULTS[ADDITIONAL_FLAGS]}"
CONDOR_MODE="${DEFAULTS[CONDOR_MODE]}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --processor)
            PROCESSOR_PATH="$2"
            shift 2
            ;;
        --dataset-metadata)
            METADATA_PATH="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --triggers)
            TRIGGERS_PATH="$2"
            shift 2
            ;;
        --luminosities)
            LUMINOSITIES_PATH="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --year)
            YEAR="$2"
            shift 2
            ;;
        --output-filename)
            OUTPUT_FILENAME="$2"
            shift 2
            ;;
        --no-test)
            TEST_MODE=""
            shift
            ;;
        --output-subdir)
            OUTPUT_SUBDIR="$2"
            shift 2
            ;;
        --condor)
            CONDOR_MODE="--condor"
            shift
            ;;
        --additional-flags)
            shift
            # Consume all remaining arguments as additional flags
            ADDITIONAL_FLAGS="$@"
            break
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

# Save our parsed values before setting up environment
declare -A SAVED_VARS=(
    ["OUTPUT_BASE"]="$OUTPUT_BASE"
    ["PROCESSOR_PATH"]="$PROCESSOR_PATH"
    ["METADATA_PATH"]="$METADATA_PATH"
    ["CONFIG_PATH"]="$CONFIG_PATH"
    ["TRIGGERS_PATH"]="$TRIGGERS_PATH"
    ["LUMINOSITIES_PATH"]="$LUMINOSITIES_PATH"
    ["DATASETS"]="$DATASETS"
    ["YEAR"]="$YEAR"
    ["OUTPUT_FILENAME"]="$OUTPUT_FILENAME"
    ["TEST_MODE"]="$TEST_MODE"
    ["DO_PROXY"]="$DO_PROXY"
    ["OUTPUT_SUBDIR"]="$OUTPUT_SUBDIR"
    ["ADDITIONAL_FLAGS"]="$ADDITIONAL_FLAGS"
    ["CONDOR_MODE"]="$CONDOR_MODE"
)

# Setup proxy if needed
setup_proxy 

# Restore our configuration variables after setup
OUTPUT_BASE="${SAVED_VARS[OUTPUT_BASE]}"
PROCESSOR_PATH="${SAVED_VARS[PROCESSOR_PATH]}"
METADATA_PATH="${SAVED_VARS[METADATA_PATH]}"
CONFIG_PATH="${SAVED_VARS[CONFIG_PATH]}"
TRIGGERS_PATH="${SAVED_VARS[TRIGGERS_PATH]}"
LUMINOSITIES_PATH="${SAVED_VARS[LUMINOSITIES_PATH]}"
DATASETS="${SAVED_VARS[DATASETS]}"
YEAR="${SAVED_VARS[YEAR]}"
OUTPUT_FILENAME="${SAVED_VARS[OUTPUT_FILENAME]}"
TEST_MODE="${SAVED_VARS[TEST_MODE]}"
OUTPUT_SUBDIR="${SAVED_VARS[OUTPUT_SUBDIR]}"
ADDITIONAL_FLAGS="${SAVED_VARS[ADDITIONAL_FLAGS]}"
CONDOR_MODE="${SAVED_VARS[CONDOR_MODE]}"

# Display configuration
display_config

OUTPUT_DIR="${OUTPUT_BASE}/${OUTPUT_SUBDIR}/"
create_output_directory "$OUTPUT_DIR"

display_section_header "Running test processor"
# Build command with proper handling of multi-word flags
cmd=(python runner.py 
    -p "$PROCESSOR_PATH" 
    -m "$METADATA_PATH" 
    -c "$CONFIG_PATH" 
    --triggers "$TRIGGERS_PATH"
    --luminosities "$LUMINOSITIES_PATH"
    -d $DATASETS 
    -y $YEAR
    -op "$OUTPUT_DIR" 
    -o "$OUTPUT_FILENAME"
)

# Add optional flags
[ -n "$TEST_MODE" ] && cmd+=( $TEST_MODE )
[ -n "$CONDOR_MODE" ] && cmd+=( $CONDOR_MODE )
[ -n "$ADDITIONAL_FLAGS" ] && cmd+=( $ADDITIONAL_FLAGS )

run_command "${cmd[@]}"
if [ $? -ne 0 ]; then
    exit 1
fi

display_section_header "Output files"
ls -R $OUTPUT_DIR