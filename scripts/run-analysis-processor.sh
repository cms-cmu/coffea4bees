#!/bin/bash

# Source common functions
source "src/scripts/common.sh"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --output-base DIR         Base output directory (default: output/)"
    echo "  --processor PATH          Path to processor file (default: coffea4bees/analysis/processors/processor_HH4b.py)"
    echo "  --dataset-metadata PATH   Path to metadata file (default: coffea4bees/metadata/datasets_HH4b.yml)"
    echo "  --config PATH             Path to config file (default: coffea4bees/analysis/metadata/HH4b.yml)"
    echo "  --triggers PATH           Path to triggers file (default: coffea4bees/metadata/triggers_HH4b.yml)"
    echo "  --luminosities PATH       Path to luminosities file (default: coffea4bees/metadata/luminosities_HH4b.yml)"
    echo "  --datasets \"DATASET1 DATASET2\"  Space-separated datasets (default: \"TTToSemiLeptonic\")"
    echo "  --year YEAR               Analysis year (default: UL18)"
    echo "  --output-filename FILE    Output filename (default: test.coffea)"
    echo "  --no-test                 Disable test mode"
    echo "  --output-subdir DIR       Output subdirectory (default: analysis_test)"
    echo "  --additional-flags FLAGS  Additional flags to pass to runner.py"
    echo "  --help                    Show this help message"
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
    echo "Additional flags:   ${ADDITIONAL_FLAGS:-"(none)"}"
    echo ""
}

# Default values
declare -A DEFAULTS=(
    ["OUTPUT_BASE"]="output/"
    ["PROCESSOR_PATH"]="coffea4bees/analysis/processors/processor_HH4b.py"
    ["METADATA_PATH"]="${DATASET:-coffea4bees/metadata/datasets_HH4b_v1p2.yml}"
    ["CONFIG_PATH"]="coffea4bees/analysis/metadata/HH4b.yml"
    ["TRIGGERS_PATH"]="coffea4bees/metadata/triggers_HH4b.yml"
    ["LUMINOSITIES_PATH"]="coffea4bees/metadata/luminosities_HH4b.yml"
    ["DATASETS"]="TTToSemiLeptonic"
    ["YEAR"]="UL18"
    ["OUTPUT_FILENAME"]="test.coffea"
    ["TEST_MODE"]="-t"
    ["OUTPUT_SUBDIR"]="analysis_test"
    ["ADDITIONAL_FLAGS"]=""
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
        --additional-flags)
            ADDITIONAL_FLAGS="$2"
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

# Display configuration
display_config

OUTPUT_DIR="${OUTPUT_BASE}/${OUTPUT_SUBDIR}/"
create_output_directory "$OUTPUT_DIR"

display_section_header "Running test processor"
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
    $TEST_MODE 
    $ADDITIONAL_FLAGS
)
run_command "${cmd[@]}"

display_section_header "Output files"
ls -R $OUTPUT_DIR