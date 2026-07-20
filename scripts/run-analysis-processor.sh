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
  --year "YEAR1 ..."              Space-separated analysis years
  --output-filename FILE         Output filename
  --output-subdir DIR            Output subdirectory
  --friends PATH                 Path to friends metadata file (default: coffea4bees/metadata/friends/friends_HH4b.yml)
  --additional-flags FLAGS       Additional flags for runner.py
  --no-test                      Disable test mode
  --condor                       Enable condor mode
  --blind                        Enable blinding (set blind: true in config)
  --run-performance              Enable memory profiling with mprof
  --log FILE                     Log file path (output is tee'd to this file)
  --not-do-proxy                 Skip grid proxy setup
  --dashboard-address PORT       Dask dashboard port (default: not set, runner.py uses 10200)
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
    echo "Friends:            $FRIENDS_PATH"
    echo "Datasets:           $DATASETS"
    echo "Year:               $YEAR"
    echo "Output filename:    ${OUTPUT_BASE}/${OUTPUT_SUBDIR}/${OUTPUT_FILENAME}"
    echo "Test mode:          $([ -n "$TEST_MODE" ] && echo "enabled" || echo "disabled")"
    echo "Output subdir:      $OUTPUT_SUBDIR"
    echo "Condor mode:        $([ -n "$CONDOR_MODE" ] && echo "enabled" || echo "disabled")"
    echo "Blind mode:         $([ "$BLIND_MODE" = true ] && echo "enabled" || echo "disabled")"
    echo "Performance:        $([ "$RUN_PERFORMANCE" = true ] && echo "enabled" || echo "disabled")"
    echo "Log file:           ${LOG_FILE:-"(none)"}"
    echo "Dashboard address:  ${DASHBOARD_ADDRESS:-"(default: 10200)"}"
    echo "Additional flags:   ${ADDITIONAL_FLAGS:-"(none)"}"
    echo ""
}

# Default values
declare -A DEFAULTS=(
    ["OUTPUT_BASE"]="output/"
    ["PROCESSOR_PATH"]="coffea4bees/analysis/processors/processor_HH4b.py"
    ["METADATA_PATH"]="${DATASET:-coffea4bees/metadata/datasets/}"
    ["CONFIG_PATH"]="coffea4bees/analysis/metadata/HH4b.yml"
    ["TRIGGERS_PATH"]="coffea4bees/metadata/triggers_HH4b.yml"
    ["LUMINOSITIES_PATH"]="coffea4bees/metadata/luminosities_HH4b.yml"
    ["FRIENDS_PATH"]="coffea4bees/metadata/friends/friends_HH4b.yml"
    ["WEIGHTS_PATH"]="coffea4bees/metadata/weights/weights_HH4b.yml"
    ["DATASETS"]="TTToSemiLeptonic"
    ["YEAR"]="UL18"
    ["OUTPUT_FILENAME"]="test.coffea"
    ["TEST_MODE"]="-t"
    ["OUTPUT_SUBDIR"]=""
    ["ADDITIONAL_FLAGS"]=""
    ["CONDOR_MODE"]=""
    ["BLIND_MODE"]=false
    ["RUN_PERFORMANCE"]=false
    ["LOG_FILE"]=""
    ["DASHBOARD_ADDRESS"]=""
)

# Initialize variables with defaults

OUTPUT_BASE="${DEFAULTS[OUTPUT_BASE]}"
PROCESSOR_PATH="${DEFAULTS[PROCESSOR_PATH]}"
METADATA_PATH="${DEFAULTS[METADATA_PATH]}"
CONFIG_PATH="${DEFAULTS[CONFIG_PATH]}"
TRIGGERS_PATH="${DEFAULTS[TRIGGERS_PATH]}"
LUMINOSITIES_PATH="${DEFAULTS[LUMINOSITIES_PATH]}"
FRIENDS_PATH="${DEFAULTS[FRIENDS_PATH]}"
WEIGHTS_PATH="${DEFAULTS[WEIGHTS_PATH]}"
DATASETS="${DEFAULTS[DATASETS]}"
YEAR="${DEFAULTS[YEAR]}"
OUTPUT_FILENAME="${DEFAULTS[OUTPUT_FILENAME]}"
TEST_MODE="${DEFAULTS[TEST_MODE]}"
OUTPUT_SUBDIR="${DEFAULTS[OUTPUT_SUBDIR]}"
ADDITIONAL_FLAGS="${DEFAULTS[ADDITIONAL_FLAGS]}"
CONDOR_MODE="${DEFAULTS[CONDOR_MODE]}"
BLIND_MODE="${DEFAULTS[BLIND_MODE]}"
RUN_PERFORMANCE="${DEFAULTS[RUN_PERFORMANCE]}"
LOG_FILE="${DEFAULTS[LOG_FILE]}"
DASHBOARD_ADDRESS="${DEFAULTS[DASHBOARD_ADDRESS]}"
DO_PROXY=true

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
        --friends)
            FRIENDS_PATH="$2"
            shift 2
            ;;
        --weights)
            WEIGHTS_PATH="$2"
            shift 2
            ;;
        --datasets)
            DATASETS=""
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DATASETS+="$1 "
                shift
            done
            DATASETS="${DATASETS%% }" # Remove trailing space
            ;;
        --year)
            YEAR=""
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                YEAR+="$1 "
                shift
            done
            YEAR="${YEAR%% }" # Remove trailing space
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
        --blind)
            BLIND_MODE=true
            shift
            ;;
        --run-performance)
            RUN_PERFORMANCE=true
            shift
            ;;
        --log)
            LOG_FILE="$2"
            shift 2
            ;;
        --not-do-proxy)
            DO_PROXY=false
            shift
            ;;
        --dashboard-address)
            DASHBOARD_ADDRESS="$2"
            shift 2
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
    ["FRIENDS_PATH"]="$FRIENDS_PATH"
    ["WEIGHTS_PATH"]="$WEIGHTS_PATH"
    ["DATASETS"]="$DATASETS"
    ["YEAR"]="$YEAR"
    ["OUTPUT_FILENAME"]="$OUTPUT_FILENAME"
    ["TEST_MODE"]="$TEST_MODE"
    ["DO_PROXY"]="$DO_PROXY"
    ["OUTPUT_SUBDIR"]="$OUTPUT_SUBDIR"
    ["ADDITIONAL_FLAGS"]="$ADDITIONAL_FLAGS"
    ["CONDOR_MODE"]="$CONDOR_MODE"
    ["BLIND_MODE"]="$BLIND_MODE"
    ["RUN_PERFORMANCE"]="$RUN_PERFORMANCE"
    ["LOG_FILE"]="$LOG_FILE"
    ["DASHBOARD_ADDRESS"]="$DASHBOARD_ADDRESS"
)

# Restore our configuration variables after setup
OUTPUT_BASE="${SAVED_VARS[OUTPUT_BASE]}"
PROCESSOR_PATH="${SAVED_VARS[PROCESSOR_PATH]}"
METADATA_PATH="${SAVED_VARS[METADATA_PATH]}"
CONFIG_PATH="${SAVED_VARS[CONFIG_PATH]}"
TRIGGERS_PATH="${SAVED_VARS[TRIGGERS_PATH]}"
LUMINOSITIES_PATH="${SAVED_VARS[LUMINOSITIES_PATH]}"
FRIENDS_PATH="${SAVED_VARS[FRIENDS_PATH]}"
WEIGHTS_PATH="${SAVED_VARS[WEIGHTS_PATH]}"
DATASETS="${SAVED_VARS[DATASETS]}"
YEAR="${SAVED_VARS[YEAR]}"
OUTPUT_FILENAME="${SAVED_VARS[OUTPUT_FILENAME]}"
TEST_MODE="${SAVED_VARS[TEST_MODE]}"
DO_PROXY="${SAVED_VARS[DO_PROXY]}"
OUTPUT_SUBDIR="${SAVED_VARS[OUTPUT_SUBDIR]}"
ADDITIONAL_FLAGS="${SAVED_VARS[ADDITIONAL_FLAGS]}"
CONDOR_MODE="${SAVED_VARS[CONDOR_MODE]}"
BLIND_MODE="${SAVED_VARS[BLIND_MODE]}"
RUN_PERFORMANCE="${SAVED_VARS[RUN_PERFORMANCE]}"
LOG_FILE="${SAVED_VARS[LOG_FILE]}"
DASHBOARD_ADDRESS="${SAVED_VARS[DASHBOARD_ADDRESS]}"

# Ensure log directory exists before any tee calls
if [ -n "$LOG_FILE" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
fi

# Display configuration
display_config

if [ -n "$OUTPUT_SUBDIR" ]; then
    OUTPUT_DIR="${OUTPUT_BASE}/${OUTPUT_SUBDIR}/"
else
    OUTPUT_DIR="${OUTPUT_BASE}/"
fi
create_output_directory "$OUTPUT_DIR"

# Setup logging helper
log_exec() {
    if [ -n "$LOG_FILE" ]; then
        mkdir -p "$(dirname "$LOG_FILE")"
        "$@" 2>&1 | tee -a "$LOG_FILE"
        return "${PIPESTATUS[0]}"
    else
        "$@"
    fi
}

log_msg() {
    if [ -n "$LOG_FILE" ]; then
        echo "$@" 2>&1 | tee -a "$LOG_FILE"
    else
        echo "$@"
    fi
}

# Setup proxy if needed
if [ "$DO_PROXY" = true ]; then
    log_msg "Setting up proxy"
    setup_proxy
else
    log_msg "Proxy setup skipped (--not-do-proxy)"
fi

# Handle blinding: patch config to set blind: true
EFFECTIVE_CONFIG="$CONFIG_PATH"
if [ "$BLIND_MODE" = true ]; then
    log_msg "Blinding SR region"
    blind_tmp="${OUTPUT_DIR}/config_blind_$(basename "$OUTPUT_FILENAME" .coffea).yml"
    mkdir -p "$(dirname "$blind_tmp")"
    sed 's/blind.*/blind: true/' "$CONFIG_PATH" > "$blind_tmp"
    EFFECTIVE_CONFIG="$blind_tmp"
fi

# Cap NumExpr threads to avoid oversubscribing shared nodes.
# Use SLURM_CPUS_PER_TASK if available (set by Slurm/REANA), otherwise default to 4.
NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_MAX_THREADS
log_msg "NUMEXPR_MAX_THREADS set to $NUMEXPR_MAX_THREADS"

# Prevent concurrent jobs sharing the same NFS workspace from racing to write
# __pycache__ files, which can produce corrupt .pyc reads and ImportErrors.
export PYTHONDONTWRITEBYTECODE=1

log_msg "Running with config file: $EFFECTIVE_CONFIG"
log_msg "Running $DATASETS $YEAR - output ${OUTPUT_DIR}${OUTPUT_FILENAME}"

display_section_header "Running analysis processor"
# Build command with proper handling of multi-word flags
cmd=(python runner.py
    -p "$PROCESSOR_PATH"
    -m "$METADATA_PATH"
    -c "$EFFECTIVE_CONFIG"
    --triggers "$TRIGGERS_PATH"
    --luminosities "$LUMINOSITIES_PATH"
    --friends "$FRIENDS_PATH"
    --weights "$WEIGHTS_PATH"
    -d $DATASETS
    -y $YEAR
    -op "$OUTPUT_DIR"
    -o "$OUTPUT_FILENAME"
)
[ -n "$DASHBOARD_ADDRESS" ] && cmd+=( --dashboard-address "$DASHBOARD_ADDRESS" )

# Add optional flags
[ -n "$TEST_MODE" ] && cmd+=( $TEST_MODE )
[ -n "$CONDOR_MODE" ] && cmd+=( $CONDOR_MODE )
[ -n "$ADDITIONAL_FLAGS" ] && cmd+=( $ADDITIONAL_FLAGS )

# Wrap with mprof if performance monitoring is enabled
USERNAME=$(whoami 2>/dev/null || echo "barista")
if [ "$RUN_PERFORMANCE" = true ]; then
    mprofile_dat="/tmp/${USERNAME}/mprofile_$(basename "$OUTPUT_FILENAME" .coffea).dat"
    mkdir -p "$(dirname "$mprofile_dat")"
    cmd=(mprof run -C -o "$mprofile_dat" "${cmd[@]}")
fi

log_msg "Command: ${cmd[*]}"

if [ -n "$LOG_FILE" ]; then
    "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
    CMD_EXIT="${PIPESTATUS[0]}"
else
    "${cmd[@]}"
    CMD_EXIT=$?
fi

if [ $CMD_EXIT -ne 0 ]; then
    exit 1
fi

# Generate performance plot if requested
if [ "$RUN_PERFORMANCE" = true ]; then
    log_msg "Generating performance plot"
    mprofile_png="${OUTPUT_DIR}/performance/mprofile_$(basename "$OUTPUT_FILENAME" .coffea).png"
    mkdir -p "$(dirname "$mprofile_png")"
    if [ -n "$LOG_FILE" ]; then
        mprof plot -o "$mprofile_png" "$mprofile_dat" 2>&1 | tee -a "$LOG_FILE"
    else
        mprof plot -o "$mprofile_png" "$mprofile_dat"
    fi
fi

display_section_header "Output files"
EXPECTED_OUTPUT="${OUTPUT_DIR}${OUTPUT_FILENAME}"
if [ -f "$EXPECTED_OUTPUT" ]; then
    log_msg "Output file created successfully: $EXPECTED_OUTPUT ($(du -h "$EXPECTED_OUTPUT" | cut -f1))"
    # Flush NFS writes so Snakemake controller can see the file immediately
    sync
    sleep 60

else
    log_msg "ERROR: Expected output file not found: $EXPECTED_OUTPUT"
    exit 1
fi