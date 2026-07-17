#!/bin/bash
# Performance profiler for runner.py + processor_HH4b.py
#
# Produces per-stage timing and memory (RSS) snapshots.
# Run BEFORE and AFTER fixes to compare.
#
# Usage (inside container):
#   bash coffea4bees/scripts/tools-perf-profile.sh [--output-base DIR]
#
# Outputs:
#   <DIR>/perf_profile/perf_profile.txt   - Human-readable report
#   <DIR>/perf_profile/perf_profile.csv   - CSV for diffing
#   <DIR>/perf_profile/mprofile.dat       - mprof raw data (if available)
#   <DIR>/perf_profile/mprofile.png       - mprof plot (if available)

source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

setup_proxy

# Create output directory
OUTPUT_DIR="$OUTPUT_BASE_DIR/perf_profile"
create_output_directory "$OUTPUT_DIR"

display_section_header "Performance Profile"
DATASETS="coffea4bees/metadata/datasets/"
echo "Using datasets file: $DATASETS"

# --- Stage 1: Per-stage instrumented run ---
# Uses --debug (iterative executor, single process) so the monkey-patched
# processor runs in the main process and stage timings are captured.
display_section_header "Stage 1: Per-stage profiling"
run_command python src/scripts/memory/perf_profile.py \
    -o "$OUTPUT_DIR/perf_profile" \
    --script runner.py \
        -o test.coffea -t --debug \
        -d GluGluToHHTo4B_cHHH1 \
        -p coffea4bees/analysis/processors/processor_HH4b.py \
        -y UL18 \
        --friends coffea4bees/metadata/friends/friends_HH4b.yml \
        -op "${OUTPUT_DIR}" \
        -m "$DATASETS" \
        -c coffea4bees/analysis/metadata/HH4b_signals.yml \
        --weights coffea4bees/metadata/weights_HH4b.yml

# --- Stage 2: mprof overall memory timeline (optional) ---
display_section_header "Stage 2: mprof memory timeline"
if command -v mprof &> /dev/null; then
    run_command mprof run -C -o "$OUTPUT_DIR/mprofile.dat" python runner.py \
        -o test.coffea -t \
        -d GluGluToHHTo4B_cHHH1 \
        -p coffea4bees/analysis/processors/processor_HH4b.py \
        -y UL18 \
        --friends coffea4bees/metadata/friends/friends_HH4b.yml \
        -op "${OUTPUT_DIR}" \
        -m "$DATASETS" \
        -c coffea4bees/analysis/metadata/HH4b_signals.yml \
        --weights coffea4bees/metadata/weights_HH4b.yml

    mprof plot -o "$OUTPUT_DIR/mprofile.png" "$OUTPUT_DIR/mprofile.dat" 2>/dev/null
    mprof peak "$OUTPUT_DIR/mprofile.dat"
else
    echo "mprof not found, skipping memory timeline. Install with: pip install memory_profiler"
fi

# --- Summary ---
display_section_header "Results"
echo "Per-stage report: $OUTPUT_DIR/perf_profile.txt"
echo "Per-stage CSV:    $OUTPUT_DIR/perf_profile.csv"
if [ -f "$OUTPUT_DIR/mprofile.png" ]; then
    echo "Memory timeline:  $OUTPUT_DIR/mprofile.png"
fi
echo ""
echo "To compare before/after, run this script again after applying fixes"
echo "and diff the CSV files:"
echo "  diff <before>/perf_profile.csv <after>/perf_profile.csv"
