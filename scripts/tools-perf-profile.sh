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
        --weights coffea4bees/metadata/weights/weights_HH4b.yml

# --- Stage 2: Enforce gating performance & memory budgets ---
display_section_header "Stage 2: Performance & memory budget check"
python - "$OUTPUT_DIR/perf_profile.txt" <<'EOF'
import re, sys

report = open(sys.argv[1]).read()
m_time = re.search(r"Total runtime:\s+([\d.]+)s", report)
m_rss = re.search(r"Peak RSS:\s+([\d.]+)\s+MB", report)
if not m_time or not m_rss:
    print(f"ERROR: Could not parse Total runtime / Peak RSS from {sys.argv[1]}")
    sys.exit(1)

runtime_s = float(m_time.group(1))
peak_mb = float(m_rss.group(1))
max_runtime_s = 120.0
max_rss_mb = 1500.0

print(f"Measured: Total runtime = {runtime_s:.1f}s (budget <= {max_runtime_s:.1f}s), Peak RSS = {peak_mb:.0f} MB (budget <= {max_rss_mb:.0f} MB)")
if runtime_s > max_runtime_s:
    print(f"FAIL: Total runtime {runtime_s:.1f}s exceeds budget {max_runtime_s:.1f}s")
    sys.exit(1)
if peak_mb > max_rss_mb:
    print(f"FAIL: Peak RSS {peak_mb:.0f} MB exceeds budget {max_rss_mb:.0f} MB")
    sys.exit(1)
print("PASS: Runtime and Peak RSS are within budget.")
EOF

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
