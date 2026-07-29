#!/bin/bash
# Run3 hemisphere-mixing accuracy test.
#
# Exercises the top-K kd-tree matcher (use_topk_matching: True) end-to-end on a
# small sample and then runs the analysis over the freshly-mixed picoAOD and
# asserts the cutflow counts against a committed reference. This guards both that
# the matcher runs (smoke) and that it keeps producing the same numbers
# (regression) -- e.g. across the awkward v1->v2 / coffea-2025 port.
#
# A single era (2022_EEG, ~25 files) keeps preprocessing fast. The mixed picoAOD
# drops most jet branches, so the analysis runs as `mixeddata_all` (JEC, lumimask,
# HLT cut and lepton-jet cleaning all auto-disabled) with FvT/SvB off (no friends).
source "src/scripts/common.sh"

OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

DATASETS=${DATASET:-"coffea4bees/metadata/datasets/"}
echo "Using datasets file: $DATASETS"

setup_proxy

OUTPUT_DIR="$OUTPUT_BASE_DIR/test_mixeddata_Run3"
create_output_directory "$OUTPUT_DIR"

display_section_header "Changing metadata"
JOB_CONFIG="$OUTPUT_DIR/mix_config.yml"
sed -e "s|base_path.*|base_path: $OUTPUT_DIR|" \
    -e "s|\#max.*|maxchunks: 1|" \
    -e "s|\#test.*|test_files: 1|" \
    -e "s|workers:.*|workers: 1|" \
    -e "s|chunksize:.*|chunksize: 1000|" \
    -e "s|subtract_ttbar.*|subtract_ttbar_with_weights: True|" \
    coffea4bees/skimmer/metadata/mixeddata_Run3.yml > $JOB_CONFIG
[[ $(hostname) = *runner* ]] && sed -i "s|T3_US_FNALLPC|T3_CH_PSI|" $JOB_CONFIG
cat $JOB_CONFIG; echo

display_section_header "Making mixed data (top-K matcher, era G)"
run_command python runner.py -s \
    -p coffea4bees/skimmer/processor/make_mixed_data.py \
    -c $JOB_CONFIG \
    -y 2022_EE -e G -d data \
    -op $OUTPUT_DIR \
    -o picoaod_datasets_mixeddata_test_2022_EEG.yml \
    -m $DATASETS

MIXED_PICO="$OUTPUT_DIR/data_2022_EEG/picoAOD_mixed_all.root"
ls -l "$MIXED_PICO"

display_section_header "Building analysis input metadata"
# Point the analysis directly at the fresh mixed picoAOD (the install yaml from a
# maxchunks-throttled make is incomplete, so we bypass it). Named `mixeddata_all`
# so the processor treats it as mixed data.
ANA_META="$OUTPUT_DIR/mixed_input.yml"
cat > "$ANA_META" <<EOF
datasets:
  mixeddata_all:
    2022_EE:
      picoAOD:
        G:
          files:
          - $MIXED_PICO
EOF
cat "$ANA_META"; echo

display_section_header "Running analysis over mixed data"
run_command python runner.py -t \
    -o testMixedData_Run3.coffea \
    -d mixeddata_all -y 2022_EE -e G \
    -p coffea4bees/analysis/processors/processor_HH4b.py \
    -m "$ANA_META" \
    -c coffea4bees/analysis/metadata/HH4b_mixed_nottcheck_Run3.yml \
    -op $OUTPUT_DIR

display_section_header "Dumping cutflow"
run_command python coffea4bees/analysis/tests/dumpCutFlow.py \
    --input $OUTPUT_DIR/testMixedData_Run3.coffea \
    -o $OUTPUT_DIR/test_dump_MixedData_Run3.yml
cat $OUTPUT_DIR/test_dump_MixedData_Run3.yml

display_section_header "Cutflow regression test vs known counts"
run_command python coffea4bees/analysis/tests/cutflow_test.py \
    --inputFile $OUTPUT_DIR/testMixedData_Run3.coffea \
    --knownCounts coffea4bees/analysis/tests/known_Counts_MixedData_Run3.yml
