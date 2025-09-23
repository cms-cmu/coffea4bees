#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

# Create output directory
JOB="synthetic_dataset_analyze"
INPUT_DIR="$OUTPUT_BASE_DIR/synthetic_dataset_make_dataset"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Creating metadata"
echo """
datasets:
  synthetic_data:
    nSamples: 1
    UL18:
      picoAOD:
        files_template:
          - ${INPUT_DIR}/data_UL18A/picoAOD_seed5.root
          - ${INPUT_DIR}/data_UL18B/picoAOD_seed5.root
          - ${INPUT_DIR}/data_UL18C/picoAOD_seed5.root
          - ${INPUT_DIR}/data_UL18D/picoAOD_seed5.root

  synthetic_mc_GluGluToHHTo4B_cHHH1:
    nSamples: 1
    UL18:
        picoAOD:
          files:
           - ${INPUT_DIR}/GluGluToHHTo4B_cHHH1_UL18/picoAOD_seed5.root
          sumw: 14442.908792499999
""" > $OUTPUT_DIR/datasets_synthetic_test.yml; echo

display_section_header "Running test processor "
time bash coffea4bees/scripts/run-analysis-processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "synthetic_data synthetic_mc_GluGluToHHTo4B_cHHH1" \
    --year "UL18" \
    --output-filename "test_synthetic_datasets.coffea" \
    --output-subdir "$JOB" \
    --dataset-metadata "$OUTPUT_DIR/datasets_synthetic_test.yml" \
    --config coffea4bees/analysis/metadata/HH4b_synthetic_data.yml \
    --no-test
    # --additional-flags "--debug"

### previous tests
# display_section_header "Modifying dataset file with skimmer ci output"
# cat coffea4bees/metadata/datasets_ci.yml
# python src/tools/merge_yaml_datasets.py -m coffea4bees/metadata/datasets_HH4b.yml -f coffea4bees/skimmer/metadata/picoaod_datasets_declustered_data_test_UL18A.yml  -o coffea4bees/metadata/datasets_synthetic_seed17_test.yml

#/builds/johnda/coffea4bees/coffea4bees/skimmer/GluGluToHHTo4B_cHHH1_UL18/picoAOD_seed5.root
#/builds/johnda/coffea4bees/python
#johnda/coffea4bees
#/builds/coffea4bees/skimmer/GluGluToHHTo4B_cHHH1_UL18/picoAOD_seed5.root


# display_section_header "Changing metadata"
# sed -e "s/apply_FvT.*/apply_FvT: false/" -e "s/apply_trig.*/apply_trigWeight: false/" -e "s/run_SvB.*/run_SvB: false/"  coffea4bees/analysis/metadata/HH4b.yml > $OUTPUT_DIR/tmp.yml
# cat $OUTPUT_DIR/tmp.yml
#display_section_header "Running test processor"
#time python runner.py -o test_synthetic_data_test.coffea -d data -p coffea4bees/analysis/processors/processor_HH4b.py -y UL18  -op $OUTPUT_DIR/ -c $OUTPUT_DIR/HH4b_synthetic_data.yml -m coffea4bees/metadata/datasets_synthetic_seed17.yml


# python src/tools/merge_yaml_datasets.py -m coffea4bees/metadata/datasets_HH4b.yml -f coffea4bees/skimmer/metadata/picoaod_datasets_declustered_test_UL18.yml  -o coffea4bees/metadata/datasets_synthetic_test.yml
# python src/tools/merge_yaml_datasets.py -m coffea4bees/metadata/datasets_synthetic_seed17.yml -f coffea4bees/skimmer/metadata/picoaod_datasets_declustered_GluGluToHHTo4B_cHHH1_Run2_seed17.yml -o coffea4bees/metadata/datasets_synthetic_seed17.yml
#cat coffea4bees/metadata/datasets_synthetic_test.yml


# time python runner.py -o test_synthetic_datasets.coffea -d data GluGluToHHTo4B_cHHH1 -p coffea4bees/analysis/processors/processor_HH4b.py -y UL18  -op $OUTPUT_DIR/ -c coffea4bees/analysis/metadata/HH4b_synthetic_data.yml -m $OUTPUT_DIR/datasets_synthetic_test.yml
# time python runner.py -o test_synthetic_datasets_Run3.coffea -d data  -p coffea4bees/analysis/processors/processor_HH4b.py -y 2022_EE  -op ${OUTPUT_DIR} -c coffea4bees/analysis/metadata/HH4b_synthetic_data.yml -m coffea4bees/metadata/datasets_HH4b_Run3_fourTag.yml


