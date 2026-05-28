config = {
    "output_path": "output/lowpt/classifier_inputs",
    "dataset_location": "coffea4bees/metadata/datasets_HH4b_Run2/2024_v2/",
    "samples": [
        "GluGluToHHTo4B_cHHH1",
        "TTTo2L2Nu",
        "TTToHadronic",
        "TTToSemiLeptonic",
    ],
}

rule all:
    input:
        [f"{config['output_path']}/classifier_input_lowpt_{sample}.json" for sample in config["samples"]],
        expand(f"{config['output_path']}/classifier_input_lowpt-data{{era}}.json", era=["A", "B", "C", "D"])

rule modify_config_file:
    input:
        config_file = "coffea4bees/analysis/metadata/HH4b_lowpt_classifier_inputs.yml"
    output:
        f"{config['output_path']}/HH4b_lowpt_classifier_inputs.yml"
    shell:
        """
        sed -e 's|make_classifier_input: .*|make_classifier_input: root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/|' {input.config_file} > {output}
        """

rule make_classifier_friendtree:
    input:
        script = "coffea4bees/analysis/processors/processor_HH4b_lowpt.py",
        config_file = f"{config['output_path']}/HH4b_lowpt_classifier_inputs.yml"
    output:
        f"{config['output_path']}/classifier_input_lowpt_{{sample}}.json"
    params:
        dataset_location = config['dataset_location'],
        output_base = config['output_path'],
    log:
        f"{config['output_path']}/logs/make_classifier_input_lowpt_{{sample}}.log"
    shell:
        """
        ./run_container bash coffea4bees/scripts/run-analysis-processor.sh \
            --processor {input.script} \
            --config {input.config_file} \
            --dataset-metadata {params.dataset_location} \
            --year UL18 \
            --datasets {wildcards.sample} \
            --output-base {params.output_base} \
            --output-filename classifier_input_lowpt_{wildcards.sample}.coffea \
            --condor \
            --no-test \
            2>&1 | tee -a {log}
        """

rule make_classifier_friendtree_data:
    input:
        script = "coffea4bees/analysis/processors/processor_HH4b_lowpt.py",
        config_file = f"{config['output_path']}/HH4b_lowpt_classifier_inputs.yml"
    output:
        f"{config['output_path']}/classifier_input_lowpt-data{{era}}.json"
    params:
        dataset_location = config['dataset_location'],
        output_base = config['output_path'],
    log:
        f"{config['output_path']}/logs/make_classifier_input_lowpt-data{{era}}.log"
    shell:
        """
        ./run_container bash coffea4bees/scripts/run-analysis-processor.sh \
            --processor {input.script} \
            --config {input.config_file} \
            --dataset-metadata {params.dataset_location} \
            --year UL18 \
            --datasets data \
            --output-base {params.output_base} \
            --output-filename classifier_input_lowpt-data{wildcards.era}.coffea \
            --condor \
            --no-test \
            --additional-flags --eras {wildcards.era} \
            2>&1 | tee -a {log}
        """