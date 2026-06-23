import os
import yaml

config.setdefault( 'output_path', "output/mixeddata_friendtrees" )
config.setdefault( 'dataset_location',  "coffea4bees/metadata/datasets_HH4b_Run2/" )
config.setdefault( 'samples', ["mixeddata_all", "data"] )
config.setdefault( 'samples_sim', ['GluGluToHHTo4B_cHHH1', 'ZH4b', 'ZZ4b'] )
config.setdefault( 'years', ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18'] )

# Valid (year, era) pairs for data in Run 2 mixeddata
year_era_pairs = [
    # UL16 preVFP
    ("UL16_preVFP", "B"), ("UL16_preVFP", "C"), ("UL16_preVFP", "D"), ("UL16_preVFP", "E"),
    # UL16 postVFP
    ("UL16_postVFP", "F"), ("UL16_postVFP", "G"), ("UL16_postVFP", "H"),
    # UL17
    ("UL17", "C"), ("UL17", "D"), ("UL17", "E"), ("UL17", "F"),
    # UL18
    ("UL18", "A"), ("UL18", "B"), ("UL18", "C"), ("UL18", "D"),
]

wildcard_constraints:
    era="[A-H]",
    year="UL16_preVFP|UL16_postVFP|UL17|UL18"

rule all:
    input:
        # expand(f"{config['output_path']}/all_friends_{{sample}}_{{year}}.json", sample=config["samples"], year=config["years"]),
        # [f"{config['output_path']}/all_friends_{sample}_{year}_{era}.json" for sample in config["samples"] for year, era in year_era_pairs if year in config["years"]],
        f"{config['output_path']}/friends_all.json"


rule make_all_other_friendtrees:
    input:
        script = "coffea4bees/analysis/processors/processor_HH4b.py",
        config_file = "coffea4bees/analysis/metadata/HH4b_make_friend_SvB.yml",
    output:
        f"{config['output_path']}/singlefiles/all_friends_{{sample}}_{{year}}_{{era}}.json"
    params:
        dataset_location = config['dataset_location'],
        output_base = config['output_path'],
    log:
        f"{config['output_path']}/logs/make_all_other_friendtrees_{{sample}}_{{year}}_{{era}}.log"
    shell:
        """
        ./run_container bash coffea4bees/scripts/run-analysis-processor.sh \
            --processor {input.script} \
            --config {input.config_file} \
            --dataset-metadata {params.dataset_location} \
            --year {wildcards.year} \
            --datasets {wildcards.sample} \
            --output-base {params.output_base}/singlefiles \
            --output-filename all_friends_{wildcards.sample}_{wildcards.year}_{wildcards.era}.coffea \
            --condor \
            --no-test \
            --additional-flags --eras {wildcards.era} \
            2>&1 | tee -a {log}
        """

rule make_all_other_friendtrees_mc:
    input:
        script = "coffea4bees/analysis/processors/processor_HH4b.py",
        config_file = "coffea4bees/analysis/metadata/HH4b_make_friend_SvB.yml",
    output:
        f"{config['output_path']}/singlefiles/all_friends_{{sample}}_{{year}}.json"
    params:
        dataset_location = config['dataset_location'],
        output_base = config['output_path'],
    log:
        f"{config['output_path']}/logs/make_all_other_friendtrees_{{sample}}_{{year}}.log"
    shell:
        """
        ./run_container bash coffea4bees/scripts/run-analysis-processor.sh \
            --processor {input.script} \
            --config {input.config_file} \
            --dataset-metadata {params.dataset_location} \
            --year {wildcards.year} \
            --datasets {wildcards.sample} \
            --output-base {params.output_base}/singlefiles \
            --output-filename all_friends_{wildcards.sample}_{wildcards.year}.coffea \
            --condor \
            --no-test \
            2>&1 | tee -a {log}
        """

rule merge_all_friendtrees:
    input:
        [f"{config['output_path']}/singlefiles/all_friends_{sample}_{year}_{era}.json" for sample in config["samples"] for year, era in year_era_pairs if year in config["years"]] + [f"{config['output_path']}/singlefiles/all_friends_{sample}_{year}.json" for sample in config["samples_sim"] for year in config["years"] if year in config["years"]]
    output:
        f"{config['output_path']}/friends_all.json"
    shell:
        """
        ./run_container python -m src.friendtrees.merge_friend_meta \
            -i {input} \
            -o {output}
        """
