config.setdefault('output_path', "output/ttHbb/")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets_HH4b_Run2/2024_v2/")
config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset', ['ttHbb'])
# config.setdefault('year', ['UL16_postVFP', 'UL16_preVFP', 'UL17', 'UL18'])
config.setdefault('year', ['UL18'])

rule all:
    input:
        expand( f"{config['output_path']}skimmer/picoaod_dataset_{{dataset}}__{{year}}.yml", dataset=config['dataset'], year=config['year']),
        expand( f"{config['output_path']}trigger_weights/trigger_weights_friends_{{dataset}}__{{year}}.json", dataset=config['dataset'], year=config['year']),
        f"{config['output_path']}trigger_weights/trigger_weights_friends_allDatasets.json"

rule skimms:
    output: f"{config['output_path']}skimmer/picoaod_dataset_{{dataset}}__{{year}}.yml"
    params:
        output_path = config['output_path'],
        config = "coffea4bees/skimmer/metadata/HH4b.yml",
        dataset_location = config['dataset_location'],
        condor_mode = "--condor"
    log: f"{config['output_path']}logs/skimmer_dataset_{{dataset}}__{{year}}.log"
    shell:
        """
        ./run_container bash coffea4bees/scripts/run-analysis-processor.sh \
            --processor coffea4bees/skimmer/processor/skimmer_4b.py \
            --output-filename "picoaod_dataset_{wildcards.dataset}__{wildcards.year}.yml" \
            --output-subdir "skimmer" \
            --output-base {params.output_path} \
            --datasets "{wildcards.dataset}" \
            --year "{wildcards.year}" \
            --dataset-metadata "{params.dataset_location}" \
            --config {params.config} \
            {params.condor_mode} \
            --no-test \
            --additional-flags '-s' 
        """        

rule modify_datasets:
    input: expand( f"{config['output_path']}skimmer/picoaod_dataset_{{dataset}}__{{year}}.yml", dataset=config['dataset'], year=config['year'])
    output: f"{config['output_path']}modified_datasets/modified_datasets.yml"
    params:
        input_dataset = config['dataset_location'],
    log: f"{config['output_path']}logs/modify_datasets.log"
    shell:
        """
        echo "Modifying datasets file to point to skimmer outputs"
        ./run_container python src/tools/merge_yaml_datasets.py \
            -m {params.input_dataset} \
            -f {input} \
            -o {output}
        """

rule trigger_weights:
    input: f"{config['output_path']}modified_datasets/modified_datasets.yml"
    output: f"{config['output_path']}trigger_weights/trigger_weights_friends_{{dataset}}__{{year}}.json"
    params:
        output_path = config['output_path'],
        config = "coffea4bees/analysis/metadata/trigger_weights.yml",
        condor_mode = "--condor"
    log: f"{config['output_path']}logs/trigger_weights_{{dataset}}__{{year}}.log"
    shell:
        """
        ./run_container bash coffea4bees/scripts/run-analysis-processor.sh \
            --processor "coffea4bees/analysis/processors/processor_trigger_weights.py" \
            --output-base {params.output_path} \
            --datasets {wildcards.dataset} \
            --dataset-metadata {input} \
            --year "{wildcards.year}" \
            --output-filename "trigger_weights_friends_{wildcards.dataset}__{wildcards.year}.json" \
            --output-subdir trigger_weights \
            --config {params.config} \
            {params.condor_mode} \
            --no-test
        """

rule merge_friendtree_json:
    input: expand( f"{config['output_path']}trigger_weights/trigger_weights_friends_{{dataset}}__{{year}}.json", dataset=config['dataset'], year=config['year'])
    output: f"{config['output_path']}trigger_weights/trigger_weights_friends_allDatasets.json"
    params:
        friendtree_file = "coffea4bees/metadata/datasets_HH4b_Run2/trigweights_2024_v1p2.json",
        output_path = config['output_path'],
    log: f"{config['output_path']}logs/merge_friendtree_json.log"
    shell:
        """
        echo "Merging all the trigger weights friendtree json files"
        ./run_container python -m src.friendtrees.merge_friend_meta \
            -i {params.friendtree_file} {input} \
            -o {output}
        """