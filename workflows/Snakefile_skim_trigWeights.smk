config.setdefault('output_path', "output/ttHbb/")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/")
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
    input:
        config_file = "coffea4bees/skimmer/metadata/HH4b.yml",
        processor_script = "coffea4bees/skimmer/processor/skimmer_4b.py"
    output: f"{config['output_path']}skimmer/picoaod_dataset_{{dataset}}__{{year}}.yml"
    params:
        output_path = config['output_path'],
        dataset_location = config['dataset_location'],
        condor_mode = "--condor"
    log: f"{config['output_path']}logs/skimmer_dataset_{{dataset}}__{{year}}.log"
    shell:
        """
        ./run_container python runner.py {input.config_file} \
            --processor {input.processor_script} \
            -s \
            --output "picoaod_dataset_{wildcards.dataset}__{wildcards.year}.yml" \
            --output-path "{params.output_path}/skimmer/" \
            --datasets "{wildcards.dataset}" \
            --years "{wildcards.year}" \
            --metadata "{params.dataset_location}" \
            {params.condor_mode} > {log} 2>&1
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
    input:
        modified_datasets = f"{config['output_path']}modified_datasets/modified_datasets.yml",
        config_file = "coffea4bees/analysis/metadata/trigger_weights.yml",
        processor_script = "coffea4bees/analysis/processors/processor_trigger_weights.py"
    output: f"{config['output_path']}trigger_weights/trigger_weights_friends_{{dataset}}__{{year}}.json"
    params:
        output_path = config['output_path'],
        condor_mode = "--condor"
    log: f"{config['output_path']}logs/trigger_weights_{{dataset}}__{{year}}.log"
    shell:
        """
        ./run_container python runner.py {input.config_file} \
            --processor {input.processor_script} \
            --output-path "{params.output_path}/trigger_weights/" \
            --datasets {wildcards.dataset} \
            --metadata {input.modified_datasets} \
            --years "{wildcards.year}" \
            --output "trigger_weights_friends_{wildcards.dataset}__{wildcards.year}.json" \
            {params.condor_mode} > {log} 2>&1
        """

rule merge_friendtree_json:
    input: expand( f"{config['output_path']}trigger_weights/trigger_weights_friends_{{dataset}}__{{year}}.json", dataset=config['dataset'], year=config['year'])
    output: f"{config['output_path']}trigger_weights/trigger_weights_friends_allDatasets.json"
    params:
        friendtree_file = "coffea4bees/metadata/friends/trigweights_2024_v1p2.json",
        output_path = config['output_path'],
    log: f"{config['output_path']}logs/merge_friendtree_json.log"
    shell:
        """
        echo "Merging all the trigger weights friendtree json files"
        ./run_container python -m src.friendtrees.merge_friend_meta \
            -i {params.friendtree_file} {input} \
            -o {output}
        """