config.setdefault('output_path', "output/trigger_weights/")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets/")
config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset', ['TTToSemiLeptonic', 'TTTo2L2Nu', 'TTToHadronic'])
config.setdefault('year', [ '2024' ])
config.setdefault('eos_path', f'root://cmseos.fnal.gov//store/user/algomez/XX4b/2025_v3/')

# Import analysis module
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

rule final_outputs:
    input:
        f"{config['output_path']}trigger_weights_friends.json"

rule create_config:
    output: f"{config['output_path']}trigger_weights_config.yml"
    params:
        eos_path = config['eos_path'],
        dataset_location = config['dataset_location']
    shell:
        """
        echo "Creating config file for trigger weights analysis"
        cat > {output} <<EOL
processor: "coffea4bees/analysis/processors/processor_trigger_weights.py"
dataset_location: "{params.dataset_location}"

runner:
  condor_cores: 2
  worker_memory: 6GB
  write_coffea_output: false
  dashboard_address: 0

config:
  use_vectorized: true
  tagger: PNet
  make_classifier_input: {params.eos_path}
EOL
        """

use rule analysis_processor from analysis as analysis_trigger_weights with:
    input: f"{config['output_path']}trigger_weights_config.yml"
    output: f"{config['output_path']}trigger_weights__{{dataset}}__{{year}}.json"
    log: f"{config['output_path']}logs/analysis_trigger_weights__{{dataset}}__{{year}}.log"
    # container: ""
    params:
        datasets = config['dataset'],
        years = config['year'],
        config = lambda wildcards, input: input[0],
        run_container_wrapper = "./run_container"

rule merge_friendtree_json:
    input: expand(f"{config['output_path']}trigger_weights__{{dataset}}__{{year}}.json", dataset=config['dataset'], year=config['year'])
    output: f"{config['output_path']}trigger_weights_friends.json"
    params:
        output_path = config['output_path'],
    log: f"{config['output_path']}logs/merge_friendtree_json.log"
    shell:
        """
        echo "Merging all the trigger weights friendtree json files"
        ./run_container python -m src.friendtrees.merge_friend_meta \
            -i {input} \
            -o {output}
        """