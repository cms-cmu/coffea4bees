# Fallback defaults
config.setdefault('output_path', "output/trigger_weights/")
config.setdefault('dataset', ['ttHbb'])
config.setdefault('config_file', "coffea4bees/analysis/metadata/trigger_weights.yml")

years = list(config['year_eras'].keys()) if 'year_eras' in config and isinstance(config['year_eras'], dict) else config.get('years', config.get('year', ['UL18']))
if isinstance(years, str):
    years = [years]

# Import analysis module
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

rule all_trigger_weights:
    input:
        f"{config['output_path']}trigger_weights_friends.json"

rule create_trigger_weights_config:
    input:
        config_file = config['config_file'],
        processor = "coffea4bees/analysis/processors/processor_trigger_weights.py"
    output: f"{config['output_path']}trigger_weights_config.yml"
    params:
        dataset_location = config.get('dataset_location', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/")
    run:
        import yaml
        with open(input.config_file, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        cfg['processor'] = input.processor
        cfg['dataset_location'] = params.dataset_location
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

use rule analysis_processor from analysis as analysis_trigger_weights with:
    input:
        runner_script = "runner.py",
        config_file = f"{config['output_path']}trigger_weights_config.yml"
    output: f"{config['output_path']}singlefiles/trigger_weights__{{dataset}}__{{year}}.json"
    log: f"{config['output_path']}logs/analysis_trigger_weights__{{dataset}}__{{year}}.log"
    params:
        datasets = lambda wildcards: wildcards.dataset,
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        run_container_wrapper = "./run_container"

rule merge_friendtree_json:
    input: expand(f"{config['output_path']}singlefiles/trigger_weights__{{dataset}}__{{year}}.json", dataset=config['dataset'], year=years)
    output: f"{config['output_path']}trigger_weights_friends.json"
    log: f"{config['output_path']}logs/merge_friendtree_json.log"
    shell:
        """
        ./run_container python -m src.friendtrees.merge_friend_meta \
            -i {input} \
            -o {output}
        """