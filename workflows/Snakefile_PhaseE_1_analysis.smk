# coffea4bees/workflows/Snakefile_PhaseE_1_analysis.smk
# Phase E_1: Mixed / Synthetic Data Analysis Processing with SvB ML Inference

import os
import glob

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb.yml"

include: "helpers/common.smk"

phase_e_cfg = resolve_config_section(config, primary_key='phase_e', fallback_keys=['phaseE', 'closure'])
for k, v in phase_e_cfg.items():
    config.setdefault(k, v)

config.setdefault('label', "ttHbb_mixeddata")
config.setdefault('output_path', "output/ttHbb/closure_studies/")
config.setdefault('phase_e_dataset', ["mixeddata_4b"])
config.setdefault('years', "UL16_preVFP UL16_postVFP UL17 UL18")
config.setdefault('n_samples', 15)
config.setdefault('additional_parameters', "--shared-dask --condor --run-performance")
config.setdefault('metadata', "coffea4bees/analysis/metadata/candidates_selection_thresholds_ttHbb.yml")
config.setdefault('datasets_file', "coffea4bees/metadata/datasets/mixeddata_4b.yml")
config.setdefault('processor', "coffea4bees/analysis/processors/processor_ttHbb.py")
config.setdefault('weights_file', "coffea4bees/metadata/weights/weights_ttHbb.yml")
config.setdefault('friend_file', "coffea4bees/metadata/friends/friends_ttHbb.yml")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets/")

config.setdefault('include_real_data_from_phase_f', True)
config.setdefault('phase_f_output_path', config.get('output_path', "output/ttHbb/"))
config.setdefault('phase_f_label', "ttHbb")

# Phase E datasets (only pseudo-data runs processor in Phase E)
PHASE_E_DATASETS = config.get('phase_e_dataset', [config.get('data_type', 'mixeddata')])
years_val = config.get('years', '2016 2017 2018')
YEARS = [str(y) for y in (years_val.split() if isinstance(years_val, str) else years_val)]
N_SAMPLES = int(config.get('n_samples', 15))
SAMPLES = [f"v{i}" for i in range(N_SAMPLES)]

# Phase F singlefiles source
config.setdefault('phase_f_label', "ttHbb_v2")

container_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container"
config.setdefault('container_wrapper', container_wrapper)
config.setdefault('analysis_container_wrapper', config.get('container_wrapper', container_wrapper))

python_bin = os.getenv("CONTAINER_PYTHON", "python")
config.setdefault('python_bin', python_bin)

wildcard_constraints:
    dataset = "|".join(PHASE_E_DATASETS),
    year = "|".join(YEARS),
    sample = "v[0-9]+"

def get_phase_e_singlefiles_for_dataset(wildcards):
    return [
        f"{config['output_path']}singlefiles/histAll_{config['label']}__{wildcards.dataset}__{y}__{s}.coffea"
        for y in YEARS for s in SAMPLES
    ]

def get_phase_e_merge_inputs(wildcards):
    files = []
    # 1. Phase E pseudo-data singlefiles (all years and samples)
    for d in PHASE_E_DATASETS:
        for y in YEARS:
            for s in SAMPLES:
                files.append(f"{config['output_path']}singlefiles/histAll_{config['label']}__{d}__{y}__{s}.coffea")
    
    # 2. Phase F singlefiles (signals and background MC, and real data if requested)
    phase_f_dir = config['phase_f_output_path'].rstrip('/') + '/singlefiles/'
    phase_f_label = config['phase_f_label']
    
    # Try finding existing Phase F singlefiles matching pattern
    existing_phase_f = glob.glob(f"{phase_f_dir}histAll_{phase_f_label}*.coffea")
    if existing_phase_f:
        valid_phase_f = [f for f in existing_phase_f if not f.endswith('.png') and not f.endswith('.dat')]
        files.extend(valid_phase_f)
    elif os.path.exists(f"{config['phase_f_output_path'].rstrip('/')}/histAll_{phase_f_label}.coffea"):
        files.append(f"{config['phase_f_output_path'].rstrip('/')}/histAll_{phase_f_label}.coffea")
    return files

original_config = workflow.configfiles[0] if workflow.configfiles else "coffea4bees/workflows/config/analysis_ttHbb.yml"

rule all_PhaseE_1:
    input:
        f"{config['output_path']}histAll_{config['label']}.coffea",
        [f"{config['output_path']}singlefiles/histAll_{config['label']}__{d}.coffea" for d in PHASE_E_DATASETS],
        f"{config['output_path']}cutflow_validation_{config['label']}.txt",
        f"{config['output_path']}cutflow_{config['label']}.yml"

rule analysis_processor_phase_e:
    input:
        runner_script = "runner.py",
        config_file = lambda wildcards: workflow.configfiles[0] if workflow.configfiles else "coffea4bees/workflows/config/analysis_ttHbb.yml"
    output:
        f"{config['output_path']}singlefiles/histAll_{config['label']}__{{dataset}}__{{year}}__{{sample}}.coffea"
    params:
        config = lambda wildcards, input: input.config_file,
        sample_idx = lambda wildcards: wildcards.sample.lstrip("v"),
        container_wrapper = config["analysis_container_wrapper"],
        python_bin = config["python_bin"],
        additional_params = config.get("additional_parameters", "")
    log:
        f"{config['output_path']}logs/analysis_processor_{config['label']}__{{dataset}}__{{year}}__{{sample}}.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} runner.py {params.config} \
            --datasets {wildcards.dataset} \
            --years {wildcards.year} \
            --samples {params.sample_idx} \
            -op $(dirname {output})/ \
            --output $(basename {output}) \
            {params.additional_params} 2>&1 | tee {log}
        """

rule merge_dataset_phase_e:
    input:
        files = get_phase_e_singlefiles_for_dataset
    output:
        f"{config['output_path']}singlefiles/histAll_{config['label']}__{{dataset}}.coffea"
    params:
        container_wrapper = config["analysis_container_wrapper"],
        python_bin = config["python_bin"],
        input_files = lambda wildcards, input: " ".join([f for f in input.files if not f.endswith('.py')])
    log:
        f"{config['output_path']}logs/merge_dataset_{config['label']}__{{dataset}}.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} src/tools/merge_coffea_files.py \
            -f {params.input_files} \
            -o {output} 2>&1 | tee {log}
        """

rule merging_files_phase_e:
    input:
        files = get_phase_e_merge_inputs
    output:
        f"{config['output_path']}histAll_{config['label']}.coffea"
    params:
        container_wrapper = config["analysis_container_wrapper"],
        python_bin = config["python_bin"],
        input_files = lambda wildcards, input: " ".join([f for f in input.files if not f.endswith('.py')])
    log:
        f"{config['output_path']}logs/merging_files_{config['label']}.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} src/tools/merge_coffea_files.py \
            -f {params.input_files} \
            -o {output} 2>&1 | tee {log}
        """

rule check_cutflow_phase_e:
    input:
        f"{config['output_path']}histAll_{config['label']}.coffea"
    output:
        txt = f"{config['output_path']}cutflow_validation_{config['label']}.txt",
        yml = f"{config['output_path']}cutflow_{config['label']}.yml"
    params:
        container_wrapper = config["analysis_container_wrapper"],
        python_bin = config["python_bin"]
    log:
        f"{config['output_path']}logs/check_cutflow_{config['label']}.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.txt}) $(dirname {log})
        {params.container_wrapper} bash coffea4bees/scripts/run-cutflow.sh \
            --input-file "{input}" \
            --output-file "{output.yml}" \
            --known-cutflow "none" \
            --python-bin "{params.python_bin}" 2>&1 | tee {log}
        touch {output.txt}
        """

localrules: all_PhaseE_1, merge_dataset_phase_e, merging_files_phase_e, check_cutflow_phase_e
