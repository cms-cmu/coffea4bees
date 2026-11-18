# ==============================================================================
# coffea4bees/workflows/Snakefile_full_syst_1_analysis.smk
#
# Stage F.1: Signal Systematic Variations & Cross-Check Quality Gate
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Processes full experimental and theoretical systematic variations for signal MC
# processes across all data-taking eras, verifies physics integrity against upstream
# nominal baselines, and generates systematic cutflow validation reports.
#
# KEY PIPELINE STAGES & RULES:
#   1. Configuration Generation (`create_analysis_config_full_syst`):
#      Constructs a resolved runtime configuration dynamically based on roast parameters,
#      injecting runner options, dataset mappings, and processor overrides.
#
#   2. Parallel Systematic Singlefiles (`analysis_signal_systematics`):
#      Executes Coffea processor jobs partitioned by (dataset, year, systematic group):
#        - `others`: Friend-tree weight variations (e.g. pileup, b-tagging SFs,
#          PDF, renormalization/factorization scales, trigger efficiencies).
#        - `jes`: Jet energy scale & resolution kinematic shifts evaluated through
#          dedicated PyTorch classifier models on 4b Signal Region events.
#      Distributes parallel Condor batch jobs via Dask on cmslpc.
#
#   3. Merging Singlefiles (`merge_signal_systematics`):
#      Aggregates all per-era and per-variation Coffea singlefiles into a single unified
#      signal systematics histogram file: `histAll_<label>_signals_syst.coffea`.
#
#   4. Nominal Signal Distribution Match Gate (`check_nominal_signal_match`):
#      Integrity validation rule: compares the nominal signal distribution inside the
#      full systematics output with the upstream nominal/stat-only reference.
#      Strict check across key kinematic variables (e.g. SvB_MA, v4j.mass, selJets.n)
#      in both SR and SB regions. If relative difference exceeds tolerance (1e-4),
#      the rule fails with a non-zero exit code to halt the workflow immediately.
#
#   5. Systematic Cutflow Validation (`cutflow_systematics_table`):
#      Extracts event yields across all nominal, up, and down systematic variations in
#      the Signal Region, compares them against canonical reference counts
#      (`known_Counts_systematics_<sig>.yml`), and generates:
#        - `cutflow_validation_systematics.txt`: Raw diagnostic report
#        - `cutflow_systematics.html`: Formatted interactive web table
#        - `cutflow_systematics.yml`: Machine-readable output counts
#
# EXECUTION TARGETS:
#   - `all_full_syst_singlefiles`: Build all per-dataset/year/group singlefiles
#   - `all_full_syst_F_1`: Master target building merged output, nominal match
#     cross-check, and systematic cutflow reports
# ==============================================================================
import os
import shutil
import yaml

config.setdefault('label', "ttHbb_full_syst")
config.setdefault('output_path', f"output/{config['label']}/")
base_output = config.get('output_path', f"output/{config['label']}/")
if not base_output.endswith('/'):
    base_output += '/'
if base_output.rstrip('/').endswith('PhaseF_1_analysis'):
    F1_OUTPUT_PATH = base_output
else:
    F1_OUTPUT_PATH = os.path.join(base_output, 'PhaseF_1_analysis/')
if not F1_OUTPUT_PATH.endswith('/'):
    F1_OUTPUT_PATH += '/'

config.setdefault('analysis_config', "coffea4bees/analysis/metadata/ttHbb_2024_v2.yml")
config.setdefault('processor', "coffea4bees/analysis/processors/processor_HH4b.py")
config.setdefault('friend_file', "coffea4bees/metadata/friends/friends_ttHbb.yml")
config.setdefault('weights_file', "coffea4bees/metadata/weights/weights_ttHbb.yml")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets/")
config.setdefault('test', False)

# Signal datasets and systematic groups
config.setdefault('signal_datasets', [k for k in config.get('channels', {}).keys()] if config.get('channels') else ['ttHbb'])
config.setdefault('syst_groups', ['others', 'jes'])
config.setdefault('year_eras', {
    'UL16_preVFP':  ['C', 'D', 'E', 'F'],
    'UL16_postVFP': ['F', 'G', 'H'],
    'UL17':         ['C', 'D', 'E', 'F'],
    'UL18':         ['A', 'B', 'C', 'D'],
})

is_test = config.get('test', False)
if isinstance(is_test, str):
    is_test = is_test.lower() in ("true", "1", "yes")
config['test'] = is_test

container_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container"
config.setdefault('container_wrapper', container_wrapper)
config.setdefault('analysis_container_wrapper', config.get('container_wrapper', container_wrapper))
config.setdefault('python_bin', os.getenv("CONTAINER_PYTHON", "python3"))

if config.get('additional_parameters') is not None:
    pass
elif config.get('test', False) or os.getenv("CI"):
    config.setdefault('additional_parameters', "")
elif "bridges2" in os.uname().nodename or "psc" in os.uname().nodename or "/ocean/" in os.getcwd():
    config.setdefault('additional_parameters', "")
else:
    config.setdefault('additional_parameters', "--shared-dask --condor --run-performance")

SIGNAL_DATASETS = config['signal_datasets']
SYST_GROUPS = config['syst_groups']
YEARS = [str(y) for y in config['year_eras'].keys()]

SIGNAL_SINGLE_FILES = expand(
    "{output_path}singlefiles/histAll_" + config['label'] + "__{dataset}__{year}__{syst_group}.coffea",
    output_path=F1_OUTPUT_PATH,
    dataset=SIGNAL_DATASETS,
    year=YEARS,
    syst_group=SYST_GROUPS,
)

MERGED_SIGNAL_SYST_COFFEA = f"{F1_OUTPUT_PATH}histAll_{config['label']}_signals_syst.coffea"

def normalize_to_posix(path: str) -> str:
    if not path:
        return ""
    if path.startswith("root://cmseos.fnal.gov/"):
        rel = path.replace("root://cmseos.fnal.gov/", "").lstrip("/")
        return f"/eos/uscms/{rel}"
    return path

def get_stat_only_coffea():
    cand = ""
    if config.get('inputs', {}).get('nominal_hists'):
        cand = config['inputs']['nominal_hists']
    elif config.get('inputs', {}).get('stat_only_coffea'):
        cand = config['inputs']['stat_only_coffea']
    elif config.get('nominal_coffea'):
        cand = config['nominal_coffea']
    else:
        upstream = config.get('inputs', {}).get('upstream_roasts', {})
        if isinstance(upstream, dict):
            nom_id = upstream.get('stat_only') or upstream.get('nominal')
            if nom_id:
                for base in [f"roasts/{nom_id}", f"/uscms_data/d3/algomez/HH4b/prod/{nom_id}/barista"]:
                    c1 = f"{base}/output/histAll_{nom_id}.coffea"
                    if os.path.exists(c1):
                        return c1
                    c2 = f"{base}/output/PhaseF_1_analysis/histAll_{nom_id}.coffea"
                    if os.path.exists(c2):
                        return c2
    posix_cand = normalize_to_posix(cand)
    if os.path.exists(posix_cand):
        return posix_cand
    if os.path.exists(cand):
        return cand
    return posix_cand if posix_cand else cand

include: "helpers/common.smk"

def get_raw_analysis_config():
    return resolve_config_section(config, primary_key='analysis_config', fallback_keys=['analysis'])

analysis_config_path = f"{F1_OUTPUT_PATH}analysis_config.yml"

def get_analysis_config_inputs(wildcards):
    inputs = list(workflow.configfiles) if workflow.configfiles else []
    ds_loc = config.get('analysis_dataset_location', config.get('dataset_location', ''))
    if isinstance(ds_loc, str) and ds_loc.endswith(('.yml', '.yaml')):
        inputs.append(ds_loc)
    return inputs

def get_f1_master_targets(wildcards):
    targets = [
        MERGED_SIGNAL_SYST_COFFEA,
        f"{F1_OUTPUT_PATH}cutflow_validation_systematics.txt",
        f"{F1_OUTPUT_PATH}cutflow_systematics.html",
        f"{F1_OUTPUT_PATH}cutflow_systematics.yml",
    ]
    nom = get_stat_only_coffea()
    if nom and os.path.exists(nom):
        targets.extend([
            f"{F1_OUTPUT_PATH}signal_crosscheck.html",
            f"{F1_OUTPUT_PATH}signal_crosscheck.txt",
        ])
    return targets

rule all_full_syst_F_1:
    input: get_f1_master_targets

rule all_full_syst_singlefiles:
    input: SIGNAL_SINGLE_FILES

rule create_analysis_config_full_syst:
    input: get_analysis_config_inputs
    output: analysis_config_path
    run:
        import yaml, os
        cfg = get_raw_analysis_config()
        for k in ['processor', 'weights_file', 'friend_file', 'dataset_location']:
            if k not in cfg and config.get(k):
                cfg[k] = config[k]
        if config.get("test", False):
            if 'runner' not in cfg or not isinstance(cfg['runner'], dict):
                cfg['runner'] = {}
            cfg['runner']['condor'] = False
            cfg['runner']['shared_dask'] = False
            cfg['runner']['run_performance'] = False
            if config.get('test_chunksize'):
                cfg['runner']['chunksize'] = int(config['test_chunksize'])
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

rule analysis_signal_systematics:
    input:
        runner_script = "runner.py",
        config_file = analysis_config_path
    output:
        f"{F1_OUTPUT_PATH}singlefiles/histAll_{config['label']}__{{dataset}}__{{year}}__{{syst_group}}.coffea"
    params:
        dataset = lambda wildcards: wildcards.dataset,
        year = lambda wildcards: wildcards.year,
        syst_group = lambda wildcards: wildcards.syst_group,
        output_path = f"{F1_OUTPUT_PATH}singlefiles/",
        output_file = lambda wildcards: f"histAll_{config['label']}__{wildcards.dataset}__{wildcards.year}__{wildcards.syst_group}.coffea",
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-t" if config.get("test", False) else "",
            config.get("additional_parameters", ""),
        ])),
        run_container_wrapper = config.get("analysis_container_wrapper", "./run_container"),
        python_bin = lambda wildcards: config.get("python_bin", "python3"),
    log:
        f"{F1_OUTPUT_PATH}logs/analysis_signal__{{dataset}}__{{year}}__{{syst_group}}.log"
    shell:
        """
        set -eo pipefail
        mkdir -p {params.output_path} $(dirname {log})
        {params.run_container_wrapper} {params.python_bin} runner.py {input.config_file} \
            --datasets {params.dataset} \
            --years {params.year} \
            --systematics {params.syst_group} \
            --output-path {params.output_path} \
            --output {params.output_file} \
            {params.extra_arguments} 2>&1 | tee {log}
        """

rule merge_signal_systematics:
    input:
        files = SIGNAL_SINGLE_FILES
    output:
        MERGED_SIGNAL_SYST_COFFEA
    params:
        run_container_wrapper = config.get("analysis_container_wrapper", "./run_container"),
        python_bin = lambda wildcards: config.get("python_bin", "python3"),
        input_files = lambda wildcards, input: " ".join(input.files)
    log:
        f"{F1_OUTPUT_PATH}logs/merge_signal_systematics.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.run_container_wrapper} {params.python_bin} src/tools/merge_coffea_files.py \
            -f {params.input_files} -o {output} 2>&1 | tee {log}
        """

rule check_nominal_signal_match:
    input:
        full_syst_signal = MERGED_SIGNAL_SYST_COFFEA,
        stat_only_signal = lambda wildcards: get_stat_only_coffea()
    output:
        html = f"{F1_OUTPUT_PATH}signal_crosscheck.html",
        txt = f"{F1_OUTPUT_PATH}signal_crosscheck.txt"
    log:
        f"{F1_OUTPUT_PATH}logs/signal_crosscheck.log"
    params:
        variables = lambda wildcards: ",".join(list(dict.fromkeys(
            [cfg['variable'] for cfg in config.get('channels', {}).values() if cfg.get('variable')] + ['v4j.mass', 'selJets.n']
        ))),
        processes = lambda wildcards: ",".join(SIGNAL_DATASETS),
        tolerance = config.get('crosscheck_tolerance', 1e-4),
        run_container_wrapper = config.get("analysis_container_wrapper", "./run_container"),
        python_bin = lambda wildcards: config.get("python_bin", "python3")
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.html}) $(dirname {log})
        {params.run_container_wrapper} {params.python_bin} src/tools/compare_coffea_hists.py \
            {input.full_syst_signal} {input.stat_only_signal} \
            -o {output.html} --txt {output.txt} \
            --label-a "full_syst" --label-b "stat_only" \
            --variables {params.variables} \
            --regions SR,SB --tags fourTag \
            --processes {params.processes} \
            --tolerance {params.tolerance} \
            --strict 2>&1 | tee {log}
        """

def get_known_counts_file():
    cand = config.get("known_counts_systematics", "")
    if not cand:
        sig = config.get('signal_datasets', ['ttHbb'])[0]
        cand = f"coffea4bees/analysis/tests/known_Counts_systematics_{sig}.yml"
    return cand if os.path.exists(cand) else ""

rule cutflow_systematics_table:
    input:
        coffea = MERGED_SIGNAL_SYST_COFFEA,
        known_counts = lambda wildcards: [f for f in [get_known_counts_file()] if f]
    output:
        txt = f"{F1_OUTPUT_PATH}cutflow_validation_systematics.txt",
        html = f"{F1_OUTPUT_PATH}cutflow_systematics.html",
        yml = f"{F1_OUTPUT_PATH}cutflow_systematics.yml",
    log:
        f"{F1_OUTPUT_PATH}logs/cutflow_systematics.log"
    params:
        tolerance = config.get("cutflow_tolerance", 1e-3),
        known_counts_arg = lambda wildcards, input: f"--known-counts {input.known_counts[0]}" if input.known_counts else "",
        run_container_wrapper = config.get("analysis_container_wrapper", "./run_container"),
        python_bin = lambda wildcards: config.get("python_bin", "python3"),
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.txt}) $(dirname {log})
        {params.run_container_wrapper} {params.python_bin} src/tools/cutflow_systematics.py \
            {input.coffea} \
            {params.known_counts_arg} \
            --output-txt {output.txt} \
            --output-html {output.html} \
            --output-yml {output.yml} \
            --tolerance {params.tolerance} 2>&1 | tee {log}
        """

localrules: create_analysis_config_full_syst, merge_signal_systematics, check_nominal_signal_match, cutflow_systematics_table, all_full_syst_singlefiles, all_full_syst_F_1
