import os
import shutil

# Fallback defaults for backwards compatibility or running direct
config.setdefault('label', "nominal_wNewSvB")
config.setdefault('output_path', "output/nominal_wNewSvB/")
base_output = config.get('output_path', "output/nominal_wNewSvB/")
if not base_output.endswith('/'):
    base_output += '/'
if base_output.rstrip('/').endswith('PhaseF_1_analysis'):
    F1_OUTPUT_PATH = base_output
else:
    F1_OUTPUT_PATH = os.path.join(base_output, 'PhaseF_1_analysis/')
if not F1_OUTPUT_PATH.endswith('/'):
    F1_OUTPUT_PATH += '/'
config.setdefault('analysis_config', "coffea4bees/analysis/metadata/HH4b_2024_v2.yml")
config.setdefault('processor', "coffea4bees/analysis/processors/processor_HH4b.py")
config.setdefault('friend_file', "coffea4bees/metadata/friends/friends_HH4b.yml")
config.setdefault('weights_file', "coffea4bees/metadata/weights/weights_HH4b.yml")
config.setdefault('plot_config', "coffea4bees/plots/metadata/plotsAll_ttbarWeights.yml")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets/")
config.setdefault('test', False)
config.setdefault('known_counts', "")

# Parse boolean for test flag
is_test = config.get('test', False)
if isinstance(is_test, str):
    is_test = is_test.lower() in ("true", "1", "yes")
config['test'] = is_test

container_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container"
config.setdefault('container_wrapper', container_wrapper)
config.setdefault('analysis_container_wrapper', config.get('container_wrapper', container_wrapper))

python_bin = os.getenv("CONTAINER_PYTHON", "python")
config.setdefault('python_bin', python_bin)

if config.get('additional_parameters') is not None:
    pass
elif config.get('test', False) or os.getenv("CI"):
    config.setdefault('additional_parameters', "")
elif "bridges2" in os.uname().nodename or "psc" in os.uname().nodename or "/ocean/" in os.getcwd():
    config.setdefault('additional_parameters', "")
else:
    config.setdefault('additional_parameters', "--shared-dask --condor --run-performance")

config.setdefault('dataset', ['GluGluToHHTo4B_cHHH1', 'GluGluToHHTo4B_cHHH0', 'GluGluToHHTo4B_cHHH2p45', 'GluGluToHHTo4B_cHHH5', 'ZH4b', 'ZZ4b', 'ggZH4b'])
config.setdefault('year_eras', {
    'UL16_preVFP':  ['C', 'D', 'E', 'F'],
    'UL16_postVFP': ['F', 'G', 'H'],
    'UL17':         ['C', 'D', 'E', 'F'],
    'UL18':         ['A', 'B', 'C', 'D'],
})

### Containers
config.setdefault('container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")

SYNTHETIC_DATA_PREFIXES = ('mixeddata', 'synthetic_data', 'datamixed', 'data_3b_for_mixed')
DATA_DATASETS = [d for d in config['dataset'] if d == 'data' or any(d.startswith(p) for p in SYNTHETIC_DATA_PREFIXES)]
MC_DATASETS = [d for d in config['dataset'] if d not in DATA_DATASETS]

DATA_YEARS = [str(y) for y in config['year_eras'].keys()]
DATA_YEAR_ERA = [(str(yr), era) for yr, eras in config['year_eras'].items() for era in eras] if 'data' in DATA_DATASETS else []
DATA_ERA_FILES = [f"{F1_OUTPUT_PATH}singlefiles/histAll_{config['label']}_data__{yr}_{era}.coffea" for yr, era in DATA_YEAR_ERA]

DATA_NOERA_DATASETS = [d for d in DATA_DATASETS if d != 'data']
DATA_NOERA_FILES = expand(
    "{output_path}singlefiles/histAll_" + config['label'] + "__{dataset}__{year}.coffea",
    output_path=F1_OUTPUT_PATH,
    dataset=DATA_NOERA_DATASETS,
    year=DATA_YEARS
) if DATA_NOERA_DATASETS else []

ALL_DATA_FILES = DATA_ERA_FILES + DATA_NOERA_FILES
MC_FILES = expand(
    "{output_path}singlefiles/histAll_" + config['label'] + "__{dataset}__{year}.coffea",
    output_path=F1_OUTPUT_PATH,
    dataset=MC_DATASETS,
    year=DATA_YEARS
) if MC_DATASETS else []

include: "helpers/common.smk"

def get_raw_analysis_config():
    cfg = resolve_config_section(config, primary_key='analysis_config', fallback_keys=['analysis'])
    if config.get('per_year_jcm', False) and 'config' in cfg and cfg['config'].get('apply_JCM'):
        if 'JCM_file' not in cfg['config'] or not cfg['config']['JCM_file']:
            inputs_jcm = (config.get('inputs') or {}).get('JCM')
            if isinstance(inputs_jcm, dict):
                cfg['config']['JCM_file'] = dict(inputs_jcm)
            else:
                tag = config.get('tag', '2024_v2')
                roast_id = config.get('roast_id', config.get('label', 'nominal'))
                jcm_candidate = {
                    yr: f"coffea4bees/metadata/weights/JCM/{roast_id}/jetCombinatoricModel_SB_{tag}_{yr}.yml"
                    for yr in config.get('year_eras', {}).keys()
                }
                if all(os.path.exists(p) for p in jcm_candidate.values()):
                    cfg['config']['JCM_file'] = jcm_candidate
                else:
                    raise ValueError(
                        f"per_year_jcm is True for {config.get('roast_id', config.get('label'))}, "
                        f"but no per-year JCM files could be found under analysis_config.config.JCM_file, "
                        f"inputs.JCM, or coffea4bees/metadata/weights/JCM/{roast_id}/. "
                        f"Failing fast to prevent silent fallback to single Run2 JCM."
                    )
    return cfg

# 2024 / 2025 data is blinded (helpers/common.smk: BLIND_YEARS)
require_blind(DATA_YEARS if 'data' in DATA_DATASETS else [], (get_raw_analysis_config().get('config') or {}).get('blind'),
              f"F.1 analysis_config.config ({config['label']})")

analysis_config_path = f"{F1_OUTPUT_PATH}analysis_config.yml"

def get_analysis_config_inputs(wildcards):
    inputs = list(workflow.configfiles) if workflow.configfiles else []
    ds_loc = config.get('analysis_dataset_location', config.get('dataset_location', ''))
    if isinstance(ds_loc, str) and ds_loc.endswith(('.yml', '.yaml')):
        inputs.append(ds_loc)
    return inputs

rule create_analysis_config:
    input: get_analysis_config_inputs
    output: analysis_config_path
    run:
        import yaml, os
        cfg = get_raw_analysis_config()
        if config.get("test", False):
            if 'runner' not in cfg or not isinstance(cfg['runner'], dict):
                cfg['runner'] = {}
            cfg['runner']['condor'] = False
            cfg['runner']['shared_dask'] = False
            cfg['runner']['run_performance'] = False
            # Test mode runs maxchunks=1, so chunksize fixes the event count the known counts were made with
            if config.get('test_chunksize'):
                cfg['runner']['chunksize'] = int(config['test_chunksize'])
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

wildcard_constraints:
    year = "|".join([str(y) for y in config['year_eras'].keys()]),
    dataset = "|".join(DATA_NOERA_DATASETS + MC_DATASETS) if (DATA_NOERA_DATASETS + MC_DATASETS) else "none"

module analysis:
    snakefile: "rules/analysis.smk"
    config: config

def get_analysis_targets(wildcards):
    return [
        f"{F1_OUTPUT_PATH}histAll_{config['label']}.coffea",
        f"{F1_OUTPUT_PATH}plots_{config['label']}/plots_done.txt",
        f"{F1_OUTPUT_PATH}cutflow_validation_{config['label']}.txt",
        f"{F1_OUTPUT_PATH}cutflow_{config['label']}.yml",
        f"{F1_OUTPUT_PATH}cutflow_{config['label']}.html",             # closure table
        f"{F1_OUTPUT_PATH}cutflow_crosscheck_{config['label']}.html",  # vs Phase C.4
    ]

rule all_analysis:
    input: get_analysis_targets
    run:
        import os
        if os.path.abspath(base_output) != os.path.abspath(F1_OUTPUT_PATH.rstrip('/')):
            os.makedirs(base_output, exist_ok=True)
            links = [
                (f"histAll_{config['label']}.coffea", f"PhaseF_1_analysis/histAll_{config['label']}.coffea"),
                (f"plots_{config['label']}", f"PhaseF_1_analysis/plots_{config['label']}"),
                (f"cutflow_{config['label']}.html", f"PhaseF_1_analysis/cutflow_{config['label']}.html"),
                (f"cutflow_{config['label']}.yml", f"PhaseF_1_analysis/cutflow_{config['label']}.yml"),
                (f"cutflow_validation_{config['label']}.txt", f"PhaseF_1_analysis/cutflow_validation_{config['label']}.txt"),
                (f"cutflow_crosscheck_{config['label']}.html", f"PhaseF_1_analysis/cutflow_crosscheck_{config['label']}.html"),
            ]
            for link_name, target in links:
                dst = os.path.join(base_output, link_name)
                if not os.path.exists(dst) and not os.path.islink(dst):
                    try:
                        os.symlink(target, dst)
                    except OSError:
                        pass

if config.get("test", False):
    # Quick Test / CI Mode: single command for all data years/eras and single command for all signals
    DATA_TEST_TARGETS = []
    MC_TEST_TARGETS = []

    if DATA_DATASETS:
        DATA_TEST_TARGETS.append(f"{F1_OUTPUT_PATH}singlefiles/histAll_{config['label']}_data.coffea")
        use rule analysis_processor from analysis as analysis_data with:
            input: 
                runner_script = "runner.py",
                config_file = analysis_config_path
            output: f"{F1_OUTPUT_PATH}singlefiles/histAll_{config['label']}_data.coffea"
            log: f"{F1_OUTPUT_PATH}logs/analysis_{config['label']}_data.log"
            params:
                datasets = " ".join(DATA_DATASETS),
                years = " ".join(DATA_YEARS),
                config = lambda wildcards, input: input.config_file,
                extra_arguments = lambda wildcards: " ".join(filter(None, [
                    "-t",
                    config.get("additional_parameters", "")
                ])),
                run_container_wrapper = config['analysis_container_wrapper']

    if MC_DATASETS:
        MC_TEST_TARGETS.append(f"{F1_OUTPUT_PATH}singlefiles/histAll_{config['label']}_signals.coffea")
        use rule analysis_processor from analysis as analysis_MC with:
            input: 
                runner_script = "runner.py",
                config_file = analysis_config_path
            output: f"{F1_OUTPUT_PATH}singlefiles/histAll_{config['label']}_signals.coffea"
            log: f"{F1_OUTPUT_PATH}logs/analysis_{config['label']}_signals.log"
            params:
                datasets = " ".join(MC_DATASETS),
                years = " ".join(DATA_YEARS),
                config = lambda wildcards, input: input.config_file,
                extra_arguments = lambda wildcards: " ".join(filter(None, [
                    "-t",
                    config.get("additional_parameters", "")
                ])),
                run_container_wrapper = config['analysis_container_wrapper']

    use rule merging_coffea_files from analysis as merging_files with:
        input:
            files = DATA_TEST_TARGETS + MC_TEST_TARGETS,
            script = "src/tools/merge_coffea_files.py"
        output: f"{F1_OUTPUT_PATH}histAll_{config['label']}.coffea"
        params:
            run_performance = False,
            run_container_wrapper = config['analysis_container_wrapper']
        container: None
        log: f"{F1_OUTPUT_PATH}logs/merging_files.log"

    rule all_data:
        input: DATA_TEST_TARGETS

    rule all_signals:
        input: MC_TEST_TARGETS

else:
    # Production Mode: granular split per year/era and dataset for distributed cluster batching
    if DATA_YEAR_ERA:
        use rule analysis_processor from analysis as analysis_data with:
            input: 
                runner_script = "runner.py",
                config_file = analysis_config_path
            output: f"{F1_OUTPUT_PATH}singlefiles/histAll_{config['label']}_data__{{year}}_{{era}}.coffea"
            log: f"{F1_OUTPUT_PATH}logs/analysis_{config['label']}_data__{{year}}_{{era}}.log"
            params:
                datasets = "data",
                years = lambda wildcards: wildcards.year,
                config = lambda wildcards, input: input.config_file,
                extra_arguments = lambda wildcards: " ".join(filter(None, [
                    f"--era {wildcards.era}",
                    config.get("additional_parameters", "")
                ])),
                run_container_wrapper = config['analysis_container_wrapper']

    if DATA_NOERA_DATASETS or MC_DATASETS:
        use rule analysis_processor from analysis as analysis_dataset with:
            input: 
                runner_script = "runner.py",
                config_file = analysis_config_path
            output: f"{F1_OUTPUT_PATH}singlefiles/histAll_{config['label']}__{{dataset}}__{{year}}.coffea"
            log: f"{F1_OUTPUT_PATH}logs/analysis_{config['label']}_{{dataset}}_{{year}}.log"
            params:
                datasets = lambda wildcards: wildcards.dataset,
                years = lambda wildcards: wildcards.year,
                config = lambda wildcards, input: input.config_file,
                extra_arguments = lambda wildcards: " ".join(filter(None, [
                    config.get("additional_parameters", "")
                ])),
                run_container_wrapper = config['analysis_container_wrapper']

    use rule merging_coffea_files from analysis as merging_files with:
        input:
            files = ALL_DATA_FILES + MC_FILES,
            script = "src/tools/merge_coffea_files.py"
        output: f"{F1_OUTPUT_PATH}histAll_{config['label']}.coffea"
        params:
            run_performance = False,
            run_container_wrapper = config['analysis_container_wrapper']
        container: None
        log: f"{F1_OUTPUT_PATH}logs/merging_files.log"

    rule all_data:
        input: ALL_DATA_FILES

    rule all_signals:
        input: MC_FILES

if os.path.abspath(base_output) != os.path.abspath(F1_OUTPUT_PATH.rstrip('/')):
    rule legacy_nominal_coffea:
        input: f"{F1_OUTPUT_PATH}histAll_{config['label']}.coffea"
        output: f"{base_output}histAll_{config['label']}.coffea"
        run:
            import os
            dst = output[0]
            src = input[0]
            if not os.path.exists(dst) and not os.path.islink(dst):
                os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)

    rule legacy_nominal_plots:
        input: f"{F1_OUTPUT_PATH}plots_{config['label']}/plots_done.txt"
        output: f"{base_output}plots_{config['label']}/plots_done.txt"
        run:
            import os
            dst = output[0]
            src = input[0]
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst) and not os.path.islink(dst):
                os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)

    rule legacy_nominal_cutflow:
        input: f"{F1_OUTPUT_PATH}cutflow_validation_{config['label']}.txt"
        output: f"{base_output}cutflow_validation_{config['label']}.txt"
        run:
            import os
            dst = output[0]
            src = input[0]
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst) and not os.path.islink(dst):
                os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)

use rule make_plots from analysis with:
    input:
        coffea_file = f"{F1_OUTPUT_PATH}histAll_{config['label']}.coffea",
        metadata_file = config['plot_config'],
        plot_script = "coffea4bees/plots/makePlots.py"
    output: f"{F1_OUTPUT_PATH}plots_{config['label']}/plots_done.txt"
    log: f"{F1_OUTPUT_PATH}logs/make_plots.log"
    params:
        output_dir = f"{F1_OUTPUT_PATH}plots_{config['label']}/",
        # the shared rule plots from params.metadata (input.metadata_file only tracks the file):
        # without this line F.1 silently plotted with the generic default plotsAll.yml, whose
        # ttbar stack is the MC (dropped: no TTTo* in the file) instead of TTbar4b_from_d3
        metadata = config['plot_config'],
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-s xW",
            "--year " + (DATA_YEARS[0] if len(DATA_YEARS) == 1 else ("Run3" if any("202" in y for y in DATA_YEARS) else "RunII")),
            config.get("plot_extra_arguments", ""),
        ])),
        png_cores = 4,
        run_container_wrapper = config['analysis_container_wrapper']
    container: None

use rule check_cutflow from analysis with:
    input:
        coffea_file = f"{F1_OUTPUT_PATH}histAll_{config['label']}.coffea"
    output:
        validation_txt = f"{F1_OUTPUT_PATH}cutflow_validation_{config['label']}.txt",
        cutflow_yml = f"{F1_OUTPUT_PATH}cutflow_{config['label']}.yml"
    log: f"{F1_OUTPUT_PATH}logs/cutflow_validation_{config['label']}.log"
    params:
        known_counts = lambda wildcards: config.get("known_counts", ""),
        error_threshold = lambda wildcards: config.get("error_threshold", "0.001"),
        cutflow_list = lambda wildcards: config.get("cutflow_list", "passJetMult,passPreSel,passDiJetMass,SR,SB"),
        run_container_wrapper = config['analysis_container_wrapper']
    container: None

# Background-closure view of the F.1 cutflow. The 3b data carries JCM x FvT (= 4b multijet) and,
# with plot_ttbar_with_weights, the ttbar is the FvT-derived estimate from 3b data (cutflow entries
# TTbar_from_d3_<era>, weights d3_to_t4 / d3_to_t3) — Phase F does not run the ttbar MC.
# Override with `closure_ttbar` (e.g. "TTToHadronic TTToSemiLeptonic TTTo2L2Nu" when it does).
use rule cutflow_closure_table from analysis as analysis_cutflow_closure_table with:
    params:
        title = lambda wildcards: f"{config.get('label', 'analysis')}_cutflow_{wildcards.label}",
        multijet = "data3b",
        ttbar = lambda wildcards: config.get("closure_ttbar", "TTbar_from_d3"),
        # further cutflow_closure.py options, e.g. "--compare ttHbb" to add the signal as extra columns
        extra_arguments = lambda wildcards: config.get("closure_extra_arguments", ""),
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = lambda wildcards: config.get("python_bin", "python")

def _get_crosscheck_ref():
    cand_c4 = f"{base_output}PhaseC_4_FvT_closure/cutflow_FvT_closure.yml"
    cand_old = f"{base_output}FvT_closure/cutflow_FvT_closure.yml"
    return cand_c4 if os.path.exists(cand_c4) else (cand_old if os.path.exists(cand_old) else cand_c4)

# Cross-check against Phase C.4 (FvT closure): same processor, same data, same JCM x FvT, so every
# shared dataset must give identical weighted and raw 3b/4b cutflows (SvB only adds histograms).
# A mismatch means the two passes did not use the same weights (e.g. a different JCM file) or
# selection (e.g. blinding). Reference path: `crosscheck_reference` (default: the C.4 dump).
use rule cutflow_crosscheck from analysis as analysis_cutflow_crosscheck with:
    params:
        reference = lambda wildcards: config.get("crosscheck_reference", _get_crosscheck_ref()),
        title = lambda wildcards: f"{config.get('label', 'analysis')}_cutflow_crosscheck_{wildcards.label}",
        label_a = lambda wildcards: f"F.1_{wildcards.label}",
        label_b = "C.4_FvT_closure",
        tolerance = lambda wildcards: config.get("crosscheck_tolerance", "0.001"),
        # blinding (SvB > 0.8 in the 4b SR) only acts in F.1 (C.4 has no SvB) -> accept the 4b data
        # differences while blind: true; override with `crosscheck_ignore`
        ignore = lambda wildcards: config.get("crosscheck_ignore",
                                              "data*:counts4*" if get_raw_analysis_config().get("config", {}).get("blind", False) else ""),
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = lambda wildcards: config.get("python_bin", "python")

localrules: merging_files, make_plots, check_cutflow, analysis_cutflow_closure_table, analysis_cutflow_crosscheck, all_analysis
