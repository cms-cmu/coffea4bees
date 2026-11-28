# coffea4bees/workflows/Snakefile_btagSF_calibration.smk
# b-tagging SF calibration: per-sample (Njet, HT) maps that remove the normalization effect of the
# b-tag shape SF (N(no SF) / N(SF), derived before any b-tag requirement).
#
#   1. processor_btagSF_calibration.py on MC NanoAOD, one job per dataset x year
#      (the picoAODs already contain a b-tag requirement and cannot be used)
#   2. merge
#   3. src/tools/compute_btagSF_calibration.py: one correctionlib JSON per year, 2D maps,
#      closure plots (no SF / SF / SF x calibration) and the yield table
#   4. copy the maps to coffea4bees/metadata/weights/btagSF_calibration/<tag>/ (not in test mode)
#
# The maps are applied in the analysis processors with
#   apply_btagSF_calibration: true
#   btagSF_calibration_file: coffea4bees/metadata/weights/btagSF_calibration/<tag>/btagSF_calibrationSF_{year}.json
#
#   ./run_container snakemake -s coffea4bees/workflows/Snakefile_btagSF_calibration.smk \
#       --configfile coffea4bees/workflows/config/btagSF_calibration_run2.yml --cores 4

import os
import copy
import yaml

config.setdefault('label', "btagSF_calibration")
base_output = config.get('output_path', 'output/btagSF_calibration/')
if not base_output.endswith('/'):
    base_output += '/'
BTAG_OUTPUT_PATH = base_output

is_test = config.get('test', False)
if isinstance(is_test, str):
    is_test = is_test.lower() in ("true", "1", "yes")
config['test'] = is_test

container_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container"
config.setdefault('container_wrapper', container_wrapper)
config.setdefault('analysis_container_wrapper', config.get('container_wrapper', container_wrapper))
config.setdefault('python_bin', os.getenv("CONTAINER_PYTHON", "python"))

if config['test'] or os.getenv("CI"):
    config.setdefault('additional_parameters', "")
else:
    config.setdefault('additional_parameters', "--shared-dask --condor --run-performance")

include: "helpers/common.smk"

BTAG_CFG = resolve_config_section(config, primary_key='btagsf_calibration')
BTAG_DATASETS = BTAG_CFG.get('datasets') or []
BTAG_YEARS = [str(y) for y in (BTAG_CFG.get('years') or [])]
if isinstance(BTAG_DATASETS, str):
    BTAG_DATASETS = [d.strip() for d in BTAG_DATASETS.split(",") if d.strip()]
if not BTAG_DATASETS or not BTAG_YEARS:
    raise ValueError("btagsf_calibration.datasets and btagsf_calibration.years must be set in the config")
BTAG_TAG = str(BTAG_CFG.get('tag', 'v1'))
BTAG_SF_HIST = BTAG_CFG.get('sf_hist', 'Njet_HT')
BTAG_EXTRA_ARGS = BTAG_CFG.get('extra_arguments', "")
BTAG_INSTALL_DIR = BTAG_CFG.get('install_dir', f"coffea4bees/metadata/weights/btagSF_calibration/{BTAG_TAG}/")
if not BTAG_INSTALL_DIR.endswith('/'):
    BTAG_INSTALL_DIR += '/'

btag_config_path = f"{BTAG_OUTPUT_PATH}processor_config.yml"
btag_hists = f"{BTAG_OUTPUT_PATH}histAll_btagSF_calibration.coffea"
btag_calib_dir = f"{BTAG_OUTPUT_PATH}calibration_{BTAG_TAG}/"
BTAG_MAPS = [f"{btag_calib_dir}btagSF_calibrationSF_{yr}.json" for yr in BTAG_YEARS]
BTAG_INSTALLED_MAPS = [f"{BTAG_INSTALL_DIR}btagSF_calibrationSF_{yr}.json" for yr in BTAG_YEARS]


def get_btag_processor_config():
    """runner.py config: the btagsf_calibration section without the workflow-only keys."""
    res = {k: copy.deepcopy(v) for k, v in BTAG_CFG.items() if k in ('processor', 'dataset_location', 'runner', 'config')}
    res.setdefault('runner', {})
    res['runner']['data_tier'] = 'nanoAOD'
    if config['test']:
        res['runner']['condor'] = False
        res['runner']['shared_dask'] = False
        res['runner']['run_performance'] = False
    return res


module analysis:
    snakefile: "rules/analysis.smk"
    config: config


rule all_btagSF_calibration:
    input:
        BTAG_MAPS,
        f"{btag_calib_dir}btagSF_calibration_yields.yml",
        [] if config['test'] else BTAG_INSTALLED_MAPS

rule create_btagSF_calibration_config:
    input: workflow.configfiles if workflow.configfiles else []
    output: btag_config_path
    run:
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(get_btag_processor_config(), f, default_flow_style=False)

use rule analysis_processor from analysis as btagSF_calibration_processor with:
    input:
        runner_script = "runner.py",
        config_file = btag_config_path
    output: f"{BTAG_OUTPUT_PATH}singlefiles/hist__{{dataset}}__{{year}}.coffea"
    log: f"{BTAG_OUTPUT_PATH}logs/processor__{{dataset}}__{{year}}.log"
    params:
        datasets = lambda wildcards: wildcards.dataset,
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-t" if config.get("test", False) else "",
            config.get("additional_parameters", "")
        ])),
        run_container_wrapper = config['analysis_container_wrapper']

use rule merging_coffea_files from analysis as merge_btagSF_calibration with:
    input:
        files = [f"{BTAG_OUTPUT_PATH}singlefiles/hist__{ds}__{yr}.coffea" for ds in BTAG_DATASETS for yr in BTAG_YEARS],
        script = "src/tools/merge_coffea_files.py"
    output: btag_hists
    log: f"{BTAG_OUTPUT_PATH}logs/merge.log"
    params:
        run_performance = False,
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = lambda wildcards: config.get("python_bin", "python"),
        input_files = lambda wildcards, input: " ".join([f for f in (input.files if hasattr(input, 'files') else input) if not f.endswith('.py')])

rule compute_btagSF_calibration:
    input:
        hists = btag_hists,
        script = "src/tools/compute_btagSF_calibration.py"
    output:
        maps = BTAG_MAPS,
        yields = f"{btag_calib_dir}btagSF_calibration_yields.yml"
    params:
        output_dir = btag_calib_dir,
        sf_hist = BTAG_SF_HIST,
        extra_arguments = BTAG_EXTRA_ARGS,
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log: f"{BTAG_OUTPUT_PATH}logs/compute_btagSF_calibration.log"
    shell:
        """
        set -eo pipefail
        export MPLCONFIGDIR="/tmp/matplotlib_btagSF_calibration"
        mkdir -p $MPLCONFIGDIR {params.output_dir}
        {params.run_container_wrapper} {params.python_bin} -m src.tools.compute_btagSF_calibration \
            -i {input.hists} -o {params.output_dir} -c -v --sf-hist {params.sf_hist} \
            {params.extra_arguments} 2>&1 | tee {log}
        """

rule install_btagSF_calibration:
    input: BTAG_MAPS
    output: BTAG_INSTALLED_MAPS
    params:
        install_dir = BTAG_INSTALL_DIR
    shell:
        """
        set -eo pipefail
        mkdir -p {params.install_dir}
        cp -f {input} {params.install_dir}
        """
