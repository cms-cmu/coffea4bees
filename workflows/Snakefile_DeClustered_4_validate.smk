# coffea4bees/workflows/Snakefile_DeClustered_4_validate.smk
# D.4: validate the synthetic data against the real 4b data it was made from.
# (scripts/synthetic-dataset-analyze-Run3-all.sh + synthetic-dataset-analyze-cutflow-Run3.sh)
#
#   D4_hist_config                    the upstream B.1 noJCM runner config, pointed at the synthetic
#                                     dataset (read from this roast's EOS handoff, as a consumer would)
#   D4_hists (per year, condor)       processor_HH4b over every seed of the synthetic dataset
#   D4_merge_hists                    + the upstream data / ttbar histAll_NoJCM
#                                     -> histAll_declustered.coffea: data 4b (data), syn(_noTT)_v<s>
#                                        (synthetic 4b), ttbar MC, one selection and binning
#   D4_cutflow                        cutflow dump (first roast: the reference to bless as
#                                     validation.known_counts)

D4_OUT = f"{out}D4/"
D4_HIST_CONFIG = f"{D4_OUT}analysis_config_declustered.yml"
D4_HISTALL = f"{D4_OUT}histAll_declustered.coffea"

rule D4_hist_config:
    input: UPSTREAM_HIST_CONFIG
    output: D4_HIST_CONFIG
    run:
        with open(input[0]) as f:
            cfg = yaml.safe_load(f) or {}
        tight = (cfg.get('config') or {}).get('fourTag_use_tight', False)
        if tight is not False:
            raise ValueError(f"upstream roast histogrammed with fourTag_use_tight={tight!r} "
                             f"({INPUTS['jcm_hists']}); the declustered data needs the non-tight selection")
        cfg['dataset_location'] = [DATASET_URL]
        cfg.get('runner', {}).pop('dataset_location', None)
        if config['test']:
            cfg.setdefault('runner', {}).update({'condor': False, 'shared_dask': False})
        write_yaml(output[0], cfg)

use rule analysis_processor from analysis as D4_hists with:
    input:
        runner_script = "runner.py",
        config_file = D4_HIST_CONFIG,
        published = D3_PUBLISHED
    output: f"{D4_OUT}singlefiles/hist__{DATASET_NAME}__{{year}}.coffea"
    log: f"{D4_OUT}logs/hists__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = DATASET_NAME,
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

use rule merging_coffea_files from analysis as D4_merge_hists with:
    input:
        files = [UPSTREAM_HISTS] + expand(f"{D4_OUT}singlefiles/hist__{DATASET_NAME}__{{year}}.coffea", year=YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: D4_HISTALL
    log: f"{D4_OUT}logs/merge_hists.log"
    params:
        run_performance = False,
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON,
        input_files = lambda wildcards, input: " ".join(input.files)

use rule check_cutflow from analysis as D4_cutflow with:
    input:
        coffea_file = D4_HISTALL
    output:
        validation_txt = f"{D4_OUT}cutflow_validation_declustered.txt",
        cutflow_yml = f"{D4_OUT}cutflow_declustered.yml"
    log: f"{D4_OUT}logs/cutflow_declustered.log"
    params:
        known_flag = lambda wildcards: (f'--known-cutflow "{VAL["known_counts"]}"'
                                        if VAL.get('known_counts') and os.path.exists(VAL['known_counts'])
                                        else '--known-cutflow "none"'),
        error_threshold = lambda wildcards: VAL.get("error_threshold", config.get("error_threshold", "0.001")),
        cutflow_list = lambda wildcards: config.get("cutflow_list", "passJetMult,passPreSel,passDiJetMass,SR,SB"),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON
    container: None

rule all_D4:
    input:
        D4_HISTALL,
        f"{D4_OUT}cutflow_validation_declustered.txt"

localrules: D4_hist_config, D4_merge_hists, D4_cutflow, all_D4
