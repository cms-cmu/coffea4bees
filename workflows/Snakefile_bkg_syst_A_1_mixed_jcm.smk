# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_A_1_mixed_jcm.smk
#
# Stage A_1: mixed-data JCM in the analysis selection
# ==============================================================================
#
# The mixed data itself (mixeddata_all) and the ttbar pseudodata come from a MakeMixedData roast
# (inputs.mixeddata_all / inputs.ttbar_psdata, its handoff/ YAMLs). What is analysis-specific is the
# mixed-data JCM the subsamples are drawn with (A_2): it is fit here against the analysis' own 4b
# data and ttbar MC (inputs.jcm_hists, the B.1 noJCM histograms), with the mixed data histogrammed
# by the very runner config those were made with -- one selection, one binning.
#
#   A1_fetch                          inputs.jcm_hists + its runner config, the mixed-data and
#                                     pseudodata YAMLs -> local
#   A1_hist_config                    that runner config, pointed at mixeddata_all (read from EOS)
#   A1_hists (per year, condor)       the analysis processor over mixeddata_all
#   A1_merge_hists                    + the analysis data / ttbar histAll_NoJCM
#   A1_jcm_config + A1_fit            make_jcm_weights.py, mixeddata_all as the "3b" sample, ttbar kept
#                                     -> jetCombinatoricModel_<region>_mixeddata.yml (A_2 splits with it)
#
# 4b mixing (subsamples.source: seeds) has no JCM: only A1_fetch runs (A_2 needs the pseudodata
# and the mixed-data YAML; B_1 the histograms).
#
# OUTPUT: {out_a1}JCM_mixeddata/jetCombinatoricModel_<region>_mixeddata.yml
# ==============================================================================

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"

MJ = config.get('mixed_jcm') or {}
A1_REGION = MJ.get('region', 'SB')
A1_JCM_TAG = "mixeddata"
A1_JCM_DIR = f"{out_a1}JCM_{A1_JCM_TAG}/"
MIXED_JCM = f"{A1_JCM_DIR}jetCombinatoricModel_{A1_REGION}_{A1_JCM_TAG}.yml"
A1_HIST_CONFIG = f"{out_a1}analysis_config_mixed.yml"
A1_HISTALL = f"{out_a1}histAll_mixedJCM.coffea"

rule A1_fetch:
    output:
        hists = JCM_HISTS,
        hist_config = JCM_HIST_CONFIG,
        psdata = PS_DATASET,
        mixed = MIXED_DATASET
    log: f"{A_INPUT_DIR}fetch.log"
    params:
        hists = JCM_HISTS_URL,
        hist_config = JCM_HIST_CONFIG_URL,
        psdata = PS_URL,
        mixed = MIXED_URL
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        mkdir -p $(dirname {output.hists})
        fetch_file() {{
            if [[ "$1" == root://* ]]; then xrdcp -f "$1" "$2"; else cp -f "$1" "$2"; fi 2>&1 | tee -a {log}
            echo "fetched $1" | tee -a {log}
        }}
        fetch_file "{params.hists}" "{output.hists}"
        fetch_file "{params.hist_config}" "{output.hist_config}"
        fetch_file "{params.psdata}" "{output.psdata}"
        fetch_file "{params.mixed}" "{output.mixed}"
        """

rule A1_hist_config:
    input: JCM_HIST_CONFIG
    output: A1_HIST_CONFIG
    run:
        with open(input[0]) as f:
            cfg = yaml.safe_load(f) or {}
        # the mixed data is built with the non-tight four-tag definition: the histograms it is fit
        # against must be too
        tight = (cfg.get('config') or {}).get('fourTag_use_tight', False)
        if tight is not False:
            raise ValueError(f"{JCM_HIST_CONFIG_URL}: fourTag_use_tight={tight!r}; the mixed data needs "
                             f"the non-tight selection")
        cfg['dataset_location'] = [MIXED_URL]
        cfg.get('runner', {}).pop('dataset_location', None)
        if config.get('test', False):
            cfg.setdefault('runner', {}).update({'condor': False, 'shared_dask': False})
        write_yaml(output[0], cfg)

use rule analysis_processor from analysis as A1_hists with:
    input:
        runner_script = "runner.py",
        config_file = A1_HIST_CONFIG
    output: f"{out_a1}singlefiles/hist__{MIX_NAME}__{{year}}.coffea"
    log: f"{out_a1}logs/hists__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = MIX_NAME,
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, condor_flags])),
        run_container_wrapper = container_wrapper,
        python_bin = python_bin

use rule merging_coffea_files from analysis as A1_merge_hists with:
    input:
        files = [JCM_HISTS] + expand(f"{out_a1}singlefiles/hist__{MIX_NAME}__{{year}}.coffea", year=YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: A1_HISTALL
    log: f"{out_a1}logs/merge_hists.log"
    params:
        run_performance = False,
        run_container_wrapper = container_wrapper,
        python_bin = python_bin,
        input_files = lambda wildcards, input: " ".join(input.files)

rule A1_jcm_config:
    input: MJ.get('config', "coffea4bees/analysis/jcm_tools/metadata/mixeddata_all_config.yml")
    output: f"{out_a1}jcm_config_mixed.yml"
    run:
        with open(input[0]) as f:
            cfg = yaml.safe_load(f) or {}
        cfg['data3bName'] = MIX_NAME        # the mixed data stands in for the 3b sample
        cfg['data4bName'] = "data"
        cfg['float_t'] = bool(MJ.get('float_t', True))
        # The fit keeps the ttbar term: the mixed data is multijet only (ttbar-subtracted 3b data)
        # while the 4b data it is fit to contains ttbar -- the samples inputs.jcm_hists holds.
        if 'ttbarProcesses' in MJ:
            cfg['ttbarProcesses'] = MJ['ttbarProcesses']
        elif TTBAR:
            cfg.setdefault('ttbarProcesses', TTBAR)
        write_yaml(output[0], cfg)

rule A1_fit:
    input:
        # the merged file: make_jcm_weights.py keeps the LAST input file holding each process
        # (jcm_tools/helpers.py:loadHistograms), so per-year inputs would fit one year of mixed data
        hists = A1_HISTALL,
        jcm_config = f"{out_a1}jcm_config_mixed.yml"
    output: MIXED_JCM
    log: f"{out_a1}logs/fit.log"
    params:
        cut = f"-c {MJ['cut']}" if MJ.get('cut') else ""
    shell:
        """
        set -eo pipefail
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR {A1_JCM_DIR}
        {container_wrapper} {python_bin} coffea4bees/analysis/jcm_tools/make_jcm_weights.py -o {A1_JCM_DIR} \
            -i {input.hists} -r {A1_REGION} -w {A1_JCM_TAG} --data4bName data {params.cut} \
            --jcm_config {input.jcm_config} 2>&1 | tee {log}
        ls {A1_JCM_DIR} 2>&1 | tee -a {log}
        """

rule all_bkg_syst_A_1:
    input: [MIXED_JCM] if SUB_SOURCE == 'split' else [JCM_HISTS]

localrules: A1_fetch, A1_hist_config, A1_merge_hists, A1_jcm_config, A1_fit, all_bkg_syst_A_1
