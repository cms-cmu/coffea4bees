# coffea4bees/workflows/Snakefile_MakeMixedData_3_validate.smk
# M.3: validate the mixed data and fit its JCM.
#
#   M3_hist_config                    the upstream B.1 noJCM runner config, pointed at mixeddata_all
#                                     (read from this roast's EOS handoff, as a consumer would)
#   M3_hists (per year, condor)       processor_HH4b over mixeddata_all
#   M3_merge_hists                    + the upstream data / ttbar histAll_NoJCM
#   M3_cutflow                        cutflow dump (first roast: the reference to bless)
#   M3_fit                            make_jcm_weights.py, mixeddata_all as the "3b" sample, float_t
#                                     -> jetCombinatoricModel_SB_mixeddata.yml (M.4 splits with it)
#   M3_study (per year) + merge       processor_study_mixed_data with that JCM: per-event subsample
#                                     assignment, pseudo-tag weights, overflow (N*w > 1) counts
#   M3_publish                        the mixed-data JCM -> <PUB>/handoff/  (MvD needs it too)

M3_OUT = f"{out}M3/"
M3_HIST_CONFIG = f"{M3_OUT}analysis_config_mixed.yml"
M3_HISTALL = f"{M3_OUT}histAll_mixedJCM.coffea"
MJ = config.get('mixed_jcm') or {}
M3_JCM_TAG = "mixeddata"
M3_JCM_DIR = f"{M3_OUT}JCM_{M3_JCM_TAG}/"
MIXED_JCM = f"{M3_JCM_DIR}jetCombinatoricModel_SB_{M3_JCM_TAG}.yml"
M3_STUDY_CONFIG = f"{M3_OUT}study_mixed_data.yml"
M3_STUDY = f"{M3_OUT}study_{MIX_NAME}.coffea"
M3_PUBLISHED = f"{M3_OUT}published.done"

rule M3_hist_config:
    input: UPSTREAM_HIST_CONFIG
    output: M3_HIST_CONFIG
    run:
        with open(input[0]) as f:
            cfg = yaml.safe_load(f) or {}
        tight = (cfg.get('config') or {}).get('fourTag_use_tight', False)
        if tight is not False:
            raise ValueError(f"upstream JCM roast histogrammed with fourTag_use_tight={tight!r} "
                             f"({INPUTS['jcm_hists']}); the mixed data needs the non-tight selection")
        cfg['dataset_location'] = [MIXED_URL]
        cfg.get('runner', {}).pop('dataset_location', None)
        if config['test']:
            cfg.setdefault('runner', {}).update({'condor': False, 'shared_dask': False})
        write_yaml(output[0], cfg)

use rule analysis_processor from analysis as M3_hists with:
    input:
        runner_script = "runner.py",
        config_file = M3_HIST_CONFIG,
        published = M2_PUBLISHED
    output: f"{M3_OUT}singlefiles/hist__{MIX_NAME}__{{year}}.coffea"
    log: f"{M3_OUT}logs/hists__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = MIX_NAME,
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

use rule merging_coffea_files from analysis as M3_merge_hists with:
    input:
        files = [UPSTREAM_HISTS] + expand(f"{M3_OUT}singlefiles/hist__{MIX_NAME}__{{year}}.coffea", year=YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: M3_HISTALL
    log: f"{M3_OUT}logs/merge_hists.log"
    params:
        run_performance = False,
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON,
        input_files = lambda wildcards, input: " ".join(input.files)

use rule check_cutflow from analysis as M3_cutflow with:
    input:
        coffea_file = M3_HISTALL
    output:
        validation_txt = f"{M3_OUT}cutflow_validation_mixedJCM.txt",
        cutflow_yml = f"{M3_OUT}cutflow_mixedJCM.yml"
    log: f"{M3_OUT}logs/cutflow_mixedJCM.log"
    params:
        known_flag = lambda wildcards: (f'--known-cutflow "{MJ["known_counts"]}"'
                                        if MJ.get('known_counts') and os.path.exists(MJ['known_counts'])
                                        else '--known-cutflow "none"'),
        error_threshold = lambda wildcards: config.get("error_threshold", "0.001"),
        cutflow_list = lambda wildcards: config.get("cutflow_list", "passJetMult,passPreSel,passDiJetMass,SR,SB"),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON
    container: None

rule M3_jcm_config:
    input: MJ.get('config', "coffea4bees/analysis/jcm_tools/metadata/mixeddata_all_config_Run3.yml")
    output: f"{M3_OUT}jcm_config_mixed.yml"
    run:
        with open(input[0]) as f:
            cfg = yaml.safe_load(f) or {}
        cfg['data3bName'] = MIX_NAME        # the mixed data stands in for the 3b sample
        cfg['float_t'] = bool(MJ.get('float_t', True))
        write_yaml(output[0], cfg)

rule M3_fit:
    input:
        hists = M3_HISTALL,
        jcm_config = f"{M3_OUT}jcm_config_mixed.yml"
    output: MIXED_JCM
    log: f"{M3_OUT}logs/fit.log"
    shell:
        """
        set -eo pipefail
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR {M3_JCM_DIR}
        {WRAPPER} {PYTHON} coffea4bees/analysis/jcm_tools/make_jcm_weights.py -o {M3_JCM_DIR} \
            -i {input.hists} -r SB -w {M3_JCM_TAG} --jcm_config {input.jcm_config} 2>&1 | tee {log}
        ls {M3_JCM_DIR} 2>&1 | tee -a {log}
        """

rule M3_study_config:
    input:
        template = MJ.get('study_template', "coffea4bees/analysis/metadata/study_mixed_data_Run3.yml"),
        jcm = MIXED_JCM
    output: M3_STUDY_CONFIG
    run:
        with open(input.template) as f:
            tmpl = yaml.safe_load(f) or {}
        # this roast's mixed-data JCM, not the template's hard-coded *_splitting.txt
        cfg = processor_config({**(tmpl.get('config') or {}), 'apply_JCM': True, 'JCM_file': input.jcm},
                               inherit_config=False,
                               processor="coffea4bees/analysis/processors/processor_study_mixed_data.py",
                               dataset_location=[MIXED_URL],
                               runner=tmpl.get('runner') or {})
        write_yaml(output[0], cfg)

use rule analysis_processor from analysis as M3_study with:
    input:
        runner_script = "runner.py",
        config_file = M3_STUDY_CONFIG,
        published = M2_PUBLISHED
    output: f"{M3_OUT}study/study__{{year}}.coffea"
    log: f"{M3_OUT}logs/study__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = MIX_NAME,
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

use rule merging_coffea_files from analysis as M3_merge_study with:
    input:
        files = expand(f"{M3_OUT}study/study__{{year}}.coffea", year=YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: M3_STUDY
    log: f"{M3_OUT}logs/merge_study.log"
    params:
        run_performance = False,
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON,
        input_files = lambda wildcards, input: " ".join(input.files)

rule M3_publish:
    input: MIXED_JCM
    output: M3_PUBLISHED
    log: f"{M3_OUT}logs/publish.log"
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        xrdcp -f -p {input} "{HANDOFF}/$(basename {input})" 2>&1 | tee {log}
        echo "published {input} -> {HANDOFF}/$(basename {input})" | tee -a {log}
        date > {output}
        """

rule all_M3:
    input:
        M3_PUBLISHED,
        M3_STUDY,
        f"{M3_OUT}cutflow_validation_mixedJCM.txt"

localrules: M3_hist_config, M3_merge_hists, M3_cutflow, M3_jcm_config, M3_fit, M3_study_config,
            M3_merge_study, M3_publish, all_M3
