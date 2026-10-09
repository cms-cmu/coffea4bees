# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_A_2_make_subsamples.smk
#
# Stage A_2: the N closure samples (+ ttbar pseudodata) -> one multi-sample dataset
# ==============================================================================
#
# subsamples.source: split (3b mixing) -- split mixeddata_all with the A_1 (analysis) mixed-data JCM:
#
# split_mixed_data.py gives each mixed 4b event a pseudo-tag weight w from the A_1 JCM and puts it
# in subsample v when one per-event uniform number lies in [v*w, (v+1)*w)
# (mixing_helpers.assign_mixed_subsamples): N unweighted, disjoint samples as long as N*w <= 1.
# Events with N*w > 1 wrap to slice (event+v) mod floor(1/w) and are shared -- the study counts them.
#
#   A2_split_config (per v)           skimmer config (subsamples.split_template + this run's values)
#   A2_split (per v, condor)          picoAODs -> <publish_base>/picoAOD/subsamples/v<v>/, registry
#   A2_clean (per v)                  registry -> plain YAML (runner leaves numpy tags in it)
#   A2_dataset_yml                    -> one multi-sample dataset `mixeddata_4b`, nSamples: N; each
#                                        sample mix_v<k> = subsample k + the ttbar pseudodata
#                                        (helpers/multisample_dataset.py, shared with MakeMixedData M.4)
#   A2_study_config / A2_study (per year) / A2_merge_study
#                                     processor_study_mixed_data with the A_1 JCM: per-event subsample
#                                     assignment, pseudo-tag weights, overflow (N*w > 1) counts
#   A2_study_report                   study plots + subsample overlap matrix + index.html
#   A2_classifier_metadata            the repo's dataset YAMLs (minus any defining the multi-sample
#                                     name) + this dataset -> the --metadata directory C reads
#
# subsamples.source: seeds (4b mixing) -- the N seeds of a MakeMixedData 4b-mixing roast's
# mixeddata_all_<tag> (inputs.mixeddata_all; seed s under .../v<s>/) are already unit-weight 4b
# samples: no JCM, no split, no study (the roast's M.6 seed study covers them).
#   A2_dataset_yml                    seeds regrouped + ttbar pseudodata -> e.g. mixeddata_4bmix_4b
#
# OUTPUT: {out_a2}<subsamples.dataset_name>.yml (read by A_3, B_1); split: {out_a2}study/index.html
# ==============================================================================

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"
if "MIXED_JCM" not in globals():           # standalone: A_1 provides the JCM
    include: "Snakefile_bkg_syst_A_1_mixed_jcm.smk"

from helpers.multisample_dataset import (build_multisample_dataset, registry_files, seed_files,
                                         template_seed_file, template_split_file)

A2_STUDY = f"{out_a2}study_{MIX_NAME}.coffea"

def _a2_runner_config(template, section, processor):
    """A runner config: the template's runner + the analysis runner config's friend / weights
    files, `section` as the processor config, reading mixeddata_all from the roast's handoff."""
    with open(JCM_HIST_CONFIG) as f:
        ana = yaml.safe_load(f) or {}
    cfg = {k: ana[k] for k in ('friend_file', 'weights_file') if k in ana}
    cfg.update({'processor': processor, 'dataset_location': [MIXED_URL],
                'runner': dict(template.get('runner') or {}), 'config': section})
    # the split was OOM-killed at the template's 4 GB (and the shared Dask daemon keeps the memory
    # of the job that started it): every Stage A runner config asks for the same
    cfg['runner']['worker_memory'] = config.get('worker_memory', "8GB")
    if config.get('test', False):
        cfg['runner'].update({'condor': False, 'shared_dask': False})
    return cfg

if SUB_SOURCE == 'split':
    rule A2_split_config:
        input:
            template = SUB.get('split_template', "coffea4bees/skimmer/metadata/split_mixeddata.yml"),
            jcm = MIXED_JCM,
            hist_config = JCM_HIST_CONFIG
        output: f"{out_a2}configs/split_v{{v}}.yml"
        wildcard_constraints:
            v = r"\d+"
        run:
            with open(input.template) as f:
                tmpl = yaml.safe_load(f) or {}
            section = {**(tmpl.get('config') or {}),
                       'base_path': f"{PUB}/picoAOD/subsamples/v{wildcards.v}",
                       'apply_JCM': True,
                       'JCM_file': input.jcm,          # not the template's *_splitting.txt
                       'mixed_subsample': int(wildcards.v),
                       'n_subsamples': N_SUBSAMPLES}
            cfg = _a2_runner_config(tmpl, section, "coffea4bees/skimmer/processor/split_mixed_data.py")
            # ONE output file per (subsample, era): the skim's final merge cuts each dataset into
            # `picosize`-event files, so with the template's 100k the larger subsamples got more chunks
            # than the smaller and no single vXXX template fits all of them.
            cfg['runner']['picosize'] = int(SUB.get('picosize', 10**9))
            cfg['runner'].update(SUB.get('runner') or {})      # subsamples.runner overrides (as M.4 had)
            write_yaml(output[0], cfg)

    use rule analysis_processor from analysis as A2_split with:
        input:
            runner_script = "runner.py",
            config_file = f"{out_a2}configs/split_v{{v}}.yml"
        output: f"{out_a2}per_subsample/picoaod_datasets_{MIX_NAME}_v{{v}}.yml"
        log: f"{out_a2}logs/split_v{{v}}.log"
        wildcard_constraints:
            v = r"\d+"
        params:
            datasets = MIX_NAME,
            years = " ".join(YEARS),
            config = lambda wildcards, input: input.config_file,
            extra_arguments = " ".join(filter(None, ["-s", TEST_FLAG, condor_flags])),
            run_container_wrapper = container_wrapper,
            python_bin = python_bin

    rule A2_clean:
        input: f"{out_a2}per_subsample/picoaod_datasets_{MIX_NAME}_v{{v}}.yml"
        output: f"{out_a2}per_subsample/clean_v{{v}}.yml"
        log: f"{out_a2}logs/clean_v{{v}}.log"
        wildcard_constraints:
            v = r"\d+"
        shell:
            """
            {container_wrapper} {python_bin} coffea4bees/workflows/scripts/merge_mixeddata_registries.py \
                {input} {output} 2>&1 | tee {log}
            """

    rule A2_dataset_yml:
        input:
            subsamples = expand(f"{out_a2}per_subsample/clean_v{{v}}.yml", v=SUBSAMPLES),
            psdata = PS_DATASET
        output: MULTISAMPLE_DATASET
        run:
            write_yaml(output[0], build_multisample_dataset([registry_files(p) for p in input.subsamples],
                                                            input.psdata, PS_NAME, YEARS, SUB_NAME,
                                                            template_split_file))

    rule A2_study_config:
        input:
            template = SUB.get('study_template', "coffea4bees/analysis/metadata/study_mixed_data.yml"),
            jcm = MIXED_JCM,
            hist_config = JCM_HIST_CONFIG
        output: f"{out_a2}configs/study_mixed_data.yml"
        run:
            with open(input.template) as f:
                tmpl = yaml.safe_load(f) or {}
            # this run's mixed-data JCM, not the template's hard-coded *_splitting.txt
            section = {**(tmpl.get('config') or {}), 'apply_JCM': True, 'JCM_file': input.jcm}
            write_yaml(output[0], _a2_runner_config(
                tmpl, section, "coffea4bees/analysis/processors/processor_study_mixed_data.py"))

    use rule analysis_processor from analysis as A2_study with:
        input:
            runner_script = "runner.py",
            config_file = f"{out_a2}configs/study_mixed_data.yml"
        output: f"{out_a2}study/study__{{year}}.coffea"
        log: f"{out_a2}logs/study__{{year}}.log"
        wildcard_constraints:
            year = "|".join(YEARS)
        params:
            datasets = MIX_NAME,
            years = lambda wildcards: wildcards.year,
            config = lambda wildcards, input: input.config_file,
            extra_arguments = " ".join(filter(None, [TEST_FLAG, condor_flags])),
            run_container_wrapper = container_wrapper,
            python_bin = python_bin

    use rule merging_coffea_files from analysis as A2_merge_study with:
        input:
            files = expand(f"{out_a2}study/study__{{year}}.coffea", year=YEARS),
            script = "src/tools/merge_coffea_files.py"
        output: A2_STUDY
        log: f"{out_a2}logs/merge_study.log"
        params:
            run_performance = False,
            run_container_wrapper = container_wrapper,
            python_bin = python_bin,
            input_files = lambda wildcards, input: " ".join(input.files)

    rule A2_study_report:
        input: A2_STUDY
        output:
            matrix = f"{out_a2}study/subsample_overlap_matrix.png",
            summary = f"{out_a2}study/summary.yml",
            index = f"{out_a2}study/index.html"
        log: f"{out_a2}logs/study_report.log"
        shell:
            """
            {container_wrapper} {python_bin} coffea4bees/workflows/scripts/mixeddata_validation_report.py study \
                {input} {out_a2}study --n-subsamples {N_SUBSAMPLES} 2>&1 | tee {log}
            """

else:
    rule A2_dataset_yml:
        input:
            mixed = MIXED_DATASET,
            psdata = PS_DATASET
        output: MULTISAMPLE_DATASET
        run:
            write_yaml(output[0], build_multisample_dataset(seed_files(input.mixed, MIX_NAME, N_SUBSAMPLES),
                                                            input.psdata, PS_NAME, YEARS, SUB_NAME,
                                                            template_seed_file(MIX_NAME)))

rule A2_classifier_metadata:
    """C's classifier merges every YAML of its --metadata directory and looks the mixed samples up by
    name: the repo's directory defines `mixeddata_4b` twice (and with other productions' files), so
    C gets a directory where this run's dataset is the only one with that name."""
    input:
        dataset = MULTISAMPLE_DATASET,
        repo = config.get('dataset_location', "coffea4bees/metadata/datasets/"),
    output: directory(CLASSIFIER_METADATA)
    run:
        from helpers.stage_configs import build_classifier_metadata
        skipped = build_classifier_metadata(input.repo, input.dataset, output[0], SUB_NAME)
        print(f"classifier metadata: {output[0]} (skipped {skipped}: they define {SUB_NAME})")

rule all_bkg_syst_A_2:
    input:
        MULTISAMPLE_DATASET,
        CLASSIFIER_METADATA,
        [f"{out_a2}study/index.html"] if SUB_SOURCE == 'split' else []

localrules: A2_dataset_yml, A2_classifier_metadata, all_bkg_syst_A_2
if SUB_SOURCE == 'split':
    localrules: A2_split_config, A2_clean, A2_study_config, A2_merge_study, A2_study_report
