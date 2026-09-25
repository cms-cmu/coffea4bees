# coffea4bees/workflows/Snakefile_MakeMixedData_5_ttbar_psdata.smk
# M.5: ttbar pseudodata -- one unweighted sample, shared by every subsample. sub_sample_MC.py
# (SubSampler) keeps a ttbar MC event when a per-event uniform number (fixed seed) is below its
# weight (MC weights incl. trigger weight, x b-tag SF), so the picoAODs are data-like. Independent
# of M.1-M.4.
#
#   M5_config                         skimmer config (as Phase E.3's, plus the trigger-weight friend)
#   M5_psdata (per year, condor)      picoAOD_PSData -> <PUB>/picoAOD/ttbar_PSData/, registry
#   M5_merge                          -> one registry
#   M5_dataset_yml                    -> dataset YAML keyed `ttbar_PSData`
#   M5_publish                        -> <PUB>/handoff/ttbar_PSData.yml

M5_OUT = f"{out}M5/"
PS = config.get('ttbar_psdata') or {}
PS_NAME = PS.get('dataset_name', 'ttbar_PSData')
M5_CONFIG = f"{M5_OUT}sub_sample_MC.yml"
M5_REGISTRY = f"{M5_OUT}picoaod_datasets_{PS_NAME}.yml"
M5_DATASET = f"{M5_OUT}handoff/{PS_NAME}.yml"
M5_PUBLISHED = f"{M5_OUT}published.done"

rule M5_config:
    output: M5_CONFIG
    run:
        # Phase E.3's SubSampler config (the only one there is), with this roast's values.
        step = int(PS.get('chunksize', 100000))
        runner = {"workers": 4, "min_workers": 1, "max_workers": 200,
                  "chunksize": step, "picosize": int(PS.get('picosize', 100000)), "basketsize": 10000,
                  "class_name": "SubSampler", "data_tier": "picoAOD", "condor_cores": 1,
                  "worker_memory": PS.get('worker_memory', "6GB"),
                  "condor_transfer_input_files": ["src", "coffea4bees/"],
                  "allowlist_sites": ["T2_US_Nebraska", "T2_US_Purdue", "T3_US_FNALLPC", "T3_US_NotreDame"]}
        section = {
            "base_path": f"{PUB}/picoAOD/{PS_NAME}",
            "sub_sampling_rand_seed": int(PS.get('seed', 5)),
            "apply_trigWeight": True,
            # SubSampler now takes `friends`, so runner.py injects the trigWeight friend; a chunk
            # without a trigger weight is an error, not a worker-log warning (sub_sample_MC.py).
            "require_trigWeight": bool(PS.get('require_trigWeight', True)),
            "friends_include": ["trigWeight"],
            "step": step,
            "skip_collections": ["notCanJet", "canJet0", "canJet1", "canJet2", "canJet3", "trigWeight"],
            "skip_branches": ["btagWeight_.*", "HHSR", "ZZSR", "ZHSR", "SR", "st", "d01TruthMatch",
                              "nMuon_selected", "fourTag", "threeTag", "xW", "leadStM", "nSelJets",
                              "d02TruthMatch", "d12TruthMatch", "truthMatch", "pseudoTagWeight",
                              "ttbarWeight", "nIsoMuons", "xt", "weight", "aveAbsEtaOth", "xWbW",
                              "nAllNotCanJets", "dRjjOther", "xWt", "nPSTJets", "passXWt", "d03TruthMatch",
                              "xbW", "mcPseudoTagWeight", "d23TruthMatch", "m4j", "sublStM", "dRjjClose",
                              "aveAbsEta", "stNotCan", "SB", "selectedViewTruthMatch", "d13TruthMatch",
                              "genWeight"],
        }
        cfg = processor_config(section, inherit_config=False,
                               processor="coffea4bees/skimmer/processor/sub_sample_MC.py",
                               runner=runner)
        write_yaml(output[0], cfg)

use rule analysis_processor from analysis as M5_psdata with:
    input:
        runner_script = "runner.py",
        config_file = M5_CONFIG
    output: f"{M5_OUT}per_year/picoaod_datasets_{PS_NAME}__{{year}}.yml"
    log: f"{M5_OUT}logs/psdata__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = " ".join(TTBAR),
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, ["-s", TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

rule M5_merge:
    input: expand(f"{M5_OUT}per_year/picoaod_datasets_{PS_NAME}__{{year}}.yml", year=YEARS)
    output: M5_REGISTRY
    log: f"{M5_OUT}logs/merge.log"
    shell:
        """
        {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/merge_mixeddata_registries.py \
            {input} {output} 2>&1 | tee {log}
        """

rule M5_dataset_yml:
    input: M5_REGISTRY
    output: M5_DATASET
    log: f"{M5_OUT}logs/dataset_yml.log"
    shell:
        """
        mkdir -p $(dirname {output})
        {WRAPPER} {PYTHON} src/tools/make_dataset_yml.py -i {input} -o {output} -n {PS_NAME} 2>&1 | tee {log}
        """

rule M5_publish:
    input: M5_DATASET
    output: M5_PUBLISHED
    log: f"{M5_OUT}logs/publish.log"
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        xrdcp -f -p {input} "{HANDOFF}/$(basename {input})" 2>&1 | tee {log}
        echo "published {input} -> {HANDOFF}/$(basename {input})" | tee -a {log}
        date > {output}
        """

rule all_M5:
    input: M5_PUBLISHED

localrules: M5_config, M5_merge, M5_dataset_yml, M5_publish, all_M5
