# coffea4bees/workflows/Snakefile_MvD_1_inputs.smk
# V.1 (cmslpc): everything the falcon steps read, published to <PUB>/handoff/.
#
#   V1_ci_config                       the nominal Phase B.2 classifier-input config, pointed at this
#                                      roast's mixeddata_all and EOS area (same selection, same JCM)
#   V1_ci (per year, condor)           mixeddata_all classifier-input friend trees -> <PUB>/classifier_inputs/
#   V1_ci_merge                        + the nominal manifest (data, ttbar, signal) -> one HCR_input manifest
#   V1_metadata                        classifier dataset metadata: the committed data / ttbar / signal
#                                      YAMLs + THIS roast's mixeddata_all (not the committed legacy one)
#   V1_hist_config                     the nominal B.1 noJCM runner config, pointed at mixeddata_all
#   V1_hists (per year, condor)        processor_HH4b over mixeddata_all (tight four-tag)
#   V1_merge_hists                     + the nominal data / ttbar histAll_NoJCM
#   V1_cutflow                         cutflow dump (first roast: the reference to bless)
#   V1_fit                             make_jcm_weights.py, mixeddata_all as the "3b" sample, float_t
#   V1_publish                         manifest, metadata, JCM -> <PUB>/handoff/

V1_OUT = f"{out}V1/"
V1_CI_CONFIG = f"{V1_OUT}classifier_inputs_config.yml"
V1_CI_DIR = f"{V1_OUT}classifier_inputs/"
V1_MANIFEST = f"{V1_OUT}handoff/{MANIFEST_NAME}"
V1_METADATA = f"{V1_OUT}handoff/{METADATA_NAME}"
V1_HIST_CONFIG = f"{V1_OUT}analysis_config_mixed_noJCM.yml"
V1_HISTALL = f"{V1_OUT}histAll_mixed_noJCM.coffea"
V1_JCM_DIR = f"{V1_OUT}JCM_{JCM_TAG}/"
MIXED_JCM = f"{V1_JCM_DIR}{JCM_NAME}"
V1_PUBLISHED = f"{V1_OUT}published.done"
V1_METADATA_FILES = config.get('classifier_metadata_files',
                               ["coffea4bees/metadata/datasets/data.yml",
                                "coffea4bees/metadata/datasets/TT.yml",
                                "coffea4bees/metadata/datasets/GluGluToHHTo4B.yml"])

rule V1_ci_config:
    input:
        upstream = UPSTREAM['ci_config'][1],
        jcm = UPSTREAM['jcm'][1]
    output: V1_CI_CONFIG
    run:
        cfg = load_yaml(input.upstream)
        require_tight(cfg, f"the upstream classifier-input config ({INPUTS['classifier_inputs_config']})")
        cfg.pop('datasets', None)
        cfg['dataset_location'] = [MIXED_URL]
        c = cfg.setdefault('config', {})
        # Same JCM as the upstream data/ttbar inputs. (The friend trees store weight_noJCM_noFvT,
        # so it does not enter the classifier inputs -- the trainings apply the JCM themselves --
        # but the processor needs a real local file whenever apply_JCM is on.)
        c['JCM_file'] = input.jcm
        c['make_classifier_input'] = f"{PUB}/classifier_inputs/"
        write_yaml(output[0], test_runner(cfg))

rule V1_ci:
    input:
        runner_script = "runner.py",
        config_file = V1_CI_CONFIG
    output: f"{V1_CI_DIR}classifier_inputs_dataset_{MIX_NAME}__{{year}}.json"
    log: f"{V1_OUT}logs/classifier_inputs__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR]))
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        mkdir -p {V1_CI_DIR} $(dirname {log})
        {WRAPPER} {PYTHON} runner.py {input.config_file} \
            --datasets {MIX_NAME} \
            --years {wildcards.year} \
            --output-path {V1_CI_DIR} \
            --output $(basename {output} .json).coffea \
            {params.extra_arguments} 2>&1 | tee {log}
        """

rule V1_ci_merge:
    input:
        nominal = UPSTREAM['manifest'][1],
        mixed = expand(f"{V1_CI_DIR}classifier_inputs_dataset_{MIX_NAME}__{{year}}.json", year=YEARS)
    output: V1_MANIFEST
    log: f"{V1_OUT}logs/ci_merge.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output})
        {WRAPPER} {PYTHON} -m src.friendtrees.merge_friend_meta \
            -i {input.mixed} {input.nominal} -o {output} 2>&1 | tee {log}
        """

rule V1_metadata:
    input:
        committed = V1_METADATA_FILES,
        mixed = UPSTREAM['mixed'][1]
    output: V1_METADATA
    run:
        # The layout `--metadata <dir>/` produces (_picoAOD._metadata_arg: every YAML merged by top-
        # level key, later files winning), as ONE file: --metadata <url-without-.yml> reads
        # <url>.yml@@datasets. The committed metadata/datasets/ directory cannot be used as is:
        # its mixeddata_all.yml is the legacy sample, which would silently replace this roast's.
        merged = {}
        for path in input.committed:
            merged.update(load_yaml(path))
        mixed = load_yaml(input.mixed)
        if set(mixed) != {MIX_NAME}:
            raise ValueError(f"{MIXED_URL}: expected one top-level key {MIX_NAME!r}, got {sorted(mixed)}")
        missing = [y for y in YEARS if not (mixed[MIX_NAME].get(y) or {}).get('picoAOD')]
        if missing:
            raise ValueError(f"{MIXED_URL}: no {MIX_NAME} picoAODs for {missing}")
        merged[MIX_NAME] = mixed[MIX_NAME]
        write_yaml(output[0], {'datasets': merged})

rule V1_hist_config:
    input: UPSTREAM['hist_config'][1]
    output: V1_HIST_CONFIG
    run:
        cfg = load_yaml(input[0])
        require_tight(cfg, f"the upstream B.1 histogram config (next to {INPUTS['jcm_hists']})")
        cfg['dataset_location'] = [MIXED_URL]
        cfg.get('runner', {}).pop('dataset_location', None)
        write_yaml(output[0], test_runner(cfg))

use rule analysis_processor from analysis as V1_hists with:
    input:
        runner_script = "runner.py",
        config_file = V1_HIST_CONFIG
    output: f"{V1_OUT}singlefiles/hist__{MIX_NAME}__{{year}}.coffea"
    log: f"{V1_OUT}logs/hists__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = MIX_NAME,
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

use rule merging_coffea_files from analysis as V1_merge_hists with:
    input:
        files = [UPSTREAM['hists'][1]] + expand(f"{V1_OUT}singlefiles/hist__{MIX_NAME}__{{year}}.coffea", year=YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: V1_HISTALL
    log: f"{V1_OUT}logs/merge_hists.log"
    params:
        run_performance = False,
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON,
        input_files = lambda wildcards, input: " ".join(input.files)

use rule check_cutflow from analysis as V1_cutflow with:
    input:
        coffea_file = V1_HISTALL
    output:
        validation_txt = f"{V1_OUT}cutflow_validation_mixed_noJCM.txt",
        cutflow_yml = f"{V1_OUT}cutflow_mixed_noJCM.yml"
    log: f"{V1_OUT}logs/cutflow_mixed_noJCM.log"
    params:
        known_flag = lambda wildcards: (f'--known-cutflow "{MJ["known_counts"]}"'
                                        if MJ.get('known_counts') and os.path.exists(MJ['known_counts'])
                                        else '--known-cutflow "none"'),
        error_threshold = lambda wildcards: config.get("error_threshold", "0.001"),
        cutflow_list = lambda wildcards: config.get("cutflow_list", "passJetMult,passPreSel,passDiJetMass,SR,SB"),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON
    container: None

rule V1_jcm_config:
    input: MJ.get('config', "coffea4bees/analysis/jcm_tools/metadata/mixeddata_all_config_Run3.yml")
    output: f"{V1_OUT}jcm_config_mixed.yml"
    run:
        cfg = load_yaml(input[0])
        cfg['data3bName'] = MIX_NAME        # the mixed data stands in for the 3b sample
        cfg['float_t'] = bool(MJ.get('float_t', True))
        if 'threeTightTagFraction' in MJ:   # the fit seed when float_t (else the template's)
            cfg['threeTightTagFraction'] = float(MJ['threeTightTagFraction'])
        write_yaml(output[0], cfg)

rule V1_fit:
    input:
        hists = V1_HISTALL,
        jcm_config = f"{V1_OUT}jcm_config_mixed.yml"
    output: MIXED_JCM
    log: f"{V1_OUT}logs/fit.log"
    shell:
        """
        set -eo pipefail
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR {V1_JCM_DIR}
        {WRAPPER} {PYTHON} coffea4bees/analysis/jcm_tools/make_jcm_weights.py -o {V1_JCM_DIR} \
            -i {input.hists} -r SB -w {JCM_TAG} --jcm_config {input.jcm_config} 2>&1 | tee {log}
        ls {V1_JCM_DIR} 2>&1 | tee -a {log}
        """

rule V1_publish:
    input:
        manifest = V1_MANIFEST,
        metadata = V1_METADATA,
        jcm = MIXED_JCM,
        cutflow = f"{V1_OUT}cutflow_mixed_noJCM.yml"
    output: touch(V1_PUBLISHED)
    log: f"{V1_OUT}logs/publish.log"
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        for f in {input.manifest} {input.metadata} {input.jcm}; do
            xrdcp -f -p $f {HANDOFF}/$(basename $f) 2>&1 | tee -a {log}
            echo "published {HANDOFF}/$(basename $f)" | tee -a {log}
        done
        """

rule all_V1:
    input: V1_PUBLISHED

localrules: V1_ci_config, V1_ci_merge, V1_metadata, V1_hist_config, V1_merge_hists, V1_cutflow,
            V1_jcm_config, V1_fit, V1_publish, all_V1
