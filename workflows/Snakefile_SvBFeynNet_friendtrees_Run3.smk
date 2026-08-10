"""SvB_FeynNet friend tree creation for data, mixeddata_all, and TTBar MC (Run 3).

FeynNet is evaluated externally (ONNX) by a separate group; we just run
inference here.  These friend trees are fully independent of the SvB/SvB_MA
friend trees produced by Snakefile_SvB_friendtrees_Run3.smk.

Usage:
    snakemake --profile software/snakemake/profiles/lpc \\
        --snakefile coffea4bees/workflows/Snakefile_SvBFeynNet_friendtrees_Run3.smk \\
        --cores 4
"""

config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset_location',   "coffea4bees/metadata/datasets/")
config.setdefault('years', ['2022_EE', '2022_preEE', '2023_BPix', '2023_preBPix'])
config.setdefault('dataset_name', 'mixeddata_all')
# Path to the installed dataset metadata yaml — declared as input to
# make_SvBFeynNet_friendtrees_mixeddata so snakemake schedules the install
# step (when invoked from Snakefile_Run3_make_mixeddata.smk) before this.
config.setdefault('install_path',
    f"coffea4bees/metadata/datasets/{config['dataset_name']}.yml")

# When True, skip the data, ttbar, and HH per-year jobs and reuse the legacy
# combined SvB_FeynNet JSON (which already contains all four sources) as a
# merge input. Only the new mixed-data per-year jobs run. Reasonable for
# rank-suffixed runs where data/TT/HH friends are unchanged from the legacy
# build. Default False preserves backward-compatible standalone behavior.
config.setdefault('reuse_legacy_friends', False)
LEGACY_FEYNET_JSON = "coffea4bees/metadata/friends/SvBFeynNetfriend_mixeddata_data.json"

# Rank/variant suffix derived from dataset_name. Empty for legacy
# 'mixeddata_all' so historical install paths stay byte-for-byte unchanged.
_dsn_tag     = "" if config['dataset_name'] == "mixeddata_all" \
               else config['dataset_name'].removeprefix("mixeddata_all_") or config['dataset_name']
_install_tag = f"_{_dsn_tag}" if _dsn_tag else ""

FEYNNET_OUT = f"output/Run3_FeynNet{_install_tag}/feynnet_friendtrees/"
INSTALL     = f"coffea4bees/metadata/friends/SvBFeynNetfriend_mixeddata_data{_install_tag}.json"

TT_DATASETS = ['TTTo2L2Nu', 'TTToHadronic', 'TTToSemiLeptonic']
# Only this HH coupling point currently has 2022/2023 picoAODs.
HH_DATASETS = ['GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00']

module analysis:
    snakefile: "rules/analysis.smk"
    config: config


rule all:
    input: INSTALL


rule install_SvBFeynNet_friend_json:
    input:  f"{FEYNNET_OUT}SvBFeynNetfriend_mixeddata_data.json"
    output: INSTALL
    shell:  "cp {input} {output}"


use rule analysis_processor from analysis as make_SvBFeynNet_friendtrees_data with:
    input: "coffea4bees/analysis/metadata/HH4b_make_friend_SvBFeynNet_Run3.yml"
    output: f"{FEYNNET_OUT}SvBFeynNet_data__{{year}}.coffea"
    log: f"{FEYNNET_OUT}logs/SvBFeynNet_data__{{year}}.log"
    params:
        datasets              = "data",
        years                 = "{year}",
        config                = lambda wildcards, input: input[0],
        processor             = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        friends               = config.get("friend_gen_friends", "coffea4bees/metadata/friends/friends_HH4b.yml"),
        run_on_condor         = True,
        extra_arguments       = "",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0


use rule analysis_processor from analysis as make_SvBFeynNet_friendtrees_mixeddata with:
    # install_path declared so this rule waits for install_mixeddata_dataset
    # when invoked from Snakefile_Run3_make_mixeddata.smk.
    input:
        config_yml   = "coffea4bees/analysis/metadata/HH4b_make_friend_SvBFeynNet_Run3.yml",
        install_path = config['install_path'],
    output: f"{FEYNNET_OUT}SvBFeynNet_{config['dataset_name']}__{{year}}.coffea"
    log: f"{FEYNNET_OUT}logs/SvBFeynNet_{config['dataset_name']}__{{year}}.log"
    params:
        datasets              = config['dataset_name'],
        years                 = "{year}",
        config                = lambda wildcards, input: input.config_yml,
        processor             = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        friends               = config.get("friend_gen_friends", "coffea4bees/metadata/friends/friends_HH4b.yml"),
        run_on_condor         = True,
        extra_arguments       = "",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0


use rule analysis_processor from analysis as make_SvBFeynNet_friendtrees_ttbar with:
    input: "coffea4bees/analysis/metadata/HH4b_make_friend_SvBFeynNet_Run3.yml"
    output: f"{FEYNNET_OUT}SvBFeynNet_{{tt_dataset}}__{{year}}.coffea"
    log: f"{FEYNNET_OUT}logs/SvBFeynNet_{{tt_dataset}}__{{year}}.log"
    wildcard_constraints:
        tt_dataset = "|".join(TT_DATASETS)
    params:
        datasets              = "{tt_dataset}",
        years                 = "{year}",
        config                = lambda wildcards, input: input[0],
        processor             = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        friends               = config.get("friend_gen_friends", "coffea4bees/metadata/friends/friends_HH4b.yml"),
        run_on_condor         = True,
        extra_arguments       = "",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0


use rule analysis_processor from analysis as make_SvBFeynNet_friendtrees_HH with:
    input: "coffea4bees/analysis/metadata/HH4b_make_friend_SvBFeynNet_Run3.yml"
    output: f"{FEYNNET_OUT}SvBFeynNet_{{hh_dataset}}__{{year}}.coffea"
    log: f"{FEYNNET_OUT}logs/SvBFeynNet_{{hh_dataset}}__{{year}}.log"
    wildcard_constraints:
        hh_dataset = "|".join(HH_DATASETS)
    params:
        datasets              = "{hh_dataset}",
        years                 = "{year}",
        config                = lambda wildcards, input: input[0],
        processor             = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        friends               = config.get("friend_gen_friends", "coffea4bees/metadata/friends/friends_HH4b.yml"),
        run_on_condor         = True,
        extra_arguments       = "",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0


if config['reuse_legacy_friends']:
    rule merge_SvBFeynNet_friendtrees:
        """Merge new mixed-data SvB_FeynNet metafiles with the legacy combined JSON.

        Skips re-making data, ttbar, and HH friend trees (unchanged across
        ranks). The legacy JSON already contains all four sources; we layer
        the new dataset_name keys on top via merge_friend_meta.py.
        """
        input:
            mixeddata_coffea = expand(
                f"{FEYNNET_OUT}SvBFeynNet_{{dataset_name}}__{{year}}.coffea",
                dataset_name=config['dataset_name'],
                year=config['years'],
            ),
            legacy_json = LEGACY_FEYNET_JSON,
        output: f"{FEYNNET_OUT}SvBFeynNetfriend_mixeddata_data.json"
        container: config['analysis_container']
        log: f"{FEYNNET_OUT}logs/merge_SvBFeynNet_friendtrees.log"
        params:
            all_jsons = lambda wildcards, input: [
                f.replace(".coffea", ".json")
                for f in list(input.mixeddata_coffea)
            ]
        shell:
            """
            ./run_container env PYTHONPATH=. python src/friendtrees/merge_friend_meta.py \
                -i {params.all_jsons} {input.legacy_json} \
                -o {output} \
                2>&1 | tee -a {log}
            """

else:
    rule merge_SvBFeynNet_friendtrees:
        """Merge per-year data, mixeddata_all, and TTBar SvB_FeynNet metafiles into one JSON.

        runner.py writes {output_path}/{output_file}.json alongside each coffea output,
        so per-year metafiles are the coffea paths with .coffea → .json.
        merge_friend_meta.py merges by key (SvB_FeynNet) using Friend.__add__.
        """
        input:
            data_coffea = expand(
                f"{FEYNNET_OUT}SvBFeynNet_data__{{year}}.coffea",
                year=config['years']
            ),
            mixeddata_coffea = expand(
                f"{FEYNNET_OUT}SvBFeynNet_{{dataset_name}}__{{year}}.coffea",
                dataset_name=config['dataset_name'],
                year=config['years'],
            ),
            ttbar_coffea = expand(
                f"{FEYNNET_OUT}SvBFeynNet_{{tt_dataset}}__{{year}}.coffea",
                tt_dataset=TT_DATASETS,
                year=config['years']
            ),
            hh_coffea = expand(
                f"{FEYNNET_OUT}SvBFeynNet_{{hh_dataset}}__{{year}}.coffea",
                hh_dataset=HH_DATASETS,
                year=config['years']
            ),
        output: f"{FEYNNET_OUT}SvBFeynNetfriend_mixeddata_data.json"
        container: config['analysis_container']
        log: f"{FEYNNET_OUT}logs/merge_SvBFeynNet_friendtrees.log"
        params:
            all_jsons = lambda wildcards, input: [
                f.replace(".coffea", ".json")
                for f in list(input.data_coffea) + list(input.mixeddata_coffea) + list(input.ttbar_coffea) + list(input.hh_coffea)
            ]
        shell:
            """
            ./run_container env PYTHONPATH=. python src/friendtrees/merge_friend_meta.py \
                -i {params.all_jsons} \
                -o {output} \
                2>&1 | tee -a {log}
            """
