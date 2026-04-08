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
config.setdefault('dataset_location',   "coffea4bees/metadata/datasets_HH4b_Run3/")
config.setdefault('years', ['2022_EE', '2022_preEE', '2023_BPix', '2023_preBPix'])

FEYNNET_OUT = "output/Run3_FeynNet/feynnet_friendtrees/"

TT_DATASETS = ['TTTo2L2Nu', 'TTToHadronic', 'TTToSemiLeptonic']

module analysis:
    snakefile: "rules/analysis.smk"
    config: config


rule all:
    input:
        "coffea4bees/metadata/datasets_HH4b_Run3/SvBFeynNetfriend_mixeddata_data.json"


rule install_SvBFeynNet_friend_json:
    input:  f"{FEYNNET_OUT}SvBFeynNetfriend_mixeddata_data.json"
    output: "coffea4bees/metadata/datasets_HH4b_Run3/SvBFeynNetfriend_mixeddata_data.json"
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
        friends               = "coffea4bees/metadata/friends_HH4b.yml",
        run_on_condor         = True,
        extra_arguments       = "",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0


use rule analysis_processor from analysis as make_SvBFeynNet_friendtrees_mixeddata with:
    input: "coffea4bees/analysis/metadata/HH4b_make_friend_SvBFeynNet_Run3.yml"
    output: f"{FEYNNET_OUT}SvBFeynNet_mixeddata_all__{{year}}.coffea"
    log: f"{FEYNNET_OUT}logs/SvBFeynNet_mixeddata_all__{{year}}.log"
    params:
        datasets              = "mixeddata_all",
        years                 = "{year}",
        config                = lambda wildcards, input: input[0],
        processor             = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        friends               = "coffea4bees/metadata/friends_HH4b.yml",
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
        friends               = "coffea4bees/metadata/friends_HH4b.yml",
        run_on_condor         = True,
        extra_arguments       = "",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0


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
            f"{FEYNNET_OUT}SvBFeynNet_mixeddata_all__{{year}}.coffea",
            year=config['years']
        ),
        ttbar_coffea = expand(
            f"{FEYNNET_OUT}SvBFeynNet_{{tt_dataset}}__{{year}}.coffea",
            tt_dataset=TT_DATASETS,
            year=config['years']
        ),
    output: f"{FEYNNET_OUT}SvBFeynNetfriend_mixeddata_data.json"
    container: config['analysis_container']
    log: f"{FEYNNET_OUT}logs/merge_SvBFeynNet_friendtrees.log"
    params:
        all_jsons = lambda wildcards, input: [
            f.replace(".coffea", ".json")
            for f in list(input.data_coffea) + list(input.mixeddata_coffea) + list(input.ttbar_coffea)
        ]
    shell:
        """
        PYTHONPATH=. python src/friendtrees/merge_friend_meta.py \
            -i {params.all_jsons} \
            -o {output} \
            2>&1 | tee -a {log}
        """
