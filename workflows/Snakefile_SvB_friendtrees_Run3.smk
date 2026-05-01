"""SvB friend tree creation for mixeddata_all (Run 3).

Mode-independent: SvB friend trees are the same regardless of analysis mode.
Run once; the installed JSON is shared by all modes of Snakefile_classifier_inputs_Run3MvD.smk.

Usage:
    snakemake --profile software/snakemake/profiles/lpc \\
        --snakefile coffea4bees/workflows/Snakefile_SvB_friendtrees_Run3.smk \\
        --cores 4
"""

config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset_location',   "coffea4bees/metadata/datasets_HH4b_Run3/")
config.setdefault('years', ['2022_EE', '2022_preEE', '2023_BPix', '2023_preBPix'])

SvB_OUT = "output/Run3_MvD/svb_friendtrees/"

# Only this HH coupling point currently has 2022/2023 picoAODs.
HH_DATASETS = ['GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00']

module analysis:
    snakefile: "rules/analysis.smk"
    config: config


rule all:
    input:
        "coffea4bees/metadata/datasets_HH4b_Run3/SvBfriend_mixeddata_data.json"


rule install_SvB_friend_json:
    input:  f"{SvB_OUT}SvBfriend_mixeddata_data.json"
    output: "coffea4bees/metadata/datasets_HH4b_Run3/SvBfriend_mixeddata_data.json"
    shell:  "cp {input} {output}"


use rule analysis_processor from analysis as make_SvB_friendtrees_mixeddata with:
    # runner.py writes a matching .json metafile alongside each .coffea output,
    # which merge_SvB_friendtrees_mixeddata picks up via the .coffea → .json swap.
    input: "coffea4bees/analysis/metadata/HH4b_make_friend_SvB_Run3.yml"
    output: f"{SvB_OUT}SvB_mixeddata_all__{{year}}.coffea"
    log: f"{SvB_OUT}logs/SvB_mixeddata_all__{{year}}.log"
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


use rule analysis_processor from analysis as make_SvB_friendtrees_HH with:
    input: "coffea4bees/analysis/metadata/HH4b_make_friend_SvB_Run3.yml"
    output: f"{SvB_OUT}SvB_{{hh_dataset}}__{{year}}.coffea"
    log: f"{SvB_OUT}logs/SvB_{{hh_dataset}}__{{year}}.log"
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
        friends               = "coffea4bees/metadata/friends_HH4b.yml",
        run_on_condor         = True,
        extra_arguments       = "",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0


rule merge_SvB_friendtrees_mixeddata:
    """Merge per-year mixeddata_all + HH SvB metafiles with the existing data SvB JSON.

    runner.py writes {output_path}/{output_file}.json alongside the coffea output,
    so the per-year metafiles are the coffea paths with .coffea → .json.
    merge_friend_meta.py merges by key (SvB, SvB_MA) using Friend.__add__.
    """
    input:
        mixeddata_coffea = expand(
            f"{SvB_OUT}SvB_mixeddata_all__{{year}}.coffea",
            year=config['years']
        ),
        hh_coffea = expand(
            f"{SvB_OUT}SvB_{{hh_dataset}}__{{year}}.coffea",
            hh_dataset=HH_DATASETS,
            year=config['years']
        ),
        data_json = "coffea4bees/metadata/datasets_HH4b_Run3/data_SvBfriend.json",
    output: f"{SvB_OUT}SvBfriend_mixeddata_data.json"
    container: config['analysis_container']
    log: f"{SvB_OUT}logs/merge_SvB_friendtrees_mixeddata.log"
    params:
        all_jsons = lambda wildcards, input: [
            f.replace(".coffea", ".json")
            for f in list(input.mixeddata_coffea) + list(input.hh_coffea)
        ]
    shell:
        """
        PYTHONPATH=. python src/friendtrees/merge_friend_meta.py \
            -i {params.all_jsons} {input.data_json} \
            -o {output} \
            2>&1 | tee -a {log}
        """
