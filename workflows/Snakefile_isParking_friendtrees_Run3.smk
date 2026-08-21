"""is_parking friend tree creation for 2023_preBPix MC (TTbar + HH).

In Run3 the trigger strategy changed mid-2023_preBPix: parking triggers are
present from era C3 onward. For inclusive MC samples we need a per-event
parking/preparking assignment whose marginal fraction matches the data
luminosity ratio. processor_isParking_friend.py uses the Squares
reproducible RNG to produce that assignment and dumps a single bool branch
(``is_parking``) as a friend tree.

Only 2023_preBPix MC is needed:
  - 2022 MC is pure preparking (parking_fraction = 0)
  - 2023_BPix MC is pure parking (parking_fraction = 1)
  - Data uses event.run >= parking_minRun, no friend tree required

Datasets covered:
  - TTbar: TTTo2L2Nu, TTToHadronic, TTToSemiLeptonic
  - HH:    GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00 (only this coupling
           point has a 2023_preBPix sample today; add more if/when they
           become available)

After running, copy the merged result.json into friends_HH4b.yml under the
2023_preBPix entry:

    is_parking: "coffea4bees/metadata/friends/isParkingfriend_2023_preBPix.json@@is_parking"

The Run3 SvB_FeynNet friend tree generation will then load this and
dispatch parking/preparking per event.

Usage:
    snakemake --profile software/snakemake/profiles/lpc \\
        --snakefile coffea4bees/workflows/Snakefile_isParking_friendtrees_Run3.smk \\
        --cores 4
"""

config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset_location',   "coffea4bees/metadata/datasets/")

ISPARKING_OUT = "output/Run3_isParking/isParking_friendtrees/"
YEAR          = "2023_preBPix"

TT_DATASETS = ['TTTo2L2Nu', 'TTToHadronic', 'TTToSemiLeptonic']
HH_DATASETS = ['GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00']

module analysis:
    snakefile: "rules/analysis.smk"
    config: config


rule all:
    input:
        f"coffea4bees/metadata/friends/isParkingfriend_{YEAR}.json"


rule install_isParking_friend_json:
    input:  f"{ISPARKING_OUT}isParkingfriend_{YEAR}.json"
    output: f"coffea4bees/metadata/friends/isParkingfriend_{YEAR}.json"
    shell:  "cp {input} {output}"


use rule analysis_processor from analysis as make_isParking_friendtrees_ttbar with:
    input:  "coffea4bees/analysis/metadata/HH4b_make_friend_isParking.yml"
    output: f"{ISPARKING_OUT}isParking_{{tt_dataset}}__{YEAR}.coffea"
    log:    f"{ISPARKING_OUT}logs/isParking_{{tt_dataset}}__{YEAR}.log"
    wildcard_constraints:
        tt_dataset = "|".join(TT_DATASETS)
    params:
        datasets              = "{tt_dataset}",
        years                 = YEAR,
        config                = lambda wildcards, input: input[0],
        run_container_wrapper = "./run_container"


use rule analysis_processor from analysis as make_isParking_friendtrees_HH with:
    input:  "coffea4bees/analysis/metadata/HH4b_make_friend_isParking.yml"
    output: f"{ISPARKING_OUT}isParking_{{hh_dataset}}__{YEAR}.coffea"
    log:    f"{ISPARKING_OUT}logs/isParking_{{hh_dataset}}__{YEAR}.log"
    wildcard_constraints:
        hh_dataset = "|".join(HH_DATASETS)
    params:
        datasets              = "{hh_dataset}",
        years                 = YEAR,
        config                = lambda wildcards, input: input[0],
        run_container_wrapper = "./run_container"


rule merge_isParking_friendtrees:
    """Merge per-dataset is_parking metafiles into one JSON.

    runner.py writes {output_path}/{output_file}.json alongside each .coffea,
    so per-dataset metafiles are the coffea paths with .coffea -> .json.
    merge_friend_meta.py merges by friend name (here: 'is_parking').
    """
    input:
        ttbar_coffea = expand(
            f"{ISPARKING_OUT}isParking_{{tt_dataset}}__{YEAR}.coffea",
            tt_dataset=TT_DATASETS,
        ),
        hh_coffea = expand(
            f"{ISPARKING_OUT}isParking_{{hh_dataset}}__{YEAR}.coffea",
            hh_dataset=HH_DATASETS,
        ),
    output: f"{ISPARKING_OUT}isParkingfriend_{YEAR}.json"
    container: config['analysis_container']
    log: f"{ISPARKING_OUT}logs/merge_isParking_friendtrees.log"
    params:
        all_jsons = lambda wildcards, input: [
            f.replace(".coffea", ".json")
            for f in list(input.ttbar_coffea) + list(input.hh_coffea)
        ]
    shell:
        """
        PYTHONPATH=. python src/friendtrees/merge_friend_meta.py \\
            -i {params.all_jsons} \\
            -o {output} \\
            2>&1 | tee -a {log}
        """
