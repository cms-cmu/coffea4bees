"""SvB friend tree creation for mixeddata_all (Run 3).

Mode-independent: SvB friend trees are the same regardless of analysis mode.
Run once; the installed JSON is shared by all modes of Snakefile_classifier_inputs_Run3MvD.smk.

Usage:
    snakemake --profile software/snakemake/profiles/lpc \\
        --snakefile coffea4bees/workflows/Snakefile_SvB_friendtrees_Run3.smk \\
        --cores 4
"""

config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
# Base datasets + friend JSONs were moved out of metadata/datasets_HH4b_Run3/
# by the repo reorg: dataset ymls now live in metadata/datasets/ and the
# committed friend JSONs (data_SvBfriend.json, SvB*friend_mixeddata_data.json)
# in metadata/friends/ (see friends_HH4b.yml). Point everything at the new
# locations.
config.setdefault('dataset_location',   "coffea4bees/metadata/datasets/")
config.setdefault('years', ['2022_EE', '2022_preEE', '2023_BPix', '2023_preBPix'])
config.setdefault('dataset_name', 'mixeddata_all')
# Path to the installed dataset metadata yaml. Declared as an input to
# make_SvB_friendtrees_mixeddata so snakemake schedules the install step
# (when invoked from Snakefile_Run3_make_mixeddata.smk) before the SvB job.
# In standalone mode this just points at the committed yaml.
config.setdefault('install_path',
    f"coffea4bees/metadata/datasets/{config['dataset_name']}.yml")

# When True, skip the HH per-year jobs and reuse the legacy combined SvB JSON
# (which already contains data + HH + legacy mixeddata friend mappings) as a
# merge input. Only the new background-dataset per-year jobs run. Reasonable for
# rank-suffixed / synthetic runs where HH and data friends are unchanged from
# the legacy build. Default False preserves backward-compatible standalone behavior.
config.setdefault('reuse_legacy_friends', False)
LEGACY_SVB_JSON = "coffea4bees/metadata/friends/SvBfriend_mixeddata_data.json"

# Friend-JSON name stub + output-dir tag, derived from dataset_name.
#   * mixeddata_all family  -> legacy 'mixeddata_data' stub (+ _rank tag), so
#     historical install paths and snakemake outputs are preserved byte-for-byte.
#   * any other background dataset (e.g. synthetic_data_noTT_ns1) -> the dataset
#     name itself, giving a clean SvBfriend_<dataset_name>.json install.
if config['dataset_name'].startswith("mixeddata_all"):
    _dsn_tag     = "" if config['dataset_name'] == "mixeddata_all" \
                   else config['dataset_name'].removeprefix("mixeddata_all_")
    _install_tag = f"_{_dsn_tag}" if _dsn_tag else ""
    _friend_stub = f"mixeddata_data{_install_tag}"
    _out_tag     = _install_tag
else:
    _friend_stub = config['dataset_name']
    _out_tag     = f"_{config['dataset_name']}"

SvB_OUT = f"output/Run3_MvD{_install_tag}/svb_friendtrees/"
INSTALL = f"coffea4bees/metadata/friends/data_SvBfriend{_install_tag}.json"


# Only this HH coupling point currently has 2022/2023 picoAODs.
HH_DATASETS = ['GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00']

module analysis:
    snakefile: "rules/analysis.smk"
    config: config


rule all:
    input: INSTALL


rule install_SvB_friend_json:
    input:  f"{SvB_OUT}SvBfriend_mixeddata_data.json"
    output: INSTALL
    shell:  "cp {input} {output}"


rule create_SvB_friend_config:
    input: "coffea4bees/analysis/metadata/HH4b_make_friend_SvB_Run3.yml"
    output: f"{SvB_OUT}analysis_config_make_friend_SvB.yml"
    params:
        processor = "coffea4bees/analysis/processors/processor_HH4b.py",
        dataset_location = config['dataset_location'],
        friends = config.get("friend_gen_friends", "coffea4bees/metadata/friends/friends_HH4b.yml")
    run:
        import yaml
        with open(input[0], 'r') as f:
            cfg = yaml.safe_load(f) or {}
        cfg['processor'] = params.processor
        cfg['dataset_location'] = params.dataset_location
        cfg['friend_file'] = params.friends
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)


use rule analysis_processor from analysis as make_SvB_friendtrees_mixeddata with:
    # runner.py writes a matching .json metafile alongside each .coffea output,
    # which merge_SvB_friendtrees_mixeddata picks up via the .coffea → .json swap.
    # install_path is the installed dataset yaml — required so this rule waits
    # for install_mixeddata_dataset when run inside the make-mixeddata pipeline.
    input:
        config_yml   = f"{SvB_OUT}analysis_config_make_friend_SvB.yml",
        install_path = config['install_path'],
    output: f"{SvB_OUT}SvB_{config['dataset_name']}__{{year}}.coffea"
    log: f"{SvB_OUT}logs/SvB_{config['dataset_name']}__{{year}}.log"
    params:
        datasets              = config['dataset_name'],
        years                 = "{year}",
        config                = lambda wildcards, input: input.config_yml,
        run_container_wrapper = "./run_container"


use rule analysis_processor from analysis as make_SvB_friendtrees_HH with:
    input: f"{SvB_OUT}analysis_config_make_friend_SvB.yml"
    output: f"{SvB_OUT}SvB_{{hh_dataset}}__{{year}}.coffea"
    log: f"{SvB_OUT}logs/SvB_{{hh_dataset}}__{{year}}.log"
    wildcard_constraints:
        hh_dataset = "|".join(HH_DATASETS)
    params:
        datasets              = "{hh_dataset}",
        years                 = "{year}",
        config                = lambda wildcards, input: input[0],
        run_container_wrapper = "./run_container"


if config['reuse_legacy_friends']:
    rule merge_SvB_friendtrees_mixeddata:
        """Merge new mixed-data SvB metafiles with the legacy combined JSON.

        Skips re-making HH friend trees (unchanged across ranks). The legacy
        JSON already contains data + HH + (legacy mixeddata) keys; we layer
        the new dataset_name keys on top via merge_friend_meta.py.
        """
        input:
            mixeddata_coffea = expand(
                f"{SvB_OUT}SvB_{{dataset_name}}__{{year}}.coffea",
                dataset_name=config['dataset_name'],
                year=config['years'],
            ),
            legacy_json = LEGACY_SVB_JSON,
        output: f"{SvB_OUT}SvBfriend_mixeddata_data.json"
        # container: None -> avoid nested apptainer under --profile lpc; ./run_container in the shell provides the container (see merge_cluster in Snakefile_Run3_make_synthetic).
        container: None
        log: f"{SvB_OUT}logs/merge_SvB_friendtrees_mixeddata.log"
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
    rule merge_SvB_friendtrees_mixeddata:
        """Merge per-year mixeddata_all + HH SvB metafiles with the existing data SvB JSON.

        runner.py writes {output_path}/{output_file}.json alongside the coffea output,
        so the per-year metafiles are the coffea paths with .coffea → .json.
        merge_friend_meta.py merges by key (SvB, SvB_MA) using Friend.__add__.
        """
        input:
            mixeddata_coffea = expand(
                f"{SvB_OUT}SvB_{{dataset_name}}__{{year}}.coffea",
                dataset_name=config['dataset_name'],
                year=config['years'],
            ),
            hh_coffea = expand(
                f"{SvB_OUT}SvB_{{hh_dataset}}__{{year}}.coffea",
                hh_dataset=HH_DATASETS,
                year=config['years']
            ),
            data_json = "coffea4bees/metadata/friends/data_SvBfriend.json",
        output: f"{SvB_OUT}SvBfriend_mixeddata_data.json"
        # container: None -> avoid nested apptainer under --profile lpc; ./run_container in the shell provides the container (see merge_cluster in Snakefile_Run3_make_synthetic).
        container: None
        log: f"{SvB_OUT}logs/merge_SvB_friendtrees_mixeddata.log"
        params:
            all_jsons = lambda wildcards, input: [
                f.replace(".coffea", ".json")
                for f in list(input.mixeddata_coffea) + list(input.hh_coffea)
            ]
        shell:
            """
            ./run_container env PYTHONPATH=. python src/friendtrees/merge_friend_meta.py \
                -i {params.all_jsons} {input.data_json} \
                -o {output} \
                2>&1 | tee -a {log}
            """
