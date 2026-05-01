import os

include: "rules/run3_variants.smk"

# Per-Snakefile defaults
config.setdefault('output_path', f"output/classifier_inputs_Run3{config['label']}/")
config.setdefault('datasets',
    ['TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu', 'data', 'mixeddata_all'])
config.setdefault('years', ['2022_EE', '2022_preEE', '2023_BPix', '2023_preBPix'])

# install path differs per mode; matches the file historically produced by
# Snakefile_Run3MvD.smk so downstream consumers (FvT training in
# Snakefile_Run3.smk for quadjet_run2) keep working unchanged.
if config["mode"] == "nominal":
    config.setdefault('classifier_inputs_install_path',
        "coffea4bees/metadata/datasets_HH4b_Run3/classifier_inputs_MvD_Run3.json")
elif config["mode"] == "quadjet_run2":
    config.setdefault('classifier_inputs_install_path',
        "coffea4bees/metadata/datasets_HH4b_Run3/classifier_inputs_MvD_Run3_quadjet_run2.json")

out = config['output_path']

module analysis:
    snakefile: "rules/analysis.smk"
    config: config

rule all:
    input: config['classifier_inputs_install_path']

# ── Classifier inputs ─────────────────────────────────────────────────────────
# Use __ (double underscore) as separator between dataset and year to avoid
# ambiguous wildcard matching, since both dataset names and years contain _.

use rule analysis_processor from analysis as classifier_inputs with:
    output: f"{out}classifier_inputs/classifier_inputs_{{dataset}}__{{year}}.coffea"
    log: f"{out}logs/classifier_inputs_{{dataset}}__{{year}}.log"
    params:
        datasets = "{dataset}",
        years = "{year}",
        config = config['classifier_config'],
        processor = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file = config['dataset_location'],
        blind = False,
        run_performance = False,
        friends = "coffea4bees/metadata/friends_HH4b.yml",
        run_on_condor = True,
        extra_arguments = "",
        run_container_wrapper = "./run_container",
        dashboard_address = 0

rule merge_json_classifier_inputs:
    input:
        expand(
            "{out}classifier_inputs/classifier_inputs_{dataset}__{year}.coffea",
            out=out,
            dataset=config['datasets'],
            year=config['years']
        )
    output: f"{out}classifier_inputs_Run3{config['label']}.json"
    log: f"{out}logs/merge_json_classifier_inputs.log"
    container: config["analysis_container"]
    shell:
        """
        echo "Merging JSON classifier input files" 2>&1 | tee -a {log}
        python -m src.friendtrees.merge_friend_meta -i $(echo {input} | sed 's/\\.coffea/.json/g') -o {output} 2>&1 | tee -a {log}
        """

rule install_classifier_inputs:
    input: f"{out}classifier_inputs_Run3{config['label']}.json"
    output: config['classifier_inputs_install_path']
    shell:
        """
        mkdir -p $(dirname {output})
        cp {input} {output}
        echo "Installed classifier inputs JSON to {output} — commit this file to git to version it."
        """
