# ==============================================================================
# Workflow: Background Systematics - Stage A_2: TTbar Pseudodata Slicing
# File: Snakefile_bkg_syst_A_2_make_ttbar_psdata.smk
#
# Description:
#   Produces unweighted ttbar pseudodata picoAODs (picoAOD_PSData.root) from the
#   stitched ttbar MC datasets across Run 2 eras (UL16_preVFP, UL16_postVFP, UL17, UL18)
#   using sub_sample_MC.py (SubSampler) with enforced trigger weights,
#   merges the registries, and generates the dataset YAML (ttbar_PSData_stitched.yml).
#
# Inputs:
#   - TT MC dataset: coffea4bees/metadata/datasets/TT_stitched.yml
#   - Friends manifest: coffea4bees/metadata/friends/friends_ttHbb.yml
#   - Staged skimmer config: {out_a2}configs/skimmer_subsample_mc_ttbar_psdata.yml
#
# Outputs:
#   - Per-year registries: {out_a2}picoaod_datasets_ttbar_PSData_stitched_{year}.yml
#   - Combined registry: {out_a2}picoaod_datasets_ttbar_PSData_stitched.yml
#   - Dataset YAML: {out}coffea4bees/metadata/datasets/ttbar_PSData_stitched.yml
#
# Usage:
#   snakemake -s Snakefile_bkg_syst_A_2_make_ttbar_psdata.smk \
#             --configfile config/analysis_ttHbb_bkg_syst.yml -np all_bkg_syst_A_2
# ==============================================================================

import os

if 'config' not in globals():
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"

# ── Stage A_2 Runtime Config Staging (Generated into {out_a2}configs/) ────────
from helpers.stage_configs import stage_phaseA_2_configs
cfg_files = stage_phaseA_2_configs(config, out_a2)

phaseA_2 = config.get('phaseA_2', {})

ttbar_datasets = phaseA_2.get('datasets', [
    "TTTo2L2Nu_stitched",
    "TTToSemiLeptonic_stitched",
    "TTToHadronic_stitched",
])
if isinstance(ttbar_datasets, str):
    ttbar_datasets = [d.strip() for d in ttbar_datasets.split() if d.strip()]

dataset_name = phaseA_2.get('dataset_name', "ttbar_PSData_stitched")
datasets_file = phaseA_2.get('datasets_file', "coffea4bees/metadata/datasets/TT_stitched.yml")
friends_file = phaseA_2.get('friends_file', "coffea4bees/metadata/friends/friends_ttHbb.yml")
install_path = phaseA_2.get('install_path', f"{out}coffea4bees/metadata/datasets/{dataset_name}.yml")

localrules: all_bkg_syst_A_2, all_bkg_syst_A_2_alias, all_ttbar_pseudodata, merge_ttbar_psdata_registries, create_ttbar_psdata_dataset_yaml

# ── Master Target ─────────────────────────────────────────────────────────────
rule all_bkg_syst_A_2:
    input:
        install_path

rule all_bkg_syst_A_2_alias:
    input:
        rules.all_bkg_syst_A_2.input

rule all_ttbar_pseudodata:
    input:
        install_path

# ── Run SubSampler per Year ───────────────────────────────────────────────────
rule make_ttbar_pseudodata:
    input:
        config_file = cfg_files['skimmer'],
        datasets_file = datasets_file,
        friends = friends_file,
    output:
        reg = f"{out_a2}picoaod_datasets_{dataset_name}_{{year}}.yml",
        done = f"{out_a2}.make_ttbar_psdata_{{year}}.done",
    log:
        f"{out_a2}logs/make_ttbar_psdata_{{year}}.log"
    params:
        processor = "coffea4bees/skimmer/processor/sub_sample_MC.py",
        datasets = " ".join(ttbar_datasets),
        datasets_file = datasets_file,
        friends = friends_file,
        output_path = out_a2,
        container_wrapper = container_wrapper,
        python_bin = python_bin,
        condor_flags = condor_flags,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.reg}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} runner.py {input.config_file} \
            -p {params.processor} \
            -d {params.datasets} \
            -c {params.datasets_file} \
            --friends {params.friends} \
            --years {wildcards.year} \
            --output-path {params.output_path} \
            --output $(basename {output.reg}) \
            -s {params.condor_flags} 2>&1 | tee {log}
        touch {output.done}
        """

# ── Merge Registries & Generate Dataset YAML ──────────────────────────────────
rule merge_ttbar_psdata_registries:
    input:
        expand(f"{out_a2}picoaod_datasets_{dataset_name}_{{year}}.yml", year=YEARS)
    output:
        f"{out_a2}picoaod_datasets_{dataset_name}.yml"
    log:
        f"{out_a2}logs/merge_ttbar_psdata_registries.log"
    params:
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.python_bin} coffea4bees/workflows/scripts/merge_mixeddata_registries.py {input} {output} 2>&1 | tee -a {log}
        """

rule create_ttbar_psdata_dataset_yaml:
    input:
        f"{out_a2}picoaod_datasets_{dataset_name}.yml"
    output:
        install_path
    log:
        f"{out_a2}logs/create_ttbar_psdata_dataset_yaml.log"
    params:
        name = dataset_name,
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.python_bin} src/tools/make_dataset_yml.py -i {input} -o {output} -n {params.name} 2>&1 | tee -a {log}
        echo "Created {output} (dataset name: {params.name})" 2>&1 | tee -a {log}
        """
