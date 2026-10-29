# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_A_1_make_mixeddata.smk
#
# Stage A_1: Mixed-Data Production & Inclusive JCM Calibration (Run 2 & Run 3)
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Generates the nominal base mixed data (hemisphere mixing at nearest-neighbor
# rank 0,0) for the ttH(bb) background systematics measurement, derives the
# inclusive Jet Combinatoric Model (JCM) against 4b Data minus ttbar MC, evaluates
# the mixed data under the calibrated JCM weights, and produces full closure
# validation plots.
#
# WORKFLOW EXECUTION PIPELINE:
#   1. Stage Baseline Input & Derive Mixing JCM:
#      - Stages baseline histograms (inputs/histAll_NoJCM.coffea).
#      - Derives or references the inclusive JCM (JCM_1) for mixing pseudo-tags.
#   2. Hemisphere Mixing Production (HemiMixer):
#      - Runs `runner.py` with `HemiMixer` over 3b Data across all eras (UL16-UL18)
#        on HTCondor/Dask to pair hemispheres and generate mixed picoAODs on EOS.
#      - Merges per-era registries and creates the master dataset definition:
#        output/ttHbb_bkg_syst/coffea4bees/metadata/datasets/mixeddata_...rank0_0.yml
#   3. Process Unweighted Mixed Data:
#      - Runs `processor_ttHbb.py` under unit weights (no FvT, no JCM, no trig SFs)
#        with `pass_nSelJets_gt6` to obtain unweighted mixed-data histograms:
#        histAll_ttHbb_mixeddata_all.coffea
#   4. Fit Calibrated Mixed-Data JCM (JCM_2):
#      - Runs `make_jcm_weights.py` using 4b Data, ttbar MC, and unweighted mixed
#        data to fit the JCM transfer function (`float_t: true`):
#        JCM_2_mixeddata_inclusive/jetCombinatoricModel_inclusive_ttHbb_mixeddata.yml
#   5. Evaluate Mixed Data with Calibrated JCM:
#      - Runs `processor_ttHbb.py` applying the calibrated JCM model to obtain:
#        histAll_ttHbb_mixeddata_all_with_mixeddata_JCM.coffea
#   6. Mixed-Data Closure Validation Plots:
#      - Runs `makePlots.py` to produce data-vs-model closure distributions across
#        inclusive and pass_nSelJets_gt6 categories in SR, SB, and sum regions.
#
# INPUTS:
#   - inputs/histAll_NoJCM.coffea (Baseline Data and ttbar MC histograms)
#   - Hemisphere library & stats (coffea4bees/skimmer/metadata/)
#   - 3b Data picoAODs & friend trees
#
# OUTPUTS:
#   - picoaod_datasets_mixeddata_*.yml (per-era & merged registries)
#   - coffea4bees/metadata/datasets/mixeddata_ttHbb_bkg_syst_rank0_0.yml
#   - JCM_2_mixeddata_inclusive/jetCombinatoricModel_inclusive_ttHbb_mixeddata.yml
#   - run_mixeddata_all_with_mixeddata_JCM/histAll_..._with_mixeddata_JCM.coffea
#   - plots_mixeddata_closure/ (Closure validation plots)
# ==============================================================================

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"

# ── JCM Inclusive Fit Setup (Phase B unmixed JCM for mixing) ───────────────────
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

jcm_source_coffea = jcm_cfg.get('source_coffea', config.get('jcm_source_coffea', "output/ttHbb/computeJCM/histAll_NoJCM.coffea"))
_raw_jcm_input = jcm_cfg.get('input_coffea', config.get('jcm_input_coffea', "inputs/histAll_NoJCM.coffea"))
if not _raw_jcm_input.startswith("/") and not _raw_jcm_input.startswith("output/"):
    jcm_input_coffea = os.path.join(out, _raw_jcm_input)
else:
    jcm_input_coffea = _raw_jcm_input

_raw_jcm_out_dir = jcm_cfg.get('output_dir', config.get('jcm_output_dir', "JCM_1_data_inclusive/"))
if not _raw_jcm_out_dir.startswith("/") and not _raw_jcm_out_dir.startswith("output/"):
    jcm_out_dir = os.path.join(out, _raw_jcm_out_dir)
else:
    jcm_out_dir = _raw_jcm_out_dir
if not jcm_out_dir.endswith('/'):
    jcm_out_dir += '/'

jcm_tag           = jcm_cfg.get('tag', config.get('tag', "ttHbb_stitched_inclusive"))
jcm_region        = jcm_cfg.get('region', config.get('jcm_region', "inclusive"))
jcm_model_file    = f"{jcm_out_dir}jetCombinatoricModel_{jcm_region}_{jcm_tag}.yml"

# ── Stage A_1 Runtime Config Staging (Generated into {out_a1}configs/) ────────
from helpers.stage_configs import stage_phaseA_1_configs
cfg_files = stage_phaseA_1_configs(config, out_a1, jcm_model_file)

localrules: all_bkg_syst_A_1, all_JCM_1_data_inclusive, stage_input_coffea, make_JCM_1_data_inclusive, all_make_mixeddata_picoAOD, all_run_analysis_mixeddata_all, all_mixeddata_all, merge_mixeddata_registries, create_mixeddata_dataset_yaml, compare_mixeddata_vs_data, make_plots_mixeddata_vs_data, make_mixeddata_JCM, all_JCM_2_mixeddata_inclusive, all_mixeddata_JCM, make_plots_mixeddata_closure, all_mixeddata_closure, all_run_mixeddata_all_with_mixeddata_JCM

# ── Default Master Target (Full Phase E1 End-to-End) ───────────────────────────
rule all_bkg_syst_A_1:
    input:
        jcm_model_file,
        mixeddata_dataset_output,
        f"{out_a1}run_analysis_mixeddata_all/histAll_{channel}_mixeddata_all.coffea",
        f"{out_a1}JCM_2_mixeddata_inclusive/jetCombinatoricModel_inclusive_{channel}_mixeddata.yml",
        f"{out_a1}run_mixeddata_all_with_mixeddata_JCM/histAll_{channel}_mixeddata_all_with_mixeddata_JCM.coffea",
        f"{out_a1}plots_mixeddata_closure/plots_done.txt",

# ── Sub-Target Aliases ─────────────────────────────────────────────────────────
rule all_JCM_1_data_inclusive:
    input:
        jcm_model_file

rule all_make_mixeddata_picoAOD:
    input:
        mixeddata_dataset_output

rule all_run_analysis_mixeddata_all:
    input:
        f"{out_a1}run_analysis_mixeddata_all/histAll_{channel}_mixeddata_all.coffea"

rule all_mixeddata_all:
    input:
        f"{out_a1}run_analysis_mixeddata_all/histAll_{channel}_mixeddata_all.coffea"

rule all_JCM_2_mixeddata_inclusive:
    input:
        f"{out_a1}JCM_2_mixeddata_inclusive/jetCombinatoricModel_inclusive_{channel}_mixeddata.yml"

rule all_mixeddata_JCM:
    input:
        f"{out_a1}JCM_2_mixeddata_inclusive/jetCombinatoricModel_inclusive_{channel}_mixeddata.yml"

rule all_run_mixeddata_all_with_mixeddata_JCM:
    input:
        f"{out_a1}run_mixeddata_all_with_mixeddata_JCM/histAll_{channel}_mixeddata_all_with_mixeddata_JCM.coffea"

rule all_mixeddata_closure:
    input:
        f"{out_a1}plots_mixeddata_closure/plots_done.txt"

rule compare_mixeddata_vs_data:
    input:
        f"{out_a1}plots_mixeddata_vs_data/plots_done.txt"

# ── Stage 1: Initial JCM for Mixing ───────────────────────────────────────────
rule stage_input_coffea:
    input:
        jcm_source_coffea
    output:
        jcm_input_coffea
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output})
        if [ "{input}" != "{output}" ]; then
            echo "Staging copy of {input} -> {output} (non-destructive)"
            cp "{input}" "{output}"
        fi
        """

use rule make_JCM from analysis as make_JCM_1_data_inclusive with:
    input:
        jcm_input_coffea
    output:
        jcm_model_file
    params:
        extra_arguments = config.get('jcm_extra_arguments', jcm_cfg.get('extra_arguments', "--jcm_config coffea4bees/analysis/jcm_tools/metadata/ttHbb_jcm_config.yml")),
        tag = jcm_tag,
        region = jcm_region,
        output_dir = jcm_out_dir,
        run_container_wrapper = config.get('analysis_container_wrapper', config.get('container_wrapper', './run_container')),
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log:
        f"{jcm_out_dir}logs/make_JCM_1_data_inclusive.log"


rule make_mixeddata_picoAOD_per_year:
    input:
        config_file = cfg_files['skimmer'],
        jcm_model = jcm_model_file,
        hemi_lib = config.get('hemi_library_yaml', 'coffea4bees/skimmer/metadata/hemisphere_library_noTT.yml'),
        hemi_stats = get_hemi_stats_file,
        data_yml = os.path.join(config.get('dataset_location', 'coffea4bees/metadata/datasets/'), 'data.yml'),
        friends = config.get('friends_file', 'coffea4bees/metadata/friends/friends_ttHbb.yml'),
    output:
        reg = f"{picoaod_dir}picoaod_datasets_{config['dataset_name']}__{{year}}.yml",
        done = f"{out_a1}.make_mixeddata_{{year}}.done",
    log:
        f"{out_a1}logs/make_mixeddata__{{year}}.log"
    params:
        friends = config.get('friends_file', 'coffea4bees/metadata/friends/friends_ttHbb.yml'),
        processor = config.get('skimmer_processor', "coffea4bees/skimmer/processor/make_mixed_data.py"),
        dataset = config.get('skimmer_dataset', "data"),
        output_path = picoaod_dir,
        container_wrapper = config['analysis_container_wrapper'],
        condor_flags = condor_flags,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.reg}) $(dirname {log})
        {params.container_wrapper} python runner.py {input.config_file} \
            -p {params.processor} \
            -d {params.dataset} \
            --friends {params.friends} \
            --years {wildcards.year} \
            --output-path {params.output_path} \
            --output $(basename {output.reg}) \
            -s {params.condor_flags} 2>&1 | tee {log}
        touch {output.done}
        """

# ── Stage 2: Merge Registries & Install Dataset YAML ──────────────────────────
rule merge_mixeddata_registries:
    input:
        expand(f"{picoaod_dir}picoaod_datasets_{config['dataset_name']}__{{year}}.yml", year=YEARS)
    output:
        reg = f"{out_a1}picoaod_datasets_{config['dataset_name']}.yml",
        done = f"{out_a1}.merge_mixeddata_registries.done",
    log:
        f"{out_a1}logs/merge_registries.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.reg}) $(dirname {log})
        python coffea4bees/workflows/scripts/merge_mixeddata_registries.py {input} {output.reg} 2>&1 | tee {log}
        touch {output.done}
        """

rule create_mixeddata_dataset_yaml:
    input:
        reg = f"{out_a1}picoaod_datasets_{config['dataset_name']}.yml",
        done = f"{out_a1}.merge_mixeddata_registries.done",
    output:
        dataset_yaml = mixeddata_dataset_output,
    log:
        f"{out_a1}logs/create_mixeddata_dataset_yaml.log"
    params:
        dataset_name = config['dataset_name'],
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.dataset_yaml}) $(dirname {log})
        python src/tools/make_dataset_yml.py -i {input.reg} -o {output.dataset_yaml} -n {params.dataset_name} 2>&1 | tee {log}
        """

rule install_mixeddata_dataset_yaml:
    input:
        mixeddata_dataset_output

# ── Stage 3: Process All Mixed Data (Unweighted Unit Weights) ─────────────────
rule run_analysis_mixeddata_all:
    input:
        ds_ready = mixeddata_dataset_file,
        analysis_cfg = cfg_files['analysis_unit'],
    output:
        coffea_out = f"{out_a1}run_analysis_mixeddata_all/histAll_{channel}_mixeddata_all.coffea",
    log:
        f"{out_a1}run_analysis_mixeddata_all/logs/run_analysis_mixeddata_all.log"
    params:
        processor = config.get('analysis_processor', f"coffea4bees/analysis/processors/processor_{channel}.py"),
        dataset = config['dataset_name'],
        output_path = f"{out_a1}run_analysis_mixeddata_all/",
        years = " ".join(YEARS),
        container_wrapper = config['analysis_container_wrapper'],
        condor_flags = condor_flags,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.coffea_out}) $(dirname {log})
        {params.container_wrapper} python runner.py {input.analysis_cfg} \
            --processor {params.processor} \
            --metadata {input.ds_ready} \
            --datasets {params.dataset} \
            --years {params.years} \
            --output-path {params.output_path} \
            --output $(basename {output.coffea_out}) \
            {params.condor_flags} 2>&1 | tee {log}
        """

# ── Stage 4: Mixed-Data Inclusive JCM Derivation ──────────────────────────────
rule make_mixeddata_JCM:
    input:
        data_tt_coffea = jcm_input_coffea,
        mixed_coffea = f"{out_a1}run_analysis_mixeddata_all/histAll_{channel}_mixeddata_all.coffea",
        jcm_cfg = cfg_files['jcm_fit'],
        plot_cfg = cfg_files['jcm_plots'],
    output:
        model_file = f"{out_a1}JCM_2_mixeddata_inclusive/jetCombinatoricModel_inclusive_{channel}_mixeddata.yml",
        seljets_plot = f"{out_a1}JCM_2_mixeddata_inclusive/selJets_noJCM_n.png",
        tagjets_plot = f"{out_a1}JCM_2_mixeddata_inclusive/tagJets_noJCM_n.png",
    log:
        f"{out_a1}JCM_2_mixeddata_inclusive/logs/make_mixeddata_JCM.log"
    params:
        output_dir = f"{out_a1}JCM_2_mixeddata_inclusive/",
        region = "inclusive",
        tag = f"{channel}_mixeddata",
        container_wrapper = config['analysis_container_wrapper'],
    shell:
        """
        set -eo pipefail
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR $(dirname {output.model_file}) $(dirname {log})
        
        echo "Computing JCM for ttbar + mixeddata vs data 4b inclusive" 2>&1 | tee {log}
        {params.container_wrapper} env PYTHONPATH=. python coffea4bees/analysis/jcm_tools/make_jcm_weights.py \
            -o {params.output_dir} \
            -r {params.region} \
            -i {input.data_tt_coffea} {input.mixed_coffea} \
            --jcm_config {input.jcm_cfg} \
            -m {input.plot_cfg} \
            -w {params.tag} \
            --combine_input_files 2>&1 | tee -a {log}
        ls -la {params.output_dir}
        """

# ── Stage 5: Evaluate Mixed Data with Calibrated JCM ──────────────────────────
rule run_analysis_mixeddata_all_with_mixeddata_JCM:
    input:
        ds_ready = mixeddata_dataset_file,
        jcm_ready = f"{out_a1}JCM_2_mixeddata_inclusive/jetCombinatoricModel_inclusive_{channel}_mixeddata.yml",
        analysis_cfg = cfg_files['analysis_jcm'],
    output:
        coffea_out = f"{out_a1}run_mixeddata_all_with_mixeddata_JCM/histAll_{channel}_mixeddata_all_with_mixeddata_JCM.coffea",
    log:
        f"{out_a1}run_mixeddata_all_with_mixeddata_JCM/logs/run_analysis_mixeddata_all_with_mixeddata_JCM.log"
    params:
        processor = config.get('analysis_processor', f"coffea4bees/analysis/processors/processor_{channel}.py"),
        dataset = config['dataset_name'],
        output_path = f"{out_a1}run_mixeddata_all_with_mixeddata_JCM/",
        years = " ".join(YEARS),
        container_wrapper = config['analysis_container_wrapper'],
        condor_flags = condor_flags,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.coffea_out}) $(dirname {log})
        {params.container_wrapper} python runner.py {input.analysis_cfg} \
            --processor {params.processor} \
            --metadata {input.ds_ready} \
            --datasets {params.dataset} \
            --years {params.years} \
            --output-path {params.output_path} \
            --output $(basename {output.coffea_out}) \
            {params.condor_flags} 2>&1 | tee {log}
        """

# ── Stage 6: Mixed-Data Closure Plots (Full Stack vs Data 4b) ─────────────────
rule make_plots_mixeddata_closure:
    input:
        data_coffea = jcm_input_coffea,
        mixed_coffea = f"{out_a1}run_mixeddata_all_with_mixeddata_JCM/histAll_{channel}_mixeddata_all_with_mixeddata_JCM.coffea",
        plot_cfg = cfg_files['closure_plots'],
    output:
        done = f"{out_a1}plots_mixeddata_closure/plots_done.txt",
    log:
        f"{out_a1}logs/make_plots_mixeddata_closure.log"
    params:
        output_dir = f"{out_a1}plots_mixeddata_closure/",
        container_wrapper = config['analysis_container_wrapper'],
        plot_year = config.get('plot_year', 'RunII'),
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.done}) $(dirname {log})
        {params.container_wrapper} python coffea4bees/plots/makePlots.py \
            {input.data_coffea} {input.mixed_coffea} \
            -o {params.output_dir} \
            -m {input.plot_cfg} \
            --combine_input_files \
            --year {params.plot_year} \
            -p 1 2>&1 | tee {log}
        touch {output.done}
        """

# ── Optional: Pre-JCM Comparison Plots vs Data 4b ─────────────────────────────
rule make_plots_mixeddata_vs_data:
    input:
        data_coffea = jcm_input_coffea,
        mixed_coffea = f"{out_a1}run_analysis_mixeddata_all/histAll_{channel}_mixeddata_all.coffea",
        plot_cfg = cfg_files['vs_data_plots'],
    output:
        done = f"{out_a1}plots_mixeddata_vs_data/plots_done.txt",
    log:
        f"{out_a1}logs/make_plots_mixeddata_vs_data.log"
    params:
        output_dir = f"{out_a1}plots_mixeddata_vs_data/",
        container_wrapper = config['analysis_container_wrapper'],
        plot_year = config.get('plot_year', 'RunII'),
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.done}) $(dirname {log})
        {params.container_wrapper} python coffea4bees/plots/makePlots.py \
            {input.data_coffea} {input.mixed_coffea} \
            -o {params.output_dir} \
            -m {input.plot_cfg} \
            --combine_input_files \
            --year {params.plot_year} 2>&1 | tee {log}
        touch {output.done}
        """
