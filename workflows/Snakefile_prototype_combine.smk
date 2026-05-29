import os

include: "helpers/common.smk"

# Define default configs (overridable via --config on CLI)
config.setdefault("stat_only", False)
config.setdefault("num_toy_jobs", 10)
config.setdefault("toys_per_job", 50)
config.setdefault("output_path", "output/v4_systematics_test/")

config.setdefault("channels", {
    "HH4b": {
        "signallabel": "ggHH_kl_1_kt_1_13p0TeV_hbbhbb",
        "othersignal": "ggHH_kl_0_kt_1_13p0TeV_hbbhbb ggHH_kl_2p45_kt_1_13p0TeV_hbbhbb ggHH_kl_5_kt_1_13p0TeV_hbbhbb",
        "signal": "GluGluToHHTo4B_cHHH1"
    }
})

# Path to the central combine rules in barista
module combine:
    snakefile: os.path.join(os.getcwd(), "src/stat_analysis/combine.smk")
    config: config

stat_only = config["stat_only"]

# Base paths and variables
output_dir = f"{config['output_path']}datacards/HH4b/"
base_workspace = f"{output_dir}datacard"

# Targets definition
targets = [
    f"{output_dir}datacard_limits__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.json",
    f"{output_dir}significance__datacard__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.log",
    f"{output_dir}datacard_likelihood_scan__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.pdf",
    f"{output_dir}datacard_postfit__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.pdf"
]

if not stat_only:
    # GoF and Impacts only make sense in systematics mode
    targets.extend([
        f"{output_dir}datacard_gof__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.pdf",
        f"{output_dir}datacard_impacts__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.pdf"
    ])

rule all:
    input: targets

# Workspace creation rule from barista
use rule workspace from combine with:
    input: f"{output_dir}datacard__HH4b.txt"
    output: f"{base_workspace}__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.root"
    log: f"{config['output_path']}logs/workspace_HH4b__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.log"
    params:
        poi_maps=lambda wildcards: make_poi_maps(
            signals=[config["channels"]["HH4b"]["signallabel"]] + config["channels"]["HH4b"].get("othersignal", "").split(),
            poi_ranges=config.get("poi_ranges", "1,-10,10")
        ),
        stat_only=config["stat_only"]

# Limits rule from barista
use rule limits from combine with:
    input: f"{base_workspace}__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.root"
    output:
        txt=f"{output_dir}datacard_limits__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.txt",
        json=f"{output_dir}datacard_limits__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.json"
    log: f"{config['output_path']}logs/limits_HH4b__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.log"
    params:
        signallabel="ggHH_kl_1_kt_1_13p0TeV_hbbhbb"

# Significance rule from barista
use rule significance from combine with:
    input: f"{base_workspace}__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.root"
    output: f"{output_dir}significance__datacard__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.log"
    log: f"{config['output_path']}logs/significance_HH4b__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.log"
    params:
        signallabel="ggHH_kl_1_kt_1_13p0TeV_hbbhbb"

# Likelihood Scan rule from barista
use rule likelihood_scan_snapshot from combine
use rule likelihood_scan_chunk from combine

num_likelihood_chunks = (int(config.get("likelihood_scan_points", 50)) + int(config.get("likelihood_scan_split_size", 10)) - 1) // int(config.get("likelihood_scan_split_size", 10))
likelihood_chunk_inputs = [
    f"{output_dir}datacard_likelihood_scan_chunk_{i}__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.root"
    for i in range(num_likelihood_chunks)
]

use rule likelihood_scan from combine with:
    input: likelihood_chunk_inputs
    output: f"{output_dir}datacard_likelihood_scan__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.pdf"
    log: f"{config['output_path']}logs/likelihood_scan_HH4b__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.log"
    params:
        signallabel="ggHH_kl_1_kt_1_13p0TeV_hbbhbb"




# Postfit Diagnostics and Plotting from barista
use rule fit_diagnostics from combine with:
    input: f"{base_workspace}__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.root"
    output: f"{output_dir}datacard_fitDiagnostics_bonly__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.root"
    log: f"{config['output_path']}logs/fit_diagnostics_HH4b__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.log"
    params:
        signallabel="ggHH_kl_1_kt_1_13p0TeV_hbbhbb"

use rule postfit from combine with:
    input:
        workspace=f"{base_workspace}__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.root",
        fit_result=f"{output_dir}datacard_fitDiagnostics_bonly__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.root"
    output: f"{output_dir}datacard_postfit__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.pdf"
    log: f"{config['output_path']}logs/postfit_HH4b__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.log"
    params:
        signallabel="ggHH_kl_1_kt_1_13p0TeV_hbbhbb",
        channel="HH4b",
        signal=config['channels']['HH4b']['signal'],
        ylog="--log",
        plot_script=lambda wildcards: config.get("postfit_plot_script", "src/stat_analysis/plots/make_postfit_plot.py"),
        metadata_template=lambda wildcards: config.get("metadata_template", "coffea4bees/stats_analysis/metadata/{channel}.yml")

# GoF rules from barista
if not stat_only:
    use rule gof_data from combine
    use rule gof_toys_chunk from combine

    num_toy_jobs = int(config.get("num_toy_jobs", 10))
    gof_toy_chunk_inputs = [
        f"{output_dir}datacard_gof_toys_{i}__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.root"
        for i in range(num_toy_jobs)
    ]

    use rule gof from combine with:
        input:
            data=f"{output_dir}datacard_gof_data__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.root",
            toys=gof_toy_chunk_inputs
        output: f"{output_dir}datacard_gof__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.pdf"
        log: f"{config['output_path']}logs/gof_HH4b__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.log"
        params:
            signallabel="ggHH_kl_1_kt_1_13p0TeV_hbbhbb"

    # Impacts rules from barista
    use rule impacts from combine with:
        input: f"{base_workspace}__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.root"
        output: f"{output_dir}datacard_impacts__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.pdf"
        log: f"{config['output_path']}logs/impacts_HH4b__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.log"
        params:
            signallabel="ggHH_kl_1_kt_1_13p0TeV_hbbhbb"

