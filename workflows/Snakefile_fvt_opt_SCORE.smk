# ==============================================================================
# coffea4bees/workflows/Snakefile_fvt_opt_SCORE.smk
#
# Phase OPT_SCORE: Paired Signal-Projected Scoring & Decision Dashboard (cmslpc)
# ==============================================================================

import os

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/ttHbb_fvt_opt_cat_year.yml"

include: "helpers/fvt_opt_common.smk"

container_wrapper = config.get("container_wrapper", "./run_container")
python_bin = config.get("python_bin", "python3")
nom_hists = (config.get("inputs") or {}).get("nominal_hists") or config.get("nominal_coffea")
if not nom_hists:
    raise ValueError("OPT_SCORE requires inputs.nominal_hists (or nominal_coffea)")

observable = str(OPT_SCORE_CFG.get("observable", "SvB_MA.ps_ttHbb"))
if observable == "SvB_MA.ps":
    observable = "SvB_MA.ps_ttHbb"
primary_rebin = int(OPT_SCORE_CFG.get("rebin", config.get("rebin", 16)))
threshold_sigma = float(OPT_SCORE_CFG.get("threshold_sigma", 2.0))


rule run_fvt_opt_score:
    input:
        f1 = expand(f"{out_opt_f}{{run}}/histAll_mixeddata_bkgs.coffea", run=OPT_RUNS),
        norm = expand(f"{out_opt_f}{{run}}/validation_normalization.done", run=OPT_RUNS),
        a3 = expand(f"{out_a3}histAll_{channel}_mixeddata_v{{m}}.coffea", m=OPT_MIXES),
        script = "coffea4bees/stats_analysis/score_fvt_opt.py",
    output:
        json = f"{out_opt_score}fvt_opt_summary.json",
        html = f"{out_opt_score}fvt_opt_dashboard.html",
    log:
        f"{out_opt_score}logs/score_fvt_opt.log",
    params:
        runs = " ".join(OPT_RUNS),
        mixes = " ".join(OPT_MIXES),
        a3_dir = out_a3,
        channel = channel,
        nom_hists = nom_hists,
        observable = observable,
        rebin = primary_rebin,
        reference = OPT_REFERENCE,
        threshold_sigma = threshold_sigma,
        container_wrapper = container_wrapper,
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.json}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} {input.script} \
            --runs {params.runs} \
            --f1-files {input.f1} \
            --mixes {params.mixes} \
            --a3-dir {params.a3_dir} \
            --channel {params.channel} \
            --nominal-hists '{params.nom_hists}' \
            --hist-key {params.observable} \
            --rebin {params.rebin} \
            --reference {params.reference} \
            --threshold-sigma {params.threshold_sigma} \
            --out-json {output.json} \
            --out-html {output.html} 2>&1 | tee {log}
        """


localrules: all_fvt_opt_SCORE, run_fvt_opt_score

rule all_fvt_opt_SCORE:
    default_target: True
    input:
        f"{out_opt_score}fvt_opt_summary.json",
        f"{out_opt_score}fvt_opt_dashboard.html",
