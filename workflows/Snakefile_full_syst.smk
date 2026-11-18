# ==============================================================================
# coffea4bees/workflows/Snakefile_full_syst.smk
#
# Master Coordinator for Full Systematics Pipeline (Signal Systs + Stats)
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Master entry-point workflow coordinating the complete full-systematics analysis
# pipeline for 4-b analyses (e.g. ttH(bb), HH(4b)). It chains Stage F.1 (Signal
# Systematic Variations & Quality Gates) with Stage F.2 (Datacard Construction,
# Systematic Variation Plots, Gallery & Combine Statistical Inference), concluding
# with automated deployment to CERNBox / EOS.
#
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │ STAGE F.1: Signal Systematics Analysis & Validation (CPU on cmslpc)   │
#   │   (Snakefile_full_syst_1_analysis.smk)                                 │
#   │ - Parallel processing per dataset, year, and systematic group          │
#   │   (weights / friend-tree variations + kinematic / PyTorch shifts)      │
#   │ - Merge singlefiles into histAll_<label>_signals_syst.coffea           │
#   │ - Strict nominal match verification against upstream stat-only         │
#   │ - Systematic cutflow table & reference count validation (HTML/YML/TXT) │
#   └───────────────────────────────────┬────────────────────────────────────┘
#                                       ▼
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │ STAGE F.2: Combine Datacards, Plots & Inference (CMS Combine Container)│
#   │   (Snakefile_full_syst_2_stats.smk)                                    │
#   │ - Convert nominal & signal systematics histograms to JSON format       │
#   │ - Integrate signal systematics with upstream background closure (pkl)  │
#   │ - Synchronize histogram rebinning with bkg_syst step (e.g. rebin 24)   │
#   │ - Generate shape comparison plots for all systematic variations        │
#   │ - Build interactive systematic variation gallery index.html            │
#   │ - Run Combine statistical inference:                                  │
#   │     * Asymptotic limits & blinded significance                         │
#   │     * Likelihood scans & post-fit distributions                        │
#   │     * Nuisance parameter impacts & grouped/split impacts               │
#   │ - Generate comprehensive HTML / Markdown statistical summary           │
#   └───────────────────────────────────┬────────────────────────────────────┘
#                                       ▼
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │ FINAL DEPLOYMENT: Remote Archive & Notification                        │
#   │ - Automated rsync / copy of analysis and stats products to CERNBox/EOS │
#   │ - Success notification email delivery (when email parameter configured)│
#   └────────────────────────────────────────────────────────────────────────┘
#
# EXECUTION TARGETS:
#   - `all_full_syst_F_1_alias`: Execute Stage F.1 only (signal systematics & tests)
#   - `all_full_syst_F_2_alias`: Execute Stage F.2 only (datacards, plots & combine)
#   - `all_full_syst`: Execute both Stage F.1 and Stage F.2 workflows locally
#   - `final_output`: [Default target] Execute full pipeline and sync to CERNBox/EOS
#
# REQUIRED UPSTREAM PRODUCTS:
#   1. Nominal / Stat-Only Histograms:
#      Specified via `inputs.nominal_hists` or `inputs.upstream_roasts.stat_only`
#      (e.g., PhaseF_1_analysis/histAll_<stat_only>.coffea).
#   2. Background Systematics Closure File:
#      Specified via `inputs.bkg_syst_file` or `inputs.upstream_roasts.bkg_syst`
#      (e.g., hists_closure_3bDvTMix4bDvT_SvB_MA_ps_<channel>_rebin*.pkl).
# ==============================================================================
import os
from datetime import datetime

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/ttHbb_full_syst.yml"

cern_user = config.get("cern_user", os.environ.get("USER", "algomez"))
cern_path = config.get("cern_path", "www/ttHbb/Plots/")
if not cern_path.startswith("/"):
    first_letter = cern_user[0]
    cern_path = f"/eos/user/{first_letter}/{cern_user}/{cern_path}"

config.setdefault('stats_container_wrapper', "./run_container combine")

include: "Snakefile_full_syst_1_analysis.smk"
include: "Snakefile_full_syst_2_stats.smk"

rule final_output:
    default_target: True
    input:
        rules.all_full_syst_F_1.input,
        rules.all_full_syst_F_2.input
    params:
        output_dir = f"{datetime.now().strftime('%Y%m%d')}_{config['label']}/",
        cern_path = cern_path,
        email = lambda wildcards: config.get('email', "")
    shell:
        """
        echo "Copying full systematics results to EOS..."
        if [ -f "proxy/x509_proxy" ]; then
            export X509_USER_PROXY="$(pwd)/proxy/x509_proxy"
        fi
        bash src/tools/copy_files_to_cernbox.sh -s {config[output_path]} -d {params.cern_path}{params.output_dir} -t || echo "Warning: copy to EOS failed. Skipping remote upload."
        if [ -n "{params.email}" ]; then
            echo "Workflow for {config[label]} completed successfully on $(date)." | mail -s "Snakemake Success: {config[label]}" "{params.email}" || echo "Warning: failed to send success notification email."
        fi
        """

rule all_full_syst:
    input:
        rules.all_full_syst_F_1.input,
        rules.all_full_syst_F_2.input

rule all_full_syst_F_1_alias:
    input:
        rules.all_full_syst_F_1.input

rule all_full_syst_F_2_alias:
    input:
        rules.all_full_syst_F_2.input

localrules: final_output, all_full_syst, all_full_syst_F_1_alias, all_full_syst_F_2_alias
