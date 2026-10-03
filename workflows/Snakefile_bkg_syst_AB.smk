# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_AB.smk
#
# Stage A & B Master Coordinator: Mixed-Data Generation & JCM Calibration
# Target: all_bkg_syst_AB (CPU on cmslpc)
# ==============================================================================

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/common.smk"
include: "helpers/bkg_syst_common.smk"

# Include Sub-workflows for Stages A and B
include: "Snakefile_bkg_syst_A_1_make_mixeddata.smk"
include: "Snakefile_bkg_syst_A_2_process_subsamples.smk"
include: "Snakefile_bkg_syst_B_1_computeJCM.smk"

rule all_bkg_syst_AB:
    default_target: True
    input:
        rules.all_bkg_syst_A_1.input,
        rules.all_bkg_syst_A_2.input,
        rules.all_bkg_syst_B_1.input,
        rules.bkg_syst_AB_handoff.output

localrules: all_bkg_syst_AB
