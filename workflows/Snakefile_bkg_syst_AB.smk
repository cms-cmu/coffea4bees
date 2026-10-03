# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_AB.smk
#
# Stages A & B (CPU, cmslpc): closure subsamples from the mixeddata roast (A_1 mixed-data JCM,
# A_2 split, A_3 processing + validation) and the per-subsample JCM fits (B_1), then the EOS
# handoff (bkg_syst_AB_handoff) that Stage C reads on the GPU host.
# Target: all_bkg_syst_AB
# ==============================================================================

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"
include: "Snakefile_bkg_syst_A_1_mixed_jcm.smk"
include: "Snakefile_bkg_syst_A_2_make_subsamples.smk"
include: "Snakefile_bkg_syst_A_3_process_subsamples.smk"
include: "Snakefile_bkg_syst_B_1_computeJCM.smk"

rule all_bkg_syst_AB:
    default_target: True
    input:
        rules.all_bkg_syst_A_1.input,
        rules.all_bkg_syst_A_2.input,
        rules.all_bkg_syst_A_3.input,
        rules.all_bkg_syst_B_1.input,
        rules.bkg_syst_AB_handoff.output

localrules: all_bkg_syst_AB
