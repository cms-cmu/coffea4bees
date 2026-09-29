# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_A_1_make_mixeddata.smk
#
# Stage A_1: Mixed-Data Production & Validation via Snakefile_MakeMixedData.smk
# ==============================================================================

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "Snakefile_MakeMixedData.smk"

rule all_bkg_syst_A_1:
    default_target: True
    input:
        rules.all_MakeMixedData.input

localrules: all_bkg_syst_A_1
