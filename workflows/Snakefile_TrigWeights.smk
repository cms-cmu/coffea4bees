# coffea4bees/workflows/Snakefile_TrigWeights.smk
# Trigger weights for samples that are already skimmed, plus their validation:
#   A.2  Snakefile_PhaseA_2_trigWeights.smk    trigWeight friends + merged friend index
#   A.3  Snakefile_PhaseA_3_trigValidation.smk  noTrig / HLT / HLT x SF overlays and report
# (Snakefile_PhaseA.smk = A.1 skim + A.2, for samples that still need skimming.)
#
#   ./run_container snakemake -s coffea4bees/workflows/Snakefile_TrigWeights.smk \
#       --configfile coffea4bees/workflows/config/trigweights_run3_signal.yml --cores 4

# Default target; the input function runs after the includes below have defined their rules.
rule all_trigWeights:
    input:
        lambda wildcards: list(rules.all_trigger_weights.input) + list(rules.all_trig_validation.input)

include: "Snakefile_PhaseA_2_trigWeights.smk"
include: "Snakefile_PhaseA_3_trigValidation.smk"
