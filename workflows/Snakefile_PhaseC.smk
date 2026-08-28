# coffea4bees/workflows/Snakefile_PhaseC.smk
# Phase C: Master FvT Classifier Workflow (Inputs -> Training -> Evaluation)

import os

# Sub-workflow includes
include: "Snakefile_PhaseC_1_inputs.smk"
include: "Snakefile_PhaseC_2_train.smk"
include: "Snakefile_PhaseC_3_evaluate.smk"

rule all_PhaseC:
    input:
        rules.all_classifier_inputs.input,
        rules.all_fvt_training.input,
        rules.all_fvt_evaluation.input
