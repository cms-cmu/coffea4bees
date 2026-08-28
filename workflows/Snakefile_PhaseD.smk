# coffea4bees/workflows/Snakefile_PhaseD.smk
# Phase D: Master SvB Classifier Workflow (Inputs -> Training -> Evaluation)

import os

# Sub-workflow includes
include: "Snakefile_PhaseD_1_inputs.smk"
include: "Snakefile_PhaseD_2_train.smk"
include: "Snakefile_PhaseD_3_evaluate.smk"

rule all_PhaseD:
    input:
        rules.all_svb_classifier_inputs.input,
        rules.all_svb_training.input,
        rules.all_svb_evaluation.input
