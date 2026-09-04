# coffea4bees/workflows/Snakefile_PhaseB.smk
# Phase B: Master Coordinator Workflow for JCM Computation and Classifier Inputs Preparation

include: "Snakefile_PhaseB_1_computeJCM.smk"
include: "Snakefile_PhaseB_2_make_classifier_friendtree.smk"

rule all_PhaseB:
    input:
        rules.output_computeJCM.input,
        rules.all_classifier_inputs.input
