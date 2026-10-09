# coffea4bees/workflows/Snakefile_MvD_2_train.smk
# MvD roast, step V.2 (falcon): train, analyze and evaluate the MvD classifier (mixed vs data)
# from the `mvd:` config block. Reads V.1's handoff (classifier-input manifest, classifier
# metadata, tight mixed-data JCM) from EOS; writes the model to <eos_base>/classifier/<label>
# and the friend (MvD, p_mix4, p_d4, p_t4, ...) to <eos_base>/friend/<label>, which V.3 and
# V.4 read. See Snakefile_MvD.smk for the whole roast.

include: "helpers/classifier_step.smk"

setup_classifier_step('mvd')

include: "../../src/classifier/workflow/Snakefile"

# default_target, not position: see Snakefile_PhaseD.smk.
rule all_MvD_train:
    default_target: True
    input:
        rules.all.input
