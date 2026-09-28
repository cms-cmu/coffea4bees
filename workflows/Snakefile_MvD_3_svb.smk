# coffea4bees/workflows/Snakefile_MvD_3_svb.smk
# MvD roast, step V.3 (falcon): train and evaluate an SvB whose multijet background is the
# four-tag mixed data x (tight mixed-data) JCM x MvD, from the `svb_mvd:` config block. Same
# templates and settings as nominal Phase D (HH4b_Run3/SvB); only the background changes, via
# workflow_modules (HCR.SvB.Background -> HCR.SvB.BackgroundMixed) and the input overrides.
# The friend (<eos_base>/friend/<label>) covers data, mixed data, ttbar and the ggF signal and
# is V.4's SvB_MA. See Snakefile_MvD.smk for the whole roast.

include: "helpers/classifier_step.smk"

setup_classifier_step('svb_mvd')

include: "../../src/classifier/workflow/Snakefile"

# default_target, not position: see Snakefile_PhaseD.smk.
rule all_MvD_svb:
    default_target: True
    input:
        rules.all.input
