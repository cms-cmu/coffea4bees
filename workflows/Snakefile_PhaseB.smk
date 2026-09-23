# coffea4bees/workflows/Snakefile_PhaseB.smk
# Phase B: Master Coordinator Workflow for JCM Computation and Classifier Inputs Preparation

include: "Snakefile_PhaseB_1_computeJCM.smk"
include: "Snakefile_PhaseB_2_make_classifier_friendtree.smk"

# ---------------------------------------------------------------------------
# Phase B -> C/D handoff
# ---------------------------------------------------------------------------
# B.1 fits the JCM and B.2 builds the classifier-input manifest; Phases C, D, C.4 and F.1 all
# read them.  Two kinds of consumer, so two transports:
#
#   * C.4 and F.1 run the analysis processor on cmslpc, and jetCombinatoricModel opens the JCM
#     with a plain open() -- so that copy has to be a local file at the repo path the configs
#     name (metadata/weights/JCM/<roast_id>/).  It never crosses a machine: B fits it on the
#     same host, so this is a copy, not a transfer, and C.4 keeps its Snakemake input edge on it.
#   * C and D run on falcon and read both files through the classifier's parse.mapping, which
#     deserializes with fsspec (src/classifier/task/parse/_dict.py) and so handles root:// URLs.
#     Publishing both to the roast's EOS area is what removes the manual per-host copy: nothing
#     has to be shipped into the falcon checkout.
#
# `handoff.eos_base` is that EOS directory.  Leave it unset and only the local copies are made,
# which is what a plain `snakemake --configfile ...` run outside roast wants.
#
# The paths below must agree with what the config's `fvt`/`svb` blocks tell C and D to read.
# The rule logs both destinations so a mismatch shows up in B's log rather than as a classifier
# that trained against the wrong friends.

_handoff = config.get('handoff') or {}
if not isinstance(_handoff, dict):
    _handoff = {}
HANDOFF_EOS = str(_handoff.get('eos_base') or "").rstrip('/')
HANDOFF_JCM = _handoff.get(
    'jcm_file', f"coffea4bees/metadata/weights/JCM/{config['roast_id']}/jetCombinatoricModel_SB_{tag}.yml")
HANDOFF_MANIFEST = _handoff.get(
    'manifest_file', f"coffea4bees/metadata/datasets/classifier_inputs_{config['roast_id']}.json")

rule phaseB_handoff:
    input:
        jcm = jcm_file_path,
        manifest = f"{config['output_path']}classifier_inputs/classifier_inputs_friends.json"
    output:
        jcm = HANDOFF_JCM,
        manifest = HANDOFF_MANIFEST,
        done = f"{config['output_path']}handoff/handoff.done"
    log: f"{config['output_path']}logs/phaseB_handoff.log"
    params:
        eos = HANDOFF_EOS
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.jcm}) $(dirname {output.manifest}) $(dirname {output.done}) $(dirname {log})
        cp -f {input.jcm} {output.jcm}
        cp -f {input.manifest} {output.manifest}
        {{
            echo "handoff local jcm      -> {output.jcm}"
            echo "handoff local manifest -> {output.manifest}"
        }} 2>&1 | tee {log}
        if [ -n "{params.eos}" ]; then
            # Same proxy fallback as the other rules that talk to EOS: roast seeds
            # ./proxy/x509_proxy in the checkout and run_container binds it into the container.
            # ${{X509_USER_PROXY:-}}, not $X509_USER_PROXY: snakemake prefixes every shell block
            # with `set -euo pipefail`, and under nounset a bare reference to an unset variable
            # aborts the rule ("X509_USER_PROXY: unbound variable") before xrdcp ever runs.
            if [ -z "${{X509_USER_PROXY:-}}" ] && [ -f ./proxy/x509_proxy ]; then
                export X509_USER_PROXY="$PWD/proxy/x509_proxy"
            fi
            # -p creates the destination directory; -f overwrites a previous run's copy.
            xrdcp -f -p "{output.jcm}"      "{params.eos}/$(basename {output.jcm})"      2>&1 | tee -a {log}
            xrdcp -f -p "{output.manifest}" "{params.eos}/$(basename {output.manifest})" 2>&1 | tee -a {log}
            {{
                echo "handoff EOS jcm        -> {params.eos}/$(basename {output.jcm})"
                echo "handoff EOS manifest   -> {params.eos}/$(basename {output.manifest})"
            }} 2>&1 | tee -a {log}
        else
            echo "handoff: no handoff.eos_base set, local copies only" 2>&1 | tee -a {log}
        fi
        date > {output.done}
        """

# default_target, not position: Snakemake takes the first rule of the *top-level* Snakefile as
# the default target (rules pulled in by `include:` do not count), so simply adding a rule above
# this one silently steals it -- and the DAG then shrinks to that rule's own inputs. That is not
# a hypothetical: adding phaseB_handoff above dropped B.1's whole wJCM second pass, its plots and
# both cutflow validations from the DAG, which C.4 needs (it reuses hist__TT*__<year>_wJCM.coffea
# for the ttbar). Marking the target explicitly makes the order irrelevant.
rule all_PhaseB:
    default_target: True
    input:
        rules.output_computeJCM.input,
        rules.all_classifier_inputs.input,
        rules.phaseB_handoff.output

localrules: phaseB_handoff
