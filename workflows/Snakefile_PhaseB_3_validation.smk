# coffea4bees/workflows/Snakefile_PhaseB_3_validation.smk
# Phase B.3: Gatekeeper validation of JCM fit, NoJCM/wJCM cutflows, and classifier_inputs friend trees.
# Included by Snakefile_PhaseB.smk after Snakefile_PhaseB_1_computeJCM.smk.

phaseB_validation_path = f"{config['output_path']}phaseB_validation.txt"

rule phaseB_validation:
    input:
        jcm = ALL_JCM_FILES,
        friends = f"{CI_OUTPUT_PATH}classifier_inputs_friends.json",
        cf_nojcm = f"{JCM_OUTPUT_PATH}cutflow_NoJCM.yml",
        cf_wjcm = f"{JCM_OUTPUT_PATH}cutflow_wJCM.yml",
        script = "coffea4bees/workflows/scripts/validate_phaseB.py"
    output:
        phaseB_validation_path
    log:
        f"{config['output_path']}logs/phaseB_validation.log"
    params:
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = lambda wildcards: config.get("python_bin", "python"),
        max_files = lambda wildcards: 0 if config.get("test", False) else 10
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.run_container_wrapper} {params.python_bin} {input.script} \
            --jcm {input.jcm} \
            --friends {input.friends} \
            --cutflow-nojcm {input.cf_nojcm} \
            --cutflow-wjcm {input.cf_wjcm} \
            --max-files {params.max_files} \
            2>&1 | tee {log} {output}
        """

localrules: phaseB_validation
