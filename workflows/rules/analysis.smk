rule analysis_processor:
    input:
        runner_script = "runner.py",
        config_file = lambda wildcards: workflow.configfiles[0] if workflow.configfiles else "coffea4bees/workflows/config/nominal_run2.yml"
    output: "{output_file}"
    retries: 3
    params:
        datasets = "",
        years = "",
        config = lambda wildcards, input: input.config_file if hasattr(input, "config_file") else (input[1] if len(input) > 1 else (workflow.configfiles[0] if workflow.configfiles else "coffea4bees/workflows/config/lowpt_run2.yml")),
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-t" if config.get("test", False) else "",
            config.get("additional_parameters", "")
        ])),
        run_container_wrapper = "",
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log: "output/logs/analysis_processor_{output_file}.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})

        {params.run_container_wrapper} {params.python_bin} runner.py {params.config} \
            --datasets {params.datasets} \
            --years {params.years} \
            --output-path $(dirname {output})/ \
            --output $(basename {output}) \
            {params.extra_arguments} 2>&1 | tee {log}
        """


rule merging_coffea_files:
    input:
        files = "{input_files}"
    output: "{output_file}"
    container: config.get("analysis_container", None)
    params:
        run_performance = False,
        run_container_wrapper = "",
        python_bin = lambda wildcards: config.get("python_bin", "python"),
        input_files = lambda wildcards, input: " ".join([f for f in (input.files if hasattr(input, 'files') else input) if not f.endswith('.py')])
    log: "logs/merging_coffea_files_{output_file}.log"
    shell:
        """
        set -eo pipefail
        echo "Merging all the coffea files" 2>&1 | tee -a {log}
        if [ "{params.run_performance}" = "True" ]; then
            cmd="{params.run_container_wrapper} {params.python_bin} -m mprof run -C -o /tmp/mprofile_merge_$(basename {log} .log).dat src/tools/merge_coffea_files.py -f {params.input_files} -o {output}"
        else
            cmd="{params.run_container_wrapper} {params.python_bin} src/tools/merge_coffea_files.py -f {params.input_files} -o {output}"
        fi
        echo $cmd 2>&1 | tee -a {log}
        $cmd 2>&1 | tee -a {log}
        echo "Output file size: $(ls -lh {output})" 2>&1 | tee -a {log}
        sync
        """

rule make_JCM:
    input: "output/histNoJCM.coffea"
    output: "output/JCM/jetCombinatoricModel_SB_reana.yml"
    container: config.get("analysis_container", None)
    params:
        extra_arguments = "",
        tag = "2024_v2",
        output_dir = "output/JCM/",
        run_container_wrapper = "",
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log: "logs/make_JCM.log"
    shell:
        """
        set -eo pipefail
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR
        
        echo "Computing JCM" 2>&1 | tee -a {log}
        {params.run_container_wrapper} {params.python_bin} coffea4bees/analysis/jcm_tools/make_jcm_weights.py -o {params.output_dir} -r SB -i {input} {params.extra_arguments} -w {params.tag} 2>&1 | tee -a {log}
        ls {params.output_dir}
        """

rule make_plots:
    input:
        coffea_file = "output/histAll.coffea",
        metadata_file = lambda wildcards, params: params.metadata,
        plot_script = "coffea4bees/plots/makePlots.py"
    output: "output/plots/plots_done.txt"
    container: config.get("analysis_container", None)
    params:
        output_dir = "output/plots/",
        metadata = "coffea4bees/plots/metadata/plotsAll.yml",
        extra_arguments = "-s xW -f png",
        run_container_wrapper = "",
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log: "logs/make_plots.log"
    shell:
        """
        set -eo pipefail
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR

        echo "Making plots" 2>&1 | tee -a {log}
        {params.run_container_wrapper} {params.python_bin} coffea4bees/plots/makePlots.py {input[0]} -o {params.output_dir} -m {params.metadata} {params.extra_arguments} 2>&1 | tee -a {log}
        # HTML gallery (barista src/plotting/make_gallery.py); skipped when the barista checkout predates it (e.g. CI against master)
        if [ -f src/plotting/make_gallery.py ]; then
            echo "Making gallery" 2>&1 | tee -a {log}
            {params.run_container_wrapper} {params.python_bin} src/plotting/make_gallery.py {params.output_dir} -m {params.metadata} --title "$(basename {params.output_dir})" 2>&1 | tee -a {log}
        else
            echo "src/plotting/make_gallery.py not found in this barista checkout; skipping gallery" 2>&1 | tee -a {log}
        fi
        touch {output}
        """

rule cutflow_closure_table:
    # Background-closure view of a cutflow dump (src/tools/cutflow_closure.py in barista):
    # cuts as rows, data 3b | tt 3b | Multijet | tt 4b | Bkg | data 4b | data/Bkg, all years + per year.
    # params.multijet: "data3b-tt3b" (JCM-only model, Phase B) or "data3b" (3b data already carries
    # the JCM x FvT weight and models multijet + 3b ttbar, Phase C.4 / F).
    input:
        cutflow_yml = "{output_path}cutflow_{label}.yml",
        validation_txt = "{output_path}cutflow_validation_{label}.txt"
    output:
        html = "{output_path}cutflow_{label}.html",
        txt = "{output_path}cutflow_{label}_table.txt"
    wildcard_constraints:
        output_path = ".*/",
        label = "[^/]+"
    log: "{output_path}logs/cutflow_closure_{label}.log"
    params:
        # no spaces/parentheses: run_container re-joins its arguments for `bash -c`, so quoting is lost
        title = lambda wildcards: f"{config.get('label', 'analysis')}_cutflow_{wildcards.label}",
        multijet = "data3b-tt3b",
        # ttbar process names in the dump: the MC samples, or "TTbar_from_d3" for the FvT-derived
        # estimate from 3b data (plot_ttbar_with_weights; Phase F runs without ttbar MC)
        ttbar = "TTToHadronic TTToSemiLeptonic TTTo2L2Nu",
        run_container_wrapper = "",
        python_bin = lambda wildcards: config.get("python_bin", "python")
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {log})
        # tool lives in barista (src/tools/cutflow_closure.py); a checkout that predates it (e.g. CI
        # against barista master) gets placeholder outputs instead of a failure
        if [ -f src/tools/cutflow_closure.py ]; then
            {params.run_container_wrapper} {params.python_bin} src/tools/cutflow_closure.py {input.cutflow_yml} \
                -o {output.html} --txt {output.txt} --title {params.title} --multijet {params.multijet} \
                --ttbar {params.ttbar} 2>&1 | tee {log}
        else
            echo "src/tools/cutflow_closure.py not found in this barista checkout; skipping closure table" 2>&1 | tee {log}
            echo "<p>cutflow closure table not available (barista checkout predates src/tools/cutflow_closure.py)</p>" > {output.html}
            cp {log} {output.txt}
        fi
        """


rule cutflow_crosscheck:
    # Cross-phase consistency check (src/tools/cutflow_compare.py in barista): compares this
    # pass's cutflow dump, dataset by dataset and cut by cut, with the dump of an earlier pass
    # that ran the same processor with the same weights (e.g. Phase F.1 vs Phase C.4: same data,
    # same JCM x FvT, so weighted and raw 3b/4b counts must agree exactly). The verdict
    # (PASS/FAIL, first line of the txt) is a report, not a gate: the rule succeeds either way so
    # the pages get published; datasets present in only one pass are listed, not failed.
    # params.reference may be missing (pass not run in this production) -> placeholder outputs.
    input:
        cutflow_yml = "{output_path}cutflow_{label}.yml"
    output:
        html = "{output_path}cutflow_crosscheck_{label}.html",
        txt = "{output_path}cutflow_crosscheck_{label}.txt"
    wildcard_constraints:
        output_path = ".*/",
        label = "[^/]+"
    log: "{output_path}logs/cutflow_crosscheck_{label}.log"
    params:
        reference = "",
        title = lambda wildcards: f"{config.get('label', 'analysis')}_cutflow_crosscheck_{wildcards.label}",
        label_a = lambda wildcards: wildcards.label,
        label_b = "reference",
        tolerance = lambda wildcards: config.get("crosscheck_tolerance", "0.001"),
        ignore = "",  # e.g. "data*:counts4*" while the 4b data is blinded in only one of the passes
        run_container_wrapper = "",
        python_bin = lambda wildcards: config.get("python_bin", "python")
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {log})
        if [ ! -f src/tools/cutflow_compare.py ]; then
            echo "src/tools/cutflow_compare.py not found in this barista checkout; skipping cross-check" 2>&1 | tee {log}
            echo "<p>cutflow cross-check not available (barista checkout predates src/tools/cutflow_compare.py)</p>" > {output.html}
            cp {log} {output.txt}
        elif [ -z "{params.reference}" ] || [ ! -f "{params.reference}" ]; then
            echo "SKIPPED: reference cutflow '{params.reference}' not found; nothing to compare {input.cutflow_yml} against" 2>&1 | tee {log}
            echo "<p>cutflow cross-check skipped: reference cutflow <code>{params.reference}</code> not found</p>" > {output.html}
            cp {log} {output.txt}
        else
            IGNORE=""
            [ -n "{params.ignore}" ] && IGNORE="--ignore {params.ignore}"
            {params.run_container_wrapper} {params.python_bin} src/tools/cutflow_compare.py {input.cutflow_yml} {params.reference} \
                -o {output.html} --txt {output.txt} --title {params.title} \
                --label-a {params.label_a} --label-b {params.label_b} --tolerance {params.tolerance} $IGNORE 2>&1 | tee {log}
        fi
        """


def get_known_cutflow_flag(wildcards):
    import os
    label = getattr(wildcards, 'label', config.get('label', ''))
    is_test = config.get("test", False)
    if is_test:
        counts_file = config.get("known_counts_test") or config.get("known_counts") or f"coffea4bees/analysis/tests/known_Counts_{label}.yml"
    else:
        counts_file = config.get("known_counts_full") or config.get("known_counts") or f"coffea4bees/analysis/tests/known_fullCounts_{label}.yml"
    
    if counts_file and os.path.exists(counts_file):
        return f'--known-cutflow "{counts_file}"'
    return '--known-cutflow "none"'

rule check_cutflow:
    input:
        coffea_file = "{output_path}histAll_{label}.coffea"
    output:
        validation_txt = "{output_path}cutflow_validation_{label}.txt",
        cutflow_yml = "{output_path}cutflow_{label}.yml"
    container: config.get("analysis_container", "")
    params:
        known_flag = get_known_cutflow_flag,
        error_threshold = lambda wildcards: config.get("error_threshold", "0.001"),
        cutflow_list = lambda wildcards: config.get("cutflow_list", "passJetMult,passPreSel,passDiJetMass,SR,SB"),
        run_container_wrapper = "",
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log:
        "{output_path}logs/cutflow_validation_{label}.log"
    shell:
        """
        set -o pipefail
        mkdir -p $(dirname {output.validation_txt}) $(dirname {log})
        echo "Running cutflow analysis and verification for {input[0]}" > {log}
        set +e
        {params.run_container_wrapper} bash coffea4bees/scripts/run-cutflow.sh \
            --input-file "{input[0]}" \
            --output-file "{output.cutflow_yml}" \
            {params.known_flag} \
            --error-threshold "{params.error_threshold}" \
            --cutflow-list "{params.cutflow_list}" \
            --python-bin "{params.python_bin}" 2>&1 | tee -a {log}
        status=$?
        set -e
        # Keep the comparison verdict (observed vs expected table) next to the cutflow yml:
        # logs/ are not published by roast, and snakemake deletes the declared outputs of a
        # failed job, so the verdict and the counts go to undeclared *_result.txt / *_failed.yml
        # siblings that survive a failure and still get published.
        result="$(dirname {output.validation_txt})/$(basename {output.validation_txt} .txt)_result.txt"
        ( grep -A80 "Running cutflow comparison" {log} || grep "Skipping cutflow comparison" {log} || true ) > "$result"
        if [ $status -ne 0 ]; then
            [ -f "{output.cutflow_yml}" ] && cp "{output.cutflow_yml}" "$(dirname {output.cutflow_yml})/$(basename {output.cutflow_yml} .yml)_failed.yml"
            echo "############### Cutflow check FAILED (exit $status): see $result" | tee -a {log}
            exit $status
        fi
        cp "$result" {output.validation_txt}
        """
