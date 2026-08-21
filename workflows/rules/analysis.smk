rule analysis_processor:
    input:
        runner_script = "runner.py",
        corrections_file = "src/physics/corrections.yml",
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
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log: "logs/make_JCM.log"
    shell:
        """
        set -eo pipefail
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR
        
        echo "Computing JCM" 2>&1 | tee -a {log}
        {params.python_bin} coffea4bees/analysis/jcm_tools/make_jcm_weights.py -o {params.output_dir} -r SB -i {input} {params.extra_arguments} -w {params.tag} 2>&1 | tee -a {log}
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
        extra_arguments = "-s xW",
        png_cores = 4,
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

        echo "Converting plots to png format" 2>&1 | tee -a {log}
        {params.run_container_wrapper} {params.python_bin} src/plotting/pb_pdf_to_png.py -r -j {params.png_cores} {params.output_dir} 2>&1 | tee -a {log}
        touch {output}
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
    return ""

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
        set -eo pipefail
        mkdir -p $(dirname {output.validation_txt}) $(dirname {log})
        echo "Running cutflow analysis and verification for {input[0]}" > {log}
        {params.run_container_wrapper} bash coffea4bees/scripts/run-cutflow.sh \
            --input-file "{input[0]}" \
            --output-file "{output.cutflow_yml}" \
            {params.known_flag} \
            --error-threshold "{params.error_threshold}" \
            --cutflow-list "{params.cutflow_list}" \
            --python-bin "{params.python_bin}" 2>&1 | tee -a {log}
        touch {output.validation_txt}
        """
