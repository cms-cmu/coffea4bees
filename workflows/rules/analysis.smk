rule analysis_processor:
    input:
        runner_script = "runner.py",
        config_file = lambda wildcards, params: params.config,
    output: "{output_file}"
    # container: config.get("analysis_container", "")
    retries: 3
    params:
        datasets = "",
        years = "",
        config = lambda wildcards: workflow.configfiles[0] if workflow.configfiles else "coffea4bees/workflows/config/lowpt_run2.yml",
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "--not-do-proxy" if config.get("not_do_proxy", False) else "",
            "--blind" if config.get("blind", False) else "",
            "--condor" if config.get("run_on_condor", False) else "",
            "--run-performance" if config.get("run_performance", False) else "",
            "-t" if config.get("test", False) else "",
            f"--dashboard-address {config.get('dashboard_address')}" if config.get("dashboard_address") else "",
            config.get("additional_parameters", "")
        ])),
        run_container_wrapper = "",
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log: "output/logs/analysis_processor_{output_file}.log"
    shell:
        """
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.run_container_wrapper} {params.python_bin} runner.py {params.config} \
            --datasets {params.datasets} \
            --years {params.years} \
            --output-path $(dirname {output})/ \
            --output $(basename {output}) \
            {params.extra_arguments} 2>&1 | tee {log}
        """
            # --tmpdir {resources.tmpdir} \


rule merging_coffea_files:
    input:
        files = "{input_files}",
        script = "src/tools/merge_coffea_files.py"
    output: "{output_file}"
    container: config["analysis_container"]
    params:
        run_performance = False,
        run_container_wrapper = "",
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log: "logs/merging_coffea_files_{output_file}.log"
    shell:
        """
        echo "Merging all the coffea files" 2>&1 | tee -a {log}
        if [ "{params.run_performance}" = "True" ]; then
            cmd="{params.run_container_wrapper} {params.python_bin} -m mprof run -C -o /tmp/mprofile_merge_$(basename {log} .log).dat src/tools/merge_coffea_files.py -f {input.files} -o {output}"
        else
            cmd="{params.run_container_wrapper} {params.python_bin} src/tools/merge_coffea_files.py -f {input.files} -o {output}"
        fi
        echo $cmd 2>&1 | tee -a {log}
        $cmd 2>&1 | tee -a {log}
        echo "Output file size: $(ls -lh {output})" 2>&1 | tee -a {log}
        sync
        """

rule make_JCM:
    input: "output/histNoJCM.coffea"
    output: "output/JCM/jetCombinatoricModel_SB_reana.yml"
    container: config["analysis_container"]
    params:
        extra_arguments = "",
        tag = "2024_v2",
        output_dir = "output/JCM/",
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log: "logs/make_JCM.log"
    shell:
        """
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
    container: config["analysis_container"]
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
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR

        echo "Making plots" 2>&1 | tee -a {log}
        {params.run_container_wrapper} {params.python_bin} coffea4bees/plots/makePlots.py {input.coffea_file} -o {params.output_dir} -m {params.metadata} {params.extra_arguments} 2>&1 | tee -a {log}

        echo "Converting plots to png format" 2>&1 | tee -a {log}
        {params.run_container_wrapper} {params.python_bin} src/plotting/pb_pdf_to_png.py -r -j {params.png_cores} {params.output_dir} 2>&1 | tee -a {log}
        touch {output}
        """

rule check_cutflow:
    input:
        coffea_file = "{output_path}histAll_{label}.coffea"
    output:
        validation_txt = "{output_path}cutflow_validation_{label}.txt",
        cutflow_yml = "{output_path}cutflow_{label}.yml"
    container: config.get("analysis_container", "")
    params:
        known_flag = lambda wildcards: f'--known-cutflow "{config["known_counts"]}"' if config.get("known_counts") else "",
        error_threshold = lambda wildcards: config.get("error_threshold", "0.001"),
        cutflow_list = lambda wildcards: config.get("cutflow_list", "passJetMult,passPreSel,passDiJetMass,SR,SB"),
        run_container_wrapper = "",
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log:
        "{output_path}logs/cutflow_validation_{label}.log"
    shell:
        """
        mkdir -p $(dirname {output.validation_txt}) $(dirname {log})
        echo "Running cutflow analysis and verification for {input.coffea_file}" > {log}
        {params.run_container_wrapper} bash coffea4bees/scripts/run-cutflow.sh \
            --input-file "{input.coffea_file}" \
            --output-file "{output.cutflow_yml}" \
            {params.known_flag} \
            --error-threshold "{params.error_threshold}" \
            --cutflow-list "{params.cutflow_list}" 2>&1 | tee -a {log}
        touch {output.validation_txt}
        """
