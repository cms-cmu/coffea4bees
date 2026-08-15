rule analysis_processor:
    input:
        runner_script = "runner.py",
        config_file = lambda wildcards, params: params.config,
        processor_script = lambda wildcards, params: params.processor,
        friend_metadata = lambda wildcards, params: params.friends if params.friends else [],
        datasets_dir = lambda wildcards, params: params.datasets_file
    output: "{output_file}"
    # container: config.get("analysis_container", "")
    retries: 3
    params:
        datasets = "",
        years = "",
        config = "coffea4bees/analysis/metadata/HH4b_noJCM.yml",
        processor = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file = config.get("datasets", "datasets/"),
        friends = "",
        weights = "",
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "--not-do-proxy" if config.get("not_do_proxy", False) else "",
            "--blind" if config.get("blind", False) else "",
            "--condor" if config.get("run_on_condor", False) else "",
            "--run-performance" if config.get("run_performance", False) else "",
            "-t" if config.get("test", False) else "",
            f"--dashboard-address {config.get('dashboard_address')}" if config.get("dashboard_address") else "",
            config.get("additional_parameters", "")
        ])),
        run_container_wrapper = ""
    log: "output/logs/analysis_processor_{output_file}.log"
    shell:
        """
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.run_container_wrapper} python runner.py -c {params.config} \
            --processor {params.processor} \
            --metadata {params.datasets_file} \
            --datasets "{params.datasets}" \
            --friends "{params.friends}" \
            --weights "{params.weights}" \
            --years "{params.years}" \
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
        # Prefix for the merge command. The `container:` directive above is NOT
        # honored when snakemake is launched via `./run_container snakemake` (it
        # runs in the pixi env, which lacks coffea), so LPC consumers set this to
        # "./run_container" to re-enter the analysis container. Default "" keeps
        # behavior unchanged for environments that do honor `container:` (reana).
        run_container_wrapper = ""
    log: "logs/merging_coffea_files_{output_file}.log"
    shell:
        """
        # export PYTHONPATH="$$PYTHONPATH:$$(pwd)/src"
        # # Set matplotlib config directory to avoid permission issues
        # mkdir -p /tmp/barista/
        
        # # Set matplotlib and fontconfig directories to avoid permission issues
        # export MPLCONFIGDIR="/tmp/barista/matplotlib"
        # export FONTCONFIG_PATH="/tmp/barista/fontconfig"
        # mkdir -p $MPLCONFIGDIR
        # mkdir -p $FONTCONFIG_PATH
        
        echo "Merging all the coffea files" 2>&1 | tee -a {log}
        if [ "{params.run_performance}" = "True" ]; then
            cmd="{params.run_container_wrapper} mprof run -C -o /tmp/mprofile_merge_$(basename {log} .log).dat python src/tools/merge_coffea_files.py -f {input.files} -o {output}"
        else
            cmd="{params.run_container_wrapper} python src/tools/merge_coffea_files.py -f {input.files} -o {output}"
        fi
        echo $cmd 2>&1 | tee -a {log}
        $cmd 2>&1 | tee -a {log}
        echo "Output file size: $(ls -lh {output})" 2>&1 | tee -a {log}
        sync
        sleep 30
        if [ "{params.run_performance}" = "True" ]; then
            echo "Running performance analysis" 2>&1 | tee -a {log}
            mkdir -p $(dirname {output})/performance/
            mprof plot -o $(dirname {output})/performance/mprofile_merge_$(basename {log} .log).png /tmp/mprofile_merge_$(basename {log} .log).dat
        fi
        """

rule make_JCM:
    input: "output/histNoJCM.coffea"
    output: "output/JCM/jetCombinatoricModel_SB_reana.yml"
    container: config["analysis_container"]
    params:
        extra_arguments = "",
        tag = "2024_v2",
        output_dir = "output/JCM/",
    log: "logs/make_JCM.log"
    shell:
        """
        # Set matplotlib config directory to avoid permission issues
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR
        
        echo "Computing JCM" 2>&1 | tee -a {log}
        python coffea4bees/analysis/jcm_tools/make_jcm_weights.py -o {params.output_dir} -r SB -i {input} {params.extra_arguments} -w {params.tag} 2>&1 | tee -a {log}
        ls {params.output_dir}
        """

rule make_plots:
    input:
        coffea_file = "output/histAll.coffea",
        metadata_file = lambda wildcards, params: params.metadata,
        plot_script = "coffea4bees/plots/makePlots.py"
    output: "output/plots/RunII/passPreSel/fourTag/SB/nPVs.pdf"
    container: config["analysis_container"]
    params:
        output_dir = "output/plots/",
        metadata = "coffea4bees/plots/metadata/plotsAll.yml",
        extra_arguments = "-s xW",
        png_cores = 4,
        run_container_wrapper = ""
    log: "logs/make_plots.log"
    shell:
        """
        # Set matplotlib config directory to avoid permission issues
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR

        echo "Making plots" 2>&1 | tee -a {log}
        {params.run_container_wrapper} python coffea4bees/plots/makePlots.py {input.coffea_file} -o {params.output_dir} -m {params.metadata} {params.extra_arguments} 2>&1 | tee -a {log}

        echo "Converting plots to png format" 2>&1 | tee -a {log}
        {params.run_container_wrapper} python src/plotting/pb_pdf_to_png.py -r -j {params.png_cores} {params.output_dir} 2>&1 | tee -a {log}
        touch {output}
        """

rule check_cutflow:
    input:
        coffea_file = "{output_path}histAll_{label}.coffea"
    output:
        "{output_path}cutflow_validation_{label}.txt"
    container: config.get("analysis_container", "")
    params:
        known_counts = lambda wildcards: config.get("known_counts", ""),
        error_threshold = lambda wildcards: config.get("error_threshold", "0.001"),
        run_container_wrapper = ""
    log:
        "{output_path}logs/cutflow_validation_{label}.log"
    shell:
        """
        mkdir -p $(dirname {output}) $(dirname {log})
        echo "Running cutflow validation for {input.coffea_file}" > {log}
        {params.run_container_wrapper} python coffea4bees/analysis/tests/cutflow_test.py \
            --inputFile {input.coffea_file} \
            --knownCounts {params.known_counts} \
            --error_threshold {params.error_threshold} 2>&1 | tee -a {log}
        touch {output}
        """
