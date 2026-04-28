rule analysis_processor:
    output: "{output_file}"
    # container: config.get("analysis_container", "")
    params:
        datasets = "",
        years = "",
        config = "coffea4bees/analysis/metadata/HH4b_noJCM.yml",
        processor = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file = config.get("datasets", "datasets/"),
        blind = False,
        run_performance = False,
        friends = "",
        run_on_condor = False,
        not_do_proxy = False,
        extra_arguments = "",
        run_container_wrapper = "",  ###include ./run_container for condor
        dashboard_address = ""
    log: "output/logs/analysis_processor_{output_file}.log"
    shell:
        """
        {params.run_container_wrapper} bash coffea4bees/scripts/run-analysis-processor.sh \
            --processor {params.processor} \
            --config {params.config} \
            --dataset-metadata {params.datasets_file} \
            --datasets "{params.datasets}" \
            --friends "{params.friends}" \
            --year "{params.years}" \
            --output-filename $(basename {output}) \
            --output-base $(dirname {output}) \
            --log {log} \
            $([ "{params.not_do_proxy}" = "True" ] && echo "--not-do-proxy") \
            $([ "{params.blind}" = "True" ] && echo "--blind") \
            $([ "{params.run_on_condor}" = "True" ] && echo "--condor") \
            $([ "{params.run_performance}" = "True" ] && echo "--run-performance") \
            $([ -n "{params.dashboard_address}" ] && echo "--dashboard-address {params.dashboard_address}") \
            --additional-flags {params.extra_arguments}
        """
            # --tmpdir {resources.tmpdir} \


rule merging_coffea_files:
    input: "{input_files}"
    output: "{output_file}"
    container: config["analysis_container"]
    params:
        run_performance = False
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
            cmd="mprof run -C -o /tmp/mprofile_merge_$(basename {log} .log).dat python src/tools/merge_coffea_files.py -f {input} -o {output}"
        else
            cmd="python src/tools/merge_coffea_files.py -f {input} -o {output}"
        fi
        echo $cmd 2>&1 | tee -a {log}
        $cmd 2>&1 | tee -a {log}
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
        python coffea4bees/analysis/jcm_tools/make_jcm_weights.py -o {params.output_dir} -c passPreSel -r SB -i {input} {params.extra_arguments} -w {params.tag} 2>&1 | tee -a {log}
        ls {params.output_dir}
        """

rule make_plots:
    input: "output/histAll.coffea"
    output: "output/plots/RunII/passPreSel/fourTag/SB/nPVs.pdf"
    container: config["analysis_container"]
    params:
        output_dir = "output/plots/",
        metadata = "coffea4bees/plots/metadata/plotsAll.yml",
        extra_arguments = "-s xW",
    log: "logs/make_plots.log"
    shell:
        """
        # Set matplotlib config directory to avoid permission issues
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR
        
        echo "Making plots" 2>&1 | tee -a {log}
        python coffea4bees/plots/makePlots.py {input} -o {params.output_dir} -m {params.metadata} {params.extra_arguments} 2>&1 | tee -a {log}

        echo "Converting plots to png format" 2>&1 | tee -a {log}
        python src/plotting/pb_pdf_to_png.py -r -j 4 {params.output_dir} 2>&1 | tee -a {log}
        """
