import os
username = os.getenv("USER", "coffea4bees_default")

rule analysis_processor:
    output: "{output_file}"
    # container: config["analysis_container"]
    params:
        datasets = "",
        years = "",
        metadata = "coffea4bees/analysis/metadata/HH4b_noJCM.yml",
        processor = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file = config.get("datasets", "datasets/"),
        blind = False,
        run_performance = False,
        extra_arguments = "",
        username = username
    log: "output/logs/analysis_processor.log"
    shell:
        """
        mkdir -p $(dirname {log})
        mkdir -p /tmp/{params.username}/
        
        # Set matplotlib config directory to avoid permission issues
        export MPLCONFIGDIR="/tmp/{params.username}/matplotlib"
        mkdir -p $MPLCONFIGDIR
        
        # Prepare metadata file
        meta_tmp="/tmp/{params.username}/metadata_$(basename {output} .coffea).yml"
        if [ "{params.blind}" = "True" ]; then
            echo "Blinding SR region"
            sed 's/blind.*/blind: true/' {params.metadata} > $meta_tmp
        else
            cp {params.metadata} $meta_tmp
        fi
        
        echo "Running with this metadata file" 2>&1 | tee {log}
        cat $meta_tmp 2>&1 | tee -a {log}
        echo "Running {params.datasets} {params.years} - output {output}" 2>&1 | tee -a {log}
        
        # Set up performance monitoring
        mprofile_dat="/tmp/{params.username}/mprofile_$(basename {log} .log).dat"
        mprofile_png="output/performance/mprofile_$(basename {log} .log).png"
        
        # Run analysis with optional performance monitoring
        cmd="python runner.py -d {params.datasets} -p {params.processor} -y {params.years} -o $(basename {output}) -op $(dirname {output})/ -m {params.datasets_file} -c $meta_tmp {params.extra_arguments}"
        if [ "{params.run_performance}" = "True" ]; then
            cmd="mprof run -C -o $mprofile_dat $cmd"
        fi
        
        echo $cmd 2>&1 | tee -a {log}
        eval ./run_container $cmd 2>&1 | tee -a {log}
        
        # Generate performance plot if requested
        if [ "{params.run_performance}" = "True" ]; then
            echo "Running performance analysis" 2>&1 | tee -a {log}
            mkdir -p output/performance/
            mprof plot -o $mprofile_png $mprofile_dat 2>&1 | tee -a {log}
        fi
        """


rule merging_coffea_files:
    input: "{input_files}"
    output: "{output_file}"
    container: config["analysis_container"]
    params:
        run_performance = False
    log: "logs/merging_{params.logname}.log"
    shell:
        """
        # Set matplotlib config directory to avoid permission issues
        mkdir -p /tmp/barista/
        
        # Set matplotlib and fontconfig directories to avoid permission issues
        export MPLCONFIGDIR="/tmp/barista/matplotlib"
        export FONTCONFIG_PATH="/tmp/barista/fontconfig"
        mkdir -p $MPLCONFIGDIR
        mkdir -p $FONTCONFIG_PATH
        
        echo "Merging all the coffea files" 2>&1 | tee -a {log}
        cmd="mprof run -C -o /tmp/mprofile_merge_$(basename {log} .log).dat python src/tools/merge_coffea_files.py -f {input} -o {output}"
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
