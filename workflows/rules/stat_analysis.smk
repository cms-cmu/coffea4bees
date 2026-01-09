rule convert_hist_to_json:
    input: "{input_file}"
    output: "{output_file}"
    container: config["analysis_container"]
    params:
        syst_flag = " "
    log:
        "logs/{input_file}.log"
    shell:
        """
        mkdir -p $(dirname {log})
        echo "[$(date)] Starting convert_hist_to_json for {input}" > {log}
        python coffea4bees/stats_analysis/convert_hist_to_json.py -o {output} -i {input} {params.syst_flag} 2>&1 | tee -a {log}
        echo "[$(date)] Completed convert_hist_to_json for {input}" >> {log}
        """

rule convert_hist_to_json_closure:
    input: "output/histAll.coffea",
    output: "output/histAll.json",
    container: config["analysis_container"]
    log:
        "logs/convert_hist_to_json_closure.log"
    shell:
        """
        mkdir -p $(dirname {log})
        echo "[$(date)] Starting convert_hist_to_json_closure" > {log}
        python coffea4bees/stats_analysis/convert_hist_to_json_closure.py -o {output} -i {input} 2>&1 | tee -a {log}
        echo "[$(date)] Completed convert_hist_to_json_closure" >> {log}
        """

rule convert_json_to_root:
    input: "output/histMixedBkg_TT.json"
    output: "output/histMixedBkg_TT.root"
    params:
        output_dir = config.get("output_path", "output/"),
        container_wrapper = config.get("container_wrapper", "./run_container combine")
    log:
        "logs/convert_json_to_root.log"
    shell:
        """
        mkdir -p $(dirname {log})
        echo "[$(date)] Starting convert_json_to_root for {input}" > {log}
        {params.container_wrapper} \
            python3 coffea4bees/stats_analysis/convert_json_to_root.py \
            -f {input} \
            --output {params.output_dir} 2>&1 | tee -a {log}

        echo "[$(date)] Completed convert_json_to_root for {input}" >> {log}
        """

rule run_two_stage_closure:
    input: 
        file_TT = "output/histMixedBkg_TT.root",
        file_mix = "output/histMixedData.root",
        file_sig = "output/histAll.root",
        file_data3b = "output/histMixedBkg_data_3b_for_mixed.root"
    output: "output/closureFits/ULHH_kfold/3bDvTMix4bDvT/SvB_MA/varrebin2/SR/hh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_fine_varrebin2.pkl"
    params:
        outputPath = "output/closureFits/ULHH_kfold",
        rebin = "2",
        variable = "SvB_MA_ps_hh_fine",
        extra_arguments = "",
        container_wrapper = config.get("container_wrapper", "./run_container combine")
    log:
        "logs/run_two_stage_closure_{variable}.log"
    shell:
        """
        mkdir -p $(dirname {log})
        echo "[$(date)] Starting run_two_stage_closure for variable {params.variable}" > {log}
        echo "[$(date)] Input files: TT={input.file_TT}, mix={input.file_mix}, sig={input.file_sig}, data3b={input.file_data3b}" >> {log}
        
        {params.container_wrapper} \
            python3 coffea4bees/stats_analysis/runTwoStageClosure.py  \
            --var {params.variable} --rebin {params.rebin} \
            {params.extra_arguments} \
            --outputPath {params.outputPath} \
            --input_file_TT {input.file_TT} \
            --input_file_mix {input.file_mix} \
            --input_file_sig {input.file_sig} \
            --input_file_data3b {input.file_data3b} 2>&1 | tee -a {log}
            
        echo "[$(date)] Completed run_two_stage_closure for variable {params.variable}" >> {log}
        """


rule make_combine_inputs:
    input:
        injson = "output/histAll.json",
        injsonsyst = "output/histAll_signals_cHHHX.json",
        bkgsyst = "output/closureFits/ULHH_kfold/3bDvTMix4bDvT/SvB_MA/varrebin2/SR/hh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_fine_varrebin2.pkl"
    output:
        "output/datacards/datacard__HHbb.txt"
    params:
        variable = "SvB_MA.ps_hh_fine",
        rebin = 2,
        output_dir = "output/datacards/",
        variable_binning =  "--variable_binning",
        stat_only = "--stat_only",
        signal = "HH4b",
        container_wrapper = config.get("container_wrapper", "./run_container combine")
    log:
        "logs/make_combine_inputs_{signal}.log"
    shell:
        """
        mkdir -p $(dirname {log})
        echo "[$(date)] Starting make_combine_inputs for signal {params.signal}" > {log}
        
        echo "[$(date)] Making combine inputs with full stats" | tee -a {log}
        {params.container_wrapper} \
            python3 coffea4bees/stats_analysis/make_combine_inputs.py \
                --var {params.variable} \
                -f {input.injson} \
                --syst_file {input.injsonsyst} \
                --bkg_syst_file {input.bkgsyst} \
                --output_dir {params.output_dir} \
                --rebin {params.rebin} \
                {params.variable_binning} \
                --metadata coffea4bees/stats_analysis/metadata/{params.signal}.yml \
                {params.stat_only} 2>&1 | tee -a {log}
                
        echo "[$(date)] Combining datacards" | tee -a {log}
        {params.container_wrapper} "cd {params.output_dir} && \
            combineCards.py {params.signal}_2016=datacard_{params.signal}_2016.txt \
            {params.signal}_2017=datacard_{params.signal}_2017.txt \
            {params.signal}_2018=datacard_{params.signal}_2018.txt > datacard__{params.signal}.txt" 2>&1 | tee -a {log}
            
        echo "[$(date)] Completed make_combine_inputs for signal {params.signal}" >> {log}
        """

rule make_syst_plots:
    input: "{datacard_file}"
    output: "{output_dir}/{variable}_nominal.pdf"
    params:
        variable="{variable}",
        output_dir="{output_dir}",
        channel="{channel}",
        signal="{signal}",
        container_wrapper = config.get("container_wrapper", "./run_container combine")
    log: "logs/make_syst_plots_{variable}.log"
    shell:
        """
        mkdir -p $(dirname {log})
        echo "[$(date)] Starting make_syst_plots for {params.variable}" > {log}
        echo "Making syst plots" 2>&1 | tee -a {log}
        {params.container_wrapper} \
            python3 coffea4bees/plots/make_syst_plots.py \
                -i {params.output_dir}/shapes.root \
                -o {params.output_dir}/systs/ \
                -d {input} \
                -s {params.signal} \
                -m coffea4bees/stats_analysis/metadata/{params.channel}.yml \
                --variable {params.variable} 2>&1 | tee -a {log}
        echo "[$(date)] Completed make_syst_plots for {params.variable}" >> {log}
        """
