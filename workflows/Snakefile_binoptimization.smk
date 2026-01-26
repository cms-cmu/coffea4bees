analysis_container="/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest"
combine_container="/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_11_3_4-combine_v9.1.0-harvester_v2.1.0"

OUTPUT_DIR="output/coffea4bees_20250301_cc8aa30"
INPUT_DIR="hists/coffea4bees_20250301_cc8aa30"
# REBIN_DIR="VarRebin"
# VARBIN="--variable_binning"
VARIABLE="SvB_MA.ps_hh"
REBIN_DIR="Rebin"
VARBIN=" "

config = {
    "rebin": [
        1,
        # 2,
        # 3,
        # 4,
        # 5, 
        # 8,
        # 10,
        # 16
    ]
}

rule all:
    input: 
        expand( f"{OUTPUT_DIR}/{REBIN_DIR}{{rebins}}/systs/{VARIABLE.replace('.','_')}_nominal.pdf", rebins=config["rebin"] ),
        expand( f"{OUTPUT_DIR}/{REBIN_DIR}{{rebins}}/plots/SvB_MA_postfitplots_prefit.pdf", rebins=config["rebin"] )


rule two_stage_closure:
    output: f"{OUTPUT_DIR}/closureFits/3bDvTMix4bDvT/SvB_MA/{REBIN_DIR.lower()}{{rebins}}/SR/hh/hists_closure_3bDvTMix4bDvT_{VARIABLE.replace('.','_')}_{REBIN_DIR.lower()}{{rebins}}.pkl"
    params:
        rebin="{rebins}",
        container = combine_container,
        outputPath=f"{OUTPUT_DIR}/closureFits/",
        variable=f"{VARIABLE.replace('.','_')}"
    shell:
        '''
        source workflows/helpers/shell_combine.sh {params.container} \
            python3 coffea4bees/stats_analysis/runTwoStageClosure.py  --var {params.variable}  \
                --rebin {params.rebin} --use_kfold {VARBIN} \
                --outputPath {params.outputPath} \
                --input_file_TT {INPUT_DIR}/histMixedBkg_TT.root \
                --input_file_mix {INPUT_DIR}/histMixedData.root \
                --input_file_sig {INPUT_DIR}/histAll.root \
                --input_file_data3b {INPUT_DIR}/histMixedBkg_data_3b_for_mixed.root
        '''

rule make_combine_inputs:
    input: f"{OUTPUT_DIR}/closureFits/3bDvTMix4bDvT/SvB_MA/{REBIN_DIR.lower()}{{rebins}}/SR/hh/hists_closure_3bDvTMix4bDvT_{VARIABLE.replace('.','_')}_{REBIN_DIR.lower()}{{rebins}}.pkl"
    output: 
        f"{OUTPUT_DIR}/{REBIN_DIR}{{rebins}}/datacard_HHbb_2016.txt",
        f"{OUTPUT_DIR}/{REBIN_DIR}{{rebins}}/shapes.root"
    params:
        rebin="{rebins}",
        container = combine_container,
        output=f"{OUTPUT_DIR}/{REBIN_DIR}{{rebins}}",
        variable=f"{VARIABLE}"
    shell:
        '''
        source workflows/helpers/shell_combine.sh {params.container} \
            python3 coffea4bees/stats_analysis/make_combine_inputs.py --var {params.variable} \
                -f {INPUT_DIR}/histAll.json {VARBIN} \
                --syst_file {INPUT_DIR}/histAll_signals_cHHHX.json \
                --bkg_syst_file {input} \
                --output_dir {params.output} \
                --rebin {params.rebin} \
                --metadata stats_analysis/metadata/HH4b.yml \
                --mixeddata_file {INPUT_DIR}/histMixedData.json
        '''

rule make_limits:
    input: f"{OUTPUT_DIR}/{REBIN_DIR}{{rebins}}/datacard_HHbb_2016.txt"
    output: 
        f"{OUTPUT_DIR}/{REBIN_DIR}{{rebins}}/datacard.txt",
        f"{OUTPUT_DIR}/{REBIN_DIR}{{rebins}}/datacard.root"
    params:
        container = combine_container,
        output=f"{OUTPUT_DIR}/{REBIN_DIR}{{rebins}}/"
    shell:
        '''
        source workflows/helpers/shell_combine.sh {params.container} \
            source stats_analysis/run_combine.sh {params.output} --limits
        '''

rule make_syst_plots:
    input: 
        root=f"{OUTPUT_DIR}/{REBIN_DIR}{{rebins}}/shapes.root",
        datacard=f"{OUTPUT_DIR}/{REBIN_DIR}{{rebins}}/datacard.txt"
    output: f"{OUTPUT_DIR}/{REBIN_DIR}{{rebins}}/systs/{VARIABLE.replace('.','_')}_nominal.pdf"
    params:
        container = combine_container,
        output=f"{OUTPUT_DIR}/{REBIN_DIR}{{rebins}}/systs/",
        variable=f"{VARIABLE.replace('.','_')}"
    shell:
        """
        source workflows/helpers/shell_combine.sh {params.container} \
            python3 coffea4bees/plots/make_syst_plots.py -i {input.root} -o {params.output} -d {input.datacard} -v {params.variable}
        """

rule run_postfit:
    input: f"{OUTPUT_DIR}/{REBIN_DIR}{{rebins}}/datacard.root"
    output: f"{OUTPUT_DIR}/{REBIN_DIR}{{rebins}}/plots/SvB_MA_postfitplots_prefit.pdf"
    params:
        container = combine_container,
        output_dir=f"{OUTPUT_DIR}/{REBIN_DIR}{{rebins}}"
    shell:
        """
        source workflows/helpers/shell_combine.sh {params.container} \
            source stats_analysis/run_combine.sh {params.output_dir} --postfit
        source workflows/helpers/shell_combine.sh {params.container} \
            python3 coffea4bees/plots/make_postfit_plot.py -i {params.output_dir}/fitDiagnostics_SvB_MA_prefit_sb.root -o {params.output_dir}/plots/ -t prefit
        source workflows/helpers/shell_combine.sh {params.container} \
            python3 coffea4bees/plots/make_postfit_plot.py -i {params.output_dir}/fitDiagnostics_SvB_MA_prefit_sb.root -o {params.output_dir}/plots/ -t fit_s
        source workflows/helpers/shell_combine.sh {params.container} \
            python3 coffea4bees/plots/make_postfit_plot.py -i {params.output_dir}/fitDiagnostics_SvB_MA_prefit_sb.root -o {params.output_dir}/plots/ -t fit_b
        """