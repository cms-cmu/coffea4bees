analysis_container="/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest"
combine_container="/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_11_3_4-combine_v9.1.0-harvester_v2.1.0"

OUTPUT_DIR="hists/coffea4bees_20250304_a100c66/datacards/mixeddata/"
INPUT_DIR="hists/coffea4bees_20250304_a100c66/"
# VARBIN="--variable_binning"
VARIABLE="SvB_MA.ps_hh"
VARBIN=" "

rule all:
    input: 
        f"{OUTPUT_DIR}/systs/{VARIABLE.replace('.','_')}_nominal.pdf",
        f"{OUTPUT_DIR}/plots/SvB_MA_postfitplots_prefit.pdf",
        f"{OUTPUT_DIR}/impacts_combine_SvB_MA_exp_HH.pdf"

rule make_combine_inputs:
    input: f"{INPUT_DIR}/closureFits/ULHH_kfold/3bDvTMix4bDvT/SvB_MA/rebin1/SR/hh/hists_closure_3bDvTMix4bDvT_{VARIABLE.replace('.','_')}_rebin1.pkl"
    output: 
        f"{OUTPUT_DIR}/datacard_HHbb_2016.txt",
        f"{OUTPUT_DIR}/shapes.root"
    params:
        rebin=1,
        container = combine_container,
        output=f"{OUTPUT_DIR}",
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
    input: f"{OUTPUT_DIR}/datacard_HHbb_2016.txt"
    output: 
        f"{OUTPUT_DIR}/datacard.txt",
        f"{OUTPUT_DIR}/datacard.root"
    params:
        container = combine_container,
        output=f"{OUTPUT_DIR}/"
    shell:
        '''
        source workflows/helpers/shell_combine.sh {params.container} \
            source stats_analysis/run_combine.sh {params.output} --limits --unblind
        '''

rule make_syst_plots:
    input: 
        root=f"{OUTPUT_DIR}/shapes.root",
        datacard=f"{OUTPUT_DIR}/datacard.txt"
    output: f"{OUTPUT_DIR}/systs/{VARIABLE.replace('.','_')}_nominal.pdf"
    params:
        container = combine_container,
        output=f"{OUTPUT_DIR}/systs/",
        variable=f"{VARIABLE.replace('.','_')}"
    shell:
        """
        source workflows/helpers/shell_combine.sh {params.container} \
            python3 coffea4bees/plots/make_syst_plots.py -i {input.root} -o {params.output} -d {input.datacard} -v {params.variable}
        """

rule run_postfit:
    input: f"{OUTPUT_DIR}/datacard.root"
    output: f"{OUTPUT_DIR}/plots/SvB_MA_postfitplots_prefit.pdf"
    params:
        container = combine_container,
        output_dir=f"{OUTPUT_DIR}"
    shell:
        """
        source workflows/helpers/shell_combine.sh {params.container} \
            source stats_analysis/run_combine.sh {params.output_dir} --postfit --unblind
        source workflows/helpers/shell_combine.sh {params.container} \
            python3 coffea4bees/plots/make_postfit_plot.py -i {params.output_dir}/fitDiagnostics_SvB_MA_unblinded_prefit_sb.root -o {params.output_dir}/plots/ -t prefit
        source workflows/helpers/shell_combine.sh {params.container} \
            python3 coffea4bees/plots/make_postfit_plot.py -i {params.output_dir}/fitDiagnostics_SvB_MA_unblinded_prefit_sb.root -o {params.output_dir}/plots/ -t fit_s
        source workflows/helpers/shell_combine.sh {params.container} \
            python3 coffea4bees/plots/make_postfit_plot.py -i {params.output_dir}/fitDiagnostics_SvB_MA_unblinded_prefit_sb.root -o {params.output_dir}/plots/ -t fit_b
        """

rule run_impacts:
    input: f"{OUTPUT_DIR}/datacard.root"
    output: f"{OUTPUT_DIR}/impacts_combine_SvB_MA_exp_HH.pdf"
    params:
        container = combine_container,
        output_dir=f"{OUTPUT_DIR}"
    shell:
        """
        source workflows/helpers/shell_combine.sh {params.container} \
            source coffea4bees/stats_analysis/run_combine.sh {params.output_dir} --impacts --unblind
        """