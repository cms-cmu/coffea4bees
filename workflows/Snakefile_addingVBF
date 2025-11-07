config = {
    "analysis_container": "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest",
    "input_path": "output/coffea4bees_20250616_af478bd_unblind_boostedVeto_wdataUL17B_wVBF",
    "output_path": "output/coffea4bees_20250616_af478bd_unblind_boostedVeto_wdataUL17B_wVBF",
    "dataset_location": "",
    'dataset': "",
    "container_wrapper": "./run_container combine ",
    'GluGlu': ['GluGluToHHTo4B_cHHH1', 'GluGluToHHTo4B_cHHH2p45', 'GluGluToHHTo4B_cHHH0', 'GluGluToHHTo4B_cHHH5'],
    'VBF': ['VBFHHTo4B_kl_1p00_cv_1p00_c2v_0p00',
                'VBFHHTo4B_kl_1p00_cv_1p00_c2v_1p00',
                'VBFHHTo4B_kl_10p2_cv_m0p012_c2v_0p03',
                'VBFHHTo4B_kl_m0p94_cv_m1p21_c2v_1p94',
                'VBFHHTo4B_kl_m19p3_cv_m0p758_c2v_1p44',
                'VBFHHTo4B_kl_m1p36_cv_m1p60_c2v_2p72',
                'VBFHHTo4B_kl_14p4_cv_1p74_c2v_1p37',
                'VBFHHTo4B_kl_m1p43_cv_m0p962_c2v_0p959',
                'VBFHHTo4B_kl_m5p96_cv_2p12_c2v_3p87',
                'VBFHHTo4B_kl_m3p39_cv_m1p83_c2v_3p57' ],
    'dataset_systematics': {
        'HH4b': ['GluGluToHHTo4B_cHHH1', 
                'GluGluToHHTo4B_cHHH2p45', 
                'GluGluToHHTo4B_cHHH0', 
                'GluGluToHHTo4B_cHHH5',
                'VBFHHTo4B_kl_1p00_cv_1p00_c2v_0p00',
                'VBFHHTo4B_kl_1p00_cv_1p00_c2v_1p00',
                'VBFHHTo4B_kl_10p2_cv_m0p012_c2v_0p03',
                'VBFHHTo4B_kl_m0p94_cv_m1p21_c2v_1p94',
                'VBFHHTo4B_kl_m19p3_cv_m0p758_c2v_1p44',
                'VBFHHTo4B_kl_m1p36_cv_m1p60_c2v_2p72',
                'VBFHHTo4B_kl_14p4_cv_1p74_c2v_1p37',
                'VBFHHTo4B_kl_m1p43_cv_m0p962_c2v_0p959',
                'VBFHHTo4B_kl_m5p96_cv_2p12_c2v_3p87',
                'VBFHHTo4B_kl_m3p39_cv_m1p83_c2v_3p57'
            ],
        # 'zz': ['ZZ4b'],
        # 'zh': ['ZH4b', 'ggZH4b']
    },
    'year': [ 'UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18' ],
    # Each case corresponds to a signal process we want to analyze
    "channels": {
        # "ZZ4b": {
        #     "datacard": "datacard_ZZ4b",
        #     "signallabel": "ZZ_bbbb",
        #     "othersignal": "",
        #     "workspace": "hists/coffea4bees_20250605_0dc846dc_unblinded_ext_ZZZH/datacards/ZZ4b/"
        # },
        # "ZH4b": {
        #     "datacard": "datacard_ZH4b",
        #     "signallabel": "ZH_bbbb",
        #     "othersignal": "",
        #     "workspace": "hists/coffea4bees_20250605_0dc846dc_unblinded_ext_ZZZH/datacards/ZH4b/"
        # },
        "HH4b": {
            "signallabel": "ggHH_kl_1_kt_1_13p0TeV_hbbhbb",
            "othersignal": "ggHH_kl_0_kt_1_13p0TeV_hbbhbb,rggHH_kl_2p45_kt_1_13p0TeV_hbbhbb,rggHH_kl_5_kt_1_13p0TeV_hbbhbb",
        },
    }
}

# Adding specific rules for analysis
module analysis:
    snakefile: "rules/analysis.smk"
    config: config
module stat_analysis:
    snakefile: "rules/stat_analysis.smk"
    config: config
module combine:
    snakefile: "rules/combine.smk"
    config: config

include: "helpers/common.smk"

rule all:
    input:
        # f"{config['output_path']}/histAll_signals_HH4b.coffea",
        f"{config['output_path']}/datacards/HH4b/limits__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.txt"


use rule merging_coffea_files from analysis as merge_signals with:
    input: expand([f'{config["output_path"]}/singlefiles/histsyst_{{idatsyst}}-{{iyear}}.coffea'], idatsyst=config['GluGlu'], iyear=config['year']) + expand([f'{config["output_path"]}/singlefiles/histsyst_others_{{idatsyst}}-{{iyear}}.coffea'], idatsyst=config['VBF'], iyear=config['year']) + expand([f'{config["output_path"]}/singlefiles/histsyst_jes_{{idatsyst}}-{{iyear}}.coffea'], idatsyst=config['VBF'], iyear=config['year']),
    output: f"{config['output_path']}/histAll_signals__HH4b.coffea"
    params:
        run_performance=False
    log: f"{config['output_path']}/logs/merging_signals_HH4b.log"

use rule convert_hist_to_json from stat_analysis as convert_hist_to_json_signals with:
    input: f"{config['output_path']}/histAll_signals__HH4b.coffea"
    output: f"{config['output_path']}/histAll_signals__HH4b.json"
    params:
        syst_flag="-s"
    log: f"{config['output_path']}/logs/convert_hist_to_json_signals__HH4b.log"

use rule make_combine_inputs from stat_analysis as make_combine_inputs with:
    input:
        injson = f"{config['output_path']}/histAll.json",
        injsonsyst = f"{config['output_path']}/histAll_signals__HH4b.json", 
        bkgsyst = f"{config['output_path']}/closureFits/ULHH_kfold/3bDvTMix4bDvT/SvB_MA/rebin1/SR/hh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_rebin1.pkl"
    output: f"{config['output_path']}/datacards/HH4b/datacard__HH4b.txt"
    params:
        variable= "SvB_MA.ps_hh",
        rebin=1,
        metadata="coffea4bees/stats_coffea4bees/analysis/metadata/HH4b.yml",
        output_dir=f"{config['output_path']}/datacards/HH4b/",
        variable_binning="",
        stat_only="",
        signal="HH4b",
        container_wrapper = config['container_wrapper']
    log: f"{config['output_path']}/logs/make_combine_inputs_HH4b.log"


use rule workspace from combine as workspace with:
    input: f"{config['output_path']}/datacards/HH4b/datacard__HH4b.txt"
    output: f"{config['output_path']}/datacards/HH4b/datacard__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.root"
    params:
        signallabel="ggHH_kl_1_kt_1_13p0TeV_hbbhbb",
        othersignal_maps=config['channels']['HH4b']['othersignal'],
        container_wrapper=config["container_wrapper"]
    log: f"{config['output_path']}/logs/workspace_HH4b__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.log"

use rule limits from combine as limits with:
    input: f"{config['output_path']}/datacards/HH4b/datacard__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.root"
    output: 
        txt=f"{config['output_path']}/datacards/HH4b/limits__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.txt",
        json=f"{config['output_path']}/datacards/HH4b/limits__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.json"
    log: f"{config['output_path']}/logs/limits_HH4b__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.log"
    params:
        signallabel="ggHH_kl_1_kt_1_13p0TeV_hbbhbb",
        set_parameters_zero=lambda wildcards: set_parameters_zero('HH4b'),
        freeze_parameters=lambda wildcards: freeze_parameters('HH4b'),
        container_wrapper=config.get("container_wrapper", "./run_container combine")