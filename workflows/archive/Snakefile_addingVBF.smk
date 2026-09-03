config = {
    "analysis_container": "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest",
    "input_path": "output/coffea4bees_20250616_af478bd_unblind_boostedVeto_wdataUL17B_wVBF_boostedVetoVBF",
    "output_path": "output/coffea4bees_20250616_af478bd_unblind_boostedVeto_wdataUL17B_wVBF_boostedVetoVBF",
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

_roc = config.get('run_on_condor', True)
if isinstance(_roc, str):
    config['run_on_condor'] = _roc.lower() not in ('false', '0', 'no')
else:
    config['run_on_condor'] = bool(_roc)

# Adding specific rules for analysis
module analysis:
    snakefile: "rules/analysis.smk"
    config: config
module stat_analysis:
    snakefile: "rules/stat_analysis.smk"
    config: config

combine_config = config.copy()
combine_config["output_path"] = os.path.join(config["output_path"], "stat_analysis/HH4b/")
combine_config["combine_container"] = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_14_1_0_pre4-combine_v10.6.0-harvester_v3.1.0"

module combine:
    snakefile: os.path.join(os.getcwd(), "src/stat_analysis/combine.smk")
    config: combine_config

include: "helpers/common.smk"

rule all:
    input:
        f"{config['output_path']}/stat_analysis/HH4b/limits/datacard_limits__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.json"


use rule merging_coffea_files from analysis as merge_signals with:
    input: expand([f'{config["output_path"]}/singlefiles/histsyst_{{idatsyst}}-{{iyear}}.coffea'], idatsyst=config['GluGlu'], iyear=config['year']) + expand([f'{config["output_path"]}/singlefiles/histsyst_others_{{idatsyst}}-{{iyear}}.coffea'], idatsyst=config['VBF'], iyear=config['year']) + expand([f'{config["output_path"]}/singlefiles/histsyst_jes_{{idatsyst}}-{{iyear}}.coffea'], idatsyst=config['VBF'], iyear=config['year'])
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
    output: f"{config['output_path']}/stat_analysis/HH4b/datacards/datacard__HH4b.txt"
    params:
        variable= "SvB_MA.ps_hh",
        rebin=1,
        metadata="coffea4bees/stats_coffea4bees/analysis/metadata/HH4b.yml",
        output_dir=f"{config['output_path']}/stat_analysis/HH4b/datacards/",
        variable_binning="",
        stat_only="",
        signal="HH4b",
        container_wrapper = config['container_wrapper'],
        syst_file = "",
        tag_flags = ""
    log: f"{config['output_path']}/logs/make_combine_inputs_HH4b.log"


use rule * from combine
use rule workspace from combine with:
    params:
        poi_maps=lambda wildcards: make_poi_maps(
            signals=["ggHH_kl_1_kt_1_13p0TeV_hbbhbb"] + config["GluGlu"] + config["VBF"],
            poi_ranges=config.get("poi_ranges", "1,-10,10")
        ),
        physics_model = lambda wildcards: config.get("physics_model", "HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel"),
        extra_t2w_args = lambda wildcards: config.get("extra_t2w_args", "--PO verbose")

localrules: all, merge_signals, convert_hist_to_json_signals, make_combine_inputs

if not config.get('run_on_condor', True):
    combine_rules = [
        "workspace", "limits", "significance", "likelihood_scan_snapshot",
        "likelihood_scan_chunk", "likelihood_scan", "impacts_initial_fit",
        "impacts_do_fits", "impacts_collect", "split_impacts", "pdf_to_png",
        "gof_data", "gof_toys_chunk", "gof", "fit_diagnostics_bonly", "fit_diagnostics_sb",
        "postfit"
    ]
    workflow._localrules.update(combine_rules)