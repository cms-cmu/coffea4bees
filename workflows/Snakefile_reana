from datetime import datetime
import os

# Define username once for reuse throughout the workflow
USERNAME = os.getenv("USER", "coffea4bees_default")

# Import rule modules
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

hash = config['extra_arguments'].split("--githash ")[1]
EOS_OUTPUT = f"/eos/user/a/algomez/www/plots/HH4b/reana/{datetime.now().strftime('%Y%m%d')}_{hash}/"
OUTPUT_PATH = config['output_path']

# Define output patterns at the top (after imports/config)
CHANNELS = list(config['channels'].keys())
SIGNALLABELS = [config['channels'][k]['signallabel'] for k in CHANNELS]
CHANNELLABELS = [k.lower().split('4b')[0] for k in CHANNELS]

OUTPUT_PATTERNS = {
    "limits": f"{config['output_path']}/datacards/{{channel}}/limits__{{signallabel}}.json",
    "significance": f"{config['output_path']}/datacards/{{channel}}/significance__{{signallabel}}.log",
    "impacts": f"{config['output_path']}/datacards/{{channel}}/impacts__{{signallabel}}.pdf",
    "likelihood_scan": f"{config['output_path']}/datacards/{{channel}}/likelihood_scan__{{signallabel}}.pdf",
    "gof": f"{config['output_path']}/datacards/{{channel}}/gof__{{signallabel}}.pdf",
    "postfit": f"{config['output_path']}/datacards/{{channel}}/plots/postfitplots__{{signallabel}}__prefit.pdf",
}

SYST_PLOTS = {
    "HH4b": f"{config['output_path']}/datacards/HH4b/systs/SvB_MA_ps_hh_nominal.pdf",
    "ZZ4b": f"{config['output_path']}/datacards/ZZ4b/systs/SvB_MA_ps_zz_nominal.pdf",
    "ZH4b": f"{config['output_path']}/datacards/ZH4b/systs/SvB_MA_ps_zh_nominal.pdf",
}

rule final_rule:
    input:
        f"{config['output_path']}/plots/RunII/passPreSel/fourTag/SB/nPVs.pdf",
        [expand(pattern, zip, channel=CHANNELS, signallabel=SIGNALLABELS, channellabel=CHANNELLABELS)
         for pattern in OUTPUT_PATTERNS.values()],
        list(SYST_PLOTS.values()),
    container: config["analysis_container"]
    resources:
        voms_proxy=True,
        kerberos=True,
    shell: 
        """
        rm -rf {config[output_path]}/datacards/*/higgsCombine_*
        cp gitdiff.txt {config[output_path]}
        echo "Copying output to cernbox "
        mkdir -p {EOS_OUTPUT}
        cp -r {config[output_path]}/* {EOS_OUTPUT}
        python src/plotting/pb_deploy_plots.py {config[output_path]}/ {EOS_OUTPUT} -r -c -j 4
        """

#######
### Running analysis processor
#######

### In the next rules, the input is commented out to not run JCM again. 
use rule analysis_processor from analysis as analysis_databkgs with:
    # input: f"{config['output_path']}/JCM/jetCombinatoricModel_SB_reana.yml"
    output: f"{config['output_path']}/singlefiles/hist__{{sample}}-{{year}}.coffea"
    params:
        datasets="{sample}",
        years="{year}",
        metadata="coffea4bees/analysis/metadata/HH4b.yml",
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        blind=False,
        run_performance=True,
        extra_arguments=config['extra_arguments'],
        username=USERNAME
    resources:
        voms_proxy=True,
        kerberos=True,
        compute_backend="kubernetes",
        kubernetes_memory_limit="9.5Gi"
    log: f"{config['output_path']}/logs/analysis_hist__{{sample}}-{{year}}.log"


use rule analysis_databkgs as analysis_data with:
    # input: f"{config['output_path']}/JCM/jetCombinatoricModel_SB_reana.yml"
    output: f"{config['output_path']}/singlefiles/histdata__data-{{year}}.coffea"
    params:
        datasets="data",
        years="{year}",
        metadata="coffea4bees/analysis/metadata/HH4b.yml",
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        blind=False,
        run_performance=True,
        extra_arguments=lambda wildcards: f"{config['extra_arguments']} -e {config['data_eras'][wildcards.year]}",
        username=USERNAME
    log: f"{config['output_path']}/logs/analysis_histdata__data-{{year}}.log"


use rule analysis_databkgs as analysis_data_UL17B with:
    output: f"{config['output_path']}/singlefiles/histdata__data-UL17B.coffea"
    params:
        datasets="data",
        years="UL17",
        metadata="coffea4bees/analysis/metadata/HH4b_dataUL17B.yml",
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        blind=False,
        run_performance=True,
        extra_arguments=f"{config['extra_arguments']} -e B",

        username=USERNAME
    log: f"{config['output_path']}/logs/analysis_histdata__data-UL17B.log"

use rule analysis_databkgs as analysis_signals with:
    # input: f"{config['output_path']}/JCM/jetCombinatoricModel_SB_reana.yml"
    output: f"{config['output_path']}/singlefiles/histsignal__{{sample_signal}}-{{year}}.coffea"
    params:
        datasets="{sample_signal}",
        years="{year}",
        metadata="coffea4bees/analysis/metadata/HH4b_signals.yml",
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        blind=False,
        run_performance=True,
        extra_arguments=config['extra_arguments'],
        username=USERNAME
    log: f"{config['output_path']}/logs/analysis_histsignal__{{sample_signal}}-{{year}}.log"

### mixdata for HH
use rule analysis_databkgs as analysis_mixedbkg_data3b with:
    output: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.coffea"
    params:
        datasets="data_3b_for_mixed",
        years=config['year_preUL'],
        metadata="coffea4bees/analysis/metadata/HH4b_mixed_data.yml",
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        blind=False,
        run_performance=True,
        extra_arguments=config['extra_arguments'],
        username=USERNAME
    log: f"{config['output_path']}/logs/analysis_mixedbkg_data3b.log"


use rule analysis_databkgs as analysis_mixedbkg with:
    output: f"{config['output_path']}/histMixedBkg_TT.coffea"
    params:
        datasets=config['dataset_for_mixed'],
        years=config['year'],
        metadata="coffea4bees/analysis/metadata/HH4b_nottcheck.yml",
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        blind=False,
        run_performance=True,
        extra_arguments=config['extra_arguments'],
        username=USERNAME
    log: f"{config['output_path']}/logs/analysis_mixedbkg_TT.log"

use rule analysis_databkgs as analysis_mixeddata with:
    output: f"{config['output_path']}/histMixedData.coffea"
    params:
        datasets="mixeddata",
        years=config['year_preUL'],
        metadata="coffea4bees/analysis/metadata/HH4b_nottcheck.yml",
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        blind=False,
        run_performance=True,
        extra_arguments=config['extra_arguments'],
        username=USERNAME
    log: f"{config['output_path']}/logs/analysis_mixeddata.log"


### mixeddata for ZZ/ZH
use rule analysis_databkgs as analysis_mixedbkg_data3b_ZZZH with:
    output: f"{config['output_path']}/histMixedBkg_ZZZH_data_3b_for_mixed.coffea"
    params:
        datasets="data_3b_for_mixed",
        years=config['year_preUL'],
        metadata="coffea4bees/analysis/metadata/HH4b_mixed_data_ZZZH.yml",
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        blind=False,
        run_performance=True,
        extra_arguments=config['extra_arguments'],
        username=USERNAME
    log: f"{config['output_path']}/logs/analysis_mixedbkg_data3b_ZZZH.log"


use rule analysis_databkgs as analysis_mixedbkg_ZZZH with:
    output: f"{config['output_path']}/histMixedBkg_ZZZH_TT.coffea"
    params:
        datasets=config['dataset_for_mixed'],
        years=config['year'],
        metadata="coffea4bees/analysis/metadata/HH4b_mixed_data_ZZZH.yml",
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        blind=False,
        run_performance=True,
        extra_arguments=config['extra_arguments'],
        username=USERNAME
    log: f"{config['output_path']}/logs/analysis_mixedbkg_TT_ZZZH.log"

use rule analysis_databkgs as analysis_mixeddata_ZZZH with:
    output: f"{config['output_path']}/histMixedData_ZZZH.coffea"
    params:
        datasets="mixeddata",
        years=config['year_preUL'],
        metadata="coffea4bees/analysis/metadata/HH4b_mixed_data_ZZZH.yml",
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        blind=False,
        run_performance=True,
        extra_arguments=config['extra_arguments'],
        username=USERNAME
    log: f"{config['output_path']}/logs/analysis_mixeddata.log"

use rule analysis_databkgs as analysis_systematics_others with:
    # input: f"{config['output_path']}/JCM/jetCombinatoricModel_SB_reana.yml"
    output: f"{config['output_path']}/singlefiles/histsyst_others_{{samplesyst}}-{{iysyst}}.coffea"
    params:
        datasets="{samplesyst}",
        years="{iysyst}",
        metadata="coffea4bees/analysis/metadata/HH4b_signals.yml",
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        blind=False,
        run_performance=True,
        extra_arguments=f"{config['extra_arguments']} --systematics others",
        username=USERNAME
    log: f"{config['output_path']}/logs/analysis_histsyst_others_{{samplesyst}}-{{iysyst}}.log"


use rule analysis_databkgs as analysis_systematics_jes with:
    # input: f"{config['output_path']}/JCM/jetCombinatoricModel_SB_reana.yml"
    output: f"{config['output_path']}/singlefiles/histsyst_jes_{{samplesyst}}-{{iysyst}}.coffea"
    params:
        datasets="{samplesyst}",
        years="{iysyst}",
        metadata="coffea4bees/analysis/metadata/HH4b_signals.yml",
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        blind=False,
        run_performance=True,
        extra_arguments=f"{config['extra_arguments']} --systematics jes",
        username=USERNAME
    log: f"{config['output_path']}/logs/analysis_histsyst_jes_{{samplesyst}}-{{iysyst}}.log"

#######
### Merging histograms 
#######

# Add helper function at top with get_channel_lower:
def get_channel_datasets(channel):
    """Return datasets for each channel"""
    datasets_map = {
        "HH4b": [d for d in config['dataset_systematics'] if d.startswith(('GluGlu', 'VBF'))],
        "ZZ4b": [d for d in config['dataset_systematics'] if 'ZZ4b' in d],
        "ZH4b": [d for d in config['dataset_systematics'] if 'ZH4b' in d]
    }
    return datasets_map.get(channel, [])

use rule merging_coffea_files from analysis as merging_coffea_files_syst with:
    input: lambda wildcards: expand([f'{config["output_path"]}/singlefiles/histsyst_others_{{idatsyst}}-{{iyear}}.coffea'], idatsyst=get_channel_datasets(wildcards.channel), iyear=config['year']) + expand([f'{config["output_path"]}/singlefiles/histsyst_jes_{{idatsyst}}-{{iyear}}.coffea'], idatsyst=get_channel_datasets(wildcards.channel), iyear=config['year'])
    output: f"{config['output_path']}/histAll_signals__{{channel}}.coffea"
    resources:
        kerberos=True,
        compute_backend="kubernetes",
        kubernetes_memory_limit="9.5Gi"
    params:
        run_performance=False
    log: f"{config['output_path']}/logs/merging_signals_{{channel}}.log"

use rule merging_coffea_files_syst as merging_coffea_files_histAll with:
    input: expand([f'{config["output_path"]}/singlefiles/histsignal__{{sample_signal}}-{{year}}.coffea'], sample_signal=config['dataset_signals'], year=config['year']) + expand([f'{config["output_path"]}/singlefiles/hist__{{idat}}-{{iyear}}.coffea'], idat=config['dataset_tt'], iyear=config['year']) + expand([f'{config["output_path"]}/singlefiles/histdata__data-{{iyear}}.coffea'], iyear=config['year']) + [ f'{config["output_path"]}/singlefiles/histdata__data-UL17B.coffea' ]
    output: f"{config['output_path']}/histAll.coffea"
    log: f"{config['output_path']}/logs/merging_histAll.log"

########
### Making plots
########    

use rule make_plots from analysis as make_plots with:
    input: f"{config['output_path']}/histAll.coffea"
    output: f"{config['output_path']}/plots/RunII/passPreSel/fourTag/SB/nPVs.pdf"
    resources:
        kerberos=True,
        compute_backend="kubernetes",
        kubernetes_memory_limit="8Gi"
    log: f"{config['output_path']}/logs/make_plots.log"

########
### Converting histograms to JSON and ROOT formats
########

use rule convert_hist_to_json from stat_analysis as convert_hist_to_json with:
    input: f"{config['output_path']}/{{histfile}}.coffea"
    output: f"{config['output_path']}/{{histfile}}.json"
    params:
        syst_flag=lambda wildcards: "-s --histos SvB_MA.ps_hh SvB_MA.ps_zz SvB_MA.ps_zh" if "signals" in wildcards.histfile else ""
    log: f"{config['output_path']}/logs/convert_hist_to_json_{{histfile}}.log"
    resources:
        kubernetes_memory_limit="8Gi"

use rule convert_hist_to_json_closure from stat_analysis as convert_hist_to_json_mixdata3b with:
    input: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.coffea"
    output: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.json"
    log: f"{config['output_path']}/logs/convert_hist_to_json_mixdata3b.log"

use rule convert_hist_to_json_mixdata3b as convert_hist_to_json_mixbkgtt with:
    input: f"{config['output_path']}/histMixedBkg_TT.coffea"
    output: f"{config['output_path']}/histMixedBkg_TT.json"
    log: f"{config['output_path']}/logs/convert_hist_to_json_mixbkgtt.log"

use rule convert_hist_to_json_mixdata3b as convert_hist_to_json_mixdata with:
    input: f"{config['output_path']}/histMixedData.coffea"
    output: f"{config['output_path']}/histMixedData.json"
    log: f"{config['output_path']}/logs/convert_hist_to_json_mixdata.log"

use rule convert_hist_to_json_mixdata3b as convert_hist_to_json_mixdata3b_ZZZH with:
    input: f"{config['output_path']}/histMixedBkg_ZZZH_data_3b_for_mixed.coffea"
    output: f"{config['output_path']}/histMixedBkg_ZZZH_data_3b_for_mixed.json"
    log: f"{config['output_path']}/logs/convert_hist_to_json_mixdata3b_ZZZH.log"

use rule convert_hist_to_json_mixdata3b as convert_hist_to_json_mixbkgtt_ZZZH with:
    input: f"{config['output_path']}/histMixedBkg_ZZZH_TT.coffea"
    output: f"{config['output_path']}/histMixedBkg_ZZZH_TT.json"
    log: f"{config['output_path']}/logs/convert_hist_to_json_mixbkgtt_ZZZH.log"

use rule convert_hist_to_json_mixdata3b as convert_hist_to_json_mixdata_ZZZH with:
    input: f"{config['output_path']}/histMixedData_ZZZH.coffea"
    output: f"{config['output_path']}/histMixedData_ZZZH.json"
    log: f"{config['output_path']}/logs/convert_hist_to_json_mixdata_ZZZH.log"
    
use rule convert_json_to_root from stat_analysis with:
    input: f"{config['output_path']}/{{jsonfile}}.json"
    output: f"{config['output_path']}/{{jsonfile}}.root"
    container: config["combine_container"]
    resources:
        compute_backend="kubernetes",
        kubernetes_memory_limit="8Gi"
    message: "Converting {input} to {output}"
    log: f"{config['output_path']}/logs/convert_json_to_root_{{jsonfile}}.log"

#######
### Closure fits (background systematics)
########

use rule run_two_stage_closure from stat_analysis as run_two_stage_closure_HH4b with:
    input: 
        file_TT = f"{config['output_path']}/histMixedBkg_TT.root",
        file_mix = f"{config['output_path']}/histMixedData.root",
        file_sig = f"{config['output_path']}/histAll.root",
        file_data3b = f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.root"
    output: f"{config['output_path']}/closureFits/3bDvTMix4bDvT/SvB_MA/rebin1/SR/hh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_rebin1.pkl"
    container: config["combine_container"]
    params:
        outputPath = f"{config['output_path']}/closureFits/",
        rebin = "1",
        variable = "SvB_MA_ps_hh",
        extra_arguments = "--use_kfold",
        container_wrapper = config['container_wrapper']
    resources:
        compute_backend="kubernetes",
        kubernetes_memory_limit="8Gi"
    log: f"{config['output_path']}/logs/run_two_stage_closure_HH4b.log"

use rule run_two_stage_closure_HH4b as run_two_stage_closure_ZZ4b with:
    input: 
        file_TT = f"{config['output_path']}/histMixedBkg_ZZZH_TT.root",
        file_mix = f"{config['output_path']}/histMixedData_ZZZH.root",
        file_sig = f"{config['output_path']}/histAll.root",
        file_data3b = f"{config['output_path']}/histMixedBkg_ZZZH_data_3b_for_mixed.root"
    output: f"{config['output_path']}/closureFits/3bDvTMix4bDvT/SvB_MA/rebin1/SR/zz/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_zz_rebin1.pkl"
    params:
        outputPath = f"{config['output_path']}/closureFits",
        rebin = "1",
        variable = "SvB_MA_ps_zz",
        extra_arguments = "",
        container_wrapper = config['container_wrapper']
    log: f"{config['output_path']}/logs/run_two_stage_closure_ZZ4b.log"

use rule run_two_stage_closure_ZZ4b as run_two_stage_closure_ZH4b with:
    output: f"{config['output_path']}/closureFits/3bDvTMix4bDvT/SvB_MA/rebin1/SR/zh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_zh_rebin1.pkl"
    params:
        outputPath = f"{config['output_path']}/closureFits",
        rebin = "1",
        variable = "SvB_MA_ps_zh",
        extra_arguments = "",
        container_wrapper = config['container_wrapper']
    log: f"{config['output_path']}/logs/run_two_stage_closure_ZH4b.log"

########
### Making combine inputs and datacards
########
def get_channel_lower(channel):
    """Convert channel name to lowercase variant used in file paths"""
    return channel.lower().split('4b')[0] 

use rule make_combine_inputs from stat_analysis with:
    input:
        injson = f"{config['output_path']}/histAll.json",
        injsonsyst = f"{config['output_path']}/histAll_signals__{{channel}}.json", 
        bkgsyst = lambda wildcards: f"{config['output_path']}/closureFits/3bDvTMix4bDvT/SvB_MA/rebin1/SR/{get_channel_lower(wildcards.channel)}/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_{get_channel_lower(wildcards.channel)}_rebin1.pkl"
    output: f"{config['output_path']}/datacards/{{channel}}/datacard__{{channel}}.txt"
    container: config["combine_container"]
    params:
        variable= lambda wildcards: config['channels'][wildcards.channel]['variable'],
        rebin=1,
        variable_binning="",
        stat_only="",
        metadata=lambda wildcards: f"coffea4bees/stats_coffea4bees/analysis/metadata/{wildcards.channel}.yml",
        output_dir=f"{config['output_path']}/datacards/{{channel}}/",
        signal="{channel}",
        container_wrapper = config['container_wrapper']
    resources:
        voms_proxy=True,
        kerberos=True,
        compute_backend="kubernetes",
        kubernetes_memory_limit="9.5Gi"
    log: f"{config['output_path']}/logs/make_combine_inputs_{{channel}}.log"

use rule workspace from combine with:
    input: f"{config['output_path']}/datacards/{{channel}}/datacard__{{channel}}.txt"
    output: f"{config['output_path']}/datacards/{{channel}}/datacard__{{signallabel}}.root"
    container: config["combine_container"]
    params:
        signallabel="{signallabel}",
        othersignal_maps=lambda wildcards: additional_poi(wildcards.channel),
        container_wrapper=config.get("container_wrapper", "./run_container combine")
    resources:
        voms_proxy=True,
        kerberos=True,
        compute_backend="kubernetes"
    log: f"{config['output_path']}/logs/workspace_{{channel}}__{{signallabel}}.log"

use rule limits from combine with:
    input: f"{config['output_path']}/datacards/{{channel}}/datacard__{{signallabel}}.root"
    output: 
        txt=f"{config['output_path']}/datacards/{{channel}}/limits__{{signallabel}}.txt",
        json=f"{config['output_path']}/datacards/{{channel}}/limits__{{signallabel}}.json"
    container: config["combine_container"]
    params:
        signallabel="{signallabel}",
        set_parameters_zero=lambda wildcards: set_parameters(wildcards.channel, 0),
        freeze_parameters=lambda wildcards: freeze_parameters(wildcards.channel),
        container_wrapper=config.get("container_wrapper", "./run_container combine")
    resources:
        voms_proxy=True,
        kerberos=True,
        compute_backend="kubernetes"
    log: f"{config['output_path']}/logs/limits_{{channel}}__{{signallabel}}.log"

use rule significance from combine with:
    input: f"{config['output_path']}/datacards/{{channel}}/datacard__{{signallabel}}.root"
    output: f"{config['output_path']}/datacards/{{channel}}/significance__{{signallabel}}.log"
    container: config["combine_container"]
    params:
        signallabel="{signallabel}",
        set_parameters_zero=lambda wildcards: set_parameters(wildcards.channel, 0),
        freeze_parameters=lambda wildcards: freeze_parameters(wildcards.channel),
        container_wrapper=config.get("container_wrapper", "./run_container combine")
    resources:
        voms_proxy=True,
        kerberos=True,
        compute_backend="kubernetes"
    log: f"{config['output_path']}/logs/significance_{{channel}}__{{signallabel}}.log"

use rule impacts from combine with:
    input: f"{config['output_path']}/datacards/{{channel}}/datacard__{{signallabel}}.root"
    output: f"{config['output_path']}/datacards/{{channel}}/impacts__{{signallabel}}.pdf"
    container: config["combine_container"]
    params:
        signallabel="{signallabel}",
        set_parameters_zero=lambda wildcards: set_parameters(wildcards.channel, 0),
        set_parameters_ranges=lambda wildcards: set_parameters_ranges(wildcards.channel),
        container_wrapper=config.get("container_wrapper", "./run_container combine")
    resources:
        voms_proxy=True,
        kerberos=True,
        compute_backend="kubernetes",
        kubernetes_memory_limit="9.5Gi"
    log: f"{config['output_path']}/logs/impacts_{{channel}}_datacard_{{channel}}__{{signallabel}}.log"

use rule likelihood_scan from combine with:
    input: f"{config['output_path']}/datacards/{{channel}}/datacard__{{signallabel}}.root"
    output: f"{config['output_path']}/datacards/{{channel}}/likelihood_scan__{{signallabel}}.pdf"
    container: config["combine_container"]
    params:
        signallabel="{signallabel}",
        set_parameters_zero=lambda wildcards: set_parameters(wildcards.channel, 0),
        freeze_parameters=lambda wildcards: freeze_parameters(wildcards.channel),
        container_wrapper=config.get("container_wrapper", "./run_container combine")
    resources:
        voms_proxy=True,
        kerberos=True,
        compute_backend="kubernetes"
    log: f"{config['output_path']}/logs/likelihood_scan_{{channel}}_datacard_{{channel}}__{{signallabel}}.log"

use rule gof from combine with:
    input: f"{config['output_path']}/datacards/{{channel}}/datacard__{{signallabel}}.root"
    output: f"{config['output_path']}/datacards/{{channel}}/gof__{{signallabel}}.pdf"
    container: config["combine_container"]
    params:
        signallabel="{signallabel}",
        set_parameters_zero=lambda wildcards: set_parameters(wildcards.channel, 0),
        container_wrapper=config.get("container_wrapper", "./run_container combine")
    resources:
        voms_proxy=True,
        kerberos=True,
        compute_backend="kubernetes"
    log: f"{config['output_path']}/logs/gof_{{channel}}_datacard_{{channel}}__{{signallabel}}.log"

use rule make_syst_plots from stat_analysis with:
    input: f"{config['output_path']}/datacards/{{channel}}/datacard__{{channel}}.txt"
    output: f"{config['output_path']}/datacards/{{channel}}/systs/SvB_MA_ps_{{channel_lower}}_nominal.pdf"
    container: config["combine_container"]
    log: f"{config['output_path']}/logs/make_syst_plots_{{channel}}_{{channel_lower}}.log"
    params:
        variable=lambda wildcards: f"SvB_MA_ps_{get_channel_lower(wildcards.channel)}",
        output_dir=f"{config['output_path']}/datacards/{{channel}}/",
        channel="{channel}",
        signal=lambda wildcards: config['channels'][wildcards.channel]['signal'],
        container_wrapper = config['container_wrapper']
    resources:
        kerberos=True,
        compute_backend="kubernetes",
        kubernetes_memory_limit="8Gi"

use rule postfit from combine as postfit with:
    input: f"{config['output_path']}/datacards/{{channel}}/datacard__{{signallabel}}.root"
    output: f"{config['output_path']}/datacards/{{channel}}/plots/postfitplots__{{signallabel}}__prefit.pdf"
    container: config["combine_container"]
    log: f"{config['output_path']}/logs/postfit__{{channel}}__{{signallabel}}.log"
    params:
        signallabel="{signallabel}",
        channel="{channel}",
        signal=lambda wildcards: config['channels'][wildcards.channel]['signal'],
        ylog=lambda wildcards: True if wildcards.channel == "HH4b" else False,
        set_parameters_zero=lambda wildcards: set_parameters(wildcards.channel, 0),
        freeze_parameters=lambda wildcards: freeze_parameters(wildcards.channel),
        container_wrapper=config.get("container_wrapper", "./run_container combine")
    resources:
        voms_proxy=True,
        kerberos=True,
        compute_backend="kubernetes"
