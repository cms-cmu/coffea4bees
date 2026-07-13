from datetime import datetime
import os
import yaml

# Set per-job Apptainer cache/tmp dirs to avoid race conditions between concurrent Slurm jobs
shell.prefix(
    "export APPTAINER_CACHEDIR=/tmp/apptainer_cache_${{SLURM_JOB_ID:-$$}} && "
    "export APPTAINER_TMPDIR=/tmp/apptainer_tmp_${{SLURM_JOB_ID:-$$}} && "
    "mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR && "
)

_roc = config.get('run_on_condor', True)
if isinstance(_roc, str):
    config['run_on_condor'] = _roc.lower() not in ('false', '0', 'no')
else:
    config['run_on_condor'] = bool(_roc)

# Import rule modules
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

module stat_analysis:
    snakefile: "rules/stat_analysis.smk"
    config: config

combine_config = config.copy()
combine_config["output_path"] = os.path.join(config["output_path"], "stat_analysis/")
combine_config["combine_container"] = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_14_1_0_pre4-combine_v10.6.0-harvester_v3.1.0"

module combine:
    snakefile: os.path.join(os.getcwd(), "src/stat_analysis/combine.smk")
    config: combine_config

def get_channel_from_path(wildcards):
    parts = wildcards.path.rstrip('/').split('/')
    for folder in ['workspace', 'limits', 'significance', 'impacts', 'likelihood_scan', 'gof', 'postfit']:
        if folder in parts:
            idx = parts.index(folder)
            if idx > 0:
                return parts[idx - 1]
    for ch in config.get('channels', {}):
        if f"/{ch}/" in wildcards.path:
            return ch
    return ""

def get_signal_from_path(wildcards):
    channel = get_channel_from_path(wildcards)
    return config.get('channels', {}).get(channel, {}).get('signal', '')

def get_workspace_input_reana(wildcards):
    channel = get_channel_from_path(wildcards)
    return f"{config['output_path']}/datacards/{channel}/datacard__{channel}.txt"

include: "helpers/common.smk"

hash = config['extra_arguments'].split("--githash ")[1]
EOS_OUTPUT = f"/eos/user/a/algomez/www/plots/HH4b/reana/{datetime.now().strftime('%Y%m%d')}_{hash}/"
OUTPUT_PATH = config['output_path']

# Define output patterns at the top (after imports/config)
CHANNELS = list(config['channels'].keys())
SIGNALLABELS = [config['channels'][k]['signallabel'] for k in CHANNELS]
CHANNELLABELS = [k.lower().split('4b')[0] for k in CHANNELS]

OUTPUT_PATTERNS = {
    "limits": f"{config['output_path']}/stat_analysis/{{channel}}/limits/datacard_limits__{{signallabel}}.json",
    "significance": f"{config['output_path']}/stat_analysis/{{channel}}/significance/datacard_significance__{{signallabel}}.log",
    "impacts": f"{config['output_path']}/stat_analysis/{{channel}}/impacts/datacard_impacts__{{signallabel}}.pdf",
    "likelihood_scan": f"{config['output_path']}/stat_analysis/{{channel}}/likelihood_scan/datacard_likelihood_scan__{{signallabel}}.pdf",
    "gof": f"{config['output_path']}/stat_analysis/{{channel}}/gof/datacard_gof__{{signallabel}}.pdf",
    "postfit": f"{config['output_path']}/stat_analysis/{{channel}}/postfit/datacard_postfit__{{signallabel}}.pdf",
}

SYST_PLOTS = {
    c: f"{config['output_path']}/stat_analysis/{c}/datacards/systs/SvB_MA_ps_{c.lower().split('4b')[0]}_nominal.pdf"
    for c in CHANNELS
}

def reana_config(name):
    return f"{config['output_path']}/configs/{name}_reana.yml"


rule all:
    input:
        f"{config['output_path']}/plots/RunII/region_SB/nPVs.pdf",
        expand(OUTPUT_PATTERNS["limits"], zip, channel=CHANNELS, signallabel=SIGNALLABELS),
        expand(OUTPUT_PATTERNS["significance"], zip, channel=CHANNELS, signallabel=SIGNALLABELS),
        expand(OUTPUT_PATTERNS["impacts"], zip, channel=CHANNELS, signallabel=SIGNALLABELS),
        expand(OUTPUT_PATTERNS["postfit"], zip, channel=CHANNELS, signallabel=SIGNALLABELS),
        expand(OUTPUT_PATTERNS["gof"], zip, channel=CHANNELS, signallabel=SIGNALLABELS),
        expand(OUTPUT_PATTERNS["likelihood_scan"], zip, channel=CHANNELS, signallabel=SIGNALLABELS),
        list(SYST_PLOTS.values()),
    container: config["analysis_container"]
    params:
        output_dir = f"{datetime.now().strftime('%Y%m%d')}_scheduled/"
    shell:
        """
        echo "Copying results to eos"
        bash src/tools/copy_files_to_cernbox.sh -s {config[output_path]} -d /eos/user/a/algomez/www/HH4b/reana/{params.output_dir} -t
        """

localrules: make_reana_config

rule make_reana_config:
    """Generate a patched analysis config with workers=32 and worker_memory=3GB for REANA."""
    input: "coffea4bees/analysis/metadata/{cfg_name}.yml"
    output: f"{config['output_path']}/configs/{{cfg_name}}_reana.yml"
    shell:
        """
        python3 -c "
import yaml, os
os.makedirs(os.path.dirname('{output}'), exist_ok=True)
with open('{input}') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('runner', {{}})
cfg['runner']['workers'] = 28
cfg['runner']['worker_memory'] = '3GB'
with open('{output}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
"
        """

#######
### Running analysis processor
#######

### In the next rules, input points to the REANA-patched config (generated by make_reana_config).
use rule analysis_processor from analysis as analysis_databkgs with:
    input: reana_config("HH4b")
    output: f"{config['output_path']}/singlefiles/hist__{{sample}}-{{year}}.coffea"
    container: config["analysis_container"]
    threads: 28
    resources:
        mem_mb=86016
    log: f"{config['output_path']}/logs/analysis_hist__{{sample}}-{{year}}.log"
    params:
        datasets="{sample}",
        years="{year}",
        config=reana_config("HH4b"),
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        friends="",
        blind=False,
        not_do_proxy=True,
        run_performance=True,
        run_on_condor=False,
        extra_arguments=config['extra_arguments'],
        run_container_wrapper="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        dashboard_address=""


use rule analysis_databkgs as analysis_data with:
    output: f"{config['output_path']}/singlefiles/histdata__data-{{year}}.coffea"
    log: f"{config['output_path']}/logs/analysis_histdata__data-{{year}}.log"
    params:
        datasets="data",
        years="{year}",
        config=reana_config("HH4b"),
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        friends="",
        blind=False,
        not_do_proxy=True,
        run_performance=True,
        run_on_condor=False,
        extra_arguments=lambda wildcards: f"{config['extra_arguments']} -e {config['data_eras'][wildcards.year]}",
        run_container_wrapper="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        dashboard_address=""


use rule analysis_databkgs as analysis_data_UL17B with:
    input: reana_config("HH4b_dataUL17B")
    output: f"{config['output_path']}/singlefiles/histdata__data-UL17B.coffea"
    log: f"{config['output_path']}/logs/analysis_histdata__data-UL17B.log"
    params:
        datasets="data",
        years="UL17",
        config=reana_config("HH4b_dataUL17B"),
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        friends="",
        blind=False,
        not_do_proxy=True,
        run_performance=True,
        run_on_condor=False,
        extra_arguments=f"{config['extra_arguments']} -e B",
        run_container_wrapper="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        dashboard_address=""

use rule analysis_databkgs as analysis_signals with:
    input: reana_config("HH4b_signals")
    output: f"{config['output_path']}/singlefiles/histsignal__{{sample_signal}}-{{year}}.coffea"
    threads: 28
    resources:
        mem_mb=86016
    log: f"{config['output_path']}/logs/analysis_histsignal__{{sample_signal}}-{{year}}.log"
    params:
        datasets="{sample_signal}",
        years="{year}",
        config=reana_config("HH4b_signals"),
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        friends="",
        blind=False,
        not_do_proxy=True,
        run_performance=True,
        run_on_condor=False,
        extra_arguments=config['extra_arguments'],
        run_container_wrapper="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        dashboard_address=""

### mixdata for HH
use rule analysis_databkgs as analysis_mixedbkg_data3b with:
    input: reana_config("HH4b_mixed_data")
    output: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.coffea"
    log: f"{config['output_path']}/logs/analysis_mixedbkg_data3b.log"
    params:
        datasets="data_3b_for_mixed",
        years=config.get('year_preUL', config['year']),
        config=reana_config("HH4b_mixed_data"),
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        friends="",
        blind=False,
        not_do_proxy=True,
        run_performance=True,
        run_on_condor=False,
        extra_arguments=config['extra_arguments'],
        run_container_wrapper="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        dashboard_address=""


use rule analysis_databkgs as analysis_mixedbkg with:
    input: reana_config("HH4b_nottcheck")
    output: f"{config['output_path']}/histMixedBkg_TT.coffea"
    log: f"{config['output_path']}/logs/analysis_mixedbkg_TT.log"
    params:
        datasets=config['dataset_for_mixed'],
        years=config['year'],
        config=reana_config("HH4b_nottcheck"),
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        friends="",
        blind=False,
        not_do_proxy=True,
        run_performance=True,
        run_on_condor=False,
        extra_arguments=config['extra_arguments'],
        run_container_wrapper="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        dashboard_address=""

use rule analysis_databkgs as analysis_mixeddata with:
    input: reana_config("HH4b_nottcheck")
    output: f"{config['output_path']}/histMixedData.coffea"
    log: f"{config['output_path']}/logs/analysis_mixeddata.log"
    params:
        datasets="mixeddata",
        years=config.get('year_preUL', config['year']),
        config=reana_config("HH4b_nottcheck"),
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        friends="",
        blind=False,
        not_do_proxy=True,
        run_performance=True,
        run_on_condor=False,
        extra_arguments=config['extra_arguments'],
        run_container_wrapper="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        dashboard_address=""


### mixeddata for ZZ/ZH
use rule analysis_databkgs as analysis_mixedbkg_data3b_ZZZH with:
    input: reana_config("HH4b_mixed_data_ZZZH")
    output: f"{config['output_path']}/histMixedBkg_ZZZH_data_3b_for_mixed.coffea"
    log: f"{config['output_path']}/logs/analysis_mixedbkg_data3b_ZZZH.log"
    params:
        datasets="data_3b_for_mixed",
        years=config.get('year_preUL', config['year']),
        config=reana_config("HH4b_mixed_data_ZZZH"),
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        friends="",
        blind=False,
        not_do_proxy=True,
        run_performance=True,
        run_on_condor=False,
        extra_arguments=config['extra_arguments'],
        run_container_wrapper="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        dashboard_address=""


use rule analysis_databkgs as analysis_mixedbkg_ZZZH with:
    input: reana_config("HH4b_mixed_data_ZZZH")
    output: f"{config['output_path']}/histMixedBkg_ZZZH_TT.coffea"
    log: f"{config['output_path']}/logs/analysis_mixedbkg_TT_ZZZH.log"
    params:
        datasets=config['dataset_for_mixed'],
        years=config['year'],
        config=reana_config("HH4b_mixed_data_ZZZH"),
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        friends="",
        blind=False,
        not_do_proxy=True,
        run_performance=True,
        run_on_condor=False,
        extra_arguments=config['extra_arguments'],
        run_container_wrapper="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        dashboard_address=""

use rule analysis_databkgs as analysis_mixeddata_ZZZH with:
    input: reana_config("HH4b_mixed_data_ZZZH")
    output: f"{config['output_path']}/histMixedData_ZZZH.coffea"
    log: f"{config['output_path']}/logs/analysis_mixeddata_ZZZH.log"
    params:
        datasets="mixeddata",
        years=config.get('year_preUL', config['year']),
        config=reana_config("HH4b_mixed_data_ZZZH"),
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        friends="",
        blind=False,
        not_do_proxy=True,
        run_performance=True,
        run_on_condor=False,
        extra_arguments=config['extra_arguments'],
        run_container_wrapper="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        dashboard_address=""

use rule analysis_databkgs as analysis_systematics_others with:
    input: reana_config("HH4b_signals")
    output: f"{config['output_path']}/singlefiles/histsyst_others_{{samplesyst}}-{{iysyst}}.coffea"
    threads: 28
    resources:
        mem_mb=86016
    log: f"{config['output_path']}/logs/analysis_histsyst_others_{{samplesyst}}-{{iysyst}}.log"
    params:
        datasets="{samplesyst}",
        years="{iysyst}",
        config=reana_config("HH4b_signals"),
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        friends="",
        blind=False,
        not_do_proxy=True,
        run_performance=True,
        run_on_condor=False,
        extra_arguments=f"{config['extra_arguments']} --systematics others",
        run_container_wrapper="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        dashboard_address=""


use rule analysis_databkgs as analysis_systematics_jes with:
    input: reana_config("HH4b_signals")
    output: f"{config['output_path']}/singlefiles/histsyst_jes_{{samplesyst}}-{{iysyst}}.coffea"
    threads: 28
    resources:
        mem_mb=86016
    log: f"{config['output_path']}/logs/analysis_histsyst_jes_{{samplesyst}}-{{iysyst}}.log"
    params:
        datasets="{samplesyst}",
        years="{iysyst}",
        config=reana_config("HH4b_signals"),
        processor="coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file=config['dataset_location'],
        friends="",
        blind=False,
        not_do_proxy=True,
        run_performance=True,
        run_on_condor=False,
        extra_arguments=f"{config['extra_arguments']} --systematics jes",
        run_container_wrapper="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        dashboard_address=""

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

def get_channel_signals(channel):
    """Return signal datasets for each channel from dataset_signals"""
    datasets_map = {
        "HH4b": [d for d in config['dataset_signals'] if d.startswith(('GluGlu', 'VBF'))],
        "ZZ4b": [d for d in config['dataset_signals'] if 'ZZ4b' in d],
        "ZH4b": [d for d in config['dataset_signals'] if 'ZH4b' in d]
    }
    return datasets_map.get(channel, [])

def get_syst_merge_inputs(wildcards):
    syst_datasets = get_channel_datasets(wildcards.channel)
    if syst_datasets:
        return expand([f'{config["output_path"]}/singlefiles/histsyst_others_{{idatsyst}}-{{iyear}}.coffea'], idatsyst=syst_datasets, iyear=config['year']) + expand([f'{config["output_path"]}/singlefiles/histsyst_jes_{{idatsyst}}-{{iyear}}.coffea'], idatsyst=syst_datasets, iyear=config['year'])
    else:
        return expand([f'{config["output_path"]}/singlefiles/histsignal__{{isig}}-{{iyear}}.coffea'], isig=get_channel_signals(wildcards.channel), iyear=config['year'])

use rule merging_coffea_files from analysis as merging_coffea_files_syst with:
    input: get_syst_merge_inputs
    output: f"{config['output_path']}/histAll_signals__{{channel}}.coffea"
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
    output: f"{config['output_path']}/plots/RunII/region_SB/nPVs.pdf"
    container: config["analysis_container"]
    threads: 8
    resources:
        mem_mb=16384
    log: f"{config['output_path']}/logs/make_plots.log"
    params:
        output_dir = f"{config['output_path']}/plots/",
        metadata = "coffea4bees/plots/metadata/plotsAll.yml",
        extra_arguments = "-s xW",
        png_cores = 8,

########
### Converting histograms to JSON and ROOT formats
########

use rule convert_hist_to_json from stat_analysis as convert_hist_to_json with:
    input: f"{config['output_path']}/{{histfile}}.coffea"
    output: f"{config['output_path']}/{{histfile}}.json"
    params:
        syst_flag=lambda wildcards: "-s --histos SvB_MA.ps_hh SvB_MA.ps_zz SvB_MA.ps_zh" if wildcards.histfile.endswith(("HH4b", "ZZ4b", "ZH4b")) else ""
    log: f"{config['output_path']}/logs/convert_hist_to_json_{{histfile}}.log"

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
    output: f"{config['output_path']}/stat_analysis/{{channel}}/datacards/datacard__{{channel}}.txt"
    container: config["combine_container"]
    params:
        variable= lambda wildcards: config['channels'][wildcards.channel]['variable'],
        rebin=1,
        variable_binning="",
        stat_only="",
        syst_file=lambda wildcards: f"-s {config['output_path']}/histAll_signals__{wildcards.channel}.json" if wildcards.channel == "HH4b" else "",
        metadata=lambda wildcards: config.get('channel_metadata', {}).get(wildcards.channel, f"coffea4bees/stats_analysis/metadata/{wildcards.channel}.yml"),
        output_dir=f"{config['output_path']}/stat_analysis/{{channel}}/datacards/",
        signal="{channel}",
        container_wrapper = config['container_wrapper'],
        tag_flags = ""
    log: f"{config['output_path']}/logs/make_combine_inputs_{{channel}}.log"

use rule make_syst_plots from stat_analysis with:
    input: f"{config['output_path']}/stat_analysis/{{channel}}/datacards/datacard__{{channel}}.txt"
    output: f"{config['output_path']}/stat_analysis/{{channel}}/datacards/systs/SvB_MA_ps_{{channel_lower}}_nominal.pdf"
    container: config["combine_container"]
    log: f"{config['output_path']}/logs/make_syst_plots_{{channel}}_{{channel_lower}}.log"
    params:
        variable=lambda wildcards: f"SvB_MA_ps_{get_channel_lower(wildcards.channel)}",
        output_dir=f"{config['output_path']}/stat_analysis/{{channel}}/datacards/",
        channel="{channel}",
        signal=lambda wildcards: config['channels'][wildcards.channel]['signal'],
        container_wrapper = config['container_wrapper']

use rule * from combine

localrules: all, make_reana_config, merging_coffea_files_syst, merging_coffea_files_histAll, make_plots, convert_hist_to_json, convert_hist_to_json_mixdata3b, convert_hist_to_json_mixbkgtt, convert_hist_to_json_mixdata, convert_hist_to_json_mixdata3b_ZZZH, convert_hist_to_json_mixbkgtt_ZZZH, convert_hist_to_json_mixdata_ZZZH, convert_json_to_root, run_two_stage_closure_HH4b, run_two_stage_closure_ZZ4b, run_two_stage_closure_ZH4b, make_combine_inputs, make_syst_plots

if not config.get('run_on_condor', True):
    combine_rules = [
        "workspace", "limits", "significance", "likelihood_scan_snapshot",
        "likelihood_scan_chunk", "likelihood_scan", "impacts_initial_fit",
        "impacts_do_fits", "impacts_collect", "split_impacts", "pdf_to_png",
        "gof_data", "gof_toys_chunk", "gof", "fit_diagnostics_bonly", "fit_diagnostics_sb",
        "postfit"
    ]
    workflow._localrules.update(combine_rules)

# Set retries to 3 for all rules to automatically restart failed cluster jobs
for r in workflow.rules:
    r.retries = 3
