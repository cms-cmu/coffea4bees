from datetime import datetime
import os
import shutil

# ==============================================================================
# Global Configuration Options & Defaults Reference
# ==============================================================================
# General Options
config.setdefault('label', "nominal_wNewSvB")
config.setdefault('output_path', "output/nominal_wNewSvB/")
config.setdefault('email', "")

# Analysis Sub-workflow Options
config.setdefault('analysis_config', "coffea4bees/analysis/metadata/HH4b_2024_v2.yml")
config.setdefault('processor', "coffea4bees/analysis/processors/processor_HH4b.py")
config.setdefault('friend_file', "coffea4bees/metadata/friends/friends_HH4b.yml")
config.setdefault('weights_file', "coffea4bees/metadata/weights/weights_HH4b.yml")
config.setdefault('additional_parameters', "--shared-dask --condor --run-performance")
config.setdefault('plot_config', "coffea4bees/plots/metadata/plotsAll_ttbarWeights.yml")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets/")
config.setdefault('dataset', ['GluGluToHHTo4B_cHHH1', 'GluGluToHHTo4B_cHHH0', 'GluGluToHHTo4B_cHHH2p45', 'GluGluToHHTo4B_cHHH5', 'ZH4b', 'ZZ4b', 'ggZH4b'])

# Stats Sub-workflow Options
config.setdefault('combine_flags', "--blind")
config.setdefault('closure_base', "coffea4bees/stats_analysis/files/HIG-24-010")
config.setdefault('channels', {})
config.setdefault('combine_outdir', "datacards/HH4b_fine")

config.setdefault('convert_hist_to_json', {})
config['convert_hist_to_json'].setdefault('syst_flag', "")
config['convert_hist_to_json'].setdefault(
    'histos',
    [ch_config['variable'] for ch_config in config.get('channels', {}).values()]
)

config.setdefault('make_combine_inputs', {})
config['make_combine_inputs'].setdefault('rebin', 1)
config['make_combine_inputs'].setdefault('variable_binning', "")
config['make_combine_inputs'].setdefault('stat_only', "--stat_only")
config['make_combine_inputs'].setdefault('bkgsyst', "")
config['make_combine_inputs'].setdefault('syst_file', "")
config['make_combine_inputs'].setdefault('metadata_template', config.get('metadata_template', "coffea4bees/stats_analysis/metadata/{channel}.yml"))
config['make_combine_inputs'].setdefault('multijet_process', "data")
config['make_combine_inputs'].setdefault('tt_processes', ["TTTo", "TTbar4b_from_d3"])

# Likelihood scan defaults
config.setdefault('likelihood_scan_points', 20)
config.setdefault('likelihood_scan_split_size', 10)
config.setdefault('likelihood_scan_r_min', -10)
config.setdefault('likelihood_scan_r_max', 10)

# Shared/Common Options
config.setdefault('year_eras', {
    'UL16_preVFP':  ['C', 'D', 'E', 'F'],
    'UL16_postVFP': ['F', 'G', 'H'],
    'UL17':         ['C', 'D', 'E', 'F'],
    'UL18':         ['A', 'B', 'C', 'D'],
})

# Container Options (DO NOT CHANGE unless you know what you are doing)
config.setdefault('container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('combine_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_14_1_0_pre4-combine_v10.6.0-harvester_v3.1.0")
config.setdefault('container_wrapper', "./run_container combine")
config.setdefault('analysis_container_wrapper', "./run_container")
config.setdefault('stats_container_wrapper', "./run_container combine")


# Resolve absolute CERNBox destination path
cern_user = config.get("cern_user", os.environ.get("USER", "algomez"))
cern_path = config.get("cern_path", "www/HH4b/Plots/")
if not cern_path.startswith("/"):
    first_letter = cern_user[0]
    cern_path = f"/eos/user/{first_letter}/{cern_user}/{cern_path}"

# Define master target endpoints dynamically using an input function
def get_master_targets(wildcards):
    master_targets = [
        f"{config['output_path']}histAll_{config['label']}.coffea",
        f"{config['output_path']}plots_{config['label']}/plots_done.txt",
        f"{config['output_path']}cutflow_validation_{config['label']}.txt",
        f"{config['output_path']}cutflow_{config['label']}.yml",
    ]
    for channel, ch_config in config.get('channels', {}).items():
        signallabel = ch_config.get('signallabel')
        if signallabel:
            master_targets.extend([
                f"{config['output_path']}stat_analysis/{channel}/limits/datacard_limits__{signallabel}.json",
                f"{config['output_path']}stat_analysis/{channel}/postfit/datacard_postfit__{signallabel}.pdf",
                f"{config['output_path']}stat_analysis/{channel}/significance/datacard_significance__{signallabel}.log",
                f"{config['output_path']}stat_analysis/{channel}/likelihood_scan/datacard_likelihood_scan__{signallabel}.pdf",
            ])
    return master_targets

# The first rule defined in the master file remains the default target
rule final_output:
    input: get_master_targets
    params:
        output_dir = f"{datetime.now().strftime('%Y%m%d')}_{config['label']}/",
        cern_path = cern_path,
        email = lambda wildcards: config.get('email', "")
    shell:
        """
        echo "Copying results to eos"
        bash src/tools/copy_files_to_cernbox.sh -s {config[output_path]} -d {params.cern_path}{params.output_dir} -t || echo "Warning: copy to EOS failed. Skipping remote upload."
        if [ -n "{params.email}" ]; then
            echo "Workflow for {config[label]} completed successfully on $(date)." | mail -s "Snakemake Success: {config[label]}" "{params.email}" || echo "Warning: failed to send success notification email."
        fi
        """

# Load both sub-workflows
include: "Snakefile_PhaseF_1_analysis.smk"
include: "Snakefile_PhaseF_2_stats.smk"
