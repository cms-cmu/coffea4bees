import os
include: "helpers/common.smk"

_roc = config.get('run_on_condor', True)
if isinstance(_roc, str):
    config['run_on_condor'] = _roc.lower() not in ('false', '0', 'no')
else:
    config['run_on_condor'] = bool(_roc)

config.setdefault('combine_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_14_1_0_pre4-combine_v10.6.0-harvester_v3.1.0")

# Set default channels configuration if not present
if 'channels' not in config:
    config['channels'] = {
        'ZZ_bbbb': {
            'datacard_run2': 'output/ZZ_ZH_combination/v3/original_datacards/run2/combine_SvB_MA.txt',
            'remove_bins_run2': "'zh*' 'hh*'",
            'remove_process_run2': "None",
            'rename_process_run2': "'ZZ=ZZ_bbbb' 'ZH=ZH_bbbb', 'HH=ggHH_bbbb'",
            'datacard_run3': 'output/ZZ_ZH_combination/v3/original_datacards/run3/datacard.txt',
            'remove_bins_run3': "'ggHHbbbb_*' 'qqHHbbbb_*' 'zhHHbbbb_*'",
            'rename_process_run3': "'ggHH_hbbhbb=ggHH_bbbb' 'qqHH_hbbhbb=qqHH_bbbb'",
            'remove_process_run3': "None",
            'signallabel': 'ZZ_bbbb',
            'othersignal': 'ZH_bbbb ggHH_bbbb qqHH_bbbb',
            'label': 'ZZ',
            'workspace': ''
        },
        'ZH_bbbb': {
            'datacard_run2': 'output/ZZ_ZH_combination/v3/original_datacards/run2/combine_SvB_MA.txt',
            'remove_bins_run2': "'zz*' 'hh*'",
            'rename_process_run2': "'ZZ=ZZ_bbbb' 'ZH=ZH_bbbb', 'HH=ggHH_bbbb'",
            'remove_process_run2': "None",
            'datacard_run3': 'output/ZZ_ZH_combination/v3/original_datacards/run3/datacard.txt',
            'remove_bins_run3': "'ggHHbbbb_*' 'qqHHbbbb_*' 'zzHHbbbb_*'",
            'remove_process_run3': "None",
            'rename_process_run3': "'ggHH_hbbhbb=ggHH_bbbb' 'qqHH_hbbhbb=qqHH_bbbb'",
            'signallabel': 'ZH_bbbb',
            'othersignal': 'ZZ_bbbb ggHH_bbbb qqHH_bbbb',
            'label': 'ZH',
            'workspace': ''
        }
    }

# Filter channels to only include ZZ and ZH
CHANNELS = [c for c in config['channels'].keys() if 'zz' in c.lower() or 'zh' in c.lower()]
CH_CFG = config['channels']

# Defensive defaults for missing keys in the channels configuration
for ch in CHANNELS:
    ch_dict = CH_CFG[ch]
    # Set default values if not defined in config file
    ch_dict.setdefault('datacard_run2', 'output/original_datacards/run2/combine_SvB_MA.txt')
    ch_dict.setdefault('remove_bins_run2', "None")
    ch_dict.setdefault('remove_process_run2', "None")
    ch_dict.setdefault('rename_process_run2', "None")
    
    ch_dict.setdefault('datacard_run3', 'output/original_datacards/run3/datacard.txt')
    ch_dict.setdefault('remove_bins_run3', "None")
    ch_dict.setdefault('remove_process_run3', "None")
    ch_dict.setdefault('rename_process_run3', "None")
    
    ch_dict.setdefault('signallabel', ch)
    ch_dict.setdefault('othersignal', "")
    ch_dict.setdefault('label', ch.replace('4b', '').replace('_bbbb', '').upper())

rule all_combination:
    input:
        expand( f"{config['output_path']}/stat_analysis/{{run}}/{{signal}}/limits/datacard_limits__{{signal}}.json", run=['run2', 'run3'], signal=CHANNELS ),
        expand( f"{config['output_path']}/stat_analysis/{{run}}/{{signal}}/significance/datacard_significance__{{signal}}.log", run=['run2', 'run3'], signal=CHANNELS ),
        expand( f"{config['output_path']}/stat_analysis/{{run}}/{{signal}}/likelihood_scan/datacard_likelihood_scan__{{signal}}.pdf", run=['run2', 'run3'], signal=CHANNELS ),
        expand( f"{config['output_path']}/stat_analysis/combination_{{signal}}/limits/datacard_limits__{{signal}}.json", signal=CHANNELS ),
        expand( f"{config['output_path']}/stat_analysis/combination_{{signal}}/significance/datacard_significance__{{signal}}.log", signal=CHANNELS ),
        expand( f"{config['output_path']}/stat_analysis/combination_{{signal}}/likelihood_scan/datacard_likelihood_scan__{{signal}}.pdf", signal=CHANNELS ),
        expand( f"{config['output_path']}/stat_analysis/combination_{{signal}}/impacts/datacard_impacts__{{signal}}.pdf", signal=CHANNELS ),
        expand( f"{config['output_path']}/stat_analysis/combination_{{signal}}/gof/datacard_gof__{{signal}}.pdf", signal=CHANNELS ),
        expand( f"{config['output_path']}/stat_analysis/combination_{{signal}}/combination_plot_{{signal}}.pdf", signal=CHANNELS ),

combine_config = config.copy()

module combine:
    snakefile: os.path.join(os.getcwd(), "src/stat_analysis/combine.smk")
    config: combine_config

def get_workspace_input_combination(wildcards):
    parts = wildcards.path.split('/')
    if len(parts) >= 2:
        parts[-2] = "workspace"
    return "/".join(parts) + ".txt"

use rule * from combine

rule prepare_datacards:
    output: f"{config['output_path']}/stat_analysis/{{run}}/{{signal}}/datacards/datacard__{{signal}}.txt"
    params:
        signal=lambda wildcards: wildcards.signal,
        datacard=lambda wildcards: CH_CFG[wildcards.signal]['datacard_' + wildcards.run],
        remove_bins=lambda wildcards: CH_CFG[wildcards.signal]['remove_bins_' + wildcards.run],
        remove_process=lambda wildcards: CH_CFG[wildcards.signal]['remove_process_' + wildcards.run],
        rename_process=lambda wildcards: CH_CFG[wildcards.signal]['rename_process_' + wildcards.run],
        output_dir=f"{config['output_path']}/stat_analysis/{{run}}/{{signal}}/datacards",
    log: f"{config['output_path']}/logs/prepare_{{run}}_datacards_{{signal}}.log"
    shell:
        '''
(
set +u
cd ../combination/inference-devel/
source setup.sh
cd ../../barista/
rm -rf {params.output_dir}
rename_processes.py {params.datacard} {params.rename_process} -d {params.output_dir}
remove_processes.py {params.output_dir}/$(basename {params.datacard}) {params.remove_process} -d none
remove_bins.py {params.output_dir}/$(basename {params.datacard}) {params.remove_bins} -d none
if grep -q ' group ' {params.output_dir}/$(basename {params.datacard}); then
    sed -e '/ group /d' {params.output_dir}/$(basename {params.datacard}) > {output}
else
    cp {params.output_dir}/$(basename {params.datacard}) {output}
fi
) 2>&1 | tee {log}
        '''

rule combine_combination_datacards:
    input:
        run2=f"{config['output_path']}/stat_analysis/run2/{{signal}}/datacards/datacard__{{signal}}.txt",
        run3=f"{config['output_path']}/stat_analysis/run3/{{signal}}/datacards/datacard__{{signal}}.txt"
    output: f"{config['output_path']}/stat_analysis/combination_{{signal}}/datacards/datacard__{{signal}}.txt"
    log: f"{config['output_path']}/logs/combine_combination_datacards_{{signal}}.log"
    shell:
        '''
./run_container combine "mkdir -p $(dirname {output}) && cd $(dirname {output}) && combineCards.py run2=/home/cmsusr/barista/{input.run2} run3=/home/cmsusr/barista/{input.run3} > $(basename {output})" 2>&1 | tee {log}
        '''

rule make_combination_plot:
    input:
        run2=f"{config['output_path']}/stat_analysis/run2/{{signal}}/limits/datacard_limits__{{signal}}.json",
        run3=f"{config['output_path']}/stat_analysis/run3/{{signal}}/limits/datacard_limits__{{signal}}.json",
        combination=f"{config['output_path']}/stat_analysis/combination_{{signal}}/limits/datacard_limits__{{signal}}.json"
    output: f"{config['output_path']}/stat_analysis/combination_{{signal}}/combination_plot_{{signal}}.pdf"
    log: f"{config['output_path']}/logs/combination_plot_{{signal}}.log"
    params:
        label=lambda wildcards: CH_CFG[wildcards.signal]['label']
    shell:
        '''
(
./run_container python coffea4bees/plots/make_limit_plot.py \
    -i {input.combination} {input.run3} {input.run2} \
    -l '"Combination" "Run 3" "Run 2"' \
    -o {output} \
    -x {params.label}

./run_container python src/plotting/pb_pdf_to_png.py {output}

) 2>&1 | tee {log}
        '''

localrules: all_combination, prepare_datacards, combine_combination_datacards, make_combination_plot

if not config.get('run_on_condor', True):
    combine_rules = [
        "workspace", "limits", "significance", "likelihood_scan_snapshot",
        "likelihood_scan_chunk", "likelihood_scan", "impacts_initial_fit",
        "impacts_do_fits", "impacts_collect", "split_impacts", "pdf_to_png",
        "gof_data", "gof_toys_chunk", "gof", "fit_diagnostics_bonly", "fit_diagnostics_sb",
        "postfit"
    ]
    workflow._localrules.update(combine_rules)