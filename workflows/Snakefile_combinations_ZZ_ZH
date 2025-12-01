config = {
    'output_path' : 'output/ZZ_ZH_combination/v3/wotherSOne',
    # 'output_path' : 'output/ZZ_ZH_combination/v3/wotherSZero',
    # 'output_path' : 'output/ZZ_ZH_combination/v3/wotherRegionsSOne',
    'container_wrapper': './run_container combine',
    'channels': {
        'ZZ_bbbb': {
            'datacard_run2': 'output/ZZ_ZH_combination/v3/original_datacards/run2/combine_SvB_MA.txt',
            'remove_bins_run2': "'zh*' 'hh*'",
            'remove_process_run2': "None", #"'ZH_bbbb' 'ggHH_bbbb'",   
            'rename_process_run2': "'ZZ=ZZ_bbbb' 'ZH=ZH_bbbb', 'HH=ggHH_bbbb'",
            'datacard_run3': 'output/ZZ_ZH_combination/v3/original_datacards/run3/datacard.txt',
            'remove_bins_run3': "'ggHHbbbb_*' 'qqHHbbbb_*' 'zhHHbbbb_*'",
            'rename_process_run3': "'ggHH_hbbhbb=ggHH_bbbb' 'qqHH_hbbhbb=qqHH_bbbb'",
            'remove_process_run3': "None", #"'ZH_bbbb' 'ggHH_bbbb' 'qqHH_bbbb'",
            'signallabel': 'ZZ_bbbb',
            'othersignal': 'ZH_bbbb ggHH_bbbb qqHH_bbbb',
            'label': 'ZZ',
            'workspace': ''
        },
        'ZH_bbbb': {
            'datacard_run2': 'output/ZZ_ZH_combination/v3/original_datacards/run2/combine_SvB_MA.txt',
            'remove_bins_run2': "'zz*' 'hh*'",
            'rename_process_run2': "'ZZ=ZZ_bbbb' 'ZH=ZH_bbbb', 'HH=ggHH_bbbb'",
            'remove_process_run2': "None", #"'ZZ_bbbb' 'ggHH_bbbb'",
            'datacard_run3': 'output/ZZ_ZH_combination/v3/original_datacards/run3/datacard.txt',
            'remove_bins_run3': "'ggHHbbbb_*' 'qqHHbbbb_*' 'zzHHbbbb_*'",
            'remove_process_run3': "None", #"'ZZ_bbbb' 'ggHH_bbbb' 'qqHH_bbbb'",
            'rename_process_run3': "'ggHH_hbbhbb=ggHH_bbbb' 'qqHH_hbbhbb=qqHH_bbbb'",
            'signallabel': 'ZH_bbbb',
            'othersignal': 'ZZ_bbbb ggHH_bbbb qqHH_bbbb',
            'label': 'ZH',
            'workspace': ''
        }
    }
}

include: "helpers/common.smk"

module combine:
    snakefile: "rules/combine.smk"
    config: config

rule all:
    input:
        expand(
            "{output_path}/{run}/{signal}/limits.txt",
            output_path=config["output_path"],
            run=["run2", "run3"],
            signal=list(config["channels"].keys()),
        ),
        expand(
            "{output_path}/{run}/{signal}/significance.txt",
            output_path=config["output_path"],
            run=["run2", "run3"],
            signal=list(config["channels"].keys()),
        ),
        expand(
            "{output_path}/{run}/{signal}/likelihood_scan.pdf",
            output_path=config["output_path"],
            run=["run2", "run3"],
            signal=list(config["channels"].keys()),
        ),
        expand(
            "{output_path}/combination_{signal}/limits.txt",
            output_path=config["output_path"],
            signal=list(config["channels"].keys()),
        ),
        expand(
            "{output_path}/combination_{signal}/significance.txt",
            output_path=config["output_path"],
            signal=list(config["channels"].keys()),
        ),
        expand(
            "{output_path}/combination_{signal}/likelihood_scan.pdf",
            output_path=config["output_path"],
            signal=list(config["channels"].keys()),
        ),
        expand(
            "{output_path}/combination_{signal}/impacts.pdf",
            output_path=config["output_path"],
            signal=list(config["channels"].keys()),
        ),
        expand(
            "{output_path}/combination_{signal}/gof.pdf",
            output_path=config["output_path"],
            signal=list(config["channels"].keys()),
        ),
        expand(
            '{output_path}/combination_{signal}/combination_plot_{signal}.pdf',
            output_path=config["output_path"],
            signal=list(config["channels"].keys()),
        ),

rule prepare_datacards:
    output:
        f'{config['output_path']}/{{run}}/{{signal}}/datacard.txt'
    params:
        signal=lambda wildcards: wildcards.signal,
        datacard=lambda wildcards: f"{config['channels'][wildcards.signal][f'datacard_{wildcards.run}']}",
        remove_bins=lambda wildcards: config['channels'][wildcards.signal][f'remove_bins_{wildcards.run}'],
        remove_process=lambda wildcards: config['channels'][wildcards.signal][f'remove_process_{wildcards.run}'],
        rename_process=lambda wildcards: config['channels'][wildcards.signal][f'rename_process_{wildcards.run}'],
        output_path=config['output_path'],
        run= lambda wildcards: wildcards.run
    log: 
        f'{config['output_path']}/logs/prepare_{{run}}_datacards_{{signal}}.log'
    shell:
        '''
(
set +u
cd ../combination/inference-devel/
source setup.sh
cd ../../barista/
rm -rf {params.output_path}/{params.run}/{params.signal}/
rename_processes.py {params.datacard} {params.rename_process} -d {params.output_path}/{params.run}/{params.signal}/
remove_processes.py {params.output_path}/{params.run}/{params.signal}/$(basename {params.datacard}) {params.remove_process} -d none
remove_bins.py {params.output_path}/{params.run}/{params.signal}/$(basename {params.datacard}) {params.remove_bins} -d none
if grep -q ' group ' {params.output_path}/{params.run}/{params.signal}/$(basename {params.datacard}); then sed -e '/ group /d' {params.output_path}/{params.run}/{params.signal}/$(basename {params.datacard}) > {output}; fi
) 2>&1 | tee {log}
        '''

use rule workspace from combine with:
    input: 
        f'{config['output_path']}/{{run}}/{{signal}}/datacard.txt'
    output:
        f'{config["output_path"]}/{{run}}/{{signal}}/workspace.root'
    log:
        f'{config["output_path"]}/logs/workspace_{{run}}_{{signal}}.log'
    params:
        signallabel = lambda wildcards: config['channels'][wildcards.signal]['signallabel'],
        othersignal_maps = lambda wildcards: additional_poi(wildcards.signal),
        container_wrapper = config['container_wrapper']

use rule limits from combine with:
    input:
        f'{config["output_path"]}/{{run}}/{{signal}}/workspace.root'
    output:
        txt=f'{config["output_path"]}/{{run}}/{{signal}}/limits.txt',
        json=f'{config["output_path"]}/{{run}}/{{signal}}/limits.json'
    log:
        f'{config["output_path"]}/logs/limits_{{run}}_{{signal}}.log'
    params:
        signallabel = lambda wildcards: config['channels'][wildcards.signal]['signallabel'],
        set_parameters_zero = lambda wildcards: set_parameters( wildcards.signal, value=1 ),
        freeze_parameters = lambda wildcards: freeze_parameters( wildcards.signal ),
        container_wrapper = config['container_wrapper']

use rule significance from combine with:
    input:
        f'{config["output_path"]}/{{run}}/{{signal}}/workspace.root'
    output:
        f'{config["output_path"]}/{{run}}/{{signal}}/significance.txt'
    log:
        f'{config["output_path"]}/logs/significance_{{run}}_{{signal}}.log'
    params:
        signallabel = lambda wildcards: config['channels'][wildcards.signal]['signallabel'],
        set_parameters_zero = lambda wildcards: set_parameters( wildcards.signal, value=1 ),
        freeze_parameters = lambda wildcards: freeze_parameters( wildcards.signal ),
        container_wrapper = config['container_wrapper']

use rule likelihood_scan from combine with:
    input:
        f'{config["output_path"]}/{{run}}/{{signal}}/workspace.root'
    output:
        f'{config["output_path"]}/{{run}}/{{signal}}/likelihood_scan.pdf'
    log:
        f'{config["output_path"]}/logs/likelihood_scan_{{run}}_{{signal}}.log'
    params:
        signallabel = lambda wildcards: config['channels'][wildcards.signal]['signallabel'],
        set_parameters_zero = lambda wildcards: set_parameters( wildcards.signal, value=1 ),
        freeze_parameters = lambda wildcards: freeze_parameters( wildcards.signal ),
        container_wrapper = config['container_wrapper']

rule combine_combination_datacards:
    input:
        run2=f'{config["output_path"]}/run2/{{signal}}/datacard.txt',
        run3=f'{config["output_path"]}/run3/{{signal}}/datacard.txt'
    output:
        f'{config["output_path"]}/combination_{{signal}}/datacard.txt'
    log:
        f'{config["output_path"]}/logs/combine_combination_datacards_{{signal}}.log'
    shell:
        '''
./run_container combine "mkdir -p $(dirname {output}) && cd $(dirname {output}) && combineCards.py run2=/home/cmsusr/barista/{input.run2} run3=/home/cmsusr/barista/{input.run3} > $(basename {output})" 2>&1 | tee {log}
        '''

use rule workspace from combine as workspace_combination with:
    input: 
        f'{config["output_path"]}/combination_{{signal}}/datacard.txt'
    output:
        f'{config["output_path"]}/combination_{{signal}}/workspace.root'
    log:
        f'{config["output_path"]}/logs/workspace_combination_{{signal}}.log'
    params:
        signallabel = lambda wildcards: config['channels'][wildcards.signal]['signallabel'],
        othersignal_maps = lambda wildcards: additional_poi(wildcards.signal),
        container_wrapper = config['container_wrapper']

use rule limits from combine as limits_combination with:
    input:
        f'{config["output_path"]}/combination_{{signal}}/workspace.root'
    output:
        txt=f'{config["output_path"]}/combination_{{signal}}/limits.txt',
        json=f'{config["output_path"]}/combination_{{signal}}/limits.json'
    log:
        f'{config["output_path"]}/logs/limits_combination_{{signal}}.log'
    params:
        signallabel = lambda wildcards: config['channels'][wildcards.signal]['signallabel'],
        set_parameters_zero = lambda wildcards: set_parameters( wildcards.signal, value=1 ),
        freeze_parameters = lambda wildcards: freeze_parameters( wildcards.signal ),
        container_wrapper = config['container_wrapper']

use rule significance from combine as significance_combination with:
    input:
        f'{config["output_path"]}/combination_{{signal}}/workspace.root'
    output:
        f'{config["output_path"]}/combination_{{signal}}/significance.txt'
    log:
        f'{config["output_path"]}/logs/significance_combination_{{signal}}.log'
    params:
        signallabel = lambda wildcards: config['channels'][wildcards.signal]['signallabel'],
        set_parameters_zero = lambda wildcards: set_parameters( wildcards.signal, value=1 ),
        freeze_parameters = lambda wildcards: freeze_parameters( wildcards.signal ),
        container_wrapper = config['container_wrapper']

use rule likelihood_scan from combine as likelihood_scan_combination with:
    input:
        f'{config["output_path"]}/combination_{{signal}}/workspace.root'
    output:
        f'{config["output_path"]}/combination_{{signal}}/likelihood_scan.pdf'
    log:
        f'{config["output_path"]}/logs/likelihood_scan_combination_{{signal}}.log'
    params:
        signallabel = lambda wildcards: config['channels'][wildcards.signal]['signallabel'],
        set_parameters_zero = lambda wildcards: set_parameters( wildcards.signal, value=1 ),
        freeze_parameters = lambda wildcards: freeze_parameters( wildcards.signal ),
        container_wrapper = config['container_wrapper']

use rule impacts from combine as impacts_combination with:
    input:
        f'{config["output_path"]}/combination_{{signal}}/workspace.root'
    output:
        f'{config["output_path"]}/combination_{{signal}}/impacts.pdf'
    log:
        f'{config["output_path"]}/logs/impacts_combination_{{signal}}.log'
    params:
        signallabel = lambda wildcards: config['channels'][wildcards.signal]['signallabel'],
        set_parameters_zero = lambda wildcards: set_parameters( wildcards.signal, value=1 ),
        set_parameters_ranges = lambda wildcards: set_parameters_ranges(wildcards.signal),
        container_wrapper = config['container_wrapper']

use rule gof from combine as gof_combination with:
    input:
        f'{config["output_path"]}/combination_{{signal}}/workspace.root'
    output:
        f'{config["output_path"]}/combination_{{signal}}/gof.pdf'
    log:
        f'{config["output_path"]}/logs/gof_combination_{{signal}}.log'
    params:
        signallabel = lambda wildcards: config['channels'][wildcards.signal]['signallabel'],
        set_parameters_zero = lambda wildcards: set_parameters( wildcards.signal, value=1 ),
        container_wrapper = config['container_wrapper']

rule make_combination_plot:
    input:
        run2=f'{config["output_path"]}/run2/{{signal}}/limits.json',
        run3=f'{config["output_path"]}/run3/{{signal}}/limits.json',
        combination=f'{config["output_path"]}/combination_{{signal}}/limits.json'
    output:
        f'{config["output_path"]}/combination_{{signal}}/combination_plot_{{signal}}.pdf'
    log:
        f'{config["output_path"]}/logs/combination_plot_{{signal}}.log'
    params:
        label=lambda wildcards: config['channels'][wildcards.signal]['label']
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