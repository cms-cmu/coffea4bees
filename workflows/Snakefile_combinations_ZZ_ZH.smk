import os
include: "helpers/common.smk"

# Include central combine rules directly
include: os.path.join(os.getcwd(), "src/stat_analysis/combine.smk")

# Define simple top-level variables for ease of use
CHANNELS = list(config['channels'].keys())
CH_CFG = config['channels']

rule all_combination:
    input:
        expand( f"{config['output_path']}/{{run}}/{{signal}}/datacard_limits__{{signal}}.json", run=['run2', 'run3'], signal=CHANNELS ),
        expand( f"significance__{config['output_path']}/{{run}}/{{signal}}/datacard__{{signal}}.log", run=['run2', 'run3'], signal=CHANNELS ),
        expand( f"{config['output_path']}/{{run}}/{{signal}}/datacard_likelihood_scan__{{signal}}.pdf", run=['run2', 'run3'], signal=CHANNELS ),
        expand( f"{config['output_path']}/combination_{{signal}}/datacard_limits__{{signal}}.json", signal=CHANNELS ),
        expand( f"significance__{config['output_path']}/combination_{{signal}}/datacard__{{signal}}.log", signal=CHANNELS ),
        expand( f"{config['output_path']}/combination_{{signal}}/datacard_likelihood_scan__{{signal}}.pdf", signal=CHANNELS ),
        expand( f"{config['output_path']}/combination_{{signal}}/datacard_impacts__{{signal}}.pdf", signal=CHANNELS ),
        expand( f"{config['output_path']}/combination_{{signal}}/datacard_gof__{{signal}}.pdf", signal=CHANNELS ),
        expand( f"{config['output_path']}/combination_{{signal}}/combination_plot_{{signal}}.pdf", signal=CHANNELS ),

rule prepare_datacards:
    output: f"{config['output_path']}/{{run}}/{{signal}}/datacard__{{signal}}.txt"
    params:
        signal=lambda wildcards: wildcards.signal,
        datacard=lambda wildcards: CH_CFG[wildcards.signal]['datacard_' + wildcards.run],
        remove_bins=lambda wildcards: CH_CFG[wildcards.signal]['remove_bins_' + wildcards.run],
        remove_process=lambda wildcards: CH_CFG[wildcards.signal]['remove_process_' + wildcards.run],
        rename_process=lambda wildcards: CH_CFG[wildcards.signal]['rename_process_' + wildcards.run],
        output_path=config['output_path'],
        run= lambda wildcards: wildcards.run
    log: f"{config['output_path']}/logs/prepare_{{run}}_datacards_{{signal}}.log"
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

rule combine_combination_datacards:
    input:
        run2=f"{config['output_path']}/run2/{{signal}}/datacard__{{signal}}.txt",
        run3=f"{config['output_path']}/run3/{{signal}}/datacard__{{signal}}.txt"
    output: f"{config['output_path']}/combination_{{signal}}/datacard__{{signal}}.txt"
    log: f"{config['output_path']}/logs/combine_combination_datacards_{{signal}}.log"
    shell:
        '''
./run_container combine "mkdir -p $(dirname {output}) && cd $(dirname {output}) && combineCards.py run2=/home/cmsusr/barista/{input.run2} run3=/home/cmsusr/barista/{input.run3} > $(basename {output})" 2>&1 | tee {log}
        '''

rule make_combination_plot:
    input:
        run2=f"{config['output_path']}/run2/{{signal}}/datacard_limits__{{signal}}.json",
        run3=f"{config['output_path']}/run3/{{signal}}/datacard_limits__{{signal}}.json",
        combination=f"{config['output_path']}/combination_{{signal}}/datacard_limits__{{signal}}.json"
    output: f"{config['output_path']}/combination_{{signal}}/combination_plot_{{signal}}.pdf"
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