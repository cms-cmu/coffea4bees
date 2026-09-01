# coffea4bees/workflows/Snakefile_PhaseE_3_closure.smk
# Phase E_3: Two-Stage Closure Fits & Background Systematics Extraction

import os

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb.yml"

include: "helpers/common.smk"

phase_e_cfg = resolve_config_section(config, primary_key='phase_e', fallback_keys=['phaseE', 'closure'])
for k, v in phase_e_cfg.items():
    config.setdefault(k, v)

config.setdefault('label', "ttHbb_mixeddata")
config.setdefault('output_path', "output/ttHbb_v2/closure_studies/")
config.setdefault('mix_name', "3bDvTMix4bDvT")
config.setdefault('nMixes', 15)
config.setdefault('classifier', "SvB_MA")
config.setdefault('variable', "SvB_MA_ps")
config.setdefault('channel', "ttHbb")
config.setdefault('rebin', "1")
config.setdefault('years_closure', "2016 2017 2018")
config.setdefault('closure_extra_args', "")

config.setdefault('input_file_data3b', f"{config['output_path']}root_inputs/histAll_{config['label']}.root")
config.setdefault('input_file_TT', f"{config['output_path']}root_inputs/histAll_{config['label']}.root")
config.setdefault('input_file_sig', f"{config['output_path']}root_inputs/histAll_{config['label']}.root")

container_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container"
config.setdefault('container_wrapper', container_wrapper)
config.setdefault('analysis_container_wrapper', config.get('container_wrapper', container_wrapper))
config.setdefault('stats_container_wrapper', "./run_container combine")

python_bin = os.getenv("CONTAINER_PYTHON", "python")
config.setdefault('python_bin', python_bin)

rule all_PhaseE_3:
    input:
        f"{config['output_path']}closure_fits/{config['mix_name']}/{config['classifier']}/rebin{config['rebin']}/SR/{config['channel']}/hists_closure_{config['mix_name']}_{config['variable']}_rebin{config['rebin']}.pkl"

rule hist_to_json_closure:
    input:
        f"{config['output_path']}histAll_{config['label']}.coffea"
    output:
        f"{config['output_path']}json_inputs/histAll_{config['label']}.json"
    params:
        container_wrapper = config['analysis_container_wrapper'],
        python_bin = config['python_bin']
    log:
        f"{config['output_path']}logs/hist_to_json_closure.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} coffea4bees/stats_analysis/convert_hist_to_json_closure.py \
            -i {input} \
            --auto_scale_mixed \
            -o {output} 2>&1 | tee {log}
        """

rule json_to_root_closure:
    input:
        injson = f"{config['output_path']}json_inputs/histAll_{config['label']}.json",
        script = "coffea4bees/stats_analysis/convert_json_to_root.py"
    output:
        f"{config['output_path']}root_inputs/histAll_{config['label']}.root"
    params:
        output_dir = f"{config['output_path']}root_inputs/",
        container_wrapper = config['stats_container_wrapper']
    log:
        f"{config['output_path']}logs/json_to_root_closure.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} python3 {input.script} \
            -f {input.injson} \
            --output {params.output_dir} 2>&1 | tee {log}
        """

rule run_two_stage_closure:
    input:
        file_mix = f"{config['output_path']}root_inputs/histAll_{config['label']}.root",
        file_TT = config["input_file_TT"],
        file_sig = config["input_file_sig"],
        file_data3b = config["input_file_data3b"],
        script = "coffea4bees/stats_analysis/runTwoStageClosure.py"
    output:
        f"{config['output_path']}closure_fits/{config['mix_name']}/{config['classifier']}/rebin{config['rebin']}/SR/{config['channel']}/hists_closure_{config['mix_name']}_{config['variable']}_rebin{config['rebin']}.pkl"
    params:
        outputPath = f"{config['output_path']}closure_fits",
        rebin = config["rebin"],
        variable = config["variable"],
        classifier = config["classifier"],
        mix_name = config["mix_name"],
        nMixes = config["nMixes"],
        years = config["years_closure"],
        channel = config["channel"],
        extra_arguments = config.get("closure_extra_args", ""),
        container_wrapper = config["stats_container_wrapper"]
    log:
        f"{config['output_path']}logs/run_two_stage_closure.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} python3 {input.script} \
            --input_file_mix {input.file_mix} \
            --input_file_TT {input.file_TT} \
            --input_file_sig {input.file_sig} \
            --input_file_data3b {input.file_data3b} \
            --mix_name {params.mix_name} \
            --nMixes {params.nMixes} \
            --classifier {params.classifier} \
            --var {params.variable} \
            --rebin {params.rebin} \
            --channel {params.channel} \
            --years {params.years} \
            --outputPath {params.outputPath} \
            {params.extra_arguments} 2>&1 | tee {log}
        """

localrules: all_PhaseE_3
