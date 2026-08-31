# coffea4bees/workflows/Snakefile_PhaseE_2_1_plots_comparison.smk
# Phase E_2_1: Comparison Plots (Data vs Mixed Data vs Synthetic Data)

import os

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb.yml"

include: "helpers/common.smk"

phase_e_cfg = resolve_config_section(config, primary_key='phase_e', fallback_keys=['phaseE', 'closure'])
for k, v in phase_e_cfg.items():
    config.setdefault(k, v)

config.setdefault('label', "ttHbb_mixeddata")
config.setdefault('output_path', "output/ttHbb_v2/closure_studies/")
config.setdefault('plot_script', "coffea4bees/plots/makePlots.py")
config.setdefault('comparison_plot_config', "coffea4bees/plots/metadata/plots_mixeddata_vs_data.yml")
config.setdefault('comparison_extra_args', "-s xW --year RunII")
config.setdefault('png_cores', 4)

container_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container"
config.setdefault('container_wrapper', container_wrapper)
config.setdefault('analysis_container_wrapper', config.get('container_wrapper', container_wrapper))

python_bin = os.getenv("CONTAINER_PYTHON", "python")
config.setdefault('python_bin', python_bin)

ROOT_DIR = os.getcwd()

rule all_PhaseE_2_1:
    input:
        f"{config['output_path']}plots_comparison/plots_done.txt"

rule make_plots_comparison:
    input:
        coffea_file = f"{config['output_path']}histAll_{config['label']}.coffea",
        metadata_file = lambda wildcards: os.path.join(ROOT_DIR, config['comparison_plot_config']),
        plot_script = lambda wildcards: os.path.join(ROOT_DIR, config['plot_script'])
    output:
        f"{config['output_path']}plots_comparison/plots_done.txt"
    log:
        f"{config['output_path']}logs/make_plots_comparison.log"
    params:
        output_dir = f"{config['output_path']}plots_comparison/",
        metadata = config['comparison_plot_config'],
        extra_arguments = config.get('comparison_extra_args', "-s xW --year RunII"),
        png_cores = config.get('png_cores', 4),
        container_wrapper = config['analysis_container_wrapper'],
        python_bin = config['python_bin']
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        echo "[$(date)] Starting comparison plots generation" > {log}
        {params.container_wrapper} {params.python_bin} {input.plot_script} \
            {input.coffea_file} \
            -m {params.metadata} \
            -o {params.output_dir} \
            -p {params.png_cores} \
            {params.extra_arguments} 2>&1 | tee -a {log}
        touch {output}
        echo "[$(date)] Completed comparison plots generation" >> {log}
        """

localrules: all_PhaseE_2_1, make_plots_comparison
