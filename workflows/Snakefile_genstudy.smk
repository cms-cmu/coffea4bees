import os
from datetime import datetime
username = os.getenv("USER")

# Use provided config or fall back to defaults
config.setdefault('output_path', 'output/gen_study/')
config.setdefault('dataset_location', "coffea4bees/metadata/datasets_HH4b_Run3/")
config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset', ['GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00'])
config.setdefault('year', '2022_EE')
config.setdefault('eos_path', f"{datetime.now().strftime('%Y%m%d')}_genstudy")

rule final_output:
    input:
        f"{config['output_path']}plots_genstudy/nJet_selected.png"
    shell:
        """
        echo "Copying results to eos"
        bash src/tools/copy_files_to_cernbox.sh -s {config[output_path]} -d www/HH4b/Plots/{config[eos_path]}/
        """
### Including modules
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

use rule analysis_processor from analysis as analysis_genstudy with:
    output: f"{config['output_path']}hists_genstudy.coffea"
    params:
        datasets = config['dataset'],
        years = config['year'],
        metadata = "coffea4bees/analysis/metadata/dropout_Run3.yml",
        processor = "coffea4bees/analysis/processors/processor_dropout_Run3.py",
        datasets_file = config['dataset_location'],
        blind = False,
        run_performance = False,
        extra_arguments = "--condor",
        username = username
    log: f"{config['output_path']}logs/analysis_genstudy.log"


rule plots:
    input: f"{config['output_path']}hists_genstudy.coffea"
    output: f"{config['output_path']}plots_genstudy/nJet_selected.png"
    params:
        output_dir = f"{config['output_path']}plots_genstudy/"
    log: f"{config['output_path']}logs/plots.log"
    shell:
        """
        ./run_container python coffea4bees/plots/makePlotsDropout.py \
    {input} -o {params.output_dir} &> {log}
        """