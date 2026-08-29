import os
import json

config.setdefault('eos_base', "root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/ttHbb_v2")
config.setdefault('output_path', "output/ttHbb_mixeddata_closure/friendtrees/")
config.setdefault('model', f"{config['eos_base']}/classifier/SvB_ttHbb_v2")
config.setdefault('years', ["2016", "2017", "2018"])
config.setdefault('nSamples', 15)

YEARS = config['years']
SAMPLES = [f"v{i}" for i in range(config['nSamples'])]
OUT = config['output_path'].rstrip("/")
MODEL = config['model']
EOS_BASE = config['eos_base']

CLASSIFIER_GPU = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:classifier_latest"
INIT = "set -e && set +u && source /entrypoint.sh && set -u && export PYTHONUNBUFFERED=1"
CLASSIFIER_CONFIG_PATHS = "coffea4bees"

def get_mixed_picoaod(wildcards):
    return f"root://cmseos.fnal.gov//store/user/jda102/condor/ZH4b/ULTrig/mixed{wildcards.year}_3bDvTMix4bDvT_{wildcards.sample}/picoAOD_3bDvTMix4bDvT_4b_wJCM_{wildcards.sample}_newSBDef.root"

def get_friend_output(wildcards):
    return f"{EOS_BASE}/friend/SvB_ttHbb_v2/mixed{wildcards.year}_3bDvTMix4bDvT_{wildcards.sample}/"

rule all:
    input:
        expand(f"{OUT}/SvB_{{year}}_{{sample}}.done", year=YEARS, sample=SAMPLES),
        f"{OUT}/friends_ttHbb_mixeddata.json"

rule evaluate_single_mixed:
    input:
        eval_yml = "coffea4bees/classifier/config/workflows/ttHbb_2024_v2/SvB/evaluate_mixed_template.yml",
        common_yml = "coffea4bees/classifier/config/workflows/ttHbb_2024_v2/common.yml",
    output:
        flag = f"{OUT}/SvB_{{year}}_{{sample}}.done",
    log:
        f"{OUT}/logs/SvB_{{year}}_{{sample}}.log",
    container: CLASSIFIER_GPU
    resources:
        runtime = 60,
        mem_mb = 16000,
        gres = "mps:25",
        slurm_partition = "work",
    threads: 4
    params:
        init = INIT,
        classifier_config_paths = CLASSIFIER_CONFIG_PATHS,
        model = MODEL,
        file = get_mixed_picoaod,
        output = get_friend_output,
        year = "{year}",
    shell:
        """
        mkdir -p $(dirname {output.flag}) $(dirname {log})
        {params.init} && \
        PORT=$(shuf -i 10000-60000 -n 1) && \
        CLASSIFIER_CONFIG_PATHS={params.classifier_config_paths} \
        python -m src.classifier.task.main \
            template "{{model: {params.model}, year: '{params.year}', file: '{params.file}', output: '{params.output}'}}" {input.eval_yml} \
            -from {input.common_yml} \
            -setting Monitor "enable: False" \
            2>&1 | tee -a {log}
        touch {output.flag}
        """

rule merge_friend_metadata:
    input:
        expand(f"{OUT}/SvB_{{year}}_{{sample}}.done", year=YEARS, sample=SAMPLES)
    output:
        json = f"{OUT}/friends_ttHbb_mixeddata.json"
    params:
        years = YEARS,
        samples = SAMPLES,
        eos_base = EOS_BASE
    run:
        friend_dict = {}
        for y in params.years:
            for s in params.samples:
                ds_name = f"mix_{s}_{y}"
                pico_file = f"root://cmseos.fnal.gov//store/user/jda102/condor/ZH4b/ULTrig/mixed{y}_3bDvTMix4bDvT_{s}/picoAOD_3bDvTMix4bDvT_4b_wJCM_{s}_newSBDef.root"
                friend_file = f"{params.eos_base}/friend/SvB_ttHbb_v2/mixed{y}_3bDvTMix4bDvT_{s}/SvB.root"
                friend_dict[ds_name] = {
                    "picoAOD": pico_file,
                    "SvB": friend_file,
                    "SvB_MA": friend_file,
                }
        with open(output.json, "w") as f:
            json.dump(friend_dict, f, indent=2)
