# coffea4bees/workflows/Snakefile_eval_friends_mixeddata_ttHbb.smk
# Evaluates SvB neural network models on ttHbb mixed data subsamples (mixeddata_4b)

import os
import shutil

YEARS = ["UL16_preVFP", "UL16_postVFP", "UL17", "UL18"]
DATASETS = ["mixeddata_4b"]

FRIEND_BASE = "root://cmseos.fnal.gov//store/user/algomez/XX4b/mixeddata/friends/ttHbb/"
OUT = "output/mixeddata_friends_ttHbb/"
FINAL_FRIEND_JSON = "coffea4bees/metadata/friends/friends_ttHbb_mixeddata_4b.json"

rule all_eval_friends:
    input:
        FINAL_FRIEND_JSON

rule create_eval_config:
    output: f"{OUT}eval_config.yml"
    run:
        import yaml
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        cfg = {
            "runner": {
                "workers": 4,
                "friend_base": FRIEND_BASE,
            },
            "dataset_location": "coffea4bees/metadata/datasets/",
            "datasets_file": "coffea4bees/metadata/datasets/mixeddata_4b.yml",
            "weights": "coffea4bees/metadata/weights/weights_ttHbb.yml",
            "config": {
                "blind": False,
                "apply_FvT": False,
                "apply_JCM": False,
                "apply_trigWeight": False,
                "apply_btagSF": True,
                "apply_boosted_veto": False,
                "run_SvB": True,
                "SvB_MA": True,
                "top_reconstruction": "fast",
                "fill_histograms": False,
                "make_friend_SvB": FRIEND_BASE,
            }
        }
        with open(output[0], "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

rule eval_friends_subsamples:
    input:
        eval_cfg = f"{OUT}eval_config.yml",
    output:
        f"{OUT}json/friends_{{dataset}}__{{year}}.json"
    log:
        f"{OUT}logs/friends_{{dataset}}__{{year}}.log"
    params:
        processor = "coffea4bees/analysis/processors/processor_ttHbb.py",
        output_path = f"{OUT}json/",
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        ./run_container python runner.py {input.eval_cfg} \
            --processor {params.processor} \
            --datasets {wildcards.dataset} \
            --years {wildcards.year} \
            --output-path {params.output_path} \
            --output $(basename {output} .json).coffea \
            --shared-dask --condor 2>&1 | tee {log}
        """

rule merge_friends_json:
    input:
        expand(f"{OUT}json/friends_{{dataset}}__{{year}}.json",
               dataset=DATASETS,
               year=YEARS)
    output:
        FINAL_FRIEND_JSON
    log:
        f"{OUT}logs/merge_friends_json.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        ./run_container python -m src.friendtrees.merge_friend_meta -i {input} -o {output} 2>&1 | tee {log}
        """
