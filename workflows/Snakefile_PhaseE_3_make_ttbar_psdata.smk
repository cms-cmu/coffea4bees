# coffea4bees/workflows/Snakefile_PhaseE_1c_make_ttbar_psdata.smk
# Phase E_1c: Stitched ttbar Pseudodata Production (sub_sample_MC.py)
#
# Produces unweighted ttbar pseudodata picoAODs (picoAOD_PSData.root) from the
# stitched ttbar MC datasets across Run 2 eras (UL16_preVFP, UL16_postVFP, UL17, UL18),
# merges the per-year registries, and installs the final dataset YAML for Phase E_2.

import os
import yaml

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_mixeddata.yml"

include: "helpers/common.smk"

# Resolve ttbar_psdata configuration block
psdata_cfg = resolve_config_section(config, primary_key='ttbar_psdata', fallback_keys=['psdata', 'ttbar_pseudodata'])
for k, v in psdata_cfg.items():
    if k != 'output_path':
        config[k] = v

# Fallbacks and defaults
config.setdefault('years', ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18'])
raw_years = config.get('years', ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18'])
if isinstance(raw_years, str):
    YEARS = [str(y).strip() for y in raw_years.split() if str(y).strip()]
else:
    YEARS = [str(y) for y in raw_years]
config['years'] = YEARS

config.setdefault('datasets_file', "coffea4bees/metadata/datasets/TT_stitched.yml")
config.setdefault('datasets', [
    "TTTo2L2Nu_stitched",
    "TTToSemiLeptonic_stitched",
    "TTToHadronic_stitched",
])
ttbar_datasets = config['datasets']
if isinstance(ttbar_datasets, str):
    ttbar_datasets = [d.strip() for d in ttbar_datasets.split() if d.strip()]

config.setdefault('friends_file', "coffea4bees/metadata/datasets/friends_ttHbb_stitched.yml")
base_output_path = config.get('output_path', "output/ttHbb_mixeddata_closure/")
if not base_output_path.endswith('/'):
    base_output_path += '/'
psdata_sub = psdata_cfg.get('output_path', "ttbar_PSData/")
if psdata_sub.startswith("output/"):
    psdata_out = os.path.join(base_output_path, "ttbar_PSData/")
else:
    psdata_out = os.path.join(base_output_path, psdata_sub)
if not psdata_out.endswith('/'):
    psdata_out += '/'
out = psdata_out
config.setdefault('base_path', "root://cmseos.fnal.gov//store/user/algomez/XX4b/mixeddata/Run2/ttbar_PSData_stitched")
config.setdefault('dataset_name', "ttbar_PSData_stitched")
config.setdefault('install_path', f"coffea4bees/metadata/datasets/{config['dataset_name']}.yml")
config.setdefault('seed', 5)
config.setdefault('worker_memory', "6GB")
config.setdefault('chunksize', 100000)
config.setdefault('picosize', 100000)

localrules: all_PhaseE_3, all_PhaseE_1c, all_ttbar_pseudodata, create_ttbar_psdata_config, merge_ttbar_psdata_registries, install_ttbar_psdata_dataset

# ── Master Target ─────────────────────────────────────────────────────────────
rule all_PhaseE_3:
    input:
        config['install_path']

rule all_PhaseE_1c:
    input:
        rules.all_PhaseE_3.input

rule all_ttbar_pseudodata:
    input:
        config['install_path']

# ── Stage 1: Generate Runner Config for SubSampler ────────────────────────────
rule create_ttbar_psdata_config:
    output:
        f"{out}ttbar_psdata_config.yml"
    params:
        base_path = config['base_path'],
        seed = int(config['seed']),
        worker_memory = str(config['worker_memory']),
        chunksize = int(config['chunksize']),
        picosize = int(config['picosize']),
        datasets_file = config['datasets_file'],
    run:
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        ps_cfg = {
            "runner": {
                "workers": 4,
                "min_workers": 1,
                "max_workers": 200,
                "chunksize": int(params.chunksize),
                "picosize": int(params.picosize),
                "basketsize": 10000,
                "class_name": "SubSampler",
                "data_tier": "picoAOD",
                "condor_cores": 1,
                "worker_memory": str(params.worker_memory),
                "condor_transfer_input_files": ["src", "coffea4bees/"],
                "allowlist_sites": ["T2_US_Nebraska", "T2_US_Purdue", "T3_US_FNALLPC", "T3_US_NotreDame"],
                "datasets_file": str(params.datasets_file),
            },
            "config": {
                "base_path": str(params.base_path),
                "sub_sampling_rand_seed": int(params.seed),
                "apply_trigWeight": True,
                "step": int(params.chunksize),
                "skip_collections": [
                    "notCanJet", "canJet0", "canJet1", "canJet2", "canJet3", "trigWeight"
                ],
                "skip_branches": [
                    "btagWeight_.*", "HHSR", "ZZSR", "ZHSR", "SR", "st", "d01TruthMatch",
                    "nMuon_selected", "fourTag", "threeTag", "xW", "leadStM", "nSelJets",
                    "d02TruthMatch", "d12TruthMatch", "truthMatch", "pseudoTagWeight",
                    "ttbarWeight", "nIsoMuons", "xt", "weight", "aveAbsEtaOth", "xWbW",
                    "nAllNotCanJets", "dRjjOther", "xWt", "nPSTJets", "passXWt", "d03TruthMatch",
                    "xbW", "mcPseudoTagWeight", "d23TruthMatch", "m4j", "sublStM", "dRjjClose",
                    "aveAbsEta", "stNotCan", "SB", "selectedViewTruthMatch", "d13TruthMatch",
                    "genWeight"
                ],
            }
        }
        with open(output[0], 'w') as f:
            yaml.dump(ps_cfg, f, default_flow_style=False)

# ── Stage 2: Run SubSampler per Year ──────────────────────────────────────────
rule make_ttbar_pseudodata_per_year:
    input:
        config_file = f"{out}ttbar_psdata_config.yml",
        datasets_file = config['datasets_file'],
        friends = config['friends_file'],
    output:
        reg = f"{out}per_year/picoaod_datasets_{config['dataset_name']}__{{year}}.yml",
        done = f"{out}.make_ttbar_psdata_{{year}}.done",
    log:
        f"{out}logs/make_ttbar_psdata__{{year}}.log"
    params:
        friends = config['friends_file'],
        processor = "coffea4bees/skimmer/processor/sub_sample_MC.py",
        datasets = " ".join(ttbar_datasets),
        datasets_file = config['datasets_file'],
        output_path = f"{out}per_year/",
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.reg}) $(dirname {log})
        ./run_container python runner.py {input.config_file} \
            -p {params.processor} \
            -d {params.datasets} \
            -c {params.datasets_file} \
            --friends {params.friends} \
            --years {wildcards.year} \
            --output-path {params.output_path} \
            --output $(basename {output.reg}) \
            -s --shared-dask --condor 2>&1 | tee {log}
        touch {output.done}
        """

# ── Stage 3: Merge Registries & Install Dataset YAML ──────────────────────────
rule merge_ttbar_psdata_registries:
    input:
        expand(f"{out}per_year/picoaod_datasets_{config['dataset_name']}__{{year}}.yml", year=YEARS)
    output:
        f"{out}picoaod_datasets_{config['dataset_name']}_combined.yml"
    log:
        f"{out}logs/merge_ttbar_psdata_registries.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        python coffea4bees/workflows/scripts/merge_mixeddata_registries.py {input} {output} 2>&1 | tee -a {log}
        top_dest="output/ttHbb_mixeddata_stitched_closure/picoaod_datasets_{config[dataset_name]}_combined.yml"
        if [ "{output}" != "$top_dest" ]; then
            mkdir -p $(dirname "$top_dest")
            cp -f {output} "$top_dest"
        fi
        """

rule install_ttbar_psdata_dataset:
    input:
        f"{out}picoaod_datasets_{config['dataset_name']}_combined.yml"
    output:
        config['install_path']
    log:
        f"{out}logs/install_ttbar_psdata_dataset.log"
    params:
        name = config['dataset_name']
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        python src/tools/make_dataset_yml.py -i {input} -o {output} -n {params.name} 2>&1 | tee -a {log}
        echo "Installed {output} (dataset name: {params.name})" 2>&1 | tee -a {log}
        """
