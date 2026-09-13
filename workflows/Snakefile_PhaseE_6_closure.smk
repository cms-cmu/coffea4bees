# coffea4bees/workflows/Snakefile_PhaseE_4_closure.smk
# Phase E_4: Two-Stage Closure Test (Stage 0 Variance, Stage 1 Bias, Stage 2 Spurious Signal)

import os

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_mixeddata.yml"

include: "helpers/common.smk"

phase_e_cfg = resolve_config_section(config, primary_key='phase_e', fallback_keys=['phaseE', 'closure'])
for k, v in phase_e_cfg.items():
    config.setdefault(k, v)

config.setdefault('label', "ttHbb_mixeddata")
config.setdefault('output_path', "output/ttHbb_mixeddata_closure/")
config.setdefault('mix_name', "3bDvTMix4bDvT")
config.setdefault('classifier', "SvB_MA")
config.setdefault('variable', "SvB_MA_ps")
config.setdefault('channel', "ttHbb")
config.setdefault('rebin', "1")
config.setdefault('years_closure', "2016 2017 2018")
config.setdefault('closure_extra_args', "")
config.setdefault('scale_mixed', 1.0)
config.setdefault('analysis_container_wrapper', "./run_container")
config.setdefault('python_bin', "python")

out = config['output_path']
if not out.endswith("/"):
    out += "/"
mix_name = config['mix_name']
classifier = config['classifier']
rebin_str = f"rebin{config['rebin']}"
channel = config['channel']
var = config['variable']

closure_output_dir = f"{out}closure_fits/{mix_name}/{classifier}/{rebin_str}/SR/{channel}/"
closure_pkl = f"{closure_output_dir}hists_closure_{mix_name}_{var}_{rebin_str}.pkl"

localrules: all_PhaseE_6, hist_to_json_closure, json_to_root_closure, make_fvt_data3b_root, run_two_stage_closure

rule all_PhaseE_6:
    input:
        closure_pkl

def get_hist_to_json_inputs(wildcards):
    nominal = f"{out}histAll_{config['label']}.coffea"
    single = f"{out}singlefiles/histAll_{config['label']}__mixeddata_4b.coffea"
    if os.path.exists(single) and not os.path.exists(nominal):
        files = [single]
    else:
        files = [nominal]
    v3_nominal = "output/ttHbb_v3/histAll_ttHbb_v3.coffea"
    if os.path.exists(v3_nominal):
        files.append(v3_nominal)
    return files

rule hist_to_json_closure:
    input:
        get_hist_to_json_inputs
    output:
        f"{out}json_inputs/histAll_{config['label']}.json"
    params:
        container_wrapper = config['analysis_container_wrapper'],
        python_bin = config['python_bin'],
        scale_mixed = config.get('scale_mixed', 1.0),
        extra_args = config.get('closure_extra_args', '')
    log:
        f"{out}logs/hist_to_json_closure.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} coffea4bees/stats_analysis/convert_hist_to_json_closure.py \
            -i {input} \
            --scale_mixed {params.scale_mixed} \
            {params.extra_args} \
            -o {output} 2>&1 | tee {log}
        """

rule json_to_root_closure:
    input:
        injson = f"{out}json_inputs/histAll_{config['label']}.json",
        script = "coffea4bees/stats_analysis/convert_json_to_root.py"
    output:
        f"{out}root_inputs/histAll_{config['label']}.root"
    params:
        container_wrapper = config['analysis_container_wrapper'],
        python_bin = config['python_bin'],
        sig_input = config.get('signal_input_json', '')
    log:
        f"{out}logs/json_to_root_closure.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} combine python3 {input.script} \
            -f {input.injson} \
            -o $(dirname {output}) 2>&1 | tee {log}
        """

n_models_closure = int(config.get('nMixes', config.get('n_models', 16)))

rule make_fvt_data3b_root:
    input:
        script = "coffea4bees/workflows/make_fvt_data3b_hists.py",
        ci = "coffea4bees/metadata/datasets/classifier_inputs_ttHbb.json",
        jcms = [f"{out}JCM_subsamples/jetCombinatoricModel_SB_mix_v{m}.yml" for m in range(n_models_closure)],
        friends = [f"{out}FvT_training/friends/friends_FvT_{mix_name}_v{m}.json" for m in range(n_models_closure)]
    output:
        f"{out}root_inputs/histMixedBkg_data_3b_for_mixed.root"
    params:
        container_wrapper = config['analysis_container_wrapper'],
        python_bin = config['python_bin'],
        mix_name = mix_name,
        svb_result = config.get('svb_result', f"{out}inputs/SvB_result.json"),
        jcm_template = lambda wildcards: f"{out}JCM_subsamples/jetCombinatoricModel_SB_mix_v{{m}}.yml",
        fvt_template = lambda wildcards: f"{out}FvT_training/friends/friends_FvT_{mix_name}_v{{m}}.json",
        n_models = n_models_closure,
        tt4bSF = config.get('tt4bSF', 1.4508),
    log:
        f"{out}logs/make_fvt_data3b_root.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} {input.script} \
            --svb_result {params.svb_result} \
            --jcm_file "{params.jcm_template}" \
            --fvt_template "{params.fvt_template}" \
            --mix_name {params.mix_name} \
            --n_models {params.n_models} \
            --tt4bSF {params.tt4bSF} \
            -o {output} 2>&1 | tee {log}
        """

rule make_signal_root_closure:
    input:
        script = "scripts/make_signal_root.py",
        injson = config.get('nominal_json', "output/ttHbb_stitched/histAll_ttHbb_stitched.json")
    output:
        f"{out}root_inputs/hist_signal_ttHbb.root"
    params:
        container_wrapper = config['analysis_container_wrapper']
    log:
        f"{out}logs/make_signal_root_closure.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} combine python3 {input.script} \
            -i {input.injson} \
            -o {output} 2>&1 | tee {log}
        """

rule run_two_stage_closure:
    input:
        inroot = f"{out}root_inputs/histAll_{config['label']}.root",
        data3b = f"{out}root_inputs/histMixedBkg_data_3b_for_mixed.root",
        sigroot = f"{out}root_inputs/hist_signal_ttHbb.root",
        script = "coffea4bees/stats_analysis/runTwoStageClosure.py"
    output:
        closure_pkl
    params:
        container_wrapper = config['analysis_container_wrapper'],
        mix_name = mix_name,
        var = var,
        channel = channel,
        rebin = config['rebin'],
        output_dir = f"{out}closure_fits/",
        years = config.get('years_closure', '2016 2017 2018'),
        nMixes = n_models_closure,
        extra_args = config.get('closure_extra_args', '')
    log:
        f"{out}logs/run_two_stage_closure.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} combine python3 {input.script} \
            --mix_name {params.mix_name} \
            --var {params.var} \
            --channel {params.channel} \
            --rebin {params.rebin} \
            --outputPath {params.output_dir} \
            --input_file_mix {input.inroot} \
            --input_file_data3b {input.data3b} \
            --input_file_sig {input.sigroot} \
            --input_file_TT {input.data3b} \
            --years {params.years} \
            --nMixes {params.nMixes} \
            {params.extra_args} 2>&1 | tee {log}
        """
