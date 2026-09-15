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
default_combine_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container combine"
config.setdefault('combine_container_wrapper', config.get('container_wrapper', default_combine_wrapper))
default_analysis_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container"
config.setdefault('analysis_container_wrapper', config.get('analysis_wrapper', default_analysis_wrapper))
python_bin = config.get('python_bin', os.getenv("CONTAINER_PYTHON", "python"))
config.setdefault('python_bin', python_bin)
combine_cmd = f"{config['combine_container_wrapper']} python3" if config.get('combine_container_wrapper') else config['python_bin']

raw_years = config.get('years', ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18'])
if isinstance(raw_years, str):
    YEARS = [str(y).strip() for y in raw_years.split() if str(y).strip()]
else:
    YEARS = [str(y) for y in raw_years]
config['years'] = YEARS

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

localrules: all_PhaseE_6, hist_to_json_closure, json_to_root_closure, make_fvt_data3b_root, make_signal_root_closure, run_two_stage_closure, check_closure_validation

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
        n_subsamples = lambda wildcards: n_models_closure,
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
            --n_subsamples {params.n_subsamples} \
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
        combine_cmd = combine_cmd,
        sig_input = config.get('signal_input_json', '')
    log:
        f"{out}logs/json_to_root_closure.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.combine_cmd} {input.script} \
            -f {input.injson} \
            -o $(dirname {output}) 2>&1 | tee {log}
        """

n_models_closure = int(config.get('n_subsamples', config.get('n_models', config.get('n_samples', config.get('nMixes', 15)))))

rule make_fvt_data3b_root:
    input:
        script = "coffea4bees/stats_analysis/make_fvt_data3b_hists.py",
        ci = config.get('nominal_classifier_inputs', "coffea4bees/metadata/datasets/classifier_inputs_ttHbb.json"),
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
        years = " ".join(YEARS),
        extra_args = "--dummy" if config.get('test', False) else "",
    log:
        f"{out}logs/make_fvt_data3b_root.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} {input.script} \
            --svb_result {params.svb_result} \
            --classifier_inputs {input.ci} \
            --jcm_file "{params.jcm_template}" \
            --fvt_template "{params.fvt_template}" \
            --mix_name {params.mix_name} \
            --n_models {params.n_models} \
            --tt4bSF {params.tt4bSF} \
            --years {params.years} \
            {params.extra_args} \
            -o {output} 2>&1 | tee {log}
        """

rule make_signal_root_closure:
    input:
        script = "coffea4bees/stats_analysis/make_signal_root.py",
    output:
        f"{out}root_inputs/hist_signal_ttHbb.root"
    params:
        combine_cmd = combine_cmd,
        injson = config.get('nominal_json', "output/ttHbb_stitched/histAll_ttHbb_stitched.json"),
        var = var,
        years = " ".join(YEARS),
        extra_args = "--dummy" if config.get('test', False) else "",
    log:
        f"{out}logs/make_signal_root_closure.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.combine_cmd} {input.script} \
            -i {params.injson} \
            --var {params.var} \
            --years {params.years} \
            {params.extra_args} \
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
        combine_cmd = combine_cmd,
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
        {params.combine_cmd} {input.script} \
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
        if [ ! -f {output} ]; then
            touch {output}
        fi
        """

rule check_closure_validation:
    input:
        closure_pkl = closure_pkl,
        script = "coffea4bees/stats_analysis/tests/dumpTwoStageInputs.py"
    output:
        validation_txt = f"{out}closure_validation_{config['label']}.txt",
        counts_yml = f"{out}closure_counts_{config['label']}.yml"
    log:
        f"{out}logs/closure_validation_{config['label']}.log"
    params:
        combine_cmd = combine_cmd,
        root_file = lambda wildcards: f"{closure_output_dir}hists_closure_{mix_name}_{var}_{rebin_str}.root",
        known_counts = lambda wildcards: config.get("known_counts_closure", ""),
        test_script = "coffea4bees/stats_analysis/tests/test_runTwoStageClosure.py",
        output_dir = closure_output_dir,
        channel = channel
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.validation_txt}) $(dirname {log})
        echo "Dumping closure counts from {params.root_file}" > {log}
        {params.combine_cmd} {input.script} \
            --inputFile {params.root_file} \
            --outputFile {output.counts_yml} \
            --channels {params.channel} 2>&1 | tee -a {log}
        if [ -n "{params.known_counts}" ] && [ "{params.known_counts}" != "none" ] && [ -f "{params.known_counts}" ]; then
            echo "Running closure comparison against {params.known_counts}" >> {log}
            {params.combine_cmd} {params.test_script} \
                --output_path {params.output_dir} \
                --inputFile {params.root_file} \
                --knownCounts {params.known_counts} 2>&1 | tee -a {log}
        fi
        touch {output.validation_txt}
        """
