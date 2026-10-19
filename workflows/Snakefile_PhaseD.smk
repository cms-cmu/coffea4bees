import os
import shutil
import yaml

include: "helpers/common.smk"

# Check if multi-classifier mode is enabled:
# e.g. both 'svb' and 'svb_ma' are configured, or 'classifiers' list is provided
svb_models = []
if 'svb' in config and isinstance(config['svb'], dict):
    svb_models.append(('svb', config['svb']))
if 'svb_ma' in config and isinstance(config['svb_ma'], dict):
    svb_models.append(('svb_ma', config['svb_ma']))
if 'classifiers' in config and isinstance(config['classifiers'], list):
    for entry in config['classifiers']:
        if isinstance(entry, dict) and 'name' in entry:
            svb_models.append((entry['name'], entry))

if len(svb_models) <= 1:
    svb_cfg = resolve_config_section(config, primary_key='svb', fallback_keys=['svb_classifier', 'classifier'])
    for k, v in svb_cfg.items():
        config[k] = v

    config.setdefault('eos_base', "root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/nominal_sel")
    config.setdefault('plot_base', "root://eosuser.cern.ch//eos/user/a/algomez/www/HH4b/Plots")
    config.setdefault('classifier_config_paths', "coffea4bees")
    config.setdefault('wfs_base', "coffea4bees/classifier/config/workflows/HH4b_2024_v2/SvB")
    config.setdefault('label', "SvB_nominal")
    config.setdefault('output_dir', f"output/{config['label']}/")
    config.setdefault('plot_inputs', True)
    config.setdefault('plot_weights', True)
    config.setdefault('evaluate', True)

    config.setdefault('model', f"{config['eos_base']}/classifier/{config['label']}")
    config.setdefault('friend', f"{config['eos_base']}/friend/{config['label']}")
    config.setdefault('train_template', f"model: {config['eos_base']}/classifier/{config['label']}, FvT: {config['eos_base']}/friend/FvT_nominal")
    config.setdefault('eval_template', f"model: {config['eos_base']}/classifier/{config['label']}, SvB: {config['eos_base']}/friend/{config['label']}")
    config.setdefault('metadata', "coffea4bees/metadata/datasets/classifier_inputs_nominal.json@@HCR_input")

    output_dir = config['output_dir'].rstrip('/')
    os.makedirs(output_dir, exist_ok=True)

    if 'classifier_setting' in config:
        common_path = f"{output_dir}/common.yml"
        with open(common_path, 'w') as f:
            yaml.dump({'setting': config['classifier_setting']}, f, default_flow_style=False)
        config['common'] = common_path

    if 'train_workflow' in config or 'eval_workflow' in config:
        wfs_dir = f"{output_dir}/wfs"
        os.makedirs(wfs_dir, exist_ok=True)
        if 'train_workflow' in config:
            with open(f"{wfs_dir}/train.yml", 'w') as f:
                yaml.dump(config['train_workflow'], f, default_flow_style=False)
        if 'eval_workflow' in config:
            with open(f"{wfs_dir}/evaluate.yml", 'w') as f:
                yaml.dump(config['eval_workflow'], f, default_flow_style=False)
        config['wfs_base'] = wfs_dir
    elif config.get('workflow_overrides') or config.get('workflow_inserts'):
        # Keep the checked-in templates, override options by flag (e.g. the roast's JCM / manifest)
        # and insert ones they lack (e.g. a wider --architecture)
        config['wfs_base'] = write_workflow_overrides(config['wfs_base'], config.get('workflow_overrides'),
                                                      f"{output_dir}/wfs", log=print,
                                                      inserts=config.get('workflow_inserts'))

    include: "../../src/classifier/workflow/Snakefile"

    all_single_inputs = list(rules.all.input)
    if config.get('compute_binning', False):
        all_single_inputs.append(f"{output_dir}/svb_quantiles.json")

    rule all_PhaseD:
        default_target: True
        input:
            all_single_inputs
else:
    # Multi-model mode: train & evaluate multiple SvB classifiers (e.g. SvB and SvB_MA)
    train_templates = config.setdefault("train_templates", {})
    eval_templates = config.setdefault("eval_templates", {})
    models_dict = config.setdefault("models", {})
    friends_dict = config.setdefault("friends_dict", {})

    eos_base_global = config.get('eos_base', "root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/nominal_sel")
    config.setdefault('classifier_config_paths', "coffea4bees")

    all_phase_d_targets = []

    for key, mcfg in svb_models:
        m_eos_base = mcfg.get('eos_base', eos_base_global)
        m_label = mcfg.get('label', key)
        m_output_dir = mcfg.get('output_dir', f"output/{m_label}/").rstrip('/')
        m_wfs_base = mcfg.get('wfs_base', "coffea4bees/classifier/config/workflows/HH4b_2024_v2/SvB")
        m_plot_inputs = mcfg.get('plot_inputs', False)
        m_plot_weights = mcfg.get('plot_weights', False)
        m_evaluate = mcfg.get('evaluate', True)

        m_model = mcfg.get('model', f"{m_eos_base}/classifier/{m_label}")
        if isinstance(m_model, str):
            m_model = m_model.format(eos_base=m_eos_base, label=m_label)
        m_friend = mcfg.get('friend', f"{m_eos_base}/friend/{m_label}")
        if isinstance(m_friend, str):
            m_friend = m_friend.format(eos_base=m_eos_base, label=m_label)
        m_train_template = mcfg.get('train_template', f"model: {m_model}, FvT: {m_eos_base}/friend/FvT_nominal")
        if isinstance(m_train_template, str):
            m_train_template = m_train_template.format(eos_base=m_eos_base, label=m_label, model=m_model, friend=m_friend)
        m_eval_template = mcfg.get('eval_template', f"model: {m_model}, SvB: {m_friend}")
        if isinstance(m_eval_template, str):
            m_eval_template = m_eval_template.format(eos_base=m_eos_base, label=m_label, model=m_model, friend=m_friend, SvB=m_friend)

        os.makedirs(m_output_dir, exist_ok=True)
        wfs_dir = f"{m_output_dir}/wfs"
        os.makedirs(wfs_dir, exist_ok=True)

        if 'classifier_setting' in config:
            common_path = f"{m_output_dir}/common.yml"
            with open(common_path, 'w') as f:
                yaml.dump({'setting': config['classifier_setting']}, f, default_flow_style=False)

        if 'train_workflow' in mcfg or 'eval_workflow' in mcfg:
            if 'train_workflow' in mcfg:
                with open(f"{wfs_dir}/train.yml", 'w') as f:
                    yaml.dump(mcfg['train_workflow'], f, default_flow_style=False)
            if 'eval_workflow' in mcfg:
                with open(f"{wfs_dir}/evaluate.yml", 'w') as f:
                    yaml.dump(mcfg['eval_workflow'], f, default_flow_style=False)
        else:
            overrides = mcfg.get('workflow_overrides', config.get('workflow_overrides'))
            inserts = mcfg.get('workflow_inserts', config.get('workflow_inserts'))
            modules = mcfg.get('workflow_modules', config.get('workflow_modules'))

            if overrides or inserts or modules:
                write_workflow_overrides(m_wfs_base, overrides, wfs_dir, log=print, inserts=inserts, modules=modules)
            else:
                if os.path.exists(f"{m_wfs_base}/train.yml"):
                    shutil.copyfile(f"{m_wfs_base}/train.yml", f"{wfs_dir}/train.yml")
                if os.path.exists(f"{m_wfs_base}/evaluate.yml"):
                    shutil.copyfile(f"{m_wfs_base}/evaluate.yml", f"{wfs_dir}/evaluate.yml")

        train_templates[m_output_dir] = m_train_template
        eval_templates[m_output_dir] = m_eval_template
        models_dict[m_output_dir] = m_model
        friends_dict[m_output_dir] = m_friend

        all_phase_d_targets.extend([f"{m_output_dir}/train.done", f"{m_output_dir}/yields.html"])
        if m_evaluate:
            all_phase_d_targets.extend([f"{m_output_dir}/evaluate.done", f"{m_output_dir}/analyze.done"])
            if mcfg.get('compute_binning', False):
                all_phase_d_targets.append(f"{m_output_dir}/svb_quantiles.json")
        if m_plot_inputs:
            all_phase_d_targets.extend([f"{m_output_dir}/plot_inputs_raw.done", f"{m_output_dir}/plot_inputs_dataprep.done"])
        if m_plot_weights:
            all_phase_d_targets.append(f"{m_output_dir}/plot_weights.done")

    # Clear top-level wfs_base and common so the generic Snakefile resolves {wildcards.model_dir}/wfs/{train,evaluate}.yml and common.yml
    config['wfs_base'] = ""
    config['common'] = ""
    first_cfg = svb_models[0][1]
    first_label = first_cfg.get('label', svb_models[0][0])
    config['output_dir'] = first_cfg.get('output_dir', f"output/{first_label}")
    config['label'] = first_label

    roast_id = config.get('roast_id', '')
    eos_prod = config.get('eos_prod', '')
    web_prod = config.get('web_prod', '')
    default_metadata = f"{eos_prod}/{roast_id}/handoff/classifier_inputs_{roast_id}.json@@HCR_input"
    default_plot_base = f"{web_prod}/{roast_id}/classifier"
    raw_metadata = first_cfg.get('metadata', default_metadata)
    raw_plot_base = first_cfg.get('plot_base', default_plot_base)
    config['metadata'] = raw_metadata.format(eos_prod=eos_prod, roast_id=roast_id, web_prod=web_prod) if isinstance(raw_metadata, str) else raw_metadata
    config['plot_base'] = raw_plot_base.format(eos_prod=eos_prod, roast_id=roast_id, web_prod=web_prod) if isinstance(raw_plot_base, str) else raw_plot_base

    include: "../../src/classifier/workflow/Snakefile"

    rule all_PhaseD:
        default_target: True
        input:
            all_phase_d_targets

rule compute_svb_binning:
    input:
        done = "{model_dir}/evaluate.done",
    output:
        json = "{model_dir}/svb_quantiles.json",
    log:
        "{model_dir}/svb_quantiles.log",
    params:
        classifier_inputs = lambda wildcards: str(config.get("metadata", "")).split("@@")[0],
        friend_result = lambda wildcards: f"{config.get('friends_dict', {}).get(wildcards.model_dir, config.get('friend', ''))}/result.json",
        sample_pattern = lambda wildcards: str(config.get("binning_sample_pattern", "ttHbb")),
        selection = lambda wildcards: str(config.get("binning_selection", "inclusive")),
        bins = lambda wildcards: int(config.get("binning_bins", 240)),
        min_ps = lambda wildcards: float(config.get("binning_min_ps", 0.01)),
        handoff_eos = lambda wildcards: str(config.get("handoff", {}).get("eos_base", "") if isinstance(config.get("handoff"), dict) else ""),
    shell:
        """
        mkdir -p $(dirname {output.json}) $(dirname {log})
        if [ -z "${{X509_USER_PROXY:-}}" ] && [ -f ./proxy/x509_proxy ]; then
            export X509_USER_PROXY="$PWD/proxy/x509_proxy"
        fi
        python3 coffea4bees/analysis/tools/rebin_SvB.py \
            --friends "{params.friend_result}" \
            --classifier-inputs "{params.classifier_inputs}" \
            --sample-pattern "{params.sample_pattern}" \
            --selection "{params.selection}" \
            --key $(basename {wildcards.model_dir}) \
            -n {params.bins} \
            --min-ps {params.min_ps} \
            --output-json {output.json} 2>&1 | tee {log}

        if [ -n "{params.handoff_eos}" ]; then
            xrdcp -f -p "{output.json}" "{params.handoff_eos}/$(basename {output.json})" 2>&1 | tee -a {log} || true
        fi
        """

localrules: compute_svb_binning

