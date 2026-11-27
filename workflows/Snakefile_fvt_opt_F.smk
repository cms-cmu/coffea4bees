# ==============================================================================
# coffea4bees/workflows/Snakefile_fvt_opt_F.smk
#
# Phase OPT_F: Per-Variant F_1 Background Histogramming on cmslpc
# Consumes this roast's own OPT_C FvT friends (never fvt_upstream) and
# produces histAll_mixeddata_bkgs.coffea per variant run.
# ==============================================================================

import os
import yaml

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/ttHbb_fvt_opt_cat_year.yml"

include: "helpers/fvt_opt_common.smk"

mix_name = config.get("mix_name", f"{channel}_fvt_opt")
roast_id = config.get("roast_id") or config.get("label") or "fvt_opt"
_handoff_in = config.get("input_handoff_eos") or globals().get("INPUT_HANDOFF_EOS") or HANDOFF_EOS


def _detect_per_year_jcm():
    if "per_year_jcm" in config:
        return bool(config["per_year_jcm"])
    if (config.get("phaseB_1", {}) or {}).get("per_year_jcm") is not None:
        return bool(config["phaseB_1"]["per_year_jcm"])
    return True


per_year_jcm = _detect_per_year_jcm()


def _resolve_jcm(m, year=None):
    filename = f"jetCombinatoricModel_SB_mix_v{m}_{year}.yml" if year else f"jetCombinatoricModel_SB_mix_v{m}.yml"
    cand = os.path.join(out_b1, filename)
    return cand


def get_all_jcm_inputs(wildcards):
    if per_year_jcm:
        return [_resolve_jcm(m, y) for m in OPT_MIXES for y in YEARS]
    return [_resolve_jcm(m) for m in OPT_MIXES]


wildcard_constraints:
    run = "|".join(OPT_RUNS),
    m = r"\d+",


rule stage_fvt_opt_friend_manifest:
    output:
        f"{out_opt_c}friends/{{run}}/friends_FvT_{mix_name}_{{run}}_v{{m}}.json",
    params:
        src = f"{HANDOFF_EOS}/friends/{{run}}/friends_FvT_{mix_name}_{{run}}_v{{m}}.json",
    shell:
        """
        mkdir -p $(dirname {output})
        if command -v xrdcp &>/dev/null; then
            xrdcp -f '{params.src}' '{output}'
        else
            cp -f '{params.src}' '{output}'
        fi
        """


if not any(r.name in ("make_subsample_jcm_b1", "make_subsample_jcm_b1_per_year") for r in workflow.rules):
    rule stage_fvt_opt_jcm_per_year:
        output:
            f"{out_b1}jetCombinatoricModel_SB_mix_v{{m}}_{{year}}.yml",
        params:
            eos_src = f"{_handoff_in}/JCM/jetCombinatoricModel_SB_mix_v{{m}}_{{year}}.yml" if _handoff_in else "",
        shell:
            """
            mkdir -p $(dirname {output})
            if [ -n '{params.eos_src}' ]; then
                if command -v xrdcp &>/dev/null; then
                    xrdcp -f '{params.eos_src}' '{output}'
                else
                    cp -f '{params.eos_src}' '{output}'
                fi
            else
                echo "ERROR: JCM v{wildcards.m}_{wildcards.year} not found" >&2
                exit 1
            fi
            """


rule stage_fvt_opt_F_config:
    input:
        base_cfg = config.get("analysis_config_mixeddata_bkgs", "coffea4bees/workflows/config/analysis_config_mixeddata_bkgs.yml"),
        jcm = get_all_jcm_inputs,
        fvt = lambda wildcards: expand(f"{out_opt_c}friends/{wildcards.run}/friends_FvT_{mix_name}_{wildcards.run}_v{{m}}.json", m=OPT_MIXES),
    output:
        cfg = f"{out_opt_f}{{run}}/analysis_config_mixeddata_bkgs.yml",
    run:
        with open(input.base_cfg, "r") as f:
            data_cfg = yaml.safe_load(f) or {}

        user_analysis_cfg = (config.get("analysis_config") or {}).get("config", {})
        if isinstance(user_analysis_cfg, dict):
            data_cfg.setdefault("config", {}).update(user_analysis_cfg)

        cand_sel_cfg = config.get("candidates_selection_cfg") or user_analysis_cfg.get("candidates_selection_cfg")
        if cand_sel_cfg:
            data_cfg.setdefault("config", {})["candidates_selection_cfg"] = cand_sel_cfg

        phase_f_cfg = config.get("phase_f_1", config.get("bkg_syst_F_1", config.get("phase_f", {})))
        if isinstance(phase_f_cfg.get("config"), dict):
            data_cfg.setdefault("config", {}).update(phase_f_cfg["config"])

        data_cfg.setdefault("config", {})["blind"] = phase_f_cfg.get("config", {}).get("blind", user_analysis_cfg.get("blind", True))
        fvt_floor = phase_f_cfg.get("config", {}).get("FvT_pd3_floor", config.get("FvT_pd3_floor", user_analysis_cfg.get("FvT_pd3_floor", 0.01)))
        data_cfg.setdefault("config", {})["FvT_pd3_floor"] = float(fvt_floor)
        data_cfg.setdefault("config", {})["pairing"] = phase_f_cfg.get("config", {}).get("pairing", user_analysis_cfg.get("pairing", config.get("pairing", "nominal")))

        jcm_file = {}
        for m in OPT_MIXES:
            v_name = f"v{m}"
            if per_year_jcm:
                jcm_file[v_name] = {y: _resolve_jcm(m, y) for y in YEARS}
            else:
                jcm_file[v_name] = _resolve_jcm(m)

        friends_dict = {
            "trigWeight": config.get("trigweights_file", "coffea4bees/metadata/friends/trigweights_Run2_v2.json@@trigWeight"),
        }
        if "data_svb_friend" in config:
            friends_dict["SvB"] = config["data_svb_friend"]
        if "data_svb_ma_friend" in config:
            friends_dict["SvB_MA"] = config["data_svb_ma_friend"]
        elif "data_svb_friend" in config:
            friends_dict["SvB_MA"] = config["data_svb_friend"]

        for m in OPT_MIXES:
            friends_dict[f"FvT_v{m}"] = f"{out_opt_c}friends/{wildcards.run}/friends_FvT_{mix_name}_{wildcards.run}_v{m}.json@@FvT"
        friends_dict["FvT"] = friends_dict[f"FvT_v{OPT_MIXES[0]}"]

        data_cfg["config"]["JCM_file"] = jcm_file
        data_cfg["config"]["subsample_names"] = [f"v{m}" for m in OPT_MIXES]
        data_cfg["config"]["friends"] = friends_dict
        data_cfg["config"]["apply_JCM"] = True
        data_cfg["config"]["apply_trigWeight"] = True
        data_cfg["config"]["apply_FvT"] = True
        data_cfg["config"]["plot_ttbar_with_weights"] = True
        data_cfg["config"]["SvB_MA"] = True
        if "SvB" in friends_dict:
            data_cfg["config"]["SvB"] = True

        os.makedirs(os.path.dirname(output.cfg), exist_ok=True)
        with open(output.cfg, "w") as f:
            yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False)


rule run_fvt_opt_F_analysis:
    input:
        cfg = f"{out_opt_f}{{run}}/analysis_config_mixeddata_bkgs.yml",
        jcm = get_all_jcm_inputs,
        fvt = lambda wildcards: expand(f"{out_opt_c}friends/{wildcards.run}/friends_FvT_{mix_name}_{wildcards.run}_v{{m}}.json", m=OPT_MIXES),
    output:
        coffea = f"{out_opt_f}{{run}}/histAll_mixeddata_bkgs.coffea",
    log:
        f"{out_opt_f}logs/analysis_{{run}}.log",
    params:
        processor = config.get("analysis_processor") or f"coffea4bees/analysis/processors/processor_{channel}.py",
        dataset = config.get("f1_dataset") or "data",
        years = " ".join(YEARS),
        output_path = lambda wildcards: f"{out_opt_f}{wildcards.run}/",
        extra_arguments = condor_flags,
        container_wrapper = container_wrapper,
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.coffea}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} runner.py {input.cfg} \
            --processor {params.processor} \
            --datasets {params.dataset} \
            --years {params.years} \
            --output-path {params.output_path} \
            --output $(basename {output.coffea}) \
            {params.extra_arguments} 2>&1 | tee {log}
        """


rule validate_fvt_opt_F_normalization:
    input:
        coffea = f"{out_opt_f}{{run}}/histAll_mixeddata_bkgs.coffea",
    output:
        done = f"{out_opt_f}{{run}}/validation_normalization.done",
    log:
        f"{out_opt_f}logs/validate_normalization_{{run}}.log",
    params:
        container_wrapper = container_wrapper,
        python_bin = python_bin,
        bounds = lambda wildcards: " ".join(
            f"--{k.replace('_', '-')} {v}" for k, v in (config.get("f1_validation") or {}).items()
            if k in ("min_sb_yield", "max_sb_yield", "max_sr_yield", "min_ratio", "max_ratio")
        ),
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.done}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} coffea4bees/analysis/tools/validate_f1_normalization.py \
            -i {input.coffea} \
            -o {output.done} {params.bounds} 2>&1 | tee {log}
        """


localrules: all_fvt_opt_F, stage_fvt_opt_friend_manifest, stage_fvt_opt_jcm_per_year, stage_fvt_opt_F_config, validate_fvt_opt_F_normalization

rule all_fvt_opt_F:
    default_target: True
    input:
        expand(f"{out_opt_f}{{run}}/histAll_mixeddata_bkgs.coffea", run=OPT_RUNS),
        expand(f"{out_opt_f}{{run}}/validation_normalization.done", run=OPT_RUNS),
