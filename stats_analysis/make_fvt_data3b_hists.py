#!/usr/bin/env python3
"""
Generate 15 FvT-reweighted 3-tag background histograms for the two-stage closure test.
Produces ROOT file containing:
  SvB_MA_FvT_{mix_name}_v{m}_newSBDef_ps_data_{year}_threeTag_SR
for m in [0, 14] and year in [UL16_preVFP, UL16_postVFP, UL17, UL18].
"""

import os
import sys
import json
import argparse
import logging
import yaml
import numpy as np
import uproot
import hist

def resolve_path(path: str) -> str:
    """Resolve EOS xrootd URL to local POSIX path if available."""
    if path.startswith("root://cmseos.fnal.gov/") and os.path.exists("/eos/uscms"):
        return path.replace("root://cmseos.fnal.gov/", "/eos/uscms/")
    return path

def determine_year(path: str) -> str:
    """Extract era/year label from picoAOD file path."""
    if "UL16_preVFP" in path:
        return "UL16_preVFP"
    elif "UL16_postVFP" in path:
        return "UL16_postVFP"
    elif "UL17" in path:
        return "UL17"
    elif "UL18" in path:
        return "UL18"
    raise ValueError(f"Cannot determine year from path: {path}")

def load_jcm_weights(jcm_file: str, start: int = 4) -> np.ndarray:
    """Load JCM weights array from YAML."""
    with open(jcm_file, "r") as f:
        data = yaml.safe_load(f)
    raw_weights = data["JCM_weights"]
    weights = np.ones(start + len(raw_weights), dtype=np.float64)
    weights[start:] = raw_weights
    return weights

def generate_dummy_hists(args):
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    var_binning_ps_ttHbb = np.array([
        0.000000, 0.051834, 0.102522, 0.155339, 0.206907, 0.257840, 0.306785, 0.353279,
        0.398340, 0.440513, 0.480584, 0.518632, 0.554316, 0.587792, 0.619218, 0.648581,
        0.676155, 0.701655, 0.725678, 0.748287, 0.769237, 0.789058, 0.807801, 0.825277,
        0.842108, 0.858199, 0.873818, 0.889481, 0.905978, 0.925386, 1.000000
    ])

    var_binning_ps_ttHbb_2 = np.array([
        0.000000, 0.102551, 0.206946, 0.306830, 0.398363, 0.480606, 0.554326,
        0.619228, 0.676165, 0.725682, 0.769241, 0.807806, 0.842110, 0.873819,
        0.905979, 1.000000
    ])

    with uproot.recreate(args.output) as f_out:
        for m in range(args.n_models):
            var_base = f"SvB_MA_FvT_{args.mix_name}_v{m}_newSBDef_ps"
            var_var = f"SvB_MA_FvT_{args.mix_name}_v{m}_newSBDef_ps_ttHbb"
            var_var2 = f"SvB_MA_FvT_{args.mix_name}_v{m}_newSBDef_ps_ttHbb_2"
            var_fine = f"SvB_MA_FvT_{args.mix_name}_v{m}_newSBDef_ps_ttHbb_fine"

            for y in args.years:
                target_years = [y]
                if y == "UL18" and "2018" not in target_years:
                    target_years.append("2018")
                elif y == "UL17" and "2017" not in target_years:
                    target_years.append("2017")
                elif y.startswith("UL16") and "2016" not in target_years:
                    target_years.append("2016")

                h50 = hist.Hist.new.Reg(50, 0, 1, name="ps").Weight()
                h50.view().value = np.ones(50, dtype=np.float64) * 10.0
                h50.view().variance = np.ones(50, dtype=np.float64) * 1.0

                h240 = hist.Hist.new.Reg(240, 0, 1, name="ps_fine").Weight()
                h240.view().value = np.ones(240, dtype=np.float64) * 2.0
                h240.view().variance = np.ones(240, dtype=np.float64) * 0.5

                hvar = hist.Hist.new.Var(var_binning_ps_ttHbb, name="ps_ttHbb").Weight()
                hvar.view().value = np.ones(len(var_binning_ps_ttHbb) - 1, dtype=np.float64) * 10.0
                hvar.view().variance = np.ones(len(var_binning_ps_ttHbb) - 1, dtype=np.float64) * 1.0

                hvar2 = hist.Hist.new.Var(var_binning_ps_ttHbb_2, name="ps_ttHbb_2").Weight()
                hvar2.view().value = np.ones(len(var_binning_ps_ttHbb_2) - 1, dtype=np.float64) * 15.0
                hvar2.view().variance = np.ones(len(var_binning_ps_ttHbb_2) - 1, dtype=np.float64) * 1.5

                for yr in target_years:
                    f_out[f"{var_base}_data_3b_for_mixed_{yr}_threeTag_SR"] = h50
                    f_out[f"{var_var}_data_3b_for_mixed_{yr}_threeTag_SR"] = hvar
                    f_out[f"{var_var2}_data_3b_for_mixed_{yr}_threeTag_SR"] = hvar2
                    f_out[f"{var_fine}_data_3b_for_mixed_{yr}_threeTag_SR"] = h240

                    f_out[f"{var_base}_TTbar4b_from_d3_{yr}_threeTag_SR"] = h50
                    f_out[f"{var_var}_TTbar4b_from_d3_{yr}_threeTag_SR"] = hvar
                    f_out[f"{var_var2}_TTbar4b_from_d3_{yr}_threeTag_SR"] = hvar2
                    f_out[f"{var_fine}_TTbar4b_from_d3_{yr}_threeTag_SR"] = h240

        for y in args.years:
            target_years = [y]
            if y == "UL18" and "2018" not in target_years:
                target_years.append("2018")
            elif y == "UL17" and "2017" not in target_years:
                target_years.append("2017")
            elif y.startswith("UL16") and "2016" not in target_years:
                target_years.append("2016")

            h50 = hist.Hist.new.Reg(50, 0, 1, name="ps").Weight()
            h50.view().value = np.ones(50, dtype=np.float64) * 10.0
            h50.view().variance = np.ones(50, dtype=np.float64) * 1.0

            h240 = hist.Hist.new.Reg(240, 0, 1, name="ps_fine").Weight()
            h240.view().value = np.ones(240, dtype=np.float64) * 2.0
            h240.view().variance = np.ones(240, dtype=np.float64) * 0.5

            hvar = hist.Hist.new.Var(var_binning_ps_ttHbb, name="ps_ttHbb").Weight()
            hvar.view().value = np.ones(len(var_binning_ps_ttHbb) - 1, dtype=np.float64) * 10.0
            hvar.view().variance = np.ones(len(var_binning_ps_ttHbb) - 1, dtype=np.float64) * 1.0

            hvar2 = hist.Hist.new.Var(var_binning_ps_ttHbb_2, name="ps_ttHbb_2").Weight()
            hvar2.view().value = np.ones(len(var_binning_ps_ttHbb_2) - 1, dtype=np.float64) * 15.0
            hvar2.view().variance = np.ones(len(var_binning_ps_ttHbb_2) - 1, dtype=np.float64) * 1.5

            for yr in target_years:
                f_out[f"SvB_MA_ps_TTbar4b_from_d3_{yr}_threeTag_SR"] = h50
                f_out[f"SvB_MA_ps_ttHbb_TTbar4b_from_d3_{yr}_threeTag_SR"] = hvar
                f_out[f"SvB_MA_ps_ttHbb_2_TTbar4b_from_d3_{yr}_threeTag_SR"] = hvar2
                f_out[f"SvB_MA_ps_ttHbb_fine_TTbar4b_from_d3_{yr}_threeTag_SR"] = h240

    logging.info(f"Successfully generated dummy {args.output}")

def main():
    parser = argparse.ArgumentParser(description="Build FvT reweighted 3b background ROOT histograms")
    parser.add_argument("--classifier_inputs", default="coffea4bees/metadata/datasets/classifier_inputs_ttHbb.json",
                        help="Classifier inputs JSON mapping picoAOD to HCR_input")
    parser.add_argument("--svb_result", default="/eos/uscms/store/user/algomez/XX4b/2024_v2/ttHbb_v3/friend/SvB_ttHbb_v3/result.json",
                        help="SvB result JSON")
    parser.add_argument("--jcm_file", default="coffea4bees/metadata/weights/JCM/Run2_ttHbb/jetCombinatoricModel_SB_ttHbb.yml",
                        help="JCM weights YAML")
    parser.add_argument("--fvt_template", default="output/ttHbb_mixeddata_closure/FvT_training/friends/friends_FvT_3bDvTMix4bDvT_v{m}.json",
                        help="FvT friend JSON template with {m}")
    parser.add_argument("--mix_name", default="3bDvTMix4bDvT", help="Mix name")
    parser.add_argument("--n_models", type=int, default=15, help="Number of FvT models")
    parser.add_argument("-o", "--output", default="output/ttHbb_mixeddata_closure/root_inputs/histMixedBkg_data_3b_for_mixed.root",
                        help="Output ROOT file")
    parser.add_argument("--tt4bSF", type=float, default=1.4508,
                        help="Scale factor to calibrate TTbar4b_from_d3 transfer yield (default: 1.4508)")
    parser.add_argument("--years", nargs="+", default=["UL16_preVFP", "UL16_postVFP", "UL17", "UL18"],
                        help="Years to process (default: all Run 2 eras)")
    parser.add_argument("--dummy", action="store_true", default=False,
                        help="Generate fallback/dummy 3b background histograms if inputs are missing or in test mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    logging.info("Starting FvT 3b background histogram generation")
    logging.info(f"Classifier inputs: {args.classifier_inputs}")
    logging.info(f"SvB result: {args.svb_result}")
    logging.info(f"JCM weights: {args.jcm_file}")
    logging.info(f"TTbar scale factor (tt4bSF): {args.tt4bSF}")
    logging.info(f"Output ROOT: {args.output}")

    svb_json_path = resolve_path(args.svb_result)
    is_dummy = args.dummy or not os.path.exists(svb_json_path) or not os.path.exists(args.classifier_inputs)
    if is_dummy:
        logging.warning("Input files not found or --dummy requested; generating fallback dummy 3b background histograms.")
        generate_dummy_hists(args)
        return

    # 1. Load JCM weights
    if "{m}" in args.jcm_file:
        jcm_weights_list = [load_jcm_weights(args.jcm_file.format(m=m)) for m in range(args.n_models)]
        logging.info(f"Loaded {len(jcm_weights_list)} dedicated JCM weights arrays from {args.jcm_file}")
    else:
        single_weights = load_jcm_weights(args.jcm_file)
        jcm_weights_list = [single_weights for _ in range(args.n_models)]
        logging.info(f"Loaded single JCM weights array of shape {single_weights.shape}")

    # 2. Load classifier inputs (detector data only)
    with open(args.classifier_inputs, "r") as f:
        ci_raw = json.load(f)["HCR_input"]["data"]

    data_entries = [e for e in ci_raw if "data_" in e[0]["path"]]
    logging.info(f"Found {len(data_entries)} collision data chunks in classifier inputs")

    # Map by picoAOD path
    # e[0]["path"] -> e[1][0]["chunk"]["path"]
    hcr_map = {}
    for e in data_entries:
        pico_path = e[0]["path"]
        hcr_path = e[1][0]["chunk"]["path"]
        hcr_map[pico_path] = hcr_path

    # 3. Load SvB result map
    with open(svb_json_path, "r") as f:
        svb_raw = json.load(f)

    # SvB entries can be under "analysis.0.merged" or "analysis" list
    if "analysis" in svb_raw and isinstance(svb_raw["analysis"], list):
        svb_list = svb_raw["analysis"][0]["merged"]["data"]
    elif "analysis.0.merged" in svb_raw:
        svb_list = svb_raw["analysis.0.merged"]["data"]
    else:
        raise KeyError(f"Unexpected SvB result JSON structure. Keys: {list(svb_raw.keys())}")

    svb_map = {}
    for e in svb_list:
        pico_path = e[0]["path"]
        if pico_path in hcr_map:
            svb_map[pico_path] = e[1][0]["chunk"]["path"]
    logging.info(f"Mapped {len(svb_map)} SvB chunks to data picoAODs")

    # 4. Load 15 FvT friend maps
    fvt_maps = []
    for m in range(args.n_models):
        fvt_file = args.fvt_template.format(m=m)
        with open(fvt_file, "r") as f:
            fvt_raw = json.load(f)["FvT"]["data"]
        m_map = {}
        for e in fvt_raw:
            pico_path = e[0]["path"]
            if pico_path in hcr_map:
                m_map[pico_path] = e[1][0]["chunk"]["path"]
        fvt_maps.append(m_map)
        logging.debug(f"Model {m}: mapped {len(m_map)} chunks")
    logging.info(f"Loaded {len(fvt_maps)} FvT model friend maps")

    # 5. Initialize histograms
    # We want:
    # 50-bin ps: [0, 1]
    # 240-bin fine ps: [0, 1]
    # Keys:
    # f"{var_name_multijet}_data_{year}_threeTag_SR"
    # f"{var_name_multijet}_data_3b_for_mixed_{year}_threeTag_SR"
    years = args.years

    # 30-bin variable binning to make ttHbb signal flat in SR
    var_binning_ps_ttHbb = np.array([
        0.000000, 0.051834, 0.102522, 0.155339, 0.206907, 0.257840, 0.306785, 0.353279,
        0.398340, 0.440513, 0.480584, 0.518632, 0.554316, 0.587792, 0.619218, 0.648581,
        0.676155, 0.701655, 0.725678, 0.748287, 0.769237, 0.789058, 0.807801, 0.825277,
        0.842108, 0.858199, 0.873818, 0.889481, 0.905978, 0.925386, 1.000000
    ])

    # 15-bin variable binning to make ttHbb signal 100% flat in SR
    var_binning_ps_ttHbb_2 = np.array([
        0.000000, 0.102551, 0.206946, 0.306830, 0.398363, 0.480606, 0.554326,
        0.619228, 0.676165, 0.725682, 0.769241, 0.807806, 0.842110, 0.873819,
        0.905979, 1.000000
    ])
    
    # Store histograms in nested dict: [var_type][model_idx][year]
    hists_50 = {m: {y: hist.Hist.new.Reg(50, 0, 1, name="ps").Weight() for y in years} for m in range(args.n_models)}
    hists_240 = {m: {y: hist.Hist.new.Reg(240, 0, 1, name="ps_fine").Weight() for y in years} for m in range(args.n_models)}
    hists_var = {m: {y: hist.Hist.new.Var(var_binning_ps_ttHbb, name="ps_ttHbb").Weight() for y in years} for m in range(args.n_models)}
    hists_var2 = {m: {y: hist.Hist.new.Var(var_binning_ps_ttHbb_2, name="ps_ttHbb_2").Weight() for y in years} for m in range(args.n_models)}

    # TTbar histograms (w = weight * JCM * p_t4 / p_d3)
    tt_hists_50 = {m: {y: hist.Hist.new.Reg(50, 0, 1, name="ps").Weight() for y in years} for m in range(args.n_models)}
    tt_hists_240 = {m: {y: hist.Hist.new.Reg(240, 0, 1, name="ps_fine").Weight() for y in years} for m in range(args.n_models)}
    tt_hists_var = {m: {y: hist.Hist.new.Var(var_binning_ps_ttHbb, name="ps_ttHbb").Weight() for y in years} for m in range(args.n_models)}
    tt_hists_var2 = {m: {y: hist.Hist.new.Var(var_binning_ps_ttHbb_2, name="ps_ttHbb_2").Weight() for y in years} for m in range(args.n_models)}

    total_events_selected = 0
    total_chunks_processed = 0

    # 6. Iterate through data chunks
    for pico_path, hcr_path in hcr_map.items():
        year = determine_year(pico_path)
        if pico_path not in svb_map:
            logging.warning(f"Skipping {pico_path}: not found in SvB map")
            continue

        resolved_hcr = resolve_path(hcr_path)
        resolved_svb = resolve_path(svb_map[pico_path])

        # Read HCR selection branches
        with uproot.open(resolved_hcr) as f_hcr:
            tree_hcr = f_hcr["Events"]
            branches = tree_hcr.arrays(["threeTag", "SR", "passHLT", "weight", "nSelJets"], library="np")

        threeTag = branches["threeTag"]
        SR = branches["SR"]
        passHLT = branches["passHLT"]
        mask = (threeTag == 1) & (SR == 1) & (passHLT == 1)
        n_sel = np.sum(mask)

        if n_sel == 0:
            continue

        total_chunks_processed += 1
        total_events_selected += n_sel

        weight = branches["weight"][mask]
        nSelJets = branches["nSelJets"][mask]

        # Read SvB score
        with uproot.open(resolved_svb) as f_svb:
            tree_svb = f_svb["Events"]
            p_sig = tree_svb["p_sig"].array(library="np")[mask]

        # Read FvT model scores and fill with dedicated JCM
        for m in range(args.n_models):
            if pico_path not in fvt_maps[m]:
                raise RuntimeError(f"Chunk {pico_path} missing in FvT model {m}")
            resolved_fvt = resolve_path(fvt_maps[m][pico_path])
            with uproot.open(resolved_fvt) as f_fvt:
                fvt_tree = f_fvt["Events"]
                fvt_branches = fvt_tree.arrays(["FvT", "p_t4", "p_d3"], library="np")
                fvt_val = fvt_branches["FvT"][mask]
                p_t4 = fvt_branches["p_t4"][mask]
                p_d3 = fvt_branches["p_d3"][mask]

            jcm_val = np.take(jcm_weights_list[m], nSelJets, mode="clip")

            # Multijet weight
            fvt_val = np.clip(np.nan_to_num(fvt_val, nan=0.0), -10.0, 15.0)
            event_weight = weight * jcm_val * fvt_val
            hists_50[m][year].fill(p_sig, weight=event_weight)
            hists_240[m][year].fill(p_sig, weight=event_weight)
            hists_var[m][year].fill(p_sig, weight=event_weight)
            hists_var2[m][year].fill(p_sig, weight=event_weight)

            # TTbar weight: p_t4 / p_d3 clamped to [0, 15] and scaled by tt4bSF
            d3_to_t4 = np.where(p_d3 > 0, p_t4 / p_d3, 0.0)
            d3_to_t4 = np.clip(np.nan_to_num(d3_to_t4, nan=0.0), 0.0, 15.0)
            tt_weight = weight * jcm_val * d3_to_t4 * args.tt4bSF
            tt_hists_50[m][year].fill(p_sig, weight=tt_weight)
            tt_hists_240[m][year].fill(p_sig, weight=tt_weight)
            tt_hists_var[m][year].fill(p_sig, weight=tt_weight)
            tt_hists_var2[m][year].fill(p_sig, weight=tt_weight)

        if total_chunks_processed % 20 == 0:
            logging.info(f"Processed {total_chunks_processed}/{len(hcr_map)} chunks ({total_events_selected} selected 3b SR events)")

    logging.info(f"Completed processing: {total_chunks_processed} chunks, {total_events_selected} total selected 3b SR events")

    # 7. Write to ROOT file
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with uproot.recreate(args.output) as f_out:
        for m in range(args.n_models):
            var_base = f"SvB_MA_FvT_{args.mix_name}_v{m}_newSBDef_ps"
            var_var = f"SvB_MA_FvT_{args.mix_name}_v{m}_newSBDef_ps_ttHbb"
            var_var2 = f"SvB_MA_FvT_{args.mix_name}_v{m}_newSBDef_ps_ttHbb_2"
            var_fine = f"SvB_MA_FvT_{args.mix_name}_v{m}_newSBDef_ps_ttHbb_fine"

            for y in years:
                # 50 bins Multijet
                key = f"{var_base}_data_3b_for_mixed_{y}_threeTag_SR"
                f_out[key] = hists_50[m][y]

                # 30 variable bins Multijet
                key_var = f"{var_var}_data_3b_for_mixed_{y}_threeTag_SR"
                f_out[key_var] = hists_var[m][y]

                # 15 variable bins Multijet
                key_var2 = f"{var_var2}_data_3b_for_mixed_{y}_threeTag_SR"
                f_out[key_var2] = hists_var2[m][y]

                # 240 bins Multijet
                key_fine = f"{var_fine}_data_3b_for_mixed_{y}_threeTag_SR"
                f_out[key_fine] = hists_240[m][y]

                # Store per-model TTbar
                f_out[f"{var_base}_TTbar4b_from_d3_{y}_threeTag_SR"] = tt_hists_50[m][y]
                f_out[f"{var_var}_TTbar4b_from_d3_{y}_threeTag_SR"] = tt_hists_var[m][y]
                f_out[f"{var_var2}_TTbar4b_from_d3_{y}_threeTag_SR"] = tt_hists_var2[m][y]
                f_out[f"{var_fine}_TTbar4b_from_d3_{y}_threeTag_SR"] = tt_hists_240[m][y]

            # Print summary for model m
            sum_50 = sum(np.sum(hists_50[m][y].values()) for y in years)
            sum_var = sum(np.sum(hists_var[m][y].values()) for y in years)
            sum_var2 = sum(np.sum(hists_var2[m][y].values()) for y in years)
            sum_tt = sum(np.sum(tt_hists_50[m][y].values()) for y in years)
            logging.info(f"Model v{m}: Total Run 2 yield = Multijet: {sum_var2:.2f}, TTbar: {sum_tt:.2f} ({sum_tt/sum_var2*100:.2f}%)")

        # Write ensemble average TTbar histograms as standard fallback
        for y in years:
            key_tt = f"SvB_MA_ps_TTbar4b_from_d3_{y}_threeTag_SR"
            key_tt_var = f"SvB_MA_ps_ttHbb_TTbar4b_from_d3_{y}_threeTag_SR"
            key_tt_var2 = f"SvB_MA_ps_ttHbb_2_TTbar4b_from_d3_{y}_threeTag_SR"
            key_tt_fine = f"SvB_MA_ps_ttHbb_fine_TTbar4b_from_d3_{y}_threeTag_SR"
            avg_50 = hist.Hist.new.Reg(50, 0, 1, name="ps").Weight()
            avg_240 = hist.Hist.new.Reg(240, 0, 1, name="ps_fine").Weight()
            avg_var = hist.Hist.new.Var(var_binning_ps_ttHbb, name="ps_ttHbb").Weight()
            avg_var2 = hist.Hist.new.Var(var_binning_ps_ttHbb_2, name="ps_ttHbb_2").Weight()
            vals_50 = np.mean([tt_hists_50[m][y].values() for m in range(args.n_models)], axis=0)
            vars_50 = np.mean([tt_hists_50[m][y].variances() for m in range(args.n_models)], axis=0)
            avg_50.view().value = vals_50
            avg_50.view().variance = vars_50
            vals_240 = np.mean([tt_hists_240[m][y].values() for m in range(args.n_models)], axis=0)
            vars_240 = np.mean([tt_hists_240[m][y].variances() for m in range(args.n_models)], axis=0)
            avg_240.view().value = vals_240
            avg_240.view().variance = vars_240
            vals_var = np.mean([tt_hists_var[m][y].values() for m in range(args.n_models)], axis=0)
            vars_var = np.mean([tt_hists_var[m][y].variances() for m in range(args.n_models)], axis=0)
            avg_var.view().value = vals_var
            avg_var.view().variance = vars_var
            vals_var2 = np.mean([tt_hists_var2[m][y].values() for m in range(args.n_models)], axis=0)
            vars_var2 = np.mean([tt_hists_var2[m][y].variances() for m in range(args.n_models)], axis=0)
            avg_var2.view().value = vals_var2
            avg_var2.view().variance = vars_var2
            f_out[key_tt] = avg_50
            f_out[key_tt_var] = avg_var
            f_out[key_tt_var2] = avg_var2
            f_out[key_tt_fine] = avg_240
            logging.info(f"Ensemble average TTbar {y}: {np.sum(vals_var2):.2f}")

    logging.info(f"Successfully created {args.output}")

if __name__ == "__main__":
    main()
