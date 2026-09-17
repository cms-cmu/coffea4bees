#!/usr/bin/env python3
"""
===============================================================================
Unbinned Signal Region Event Extractor for SvB Rebinning
===============================================================================

Purpose:
  Extracts unbinned (ps, weight, nSelJets) arrays for ttH(bb) signal events in the
  Signal Region (SR) across all available chunks in parallel via XRootD/uproot.
  Saves the arrays to an .npz file used downstream by `rebin_SvB.py`.

Usage Instructions:
  1. Ensure network or XRootD access to the input picoAOD and SvB files.
  2. Run inside the coffea container:
     ./run_container python coffea4bees/analysis/tools/extract_unbinned_SR.py

  3. Output:
     - `output/ttHbb/unbinned_SR_ttHbb.npz` containing arrays: `ps`, `weights`, `nSelJets`.

Downstream:
  - Feed `output/ttHbb/unbinned_SR_ttHbb.npz` into `rebin_SvB.py` to calculate
    flat-yield quantile binning schemes.
===============================================================================
"""
import os
import json
import time
import glob
import uproot
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_svb_mapping(url_svb_json):
    """Read SvB result.json via XRootD or local file and map picoAOD path -> SvB chunk path."""
    import XRootD.client
    if url_svb_json.startswith("root://"):
        client = XRootD.client.File()
        client.open(url_svb_json, XRootD.client.flags.OpenFlags.READ)
        content = b""
        while True:
            st, d = client.read(len(content), 65536)
            if not d:
                break
            content += d
        client.close()
        svb_data = json.loads(content.decode("utf-8"))
    else:
        with open(url_svb_json) as f:
            svb_data = json.load(f)
            
    svb_map = {}
    for entry in svb_data["analysis"][0]["merged"]["data"]:
        pico_path = entry[0]["path"]
        svb_chunk = entry[1][0]["chunk"]["path"]
        svb_map[pico_path] = svb_chunk
    return svb_map

def process_chunk(pico_path, hcr_path, svb_path):
    """Extract (ps, weight, nSelJets) for events passing SR and fourTag."""
    try:
        with uproot.open(hcr_path) as f_hcr, uproot.open(svb_path) as f_svb:
            t_hcr = f_hcr["Events"]
            t_svb = f_svb["Events"]
            
            sr = t_hcr["SR"].array(library="np")
            four_tag = t_hcr["fourTag"].array(library="np")
            mask = sr & four_tag
            
            if np.sum(mask) == 0:
                return np.array([]), np.array([]), np.array([])
                
            w = t_hcr["weight"].array(library="np")[mask]
            n_sel = t_hcr["nSelJets"].array(library="np")[mask] if "nSelJets" in t_hcr else np.zeros(len(w), dtype=int)
            
            p_sig = t_svb["p_ttHbb"].array(library="np")[mask]
            p_mj = t_svb["p_multijet"].array(library="np")[mask]
            p_tt = t_svb["p_ttbar"].array(library="np")[mask]
            
            ps = p_sig / np.maximum(p_sig + p_mj + p_tt, 1e-10)
            return ps, w, n_sel
    except Exception as e:
        print(f"Error processing {pico_path}: {e}")
        return np.array([]), np.array([]), np.array([])

def extract_all_ttHbb(output_npz="output/ttHbb/unbinned_SR_ttHbb.npz", max_workers=16):
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    url_svb_json = "root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/ttHbb_v3/friend/SvB_ttHbb_v3/result.json"
    print("Loading SvB result.json mapping...")
    svb_map = load_svb_mapping(url_svb_json)
    print(f"Loaded {len(svb_map)} SvB chunk mappings.")
    
    # Find all classifier input JSON files for ttHbb
    cfg_files = sorted(glob.glob("output/ttHbb/classifier_inputs/classifier_inputs_dataset_ttHbb__*.json"))
    tasks = []
    for cfg_file in cfg_files:
        with open(cfg_file) as f:
            cfg = json.load(f)
        for entry in cfg["HCR_input"]["data"]:
            pico_path = entry[0]["path"]
            hcr_path = entry[1][0]["chunk"]["path"]
            svb_path = svb_map.get(pico_path)
            if svb_path:
                tasks.append((pico_path, hcr_path, svb_path))
                
    print(f"Submitting {len(tasks)} chunk extraction tasks using {max_workers} worker threads...")
    t0 = time.time()
    all_ps = []
    all_wt = []
    all_nsel = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_chunk, p, h, s): (p, h, s) for p, h, s in tasks}
        for fut in as_completed(futures):
            ps, w, n_sel = fut.result()
            if len(ps) > 0:
                all_ps.append(ps)
                all_wt.append(w)
                all_nsel.append(n_sel)
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                print(f"  Progress: {completed}/{len(tasks)} chunks completed ({time.time()-t0:.1f} s)...")
                
    all_ps = np.concatenate(all_ps)
    all_wt = np.concatenate(all_wt)
    all_nsel = np.concatenate(all_nsel)
    dt = time.time() - t0
    
    print(f"\nExtraction complete in {dt:.1f} s!")
    print(f"Total unbinned SR events: {len(all_ps):,}")
    print(f"Total signal weight in SR: {np.sum(all_wt):.2f}")
    
    np.savez_compressed(output_npz, ps=all_ps, weight=all_wt, nSelJets=all_nsel)
    print(f"Saved unbinned data to {output_npz} ({os.path.getsize(output_npz)/1024:.1f} KB)")

if __name__ == "__main__":
    extract_all_ttHbb()
