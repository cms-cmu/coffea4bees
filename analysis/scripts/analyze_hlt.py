#!/usr/bin/env python
import argparse
import numpy as np
import yaml
import re
from coffea.util import load


def is_physics_trigger(name):
    name_lower = name.lower()
    
    # 1. Exclude calibration / zero-bias / helper triggers
    exclude_keywords = [
        'zerobias', 'physics', 'alignment', 'calibration', 'random', 'beamspot', 
        'l1notbptx', 'unpairedbunch', 'nzs', 'phisym', 'isolatedbunch', 'l2cosmic', 
        'part', 'ecal', 'hcal', 'l1_', 'firstcollision', 'isolatedbunches', 'part0',
        'part1', 'part2', 'part3', 'part4', 'part5', 'part6', 'part7'
    ]
    for kw in exclude_keywords:
        if kw in name_lower:
            return False
            
    # 2. Filter to include only standard analysis triggers:
    include_patterns = [
        'pfht', 'quadpfjet', 'quadjet', 'sixpfjet', 'sixjet', 'doublepfjet', 'doublejet',
        'btag', 'csv', 'pnet', 'particlenet', 'deepcsv', 'deepjet',
        'isomu', 'mu50', 'ele', 'wptight', 'pfmet', 'pfmht', 'ak8pfjet'
    ]
    
    has_pattern = False
    for pat in include_patterns:
        if pat in name_lower:
            has_pattern = True
            break
    if not has_pattern:
        return False

    # 3. Exclude low-threshold jet/HT/MET/lepton triggers that are heavily prescaled:
    # Exclude single jet triggers with pT < 300
    jet_match = re.search(r'(?:ak\d+|pf|calo)?jet(?:ave)?(?:fwd)?(\d+)', name_lower)
    if jet_match:
        pt = int(jet_match.group(1))
        is_multijet = any(k in name_lower for k in ['quad', 'six', 'double', 'triple', 'di', 'mjj'])
        if not is_multijet and pt < 300:
            return False

    # Exclude single muon triggers with pT < 24
    mu_match = re.search(r'mu(\d+)', name_lower)
    if mu_match:
        pt = int(mu_match.group(1))
        if pt < 24 and 'double' not in name_lower and 'triple' not in name_lower and 'di' not in name_lower:
            return False

    # Exclude single electron triggers with pT < 28
    ele_match = re.search(r'ele(\d+)', name_lower)
    if ele_match:
        pt = int(ele_match.group(1))
        if pt < 28 and 'double' not in name_lower and 'di' not in name_lower:
            return False

    # Exclude HT triggers with HT < 250
    ht_match = re.search(r'pfht(\d+)', name_lower)
    if ht_match:
        ht = int(ht_match.group(1))
        if ht < 250:
            return False

    return True


def is_unprescaled(name):
    name_lower = name.lower()
    if "pfht330pt30_quadpfjet_75_60_45_40_triplepfbtagdeepcsv_4p5" in name_lower:
        return True
    if "doublepfbtag" in name_lower and "vbf" in name_lower:
        return True
    if "pfht1050" in name_lower:
        return True
    if "pfjet500" in name_lower or "ak8pfjet500" in name_lower:
        return True
    if "ak8pfjet" in name_lower and "trimmass30" in name_lower:
        pt_match = re.search(r'ak8pfjet(\d+)', name_lower)
        if pt_match:
            pt = int(pt_match.group(1))
            if pt >= 330:
                return True
    if "ak8pfht" in name_lower and "trimmass50" in name_lower:
        return True
    if "isomu24" in name_lower or "isomu27" in name_lower or "mu50" in name_lower:
        return True
    if "ele32_wptight" in name_lower or "ele35_wptight" in name_lower:
        return True
    if "pfmet120_pfmht120" in name_lower or "pfmet140_pfmht140" in name_lower:
        return True
    return False


def print_pairwise_addition_matrix(top_10, n_triggered, title):
    print(f"\n### {title}: [Overlap % | Absolute Gain %]")
    print("How to read: If you already have the column trigger, and you add the row trigger:")
    print("  - The first number is the % of the row trigger's events that are already covered by the column trigger.")
    print("  - The second number is the absolute percentage of the triggered pool you GAIN by adding the row trigger.")
    print()
    
    header = f"{'Trigger':<3} | " + " | ".join(f"  T{i:<2}   " for i in range(1, len(top_10) + 1))
    print(header)
    print("-" * (6 + len(top_10) * 10))
    
    for i, (name_i, n_abs_i, dec_i) in enumerate(top_10, 1):
        row_vals = []
        for j, (name_j, _, dec_j) in enumerate(top_10, 1):
            if i == j:
                row_vals.append("  diag  ")
                continue
            # Overlap: fraction of T_i that passes T_j
            n_both = np.sum(dec_i & dec_j)
            overlap_pct = (n_both / n_abs_i) * 100
            
            # Gain: fraction of pool gained by adding T_i on top of T_j
            n_new = np.sum(dec_i & ~dec_j)
            gain_pct = (n_new / n_triggered) * 100
            
            row_vals.append(f"{overlap_pct:3.0f}%|+{gain_pct:3.1f}%")
        print(f"T{i:<2}  | " + " | ".join(row_vals))

    print("\n### Trigger Key Legend:")
    for rank, (name, n_abs, _) in enumerate(top_10, 1):
        print(f"T{rank:<2} [Abs. Share: {n_abs/n_triggered*100:4.1f}%]: {name}")


def analyze_region(dataset, accum, num_events, mask, region_name):
    # Gather active unprescaled triggers on masked events
    all_unprescaled_decs = {}
    for k in accum:
        if k.startswith("HLT_") and is_physics_trigger(k) and is_unprescaled(k):
            dec = accum[k].value & mask
            if np.sum(dec) > 0:
                all_unprescaled_decs[k] = dec

    if not all_unprescaled_decs:
        print(f"\nNo active unprescaled triggers found in {region_name} region.")
        return

    # Total triggered pool within the selection mask
    all_or = np.zeros(len(mask), dtype=bool)
    for dec in all_unprescaled_decs.values():
        all_or = all_or | dec
    n_triggered = np.sum(all_or)

    # Sort triggers by absolute efficiency in this region
    top_triggers = []
    for name, dec in all_unprescaled_decs.items():
        top_triggers.append((name, np.sum(dec), dec))
    top_triggers.sort(key=lambda x: x[1], reverse=True)
    top_10 = top_triggers[:10]

    n_region_total = np.sum(mask)
    print(f"\n========================================================")
    print(f"Region: {region_name} (Events: {n_region_total}/{num_events} = {n_region_total/num_events:.4f})")
    print(f"Triggered region events (OR pool): {n_triggered} ({n_triggered/n_region_total*100:5.2f}%)")
    print(f"========================================================")

    # Shares table
    print(f"\n### Top Triggers: Shares of Triggered Pool in {region_name}")
    print(f"{'Rank':<5} | {'Trigger Path':<70} | {'Abs. Share':<12} | {'Excl. Share':<12}")
    print("-" * 105)
    for rank, (name, n_abs, _) in enumerate(top_10, 1):
        other_dec = np.zeros(len(mask), dtype=bool)
        for other_name, other_val in all_unprescaled_decs.items():
            if other_name != name:
                other_dec = other_dec | other_val
        n_excl = np.sum(all_unprescaled_decs[name] & ~other_dec)
        print(f"{rank:<5} | {name:<70} | {n_abs/n_triggered:10.4f}   | {n_excl/n_triggered:10.4f}")

    # Overlaps & gains matrix
    print_pairwise_addition_matrix(top_10, n_triggered, f"Pairwise Trigger Addition ({region_name})")

    # Overlap Breakdown
    matrix = np.column_stack(list(all_unprescaled_decs.values()))
    n_triggers_per_event = np.sum(matrix, axis=1)
    triggered_events = n_triggers_per_event[n_triggers_per_event > 0]
    
    print(f"\n### Overlap Breakdown in {region_name}:")
    for i in range(1, 6):
        count = np.sum(triggered_events == i)
        print(f"Events passing EXACTLY {i} triggers: {count:<8} ({count/n_triggered*100:6.2f}%)")
    count_more = np.sum(triggered_events > 5)
    print(f"Events passing MORE THAN 5 triggers: {count_more:<8} ({count_more/n_triggered*100:6.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Consolidated HLT trigger study script")
    parser.add_argument("-i", "--input", required=True, help="Input .coffea file path")
    parser.add_argument("-m", "--mode", choices=["all", "unselected", "fourtag"], default="all",
                        help="Analysis mode: 'all' (runs both unselected and fourTag), 'unselected', or 'fourtag'")
    args = parser.parse_args()

    data = load(args.input)

    for dataset, accum in data.items():
        if not isinstance(accum, dict) or 'numEvents' not in accum:
            continue
        
        num_events = accum['numEvents']
        print(f"\n========================================================")
        print(f"Dataset: {dataset}")
        print(f"Total events in sample: {num_events}")
        print(f"========================================================")

        # 1. Unselected / All Events study
        if args.mode in ["all", "unselected"]:
            all_mask = np.ones(num_events, dtype=bool)
            analyze_region(dataset, accum, num_events, all_mask, "All Events (Unselected)")

        # 2. Offline Resolved / fourTag study
        if args.mode in ["all", "fourtag"]:
            if 'fourTag' in accum:
                fourTag_mask = accum['fourTag'].value
                analyze_region(dataset, accum, num_events, fourTag_mask, "Offline Resolved (fourTag)")
            else:
                print("\n'fourTag' selection mask not found in the input coffea file. Skipping fourTag study.")


if __name__ == "__main__":
    main()
