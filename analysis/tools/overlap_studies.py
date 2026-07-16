#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try to load coffea.util
try:
    from coffea.util import load
except ImportError:
    # Fallback using standard pickle if coffea is not available in current environment
    import gzip
    import pickle
    def load(filename):
        with gzip.open(filename, "rb") as f:
            return pickle.load(f)

def generate_overlap_report(input_file, output_dir=None):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)

    if output_dir is None:
        output_dir = os.path.dirname(input_file) or "."
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from {input_file}...")
    try:
        results = load(input_file)
    except Exception as e:
        print(f"Failed to load file: {e}")
        sys.exit(1)

    # The file could contain multiple datasets
    for dataset, stats in results.items():
        if not isinstance(stats, dict):
            continue
        print(f"\nProcessing dataset: {dataset}")
        
        # Get base quantities
        num_events = stats.get('numEvents', 0)
        pass_base = stats.get('passBase', 0)
        any_selection = stats.get('anySelection', 0)
        none_all = stats.get('none_all', 0)
        
        # Define combinations
        exclusive_cats = {
            'onlyResolved': 'Only Resolved',
            'onlyLowPt': 'Only Low-pT',
            'onlyBoosted': 'Only Boosted',
            'onlySemiResolved': 'Only Semi-Resolved'
        }
        
        pairwise_cats = {
            'resolved_and_boosted': 'Resolved & Boosted',
            'resolved_and_semiresolved': 'Resolved & Semi-Resolved',
            'resolved_and_lowpt': 'Resolved & Low-pT',
            'boosted_and_semiresolved': 'Boosted & Semi-Resolved',
            'boosted_and_lowpt': 'Boosted & Low-pT',
            'semiresolved_and_lowpt': 'Semi-Resolved & Low-pT'
        }
        
        threeway_cats = {
            'resolved_boosted_semiresolved': 'Resolved & Boosted & Semi-Resolved',
            'resolved_boosted_lowpt': 'Resolved & Boosted & Low-pT',
            'resolved_semiresolved_lowpt': 'Resolved & Semi-Resolved & Low-pT',
            'boosted_semiresolved_lowpt': 'Boosted & Semi-Resolved & Low-pT'
        }
        
        fourway_cats = {
            'all_four': 'All Four Selections'
        }

        # Build table entries
        table_lines = []
        table_lines.append(f"# Overlap Study Report for {dataset}")
        table_lines.append(f"\n- **Total Input Events:** {num_events:,}")
        table_lines.append(f"- **Passed Baseline (Lumimask + Noise Filter):** {pass_base:,}")
        table_lines.append(f"- **Passed Any Selection:** {any_selection:,} ({any_selection/pass_base*100:.2f}% of Baseline)" if pass_base else "- **Passed Any Selection:** 0")
        table_lines.append(f"- **Failed All Selections:** {none_all:,} ({none_all/pass_base*100:.2f}% of Baseline)" if pass_base else "- **Failed All Selections:** 0")
        table_lines.append("\n## Event Selection Breakdown")
        table_lines.append("\n| Selection Category | Events | % of Baseline | % of Selected |")
        table_lines.append("| :--- | :---: | :---: | :---: |")

        def add_row(key, label):
            val = stats.get(key, 0)
            pct_base = (val / pass_base * 100) if pass_base else 0.0
            pct_sel = (val / any_selection * 100) if any_selection else 0.0
            table_lines.append(f"| {label} | {val:,} | {pct_base:.2f}% | {pct_sel:.2f}% |")
            return val

        # Exclusive categories
        table_lines.append("| **Exclusive (Only One)** | | | |")
        exclusive_vals = {}
        for k, label in exclusive_cats.items():
            exclusive_vals[label] = add_row(k, f"  - {label}")
            
        # Pairwise
        table_lines.append("| **Pairwise Overlaps** | | | |")
        pairwise_vals = {}
        for k, label in pairwise_cats.items():
            pairwise_vals[label] = add_row(k, f"  - {label}")

        # 3-Way
        table_lines.append("| **Three-way Overlaps** | | | |")
        threeway_vals = {}
        for k, label in threeway_cats.items():
            threeway_vals[label] = add_row(k, f"  - {label}")

        # 4-Way
        table_lines.append("| **Four-way Overlaps** | | | |")
        fourway_vals = {}
        for k, label in fourway_cats.items():
            fourway_vals[label] = add_row(k, f"  - {label}")

        # ── Only Semi-Resolved diagnostics cutflow ──
        if 'only_sr_diagnostics' in stats:
            sr_diag = stats['only_sr_diagnostics']
            total_sr = sr_diag.get('total_only_sr', 0)
            
            table_lines.append("\n## Only Semi-Resolved Failure Modes (Why they failed Resolved)")
            table_lines.append("\n| Failure Mode | Events | % of Only Semi-Resolved |")
            table_lines.append("| :--- | :---: | :---: |")
            
            def add_sr_diag_row(key, label):
                val = sr_diag.get(key, 0)
                pct = (val / total_sr * 100) if total_sr else 0.0
                table_lines.append(f"| {label} | {val:,} | {pct:.2f}% |")
                
            add_sr_diag_row('fail_resolved_jets', "Failed Resolved Jet Multiplicity (selected jets < 4)")
            add_sr_diag_row('fail_resolved_tags', "Failed Resolved B-Tag Multiplicity (medium tags < 4, but jets >= 4)")
            add_sr_diag_row('fail_resolved_trigger', "Failed Resolved Trigger (passed only boosted triggers)")
            add_sr_diag_row('fail_resolved_other', "Other / Uncategorized failure modes")

            table_lines.append("\n## Only Semi-Resolved Failure Modes (Why they failed Low-pT)")
            table_lines.append("\n| Failure Mode | Events | % of Only Semi-Resolved |")
            table_lines.append("| :--- | :---: | :---: |")
            
            add_sr_diag_row('fail_lowpt_jets', "Failed Resolved Jet Multiplicity (selected jets < 4)")
            add_sr_diag_row('fail_lowpt_tags_short', "Failed Low-pT Resolved Tag Count: tags < 3")
            add_sr_diag_row('fail_lowpt_tags_long', "Failed Low-pT Resolved Tag Count: tags > 3")
            add_sr_diag_row('fail_lowpt_no_lowpt_tag', "Failed Low-pT: exactly 3 tags, but 0 low-pT b-tags")
            add_sr_diag_row('fail_lowpt_trigger', "Failed Low-pT Resolved Trigger")
            add_sr_diag_row('fail_lowpt_other', "Other / Uncategorized failure modes")

        # ── Only Boosted diagnostics cutflow ──
        if 'only_boosted_diagnostics' in stats:
            boosted_diag = stats['only_boosted_diagnostics']
            total_boosted = boosted_diag.get('total_only_boosted', 0)
            
            table_lines.append("\n## Only Boosted Failure Modes (Why they failed Resolved)")
            table_lines.append("\n| Failure Mode | Events | % of Only Boosted |")
            table_lines.append("| :--- | :---: | :---: |")
            
            def add_boosted_diag_row(key, label):
                val = boosted_diag.get(key, 0)
                pct = (val / total_boosted * 100) if total_boosted else 0.0
                table_lines.append(f"| {label} | {val:,} | {pct:.2f}% |")
                
            add_boosted_diag_row('fail_resolved_jets', "Failed Resolved Jet Multiplicity (selected jets < 4)")
            add_boosted_diag_row('fail_resolved_tags', "Failed Resolved B-Tag Multiplicity (medium tags < 4, but jets >= 4)")
            add_boosted_diag_row('fail_resolved_trigger', "Failed Resolved Trigger (passed only boosted triggers)")
            add_boosted_diag_row('fail_resolved_other', "Other / Uncategorized failure modes")

            if 'sublead_missing_total' in boosted_diag:
                total_missing = boosted_diag.get('sublead_missing_total', 0)
                table_lines.append("\n## Only Boosted Subleading Higgs Missing Jet Breakdown (Events with exactly 1 selected AK4 jet inside Subleading FatJet)")
                table_lines.append("\n| Missing Jet Classification | Events | % of Target Events |")
                table_lines.append("| :--- | :---: | :---: |")
                
                def add_missing_row(key, label):
                    val = boosted_diag.get(key, 0)
                    pct = (val / total_missing * 100) if total_missing else 0.0
                    table_lines.append(f"| {label} | {val:,} | {pct:.2f}% |")
                    
                add_missing_row('sublead_missing_merged', "Merged AK4 Jets (daughters are collimated: $\Delta R(b_1, b_2) < 0.4$)")
                add_missing_row('sublead_missing_soft', "Soft $p_T$ (at least one daughter has $p_T < 40$ GeV)")
                add_missing_row('sublead_missing_out_eta', "Out of acceptance (at least one daughter has $|\eta| > 2.4$)")
                add_missing_row('sublead_missing_other', "Other / Uncategorized")

        # 5-Selection Breakdown (including Resolved3b)
        has_5sel = "5_onlyResolved" in stats
        if has_5sel:
            any_selection_5 = stats.get('anySelection_5', 0)
            none_all_5 = stats.get('none_all_5', 0)
            
            table_lines.append("\n## Expanded Event Selection Breakdown (5 Categories, including Resolved 3-tag)")
            table_lines.append(f"\n- **Passed Any Selection (5 Cats):** {any_selection_5:,} ({any_selection_5/pass_base*100:.2f}% of Baseline)" if pass_base else "- **Passed Any Selection (5 Cats):** 0")
            table_lines.append(f"- **Failed All Selections (5 Cats):** {none_all_5:,} ({none_all_5/pass_base*100:.2f}% of Baseline)" if pass_base else "- **Failed All Selections (5 Cats):** 0")
            table_lines.append("\n| Selection Category | Events | % of Baseline | % of Selected |")
            table_lines.append("| :--- | :---: | :---: | :---: |")
            
            def add_row_5(key, label):
                val = stats.get(key, 0)
                pct_base = (val / pass_base * 100) if pass_base else 0.0
                pct_sel = (val / any_selection_5 * 100) if any_selection_5 else 0.0
                table_lines.append(f"| {label} | {val:,} | {pct_base:.2f}% | {pct_sel:.2f}% |")
                return val

            # Exclusive
            table_lines.append("| **Exclusive (Only One)** | | | |")
            add_row_5('5_onlyResolved', '  - Only Resolved')
            add_row_5('5_onlyResolved3b', '  - Only Resolved 3-tag')
            add_row_5('5_onlyBoosted', '  - Only Boosted')
            add_row_5('5_onlySemiResolved', '  - Only Semi-Resolved')
            add_row_5('5_onlyLowPt', '  - Only Low-pT')
            
            # Pairwise
            table_lines.append("| **Pairwise Overlaps** | | | |")
            add_row_5('5_resolved_and_boosted', '  - Resolved & Boosted')
            add_row_5('5_resolved_and_semiresolved', '  - Resolved & Semi-Resolved')
            add_row_5('5_resolved_and_lowpt', '  - Resolved & Low-pT')
            add_row_5('5_resolved3b_and_boosted', '  - Resolved 3-tag & Boosted')
            add_row_5('5_resolved3b_and_semiresolved', '  - Resolved 3-tag & Semi-Resolved')
            add_row_5('5_resolved3b_and_lowpt', '  - Resolved 3-tag & Low-pT')
            add_row_5('5_boosted_and_semiresolved', '  - Boosted & Semi-Resolved')
            add_row_5('5_boosted_and_lowpt', '  - Boosted & Low-pT')
            add_row_5('5_semiresolved_and_lowpt', '  - Semi-Resolved & Low-pT')
            
            # 3-Way
            table_lines.append("| **Three-way Overlaps** | | | |")
            add_row_5('5_resolved_boosted_semiresolved', '  - Resolved & Boosted & Semi-Resolved')
            add_row_5('5_resolved_boosted_lowpt', '  - Resolved & Boosted & Low-pT')
            add_row_5('5_resolved_semiresolved_lowpt', '  - Resolved & Semi-Resolved & Low-pT')
            add_row_5('5_resolved3b_boosted_semiresolved', '  - Resolved 3-tag & Boosted & Semi-Resolved')
            add_row_5('5_resolved3b_boosted_lowpt', '  - Resolved 3-tag & Boosted & Low-pT')
            add_row_5('5_resolved3b_semiresolved_lowpt', '  - Resolved 3-tag & Semi-Resolved & Low-pT')
            add_row_5('5_boosted_semiresolved_lowpt', '  - Boosted & Semi-Resolved & Low-pT')
            
            # 4-Way
            table_lines.append("| **Four-way Overlaps** | | | |")
            add_row_5('5_resolved_boosted_semiresolved_lowpt', '  - Resolved & Boosted & Semi-Resolved & Low-pT')
            add_row_5('5_resolved3b_boosted_semiresolved_lowpt', '  - Resolved 3-tag & Boosted & Semi-Resolved & Low-pT')
            
            # 5-Way
            table_lines.append("| **Five-way Overlaps** | | | |")
            add_row_5('5_all_five', '  - All Five Selections')

        # Print report to stdout
        report_text = "\n".join(table_lines)
        print(report_text)
        
        # Save report to file
        report_path = os.path.join(output_dir, f"overlap_report_{dataset}.md")
        with open(report_path, "w") as f:
            f.write(report_text)
        print(f"\nSaved report to: {report_path}")

        # -------------------------------------------------------------
        # Pie Chart: Breakdown of Selected Events (anySelection)
        # -------------------------------------------------------------
        # For the pie chart, group all overlaps together to keep it clean
        sum_overlaps = sum(pairwise_vals.values()) + sum(threeway_vals.values()) + sum(fourway_vals.values())
        
        pie_labels = []
        pie_sizes = []
        
        for label, val in exclusive_vals.items():
            if val > 0:
                pie_labels.append(label)
                pie_sizes.append(val)
        if sum_overlaps > 0:
            pie_labels.append("Overlapping Selections")
            pie_sizes.append(sum_overlaps)
            
        if pie_sizes:
            plt.figure(figsize=(8, 8))
            # Elegant colors matching a physics aesthetic
            colors = ['#4f46e5', '#06b6d4', '#10b981', '#f59e0b', '#ec4899']
            
            plt.pie(
                pie_sizes, 
                labels=pie_labels, 
                autopct='%1.1f%%', 
                startangle=140, 
                colors=colors[:len(pie_sizes)],
                textprops={'fontsize': 12},
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True}
            )
            plt.title(f"Exclusive & Overlap Breakdown ({dataset})\nTotal Selected: {any_selection:,} Events", fontsize=14, fontweight='bold')
            plt.tight_layout()
            pie_path = os.path.join(output_dir, f"overlap_pie_{dataset}.png")
            plt.savefig(pie_path, dpi=150)
            plt.close()
            print(f"Saved pie chart to: {pie_path}")

        # -------------------------------------------------------------
        # Horizontal Bar Chart of All 15 Categories
        # -------------------------------------------------------------
        all_combos = {}
        all_combos.update(exclusive_vals)
        all_combos.update(pairwise_vals)
        all_combos.update(threeway_vals)
        all_combos.update(fourway_vals)
        
        # Sort by value
        sorted_combos = sorted(all_combos.items(), key=lambda x: x[1], reverse=False)
        labels = [item[0] for item in sorted_combos if item[1] > 0]
        values = [item[1] for item in sorted_combos if item[1] > 0]
        
        if values:
            plt.figure(figsize=(10, 6))
            y_pos = np.arange(len(labels))
            
            plt.barh(y_pos, values, color='#6366f1', edgecolor='#4338ca', height=0.6)
            plt.yticks(y_pos, labels, fontsize=10)
            plt.xlabel("Events", fontsize=11)
            plt.title(f"Event Count per Selection/Overlap Category ({dataset})", fontsize=13, fontweight='bold')
            
            # Add values next to the bars
            for i, val in enumerate(values):
                plt.text(val + (max(values)*0.01), i, f"{val:,}", va='center', fontsize=9, fontweight='semibold')
                
            # Expand x limit slightly to accommodate labels
            plt.xlim(0, max(values) * 1.15)
            plt.grid(axis='x', linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            bar_path = os.path.join(output_dir, f"overlap_bar_{dataset}.png")
            plt.savefig(bar_path, dpi=150)
            plt.close()
            print(f"Saved bar chart to: {bar_path}")

        # -------------------------------------------------------------
        # Only Semi-Resolved Diagnostic Plots
        # -------------------------------------------------------------
        if 'only_sr_diagnostics' in stats:
            sr_diag = stats['only_sr_diagnostics']
            
            # Static bins matching processor definition
            njets_bins = np.arange(0, 11)
            pt_bins = np.linspace(200, 1000, 41)
            mass_bins = np.linspace(0, 250, 26)
            xbb_bins = np.linspace(0, 1, 26)
            
            # 1. Jet and Tag multiplicity plot
            if 'njets_hist' in sr_diag and 'ntags_hist' in sr_diag:
                njets_hist = np.array(sr_diag['njets_hist'])
                ntags_hist = np.array(sr_diag['ntags_hist'])
                
                bin_centers = njets_bins[:-1]
                
                plt.figure(figsize=(9, 5))
                width = 0.35
                plt.bar([c - width/2 for c in bin_centers], njets_hist, width, label='Selected AK4 Jets', color='#3b82f6')
                plt.bar([c + width/2 for c in bin_centers], ntags_hist, width, label='Medium B-Tagged AK4 Jets', color='#ef4444')
                
                plt.xlabel('AK4 Jet Multiplicity', fontsize=11)
                plt.ylabel('Events', fontsize=11)
                plt.title(f"AK4 Jet and Tag Multiplicity for 'Only Semi-Resolved' Events ({dataset})", fontsize=12, fontweight='bold')
                plt.xticks(bin_centers)
                plt.legend(fontsize=10)
                plt.grid(axis='y', linestyle='--', alpha=0.5)
                plt.tight_layout()
                
                njets_plot_path = os.path.join(output_dir, f"only_sr_njets_{dataset}.png")
                plt.savefig(njets_plot_path, dpi=150)
                plt.close()
                print(f"Saved Only SR jet multiplicity plot to: {njets_plot_path}")

            # 2. Leading FatJet kinematics subplots
            if 'pt_hist' in sr_diag and 'mass_hist' in sr_diag and 'xbb_hist' in sr_diag:
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                
                # Plot pT
                pt_hist = np.array(sr_diag['pt_hist'])
                pt_centers = [(pt_bins[i] + pt_bins[i+1])/2 for i in range(len(pt_hist))]
                pt_widths = [pt_bins[i+1] - pt_bins[i] for i in range(len(pt_hist))]
                axes[0].bar(pt_centers, pt_hist, width=pt_widths, color='#4f46e5', edgecolor='white', alpha=0.8)
                axes[0].set_xlabel('Leading FatJet $p_T$ [GeV]', fontsize=11)
                axes[0].set_ylabel('Events', fontsize=11)
                axes[0].set_title('Leading FatJet $p_T$', fontweight='bold')
                axes[0].grid(linestyle='--', alpha=0.5)
                
                # Plot Mass
                m_hist = np.array(sr_diag['mass_hist'])
                m_centers = [(mass_bins[i] + mass_bins[i+1])/2 for i in range(len(m_hist))]
                m_widths = [mass_bins[i+1] - mass_bins[i] for i in range(len(m_hist))]
                axes[1].bar(m_centers, m_hist, width=m_widths, color='#06b6d4', edgecolor='white', alpha=0.8)
                axes[1].set_xlabel('Leading FatJet $m_{softdrop}$ [GeV]', fontsize=11)
                axes[1].set_ylabel('Events', fontsize=11)
                axes[1].set_title('Leading FatJet $m_{softdrop}$', fontweight='bold')
                axes[1].grid(linestyle='--', alpha=0.5)
                
                # Plot Xbb
                x_hist = np.array(sr_diag['xbb_hist'])
                x_centers = [(xbb_bins[i] + xbb_bins[i+1])/2 for i in range(len(x_hist))]
                x_widths = [xbb_bins[i+1] - xbb_bins[i] for i in range(len(x_hist))]
                axes[2].bar(x_centers, x_hist, width=x_widths, color='#10b981', edgecolor='white', alpha=0.8)
                axes[2].set_xlabel('Leading FatJet ParticleNet MD Xbb', fontsize=11)
                axes[2].set_ylabel('Events', fontsize=11)
                axes[2].set_title('Leading FatJet Xbb Score', fontweight='bold')
                axes[2].grid(linestyle='--', alpha=0.5)
                
                fig.suptitle(f"Leading FatJet Kinematics for 'Only Semi-Resolved' Events ({dataset})", fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                kin_plot_path = os.path.join(output_dir, f"only_sr_fatjet_kin_{dataset}.png")
                plt.savefig(kin_plot_path, dpi=150)
                plt.close()
                print(f"Saved Only SR FatJet kinematics plot to: {kin_plot_path}")

            # 3. Low-pT Jet and Tag multiplicity plot
            if 'nlowpt_jets_hist' in sr_diag and 'nlowpt_tags_hist' in sr_diag:
                nlowpt_jets_hist = np.array(sr_diag['nlowpt_jets_hist'])
                nlowpt_tags_hist = np.array(sr_diag['nlowpt_tags_hist'])
                
                bin_centers = njets_bins[:-1]
                
                plt.figure(figsize=(9, 5))
                width = 0.35
                plt.bar([c - width/2 for c in bin_centers], nlowpt_jets_hist, width, label='Selected Low-pT AK4 Jets ($15 < p_T < 40$ GeV)', color='#8b5cf6')
                plt.bar([c + width/2 for c in bin_centers], nlowpt_tags_hist, width, label='Medium B-Tagged Low-pT Jets', color='#f43f5e')
                
                plt.xlabel('Low-pT AK4 Jet Multiplicity', fontsize=11)
                plt.ylabel('Events', fontsize=11)
                plt.title(f"Low-pT Jet and Tag Multiplicity for 'Only Semi-Resolved' Events ({dataset})", fontsize=12, fontweight='bold')
                plt.xticks(bin_centers)
                plt.legend(fontsize=10)
                plt.grid(axis='y', linestyle='--', alpha=0.5)
                plt.tight_layout()
                
                lowpt_njets_plot_path = os.path.join(output_dir, f"only_sr_lowpt_njets_{dataset}.png")
                plt.savefig(lowpt_njets_plot_path, dpi=150)
                plt.close()
                print(f"Saved Only SR low-pT jet multiplicity plot to: {lowpt_njets_plot_path}")

            # 4. Matched jet btag score distributions
            if 'matched_btag_1_hist' in sr_diag:
                btag_bins = np.linspace(0, 1, 51)
                
                plt.figure(figsize=(10, 6))
                
                h1 = np.array(sr_diag['matched_btag_1_hist'])
                plt.step(btag_bins, np.concatenate(([0], h1)), where='pre', label='1st Highest Score', color='#3b82f6', linewidth=2)
                
                h2 = np.array(sr_diag['matched_btag_2_hist'])
                plt.step(btag_bins, np.concatenate(([0], h2)), where='pre', label='2nd Highest Score', color='#10b981', linewidth=2)
                
                h3 = np.array(sr_diag['matched_btag_3_hist'])
                plt.step(btag_bins, np.concatenate(([0], h3)), where='pre', label='3rd Highest Score', color='#f59e0b', linewidth=2)
                
                h4 = np.array(sr_diag['matched_btag_4_hist'])
                plt.step(btag_bins, np.concatenate(([0], h4)), where='pre', label='4th Highest Score', color='#ef4444', linewidth=2)
                
                # Draw Medium WP line if present
                if 'btag_wp_m' in sr_diag:
                    wp = sr_diag['btag_wp_m']
                    plt.axvline(wp, color='black', linestyle='--', linewidth=1.5, label=f'Medium WP ({wp:.4f})')
                
                plt.xlabel('DeepJet B-Tagging Score (matched reco jet)', fontsize=11)
                plt.ylabel('Events', fontsize=11)
                plt.title(f"Gen-Matched AK4 Jet B-Tagging Scores for 'Only Semi-Resolved' Events ({dataset})", fontsize=12, fontweight='bold')
                plt.yscale('log')
                plt.ylim(bottom=0.5)
                plt.legend(fontsize=10, loc='upper left')
                plt.grid(linestyle='--', alpha=0.5)
                plt.xlim(0, 1)
                plt.tight_layout()
                
                btag_plot_path = os.path.join(output_dir, f"only_sr_matched_btags_{dataset}.png")
                plt.savefig(btag_plot_path, dpi=150)
                plt.close()
                print(f"Saved Only SR matched btag scores plot to: {btag_plot_path}")

            # 5. Delta R plots (cross-Higgs and same-Higgs)
            if 'match_dr_h1h2_hist' in sr_diag:
                dr_bins = np.linspace(0, 5, 51)
                dr_hist = np.array(sr_diag['match_dr_h1h2_hist'])
                
                plt.figure(figsize=(9, 5))
                plt.step(dr_bins, np.concatenate(([0], dr_hist)), where='pre', label=r'Min $\Delta R(j_{H1}, j_{H2})$', color='#4f46e5', linewidth=2)
                
                if 'match_dr_h0_hist' in sr_diag:
                    dr_h0 = np.array(sr_diag['match_dr_h0_hist'])
                    plt.step(dr_bins, np.concatenate(([0], dr_h0)), where='pre', label=r'$\Delta R(j, j)$ from Higgs 1', color='#10b981', linewidth=2, linestyle='--')
                    
                if 'match_dr_h1_hist' in sr_diag:
                    dr_h1 = np.array(sr_diag['match_dr_h1_hist'])
                    plt.step(dr_bins, np.concatenate(([0], dr_h1)), where='pre', label=r'$\Delta R(j, j)$ from Higgs 2', color='#f59e0b', linewidth=2, linestyle='-.')
                
                plt.xlabel(r'$\Delta R$', fontsize=11)
                plt.ylabel('Events', fontsize=11)
                plt.title(f"Matched Jets $\Delta R$ Distributions for 'Only Semi-Resolved' Events ({dataset})", fontsize=12, fontweight='bold')
                plt.legend(fontsize=10, loc='upper right')
                plt.grid(linestyle='--', alpha=0.5)
                plt.xlim(0, 5)
                plt.tight_layout()
                
                dr_plot_path = os.path.join(output_dir, f"only_sr_matched_dr_{dataset}.png")
                plt.savefig(dr_plot_path, dpi=150)
                plt.close()
                print(f"Saved Only SR matched Delta R plot to: {dr_plot_path}")

            # 6. Higgs Pt (lead and sublead pt from matched reco jets)
            if 'match_h_lead_pt_hist' in sr_diag and 'match_h_sublead_pt_hist' in sr_diag:
                pt_bins = np.linspace(0, 800, 41)
                lead_pt = np.array(sr_diag['match_h_lead_pt_hist'])
                sublead_pt = np.array(sr_diag['match_h_sublead_pt_hist'])
                
                plt.figure(figsize=(9, 5))
                plt.step(pt_bins, np.concatenate(([0], lead_pt)), where='pre', label='Leading Higgs Candidate', color='#2563eb', linewidth=2)
                plt.step(pt_bins, np.concatenate(([0], sublead_pt)), where='pre', label='Subleading Higgs Candidate', color='#db2777', linewidth=2)
                plt.xlabel('Reconstructed Higgs Candidate $p_T$ [GeV]', fontsize=11)
                plt.ylabel('Events', fontsize=11)
                plt.title(f"Reconstructed Higgs $p_T$ for 'Only Semi-Resolved' Events ({dataset})", fontsize=12, fontweight='bold')
                plt.legend(fontsize=10)
                plt.grid(linestyle='--', alpha=0.5)
                plt.xlim(0, 800)
                plt.tight_layout()
                
                pt_plot_path = os.path.join(output_dir, f"only_sr_reco_h_pt_{dataset}.png")
                plt.savefig(pt_plot_path, dpi=150)
                plt.close()
                print(f"Saved Only SR reco Higgs pT plot to: {pt_plot_path}")

            # 7. Invariant mass of matched reco jets
            if 'match_m4j_hist' in sr_diag:
                m4j_bins = np.linspace(0, 1200, 61)
                m4j_hist = np.array(sr_diag['match_m4j_hist'])
                
                plt.figure(figsize=(8, 5))
                plt.step(m4j_bins, np.concatenate(([0], m4j_hist)), where='pre', color='#0d9488', linewidth=2)
                plt.xlabel('Invariant Mass of all matched reco jets [GeV]', fontsize=11)
                plt.ylabel('Events', fontsize=11)
                plt.title(f"Invariant Mass of Matched Reco Jets for 'Only Semi-Resolved' Events ({dataset})", fontsize=12, fontweight='bold')
                plt.grid(linestyle='--', alpha=0.5)
                plt.xlim(0, 1200)
                plt.tight_layout()
                
                m4j_plot_path = os.path.join(output_dir, f"only_sr_m4j_{dataset}.png")
                plt.savefig(m4j_plot_path, dpi=150)
                plt.close()
                print(f"Saved Only SR matched invariant mass plot to: {m4j_plot_path}")

            # 8. Eta of 4th highest b-tag score jet (reco vs gen)
            if 'match_4th_reco_eta_hist' in sr_diag and 'match_4th_gen_eta_hist' in sr_diag:
                eta_bins = np.linspace(-2.5, 2.5, 51)
                reco_eta = np.array(sr_diag['match_4th_reco_eta_hist'])
                gen_eta = np.array(sr_diag['match_4th_gen_eta_hist'])
                
                plt.figure(figsize=(9, 5))
                plt.step(eta_bins, np.concatenate(([0], reco_eta)), where='pre', label='Reco Jet', color='#ea580c', linewidth=2)
                plt.step(eta_bins, np.concatenate(([0], gen_eta)), where='pre', label='Gen b-quark', color='#0284c7', linewidth=2, linestyle='--')
                plt.xlabel('Pseudorapidity $\eta$', fontsize=11)
                plt.ylabel('Events', fontsize=11)
                plt.title(f"Pseudorapidity $\eta$ of the 4th B-Tag Score Jet ({dataset})", fontsize=12, fontweight='bold')
                plt.legend(fontsize=10)
                plt.grid(linestyle='--', alpha=0.5)
                plt.xlim(-2.5, 2.5)
                plt.tight_layout()
                
                eta_plot_path = os.path.join(output_dir, f"only_sr_4th_eta_{dataset}.png")
                plt.savefig(eta_plot_path, dpi=150)
                plt.close()
                print(f"Saved Only SR 4th jet eta plot to: {eta_plot_path}")

            # 9. 2D distribution: pt of 4th leading matched jet vs mHH for ALL events with 4 matched jets
            if 'match_mHH_vs_pt4_hist2d' in sr_diag:
                m4j_bins = np.linspace(0, 1200, 61)
                pt_4th_bins = np.linspace(0, 250, 51)
                hist_2d = np.array(sr_diag['match_mHH_vs_pt4_hist2d'])
                
                plt.figure(figsize=(9, 7))
                X, Y = np.meshgrid(m4j_bins, pt_4th_bins)
                cmap = plt.cm.Blues.copy()
                cmap.set_under('white')
                mesh = plt.pcolormesh(X, Y, hist_2d.T, cmap=cmap, shading='auto', vmin=1e-5)
                
                plt.xlabel(r'$m_{HH}$ (Invariant Mass of 4 Matched Jets) [GeV]', fontsize=11)
                plt.ylabel(r'$p_T$ of 4th Leading Matched Jet [GeV]', fontsize=11)
                plt.title(f"4th Matched Jet $p_T$ vs $m_{{HH}}$ (Events with 4 Matched Jets) ({dataset})", fontsize=12, fontweight='bold')
                
                cbar = plt.colorbar(mesh)
                cbar.set_label('Events', fontsize=11)
                plt.grid(linestyle='--', alpha=0.3)
                plt.xlim(0, 1200)
                plt.ylim(0, 250)
                plt.tight_layout()
                
                hist2d_plot_path = os.path.join(output_dir, f"match_mHH_vs_pt4_2d_{dataset}.png")
                plt.savefig(hist2d_plot_path, dpi=150)
                plt.close()
                print(f"Saved 2D histogram of mHH vs pt4 to: {hist2d_plot_path}")

            # 10. Only Boosted diagnostics plots
            if 'only_boosted_diagnostics' in stats:
                boosted_diag = stats['only_boosted_diagnostics']
                
                # Plot AK4 count inside leading FatJet
                if 'ak4_in_fj_hist' in boosted_diag:
                    ak4_hist = np.array(boosted_diag['ak4_in_fj_hist'])
                    bin_centers = np.arange(0, 6)
                    
                    plt.figure(figsize=(9, 5.5))
                    plt.bar(bin_centers - 0.2, ak4_hist, width=0.4, label='Leading FatJet (AK8)', color='#6366f1', edgecolor='black', alpha=0.8)
                    if 'ak4_in_sublead_fj_hist' in boosted_diag:
                        ak4_sub_hist = np.array(boosted_diag['ak4_in_sublead_fj_hist'])
                        plt.bar(bin_centers + 0.2, ak4_sub_hist, width=0.4, label='Subleading FatJet (AK8)', color='#3b82f6', edgecolor='black', alpha=0.8)
                    plt.xlabel('Number of Selected AK4 Jets inside FatJet Cone ($\Delta R < 0.8$)', fontsize=11)
                    plt.ylabel('Events', fontsize=11)
                    plt.title(f"AK4 Jet Multiplicity inside FatJets for 'Only Boosted' Events ({dataset})", fontsize=12, fontweight='bold')
                    plt.grid(axis='y', linestyle='--', alpha=0.5)
                    plt.xticks(bin_centers)
                    plt.legend(fontsize=10)
                    plt.tight_layout()
                    
                    ak4_plot_path = os.path.join(output_dir, f"only_boosted_ak4_in_fatjet_{dataset}.png")
                    plt.savefig(ak4_plot_path, dpi=150)
                    plt.close()
                    print(f"Saved Only Boosted AK4 in FatJet plot to: {ak4_plot_path}")

                # Plot Gen-level Delta R
                if 'dr_b_H0_hist' in boosted_diag and 'dr_b_H1_hist' in boosted_diag:
                    dr_b_bins = np.linspace(0, 5.0, 51)
                    dr_h0 = np.array(boosted_diag['dr_b_H0_hist'])
                    dr_h1 = np.array(boosted_diag['dr_b_H1_hist'])
                    
                    plt.figure(figsize=(9, 5))
                    plt.step(dr_b_bins, np.concatenate(([0], dr_h0)), where='pre', label='Higgs 1 ($b\overline{b}$)', color='#ec4899', linewidth=2)
                    plt.step(dr_b_bins, np.concatenate(([0], dr_h1)), where='pre', label='Higgs 2 ($b\overline{b}$)', color='#8b5cf6', linewidth=2, linestyle='--')
                    plt.xlabel('Gen-level $\Delta R(b, \overline{b})$', fontsize=11)
                    plt.ylabel('Events', fontsize=11)
                    plt.title(f"Gen-level $\Delta R(b, \overline{{b}})$ for 'Only Boosted' Events ({dataset})", fontsize=12, fontweight='bold')
                    plt.legend(fontsize=10)
                    plt.grid(linestyle='--', alpha=0.5)
                    plt.xlim(0, 5.0)
                    plt.tight_layout()
                    
                    dr_plot_path = os.path.join(output_dir, f"only_boosted_gen_dr_{dataset}.png")
                    plt.savefig(dr_plot_path, dpi=150)
                    plt.close()
                    print(f"Saved Only Boosted gen Delta R plot to: {dr_plot_path}")

                # Plot Matched Reco Jet B-tagging Scores
                if all(f'matched_btag_{i}_hist' in boosted_diag for i in range(1, 5)):
                    btag_bins = np.linspace(0, 1, 51)
                    hist_1 = np.array(boosted_diag['matched_btag_1_hist'])
                    hist_2 = np.array(boosted_diag['matched_btag_2_hist'])
                    hist_3 = np.array(boosted_diag['matched_btag_3_hist'])
                    hist_4 = np.array(boosted_diag['matched_btag_4_hist'])
                    
                    plt.figure(figsize=(9, 5.5))
                    plt.step(btag_bins, np.concatenate(([0], hist_1)), where='pre', label='1st Matched Reco Jet (Highest score)', color='#dc2626', linewidth=2)
                    plt.step(btag_bins, np.concatenate(([0], hist_2)), where='pre', label='2nd Matched Reco Jet', color='#ea580c', linewidth=2)
                    plt.step(btag_bins, np.concatenate(([0], hist_3)), where='pre', label='3rd Matched Reco Jet', color='#eab308', linewidth=2)
                    plt.step(btag_bins, np.concatenate(([0], hist_4)), where='pre', label='4th Matched Reco Jet', color='#16a34a', linewidth=2)
                    
                    btag_wp = 0.6
                    plt.axvline(btag_wp, color='black', linestyle='--', linewidth=1.5, label=f'Medium WP ({btag_wp:.1f})')
                    
                    plt.xlabel('B-Tagging Score (DeepJet)', fontsize=11)
                    plt.ylabel('Events', fontsize=11)
                    plt.title(f"B-Tagging Scores of Matched Reco Jets for 'Only Boosted' Events ({dataset})", fontsize=12, fontweight='bold')
                    plt.legend(fontsize=10)
                    plt.grid(linestyle='--', alpha=0.5)
                    plt.xlim(0, 1.0)
                    plt.yscale('log')
                    plt.tight_layout()
                    
                    btag_plot_path = os.path.join(output_dir, f"only_boosted_matched_btags_{dataset}.png")
                    plt.savefig(btag_plot_path, dpi=150)
                    plt.close()
                    print(f"Saved Only Boosted matched btag scores plot to: {btag_plot_path}")

                # Plot low-btag matched jet diagnostics
                if 'lowtag_pt_hist' in boosted_diag:
                    pt_bins = np.linspace(0, 200, 41)
                    eta_bins = np.linspace(-2.5, 2.5, 51)
                    hist_pt = np.array(boosted_diag['lowtag_pt_hist'])
                    hist_eta = np.array(boosted_diag['lowtag_eta_hist'])
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
                    ax1.step(pt_bins, np.concatenate(([0], hist_pt)), where='pre', color='#4f46e5', linewidth=2)
                    ax1.set_xlabel(r'4th Matched Jet $p_T$ [GeV]', fontsize=11)
                    ax1.set_ylabel('Events', fontsize=11)
                    ax1.set_title(r'Jet $p_T$ distribution', fontsize=12, fontweight='bold')
                    ax1.grid(linestyle='--', alpha=0.5)
                    
                    ax2.step(eta_bins, np.concatenate(([0], hist_eta)), where='pre', color='#0891b2', linewidth=2)
                    ax2.set_xlabel(r'4th Matched Jet $\eta$', fontsize=11)
                    ax2.set_ylabel('Events', fontsize=11)
                    ax2.set_title(r'Jet $\eta$ distribution', fontsize=12, fontweight='bold')
                    ax2.grid(linestyle='--', alpha=0.5)
                    
                    fig.suptitle(f"Kinematics of 4th Matched Jet with B-Tag < 0.05 ({dataset})", fontsize=13, fontweight='bold')
                    plt.tight_layout()
                    kin_plot_path = os.path.join(output_dir, f"only_boosted_lowtag_kin_{dataset}.png")
                    plt.savefig(kin_plot_path, dpi=150)
                    plt.close()
                    print(f"Saved Only Boosted lowtag kinematics plot to: {kin_plot_path}")
                    
                    # Hadron Flavour (Bar plot)
                    flav_hist = np.array(boosted_diag['lowtag_flav_hist'])
                    filtered_flavours = ['Light / Gluon (0)', 'Charm (4)', 'Beauty (5)']
                    filtered_counts = [flav_hist[0], flav_hist[4], flav_hist[5]]
                    
                    plt.figure(figsize=(8, 5.5))
                    bars = plt.bar(filtered_flavours, filtered_counts, color=['#6b7280', '#fb923c', '#3b82f6'], edgecolor='black', alpha=0.8, width=0.6)
                    plt.ylabel('Events', fontsize=11)
                    plt.title(f"True Hadron Flavor of 4th Matched Jet with B-Tag < 0.05 ({dataset})", fontsize=12, fontweight='bold')
                    plt.grid(axis='y', linestyle='--', alpha=0.5)
                    
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.05 * (max(filtered_counts)+1), f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
                        
                    plt.tight_layout()
                    flav_plot_path = os.path.join(output_dir, f"only_boosted_lowtag_flavor_{dataset}.png")
                    plt.savefig(flav_plot_path, dpi=150)
                    plt.close()
                    print(f"Saved Only Boosted lowtag flavor plot to: {flav_plot_path}")
                    
                    # Matching Delta R
                    dr_bins = np.linspace(0, 0.8, 41)
                    hist_dr = np.array(boosted_diag['lowtag_dr_hist'])
                    
                    plt.figure(figsize=(8, 5.5))
                    plt.step(dr_bins, np.concatenate(([0], hist_dr)), where='pre', color='#ea580c', linewidth=2)
                    plt.xlabel(r'$\Delta R(\mathrm{Jet}, \mathrm{Gen\text{-}b})$', fontsize=11)
                    plt.ylabel('Events', fontsize=11)
                    plt.title(f"Gen-matching $\Delta R$ for 4th Matched Jet with B-Tag < 0.05 ({dataset})", fontsize=12, fontweight='bold')
                    plt.grid(linestyle='--', alpha=0.5)
                    plt.xlim(0, 0.8)
                    plt.tight_layout()
                    dr_plot_path = os.path.join(output_dir, f"only_boosted_lowtag_dr_{dataset}.png")
                    plt.savefig(dr_plot_path, dpi=150)
                    plt.close()
                    print(f"Saved Only Boosted lowtag Delta R plot to: {dr_plot_path}")

                    # Matching FatJet ParticleNetMD_Xbb
                    if 'lowtag_xbb_b_hist' in boosted_diag and 'lowtag_xbb_light_hist' in boosted_diag:
                        xbb_bins = np.linspace(0, 1.0, 51)
                        hist_xbb_b = np.array(boosted_diag['lowtag_xbb_b_hist'])
                        hist_xbb_light = np.array(boosted_diag['lowtag_xbb_light_hist'])
                        
                        plt.figure(figsize=(8.5, 5.5))
                        plt.step(xbb_bins, np.concatenate(([0], hist_xbb_b)), where='pre', color='#3b82f6', linewidth=2.5, label='Beauty Matched (Flavor 5)')
                        plt.step(xbb_bins, np.concatenate(([0], hist_xbb_light)), where='pre', color='#ef4444', linewidth=2, linestyle='--', label='Light / Gluon Matched (Flavor 0)')
                        
                        plt.axvline(0.8, color='black', linestyle=':', linewidth=1.5, label='Loose Xbb Selection (0.8)')
                        
                        plt.xlabel('Matching FatJet ParticleNetMD_Xbb', fontsize=11)
                        plt.ylabel('Events', fontsize=11)
                        plt.title(f"FatJet Xbb Discriminant for 4th Jet with B-Tag < 0.05 ({dataset})", fontsize=12, fontweight='bold')
                        plt.legend(fontsize=10)
                        plt.grid(linestyle='--', alpha=0.5)
                        plt.xlim(0, 1.0)
                        plt.tight_layout()
                        
                        xbb_plot_path = os.path.join(output_dir, f"only_boosted_lowtag_xbb_{dataset}.png")
                        plt.savefig(xbb_plot_path, dpi=150)
                        plt.close()
                        print(f"Saved Only Boosted lowtag matching FatJet Xbb plot to: {xbb_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze overlap study processor output and generate plots.")
    parser.add_argument("input_file", help="Path to the input .coffea file")
    parser.add_argument("-o", "--output-dir", help="Directory to save output files (default: same as input file)")
    args = parser.parse_args()
    
    generate_overlap_report(args.input_file, args.output_dir)
