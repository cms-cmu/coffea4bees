import uproot
import numpy as np
import hist
import matplotlib
matplotlib.use("Agg")  # no GUI, renders directly to files
import matplotlib.pyplot as plt
import os
import yaml
import sys
sys.path.insert(0, os.getcwd())
from coffea4bees.hemisphere_mixing.mixing_helpers   import read_hemi_files, get_grouped_hemispheres_data, get_filter


def count_all_hemispheres(hemi_data, do_print=False):

    tags, tag_counts = np.unique(hemi_data["nTagJet"], return_counts=True)
    hemi_count_data = {}
    hemi_count_data["tagJet_counts"] = list(zip(tags, tag_counts))

    for itag, t in enumerate(tags):
        if do_print: print(f"\nnTagJet = {t} : count = {tag_counts[itag]}")
        hemi_count_data[f"nTagJet{t}"] = {}

        tag_filter = (hemi_data["nTagJet"] == t)

        selJet, selJet_counts = np.unique(hemi_data["nSelJet"][tag_filter], return_counts=True)
        hemi_count_data[f"nTagJet{t}"]["selJet_counts"] = list(zip(selJet, selJet_counts))

        for isel, s in enumerate(selJet):
            if do_print: print(f"\tnSelJet = {s} : count = {selJet_counts[isel]}")
            hemi_count_data[f"nTagJet{t}"][f"nSelJet{s}"] = {}

            selJet_filter = (hemi_data["nSelJet"] == s) & tag_filter

            allJet, allJet_counts = np.unique(hemi_data["nJet"][selJet_filter], return_counts=True)
            hemi_count_data[f"nTagJet{t}"][f"nSelJet{s}"]["allJet_counts"] = list(zip(allJet, allJet_counts))

            for ijet, j in enumerate(allJet):
                if do_print: print(f"\t\tnJet = {j} : count = {allJet_counts[ijet]}")

    return hemi_count_data


def get_hemi_ranges(hemi_count_data, threshold=300):

    tagJet_ranges = {}

    for tag, tag_count in hemi_count_data["tagJet_counts"]:

        # keep track of tag rages
        if tag_count > threshold:

            tagJet_ranges[int(tag)] = {}

            for sel, sel_count in hemi_count_data[f"nTagJet{tag}"]["selJet_counts"]:

                # keep track of sel ranges
                if sel_count > threshold:

                    tagJet_ranges[int(tag)][int(sel)] = []

                    for jet, jet_count in hemi_count_data[f"nTagJet{tag}"][f"nSelJet{sel}"]["allJet_counts"]:

                        # keep track of sel ranges
                        if jet_count > threshold:
                            tagJet_ranges[int(tag)][int(sel)].append(int(jet))
    return tagJet_ranges



def count_combined_hemispheres(hemi_ranges, hemi_data, do_print=False):
    hemi_count_data = {}

    # Outer loop: tag multiplicity bins
    tag_keys = list(hemi_ranges.keys())
    for itag, tag in enumerate(tag_keys):
        if do_print:
            print(f"tag = {tag}  itag = {itag}")

        hemi_count_data[tag] = {}

        # --- tag filter ----------------------------------------------------------
        tag_filter = get_filter(hemi_data, "nTagJet", tag, low_edge=(itag==0), high_edge=(itag==len(tag_keys)-1))

        if do_print:
            print(f"\ttag_filter: == {tag}" + (" or <" if itag==0 else " or >" if itag==len(tag_keys)-1 else ""))

        # skip empty sub-ranges
        if not hemi_ranges[tag]:
            print(f"ERROR: no sel jets for tag = {tag}")
            continue

        # -------------------------------------------------------------------------
        # Middle loop: selected-jet multiplicity bins
        sel_keys = list(hemi_ranges[tag].keys())
        for isel, sel in enumerate(sel_keys):
            if do_print:
                print(f"\t sel = {sel}  isel = {isel}")

            hemi_count_data[tag][sel] = {}

            # --- sel filter ------------------------------------------------------
            sel_filter = get_filter(hemi_data, "nSelJet", sel, low_edge=(isel==0), high_edge=(isel==len(sel_keys)-1))

            if do_print:
                print(f"\t\tsel_filter: == {sel}" + (" or <" if isel==0 else " or >" if isel==len(sel_keys)-1 else ""))

            # ---------------------------------------------------------------------
            # Inner loop: total-jet multiplicity bins
            jet_bins = hemi_ranges[tag][sel]
            if not jet_bins:
                # special case: no jet bins defined
                count = len(hemi_data["nSelJet"][tag_filter & sel_filter])
                hemi_count_data[tag][sel][-1] = count
                if do_print:
                    print("\t\t jet_filter: == True  (no jet bins)")
                continue

            for ijet, jet in enumerate(jet_bins):
                if do_print:
                    print(f"\t\t jet = {jet}  ijet = {ijet}")

                jet_filter = get_filter(hemi_data, "nJet", jet, low_edge=(ijet==0), high_edge=(ijet==len(jet_bins)-1))

                if do_print:
                    bounds = (" or <" if ijet==0 else " or >" if ijet==len(jet_bins)-1 else "")
                    print(f"\t\t\tjet_filter: == {jet}{bounds}")

                # --- final selection ---------------------------------------------
                mask = tag_filter & sel_filter & jet_filter
                hemi_count_data[tag][sel][jet] = len(hemi_data["nSelJet"][mask])

    return hemi_count_data




def study_hemis(hemifiles, tree_name="Events", year_str="UL18"):

    #
    #  Read in hemisphere data
    #
    branch_list = ["nJet", "nSelJet", "nTagJet", "sumPt_T_minor", "sumPt_T", "combinedMass", "pz"]
    #branch_list = ["nJet", "nSelJet", "nTagJet"]
    hemi_data = read_hemi_files(hemifiles, tree_name=tree_name, branch_list=branch_list)


    #
    # Count the hemispheres by nTagJet, nSelJet, nJet
    #
    hemi_counts = count_all_hemispheres(hemi_data, do_print=False)


    #
    #  Apply thresholds to get ranges
    #
    hemi_ranges = get_hemi_ranges(hemi_counts, threshold=300)

    combined_hemi_counts = count_combined_hemispheres(hemi_ranges, hemi_data, do_print=False)

    #
    # Check we got them all!
    #
    total = 0
    nHemiLibraries = 0
    for tag in combined_hemi_counts.keys():
        this_tag = sum(v for outer in combined_hemi_counts[tag].values() for v in outer.values())
        print(f"tag={tag}, {this_tag}")
        total += this_tag
        nHemiLibraries += len([v for outer in combined_hemi_counts[tag].values() for v in outer.values()])

    print("total hemisphere =", total)
    print("nHemiLibraries =", nHemiLibraries)
    print("combined counts", total, "vs total counts", sum(s[1] for s in hemi_counts["tagJet_counts"]))


    #
    #  Get summary data by grouping
    #
    hemi_vars=["sumPt_T_minor", "sumPt_T", "combinedMass", "pz"]
    grouped_hemi_data = get_grouped_hemispheres_data(hemi_ranges, hemi_data, hemi_vars=hemi_vars)

    #
    #  Make histograms
    #
    binning = {"sumPt_T_minor": (50, 0, 500),
               "sumPt_T":       (50, 0, 1000),
               "pz":            (50, -1500, 1500),
               "combinedMass":  (50, 0, 2000),
               }

    output_path = "coffea4bees/hemisphere_mixing/hemi_plots"
    os.makedirs(output_path, exist_ok=True)
    print(f"Saveing plots to {output_path}")

    hemi_statistics = {}
    hemi_statistics["jet_mult_ranges"] = hemi_ranges
    hemi_statistics["hemi_summary_vars"] = {}


    for jet_mult_key in grouped_hemi_data.keys():

        tag, sel, jet = jet_mult_key

        hemi_statistics["hemi_summary_vars"][jet_mult_key] = {"count": len(grouped_hemi_data[jet_mult_key][hemi_vars[0]])}

        for var_name in hemi_vars:

            output_dir = f"{output_path}/nTag{tag}_nSel{sel}_nJet{jet}/"
            os.makedirs(output_dir, exist_ok=True)
            this_hist = hist.Hist(hist.axis.Regular(*binning[var_name], name=var_name, label=var_name))
            this_hist.fill(grouped_hemi_data[jet_mult_key][var_name])
            this_hist.plot()
            plt.savefig(f"{output_dir}/{var_name}.pdf")
            plt.close()


            _hemi_var_mean = np.mean(grouped_hemi_data[jet_mult_key][var_name])
            _hemi_var_RMS  = np.sqrt(np.mean(grouped_hemi_data[jet_mult_key][var_name]**2))
            hemi_statistics["hemi_summary_vars"][jet_mult_key][var_name] = {"mean": float(_hemi_var_mean), "RMS": float(_hemi_var_RMS)}

            this_hist = hist.Hist(hist.axis.Regular(50, -3, 3, name=f"zscore {var_name}", label=var_name))
            this_hist.fill( (grouped_hemi_data[jet_mult_key][var_name] - _hemi_var_mean) / _hemi_var_RMS)  # for z-score, divide by RMS if needed
            this_hist.plot()
            plt.savefig(f"{output_dir}/zscore_{var_name}.pdf")
            plt.close()


    # Extract all count values from hemi_statistics
    counts = []
    for key, value in hemi_statistics["hemi_summary_vars"].items():
        counts.append(value["count"])

    # Sort counts from low to high
    counts.sort()

    # Create histogram of counts
    count_hist = hist.Hist(hist.axis.Regular(100, 300, 300*100, name="count", label="Event Count"))
    count_hist.fill(counts)
    count_hist.plot()
    plt.xlabel("Hemisphere Counts")
    plt.ylabel("Number of Hemi Libraries")
    plt.title("Distribution of Event Counts")
    plt.savefig(f"{output_path}/count_distribution.pdf")
    plt.close()

    def make_yaml_safe(d):
        #print("Making YAML safe...",d)
        import numpy as np, awkward as ak
        if isinstance(d, dict):
            return {str(k) : make_yaml_safe(v) for k, v in d.items()}  # <-- cast keys to str
        elif isinstance(d, (list, tuple)):
            return [make_yaml_safe(v) for v in d]
#        elif isinstance(d, np.generic):
#            return d.item()
#        elif isinstance(d, np.ndarray):
#            return d.tolist()
#        elif "awkward" in type(d).__module__:
#            return ak.to_list(d)
        else:
            return d

    # Save hemi statistics to a YAML file
    with open(f'{output_path}/hemi_statistics_{year_str}.yml', 'w') as hemi_stats_yaml_file:
        yaml.dump(make_yaml_safe(hemi_statistics), hemi_stats_yaml_file) #, default_flow_style=False)


def doStudy():
    year_str = "UL18"
    study_hemis(hemifiles = f"output/mixeddata_cluster/data_{year_str}*/*.root", year_str=year_str)


if __name__ == "__main__":
    doStudy()
