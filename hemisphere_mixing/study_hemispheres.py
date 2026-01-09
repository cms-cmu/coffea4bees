import uproot
import numpy as np
import hist
import matplotlib
matplotlib.use("Agg")  # no GUI, renders directly to files
import matplotlib.pyplot as plt
import os
import yaml
import sys
import argparse
sys.path.insert(0, os.getcwd())
from coffea4bees.hemisphere_mixing.mixing_helpers   import read_hemi_files, get_grouped_hemispheres_data, get_filter, iter_hemi_filters


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

    for jet_mult_key, mask in iter_hemi_filters(hemi_ranges, hemi_data):

        hemi_count_data[jet_mult_key] = len(hemi_data["nSelJet"][mask])

    return hemi_count_data




def study_hemis(hemifiles, tree_name="Events", year_str="UL18", do_plots=False, output_path="coffea4bees/hemisphere_mixing/hemi_plots/", threshold=300):

    print(f"Studying hemispheres from files: {hemifiles} with threshold {threshold}")
    print(f"Output path: {output_path}")

    #
    #  Read in hemisphere data
    #
    branch_list = ["nJet", "nSelJet", "nTagJet", "sumPt_T_minor", "sumPt_T", "combinedMass", "pz"]

    hemi_data = read_hemi_files(hemifiles, year=year_str, tree_name=tree_name, branch_list=branch_list)


    #
    # Count the hemispheres by nTagJet, nSelJet, nJet
    #
    hemi_counts = count_all_hemispheres(hemi_data, do_print=False)

    #
    #  Apply thresholds to get ranges
    #
    hemi_ranges = get_hemi_ranges(hemi_counts, threshold=threshold)

    combined_hemi_counts = count_combined_hemispheres(hemi_ranges, hemi_data, do_print=False)

    #
    # Check we got them all!
    #
    total = sum(combined_hemi_counts.values())
    nHemiLibraries = len(combined_hemi_counts.values())

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


    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/{year_str}", exist_ok=True)
    if do_plots:
        print(f"Saveing plots to {output_path}/{year_str}")

    hemi_statistics = {}
    hemi_statistics["jet_mult_ranges"] = hemi_ranges
    hemi_statistics["hemi_summary_vars"] = {}


    for jet_mult_key in grouped_hemi_data.keys():

        tag, sel, jet = jet_mult_key

        hemi_statistics["hemi_summary_vars"][jet_mult_key] = {"count": len(grouped_hemi_data[jet_mult_key][hemi_vars[0]])}

        for var_name in hemi_vars:

            _hemi_var_mean = np.mean(grouped_hemi_data[jet_mult_key][var_name])
            _hemi_var_RMS  = np.sqrt(np.mean(grouped_hemi_data[jet_mult_key][var_name]**2))
            hemi_statistics["hemi_summary_vars"][jet_mult_key][var_name] = {"mean": float(_hemi_var_mean), "RMS": float(_hemi_var_RMS)}

            if do_plots:
                output_dir = f"{output_path}/{year_str}/nTag{tag}_nSel{sel}_nJet{jet}/"
                os.makedirs(output_dir, exist_ok=True)
                this_hist = hist.Hist(hist.axis.Regular(*binning[var_name], name=var_name, label=var_name))
                this_hist.fill(grouped_hemi_data[jet_mult_key][var_name])
                this_hist.plot()
                plt.savefig(f"{output_dir}/{var_name}.pdf")
                plt.close()



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
    if do_plots:
        count_hist = hist.Hist(hist.axis.Regular(100, 300, 300*100, name="count", label="Event Count"))
        count_hist.fill(counts)
        count_hist.plot()
        plt.xlabel("Hemisphere Counts")
        plt.ylabel("Number of Hemi Libraries")
        plt.title("Distribution of Event Counts")
        plt.savefig(f"{output_path}/{year_str}/count_distribution.pdf")
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
    print(f'writting {output_path}/hemi_statistics_{year_str}.yml')
    with open(f'{output_path}/hemi_statistics_{year_str}.yml', 'w') as hemi_stats_yaml_file:
        yaml.dump(make_yaml_safe(hemi_statistics), hemi_stats_yaml_file) #, default_flow_style=False)


def doStudy():

    parser = argparse.ArgumentParser(description='study_hemispheres', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hemifiles', default="coffea4bees/skimmer/metadata/hemisphere_library_noTT.yml" )
    parser.add_argument('--year', default="UL18", nargs='+')
    parser.add_argument('--threshold', default=300)
    #parser.add_argument('--m4j_xmax', default=1200)
    #parser.add_argument('--variable_binning', action="store_true")
    parser.add_argument('--output_path', default="coffea4bees/hemisphere_mixing/hemi_plots")
    parser.add_argument('--do_plots',   action="store_true")

    args = parser.parse_args()
    print(f"\nRunning with these parameters: {args}")

    #hemifiles = "output/mixeddata_cluster/data_UL18*/*.root"
    for year in args.year:
        study_hemis(hemifiles = args.hemifiles, year_str=year, do_plots=args.do_plots, output_path=args.output_path, threshold=int(args.threshold))


if __name__ == "__main__":
    doStudy()
