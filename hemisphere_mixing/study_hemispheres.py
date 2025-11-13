import uproot
import numpy as np
import hist
import matplotlib
matplotlib.use("Agg")  # no GUI, renders directly to files
import matplotlib.pyplot as plt



def read_hemi_files(hemifiles, tree_name="Events", branch_list=None):

    hemi_vars = { var_name: [] for var_name in branch_list }

    for batch in uproot.iterate(
            f"{hemifiles}:{tree_name}",  #"coffea4bees/hemisphere_mixing/tests/*.root:Events",
            branch_list,
            step_size=200_000,  # entries per chunk
            library="np",
    ):

        #nTag2Sel2Jet2 = ( (batch["nTagJet"] == 2) & (batch["nSelJet"] == 2) & (batch["nJet"] == 2) )

        if hemi_vars[branch_list[0]] is None:
            for var_name in branch_list:
                hemi_vars[var_name] = batch[var_name]
        else:
            for var_name in branch_list:
                hemi_vars[var_name] = np.concatenate( (hemi_vars[var_name], batch[var_name]) )

    return hemi_vars


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


def get_hemi_ranges(hemi_count_data, threshold=200):

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


def get_filter(data, key, val, low_edge=False, high_edge=False):
    this_filter = (data[key] == val)

    if low_edge:
        this_filter |= (data[key] < val)
    elif high_edge:
        this_filter |= (data[key] > val)

    return this_filter

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



def get_grouped_hemispheres_data(hemi_ranges, hemi_data, hemi_vars):
    grouped_hemi_data = {}

    # Outer loop: tag multiplicity bins
    tag_keys = list(hemi_ranges.keys())
    for itag, tag in enumerate(tag_keys):

        grouped_hemi_data[tag] = {}

        # --- tag filter ----------------------------------------------------------
        tag_filter = get_filter(hemi_data, "nTagJet", tag, low_edge=(itag==0), high_edge=(itag==len(tag_keys)-1))

        # skip empty sub-ranges
        if not hemi_ranges[tag]:
            print(f"ERROR: no sel jets for tag = {tag}")
            continue

        # -------------------------------------------------------------------------
        # Middle loop: selected-jet multiplicity bins
        sel_keys = list(hemi_ranges[tag].keys())
        for isel, sel in enumerate(sel_keys):

            grouped_hemi_data[tag][sel] = {}

            # --- sel filter ------------------------------------------------------
            sel_filter = get_filter(hemi_data, "nSelJet", sel, low_edge=(isel==0), high_edge=(isel==len(sel_keys)-1))

            # ---------------------------------------------------------------------
            # Inner loop: total-jet multiplicity bins
            jet_bins = hemi_ranges[tag][sel]
            if not jet_bins:
                # special case: no jet bins defined
                grouped_hemi_data[tag][sel][-1] = {}
                for var_name in hemi_vars:
                    grouped_hemi_data[tag][sel][-1][var_name] = hemi_data[var_name][tag_filter & sel_filter]

                continue

            for ijet, jet in enumerate(jet_bins):

                jet_filter = get_filter(hemi_data, "nJet", jet, low_edge=(ijet==0), high_edge=(ijet==len(jet_bins)-1))

                # --- final selection ---------------------------------------------
                mask = tag_filter & sel_filter & jet_filter
                grouped_hemi_data[tag][sel][jet] = {}
                for var_name in hemi_vars:
                    grouped_hemi_data[tag][sel][jet][var_name] = hemi_data[var_name][tag_filter & sel_filter & jet_filter]


    return grouped_hemi_data




def study_hemis(hemifiles, tree_name="Events"):

    #
    #  Read in hemisphere data
    #
    branch_list = ["nJet", "nSelJet", "nTagJet", "sumPt_T_minor", "sumPt_T", "combinedMass", "pz"]
    #branch_list = ["nJet", "nSelJet", "nTagJet"]
    hemi_data = read_hemi_files(hemifiles, tree_name=tree_name, branch_list=branch_list)


    #
    # Count the hemispheres by nTagJet, nSelJet, nJet
    #
    hemi_counts = count_all_hemispheres(hemi_data, do_print=True)


    #
    #  Apply thresholds to get ranges
    #
    hemi_ranges = get_hemi_ranges(hemi_counts, threshold=200)

    combined_hemi_counts = count_combined_hemispheres(hemi_ranges, hemi_data, do_print=False)

    total = 0
    for tag in combined_hemi_counts.keys():
        this_tag = sum(v for outer in combined_hemi_counts[tag].values() for v in outer.values())
        print(f"tag={tag}, {this_tag}")
        total += this_tag
    print("total =", total)

    #
    # Check we got them all!
    #
    print("combined counts", total, "vs total counts", sum(s[1] for s in hemi_counts["tagJet_counts"]))


    #
    #  Get summary data by grouping
    #
    hemi_vars=["sumPt_T_minor", "sumPt_T", "combinedMass", "pz"]
    grouped_hemi_data = get_grouped_hemispheres_data(hemi_ranges, hemi_data, hemi_vars=hemi_vars)

    print("making Histogrmas")

    #
    #  Make histograms
    #
    binning = {"sumPt_T_minor": (50, -100, 100),
               "sumPt_T":       (50, -1000, 1000),
               "pz":            (50, -1500, 1500),
               "combinedMass":  (50, 0, 2000),
               }

    for tag in grouped_hemi_data.keys():
        print(f"tag = {tag}")
        for sel in grouped_hemi_data[tag].keys():
            print(f"\tsel = {sel}")
            for jet in grouped_hemi_data[tag][sel].keys():
                for var_name in hemi_vars:
                    this_hist = hist.Hist(hist.axis.Regular(*binning[var_name], name=var_name, label=var_name))
                    this_hist.fill(grouped_hemi_data[tag][sel][jet][var_name])
                    this_hist.plot()
                    plt.savefig(f"hemi_nTag{tag}_nSel{sel}_nJet{jet}_{var_name}.pdf")
                    plt.close()
    #Hist(Regular(50, -100, 100, name='x', label='x variable'), storage=Double())

def doStudy():
    study_hemis(hemifiles = "output/mixeddata_cluster/data_UL18*/*.root")


if __name__ == "__main__":
    doStudy()
