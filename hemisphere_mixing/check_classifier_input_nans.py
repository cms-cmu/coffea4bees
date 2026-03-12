"""
Check HCR classifier input ROOT files for NaN values.

Checks CanJet, NotCanJet (excluding -1 padding), and ancillary branches.

Usage:
    python coffea4bees/hemisphere_mixing/check_classifier_input_nans.py <file.root> [<file2.root> ...]

Paths starting with /store/ are auto-prefixed with root://cmseos.fnal.gov/.
"""
import sys
import numpy as np
import awkward as ak
import uproot


CANJET_FIELDS    = ["CanJet_pt", "CanJet_eta", "CanJet_phi", "CanJet_mass"]
NOTCANJET_FIELDS = ["NotCanJet_pt", "NotCanJet_eta", "NotCanJet_phi", "NotCanJet_mass"]
ANCILLARY_FIELDS = ["year", "nSelJets", "xW", "xbW"]


def resolve_path(path):
    if path.startswith("/store/"):
        return "root://cmseos.fnal.gov/" + path
    return path


def check_jagged_field(tree, field, n_events, pad_value=None):
    """Check a jagged branch for NaN. Optionally exclude pad_value entries."""
    flat = ak.to_numpy(ak.flatten(tree[field].array(library="ak")))
    if pad_value is not None:
        flat = flat[flat != pad_value]
    n_total = len(flat)
    n_nan   = int(np.isnan(flat).sum())
    frac    = n_nan / n_total if n_total > 0 else 0.0
    return n_nan, n_total, frac


def check_flat_field(tree, field):
    """Check a per-event flat branch for NaN."""
    arr   = tree[field].array(library="np")
    n_nan = int(np.isnan(arr.astype(float)).sum())
    return n_nan, len(arr)


def check_file(path):
    xrd_path = resolve_path(path)
    print(f"\n{'='*70}")
    print(f"File: {path}")
    try:
        with uproot.open(f"{xrd_path}:Events") as tree:
            n_events = tree.num_entries
            print(f"  Events: {n_events}")

            any_nan = False

            # --- CanJet (always 4 per event, no padding) ---
            print(f"\n  CanJet (4 jets/event, no padding):")
            for field in CANJET_FIELDS:
                n_nan, n_total, frac = check_jagged_field(tree, field, n_events)
                flag = " <-- NaN!" if n_nan > 0 else ""
                print(f"    {field:<22}: {n_nan:>6} / {n_total}  ({100*frac:.4f}%){flag}")
                if n_nan > 0:
                    any_nan = True

            # --- NotCanJet (variable, padded with -1) ---
            print(f"\n  NotCanJet (variable, -1 padding excluded):")
            for field in NOTCANJET_FIELDS:
                n_nan, n_total, frac = check_jagged_field(tree, field, n_events, pad_value=-1)
                flag = " <-- NaN!" if n_nan > 0 else ""
                print(f"    {field:<22}: {n_nan:>6} / {n_total}  ({100*frac:.4f}%){flag}")
                if n_nan > 0:
                    any_nan = True

            # --- Ancillary (flat per-event) ---
            print(f"\n  Ancillary (per-event scalars):")
            for field in ANCILLARY_FIELDS:
                if field not in tree:
                    print(f"    {field:<22}: (not present)")
                    continue
                n_nan, n_total = check_flat_field(tree, field)
                frac = n_nan / n_total if n_total > 0 else 0.0
                flag = " <-- NaN!" if n_nan > 0 else ""
                print(f"    {field:<22}: {n_nan:>6} / {n_total}  ({100*frac:.4f}%){flag}")
                if n_nan > 0:
                    any_nan = True

            print(f"\n  Summary: {'NaN values found!' if any_nan else 'No NaN values found.'}")

            # --- Detail: show first few NaN CanJet_mass events ---
            if "CanJet_mass" in tree:
                mass_flat = ak.to_numpy(ak.flatten(tree["CanJet_mass"].array(library="ak")))
                n_nan_mass = int(np.isnan(mass_flat).sum())
                if n_nan_mass > 0:
                    nJet_arr  = np.full(n_events, 4, dtype=int)  # CanJet always 4
                    pt_flat   = ak.to_numpy(ak.flatten(tree["CanJet_pt"].array(library="ak")))
                    eta_flat  = ak.to_numpy(ak.flatten(tree["CanJet_eta"].array(library="ak")))
                    evt_arr   = tree["event"].array(library="np") if "event" in tree else np.arange(n_events)
                    evt_idx   = np.repeat(np.arange(n_events), nJet_arr)
                    nan_evt_indices = np.unique(evt_idx[np.isnan(mass_flat)])
                    print(f"\n  First 5 events with NaN CanJet_mass:")
                    for ei in nan_evt_indices[:5]:
                        start = ei * 4
                        end   = start + 4
                        jets_pt   = pt_flat[start:end]
                        jets_eta  = eta_flat[start:end]
                        jets_mass = mass_flat[start:end]
                        nan_j = np.where(np.isnan(jets_mass))[0]
                        print(f"    event={evt_arr[ei]}  nCanJet=4")
                        for j in nan_j:
                            print(f"      CanJet[{j}]: pt={jets_pt[j]:.2f} eta={jets_eta[j]:.3f} mass=NaN")

    except Exception as ex:
        print(f"  ERROR: {ex}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    for path in sys.argv[1:]:
        check_file(path)
