"""
Check a picoAOD ROOT file for NaN values in Jet four-vector fields.
Reports the fraction of jets with NaN in each field.

Usage:
    python coffea4bees/hemisphere_mixing/check_picoAOD_nans.py <picoAOD.root> [<picoAOD2.root> ...]
"""
import sys
import numpy as np
import awkward as ak
import uproot

JET_FIELDS = ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass"]


def check_file(path):
    print(f"\n{'='*70}")
    print(f"File: {path}")
    try:
        with uproot.open(f"{path}:Events") as tree:
            n_events = tree.num_entries
            nJet_arr = tree["nJet"].array(library="np")
            n_jets_total = int(nJet_arr.sum())
            print(f"  Events: {n_events}   Total jets: {n_jets_total}")

            any_nan = np.zeros(n_events, dtype=bool)

            for field in JET_FIELDS:
                flat = ak.to_numpy(ak.flatten(tree[field].array(library="ak")))
                nan_mask = np.isnan(flat)
                n_nan = nan_mask.sum()
                frac = n_nan / n_jets_total if n_jets_total > 0 else 0.0
                print(f"  {field:<12}: {n_nan:>6} NaN jets / {n_jets_total}  ({100*frac:.4f}%)")

                if field == "Jet_mass" and n_nan > 0:
                    # Find which events have NaN mass jets
                    run_arr   = tree["run"].array(library="np")
                    event_arr = tree["event"].array(library="np")
                    nJet_arr2 = tree["nJet"].array(library="np")
                    pt_arr    = ak.to_numpy(ak.flatten(tree["Jet_pt"].array(library="ak")))
                    eta_arr   = ak.to_numpy(ak.flatten(tree["Jet_eta"].array(library="ak")))
                    mass_arr  = ak.to_numpy(ak.flatten(tree["Jet_mass"].array(library="ak")))

                    # Expand nJet per event to per-jet event index
                    evt_idx = np.repeat(np.arange(n_events), nJet_arr2)
                    nan_evt_indices = np.unique(evt_idx[nan_mask])

                    print(f"\n  First 5 events with NaN Jet_mass:")
                    for ei in nan_evt_indices[:5]:
                        start = int(nJet_arr2[:ei].sum())
                        end   = start + int(nJet_arr2[ei])
                        jets_pt   = pt_arr[start:end]
                        jets_eta  = eta_arr[start:end]
                        jets_mass = mass_arr[start:end]
                        nan_j = np.where(np.isnan(jets_mass))[0]
                        print(f"    run={run_arr[ei]} event={event_arr[ei]} nJet={int(nJet_arr2[ei])}")
                        for j in nan_j:
                            print(f"      Jet[{j}]: pt={jets_pt[j]:.2f} eta={jets_eta[j]:.3f} mass=NaN")

    except Exception as ex:
        print(f"  ERROR: {ex}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    for path in sys.argv[1:]:
        check_file(path)
