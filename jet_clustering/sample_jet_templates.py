import numpy as np
import awkward as ak
import sys
try:
    from src.math_tools.random import Squares
except:
    print("Warning ... Squares not availible")
    pass


#def sample_PDFs(input_jets_decluster, input_pdfs, splittings):
#
#    n_jets   = np.sum(ak.num(input_jets_decluster))
#
#    #
#    #  Sample the PDFs for the jets we will uncluster
#    #
#    for _var_name in input_pdfs["varNames"]:
#
#        if _var_name.find("_vs_") == -1:
#            is_1d_pdf = True
#            _sampled_data = np.ones(n_jets)
#        else:
#            is_1d_pdf = False
#            _sampled_data_x = np.ones(n_jets)
#            _sampled_data_y = np.ones(n_jets)
#
#        # Sample the pdfs from the different splitting options
#        for _splitting_name, _num_samples, _indicies_tuple in splittings:
#
#            if is_1d_pdf:
#                probs   = np.array(input_pdfs[_splitting_name][_var_name]["probs"], dtype=float)
#                centers = np.array(input_pdfs[_splitting_name][_var_name]["bin_centers"], dtype=float)
#                _sampled_data[_indicies_tuple] = np.random.choice(centers, size=_num_samples, p=probs)
#            else:
#                probabilities_flat   = np.array(input_pdfs[_splitting_name][_var_name]["probabilities_flat"], dtype=float)
#                xcenters        = np.array(input_pdfs[_splitting_name][_var_name]["xcenters"],      dtype=float)
#                ycenters        = np.array(input_pdfs[_splitting_name][_var_name]["ycenters"],      dtype=float)
#
#                xcenters_flat = np.repeat(xcenters, len(ycenters))
#                ycenters_flat = np.tile(ycenters, len(xcenters))
#
#                sampled_indices = np.random.choice(len(probabilities_flat), size=_num_samples, p=probabilities_flat)
#
#                _sampled_data_x[_indicies_tuple] = xcenters_flat[sampled_indices]
#                _sampled_data_y[_indicies_tuple] = ycenters_flat[sampled_indices]
#
#        #
#        # Save the sampled data to the jets to be uclustered for use in decluster_combined_jets
#        #
#        if is_1d_pdf:
#            input_jets_decluster[_var_name]         = ak.unflatten(_sampled_data,    ak.num(input_jets_decluster))
#        else:
#            input_jets_decluster["zA"]         = ak.unflatten(_sampled_data_x,    ak.num(input_jets_decluster))
#            input_jets_decluster["thetaA"]     = ak.unflatten(_sampled_data_y,    ak.num(input_jets_decluster))


def sample_PDFs_vs_pT(input_jets_decluster, input_pdfs, rand_seed, splittings, chunk=None, debug=False):

    n_jets   = np.sum(ak.num(input_jets_decluster))

    if debug:
        print(f"{chunk} sample_PDFs_vs_pT n_jets {n_jets}\n")

    n_pt_bins = len(input_pdfs["pt_bins"]) - 1
    pt_masks = []
    for iPt in range(n_pt_bins):
        min_pt = float(input_pdfs["pt_bins"][iPt])
        max_pt = float(input_pdfs["pt_bins"][iPt + 1]) if input_pdfs["pt_bins"][iPt + 1] != "inf" else np.inf
        pt_masks.append((input_jets_decluster.pt > min_pt) & (input_jets_decluster.pt < max_pt))


    if debug:
        print(f"{chunk} len pt_masks  {len(pt_masks)}\n")

    #
    #  Sample the PDFs for the jets we will uncluster
    #
    for iVar, var_name in enumerate(input_pdfs["varNames"]):

        is_1d_pdf = "_vs_" not in var_name

        # Initialize sample storage arrays
        if is_1d_pdf:
            sampled_data = np.ones(n_jets)
            sampled_data_vs_pT = [np.ones(n_jets) for _ in range(n_pt_bins)]
        else:
            sampled_data_x = np.ones(n_jets)
            sampled_data_y = np.ones(n_jets)
            sampled_data_x_vs_pT = [np.ones(n_jets) for _ in range(n_pt_bins)]
            sampled_data_y_vs_pT = [np.ones(n_jets) for _ in range(n_pt_bins)]


        # Sample the pdfs from the different splitting options
        for splitting_name, num_samples, indicies_tuple in splittings:
            if debug:
                print(f"{chunk} sample_PDFs_vs_pT {iVar} {var_name} {splitting_name} {num_samples} {len(indicies_tuple)}\n")
                print(f"{chunk} len jets {np.sum(ak.num(input_jets_decluster))}\n")
                print(f"{chunk} pt: {len(input_jets_decluster.pt)}\n")
                print(f"{chunk} pt_flat: {len(ak.flatten(input_jets_decluster.pt))}\n")
                print(input_jets_decluster.pt)

            # Get jet kinematics for random seeding
            pts  = ak.flatten(input_jets_decluster.pt) [indicies_tuple]
            etas = ak.flatten(input_jets_decluster.eta)[indicies_tuple]
            phis = ak.flatten(input_jets_decluster.phi)[indicies_tuple]

            # Create counter for RNG seeding
            counter = np.zeros((num_samples, 3), dtype=np.uint64)
            counter[:, 0] = np.round(np.asarray(pts),  1).view(np.uint64)
            counter[:, 1] = np.round(np.asarray(etas), 3).view(np.uint64)
            counter[:, 2] = np.round(np.asarray(phis), 3).view(np.uint64)

            rng = Squares("sample_jet_templates", iVar, rand_seed, splitting_name)


            # Handle missing splitting names
            if splitting_name not in input_pdfs:
                old_splitting_name = splitting_name
                splitting_name = list(input_pdfs.keys())[-1]
                print(f"ERROR {old_splitting_name} not in inputPDFs using last splitting {splitting_name}")


            for iPt in range(n_pt_bins):

                if is_1d_pdf:
                    probs   = np.array(input_pdfs[splitting_name][var_name][iPt]["probs"],       dtype=float)
                    centers = np.array(input_pdfs[splitting_name][var_name][iPt]["bin_centers"], dtype=float)
                    sampled_data_vs_pT[iPt][indices_tuple] = rng.choice(counter, a=centers, p=probs).astype(np.float32)

                else:
                    probabilities_flat = np.array(input_pdfs[splitting_name][var_name][iPt]["probabilities_flat"], dtype=float)
                    xcenters           = np.array(input_pdfs[splitting_name][var_name][iPt]["xcenters"],           dtype=float)
                    ycenters           = np.array(input_pdfs[splitting_name][var_name][iPt]["ycenters"],           dtype=float)

                    xcenters_flat = np.repeat(xcenters, len(ycenters))
                    ycenters_flat = np.tile(ycenters, len(xcenters))

                    sampled_indices = rng.choice(counter, a=len(probabilities_flat), p=probabilities_flat)
                    sampled_data_x_vs_pT[iPt][indices_tuple] = xcenters_flat[sampled_indices]
                    sampled_data_y_vs_pT[iPt][indices_tuple] = ycenters_flat[sampled_indices]

            #
            #  Now work out which pT bins to use
            #    (Combine pT bins into final sampled data)
            if is_1d_pdf:

                for iPt in range(n_pt_bins):
                    pt_indices = np.where(ak.flatten(pt_masks[iPt]))[0]
                    sampled_data[pt_indices] = sampled_data_vs_pT[iPt][pt_indices]

                input_jets_decluster[var_name] = ak.unflatten(sampled_data, ak.num(input_jets_decluster))

            else:

                for iPt in range(n_pt_bins):
                    pt_indices = np.where(ak.flatten(pt_masks[iPt]))[0]
                    sampled_data_x[pt_indices] = sampled_data_x_vs_pT[iPt][pt_indices]
                    sampled_data_y[pt_indices] = sampled_data_y_vs_pT[iPt][pt_indices]

                input_jets_decluster["zA"]     = ak.unflatten(sampled_data_x, ak.num(input_jets_decluster))
                input_jets_decluster["thetaA"] = ak.unflatten(sampled_data_y, ak.num(input_jets_decluster))
