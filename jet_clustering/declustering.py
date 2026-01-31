import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector
from coffea4bees.jet_clustering.sample_jet_templates import sample_PDFs_vs_pT

# _MAX_NUM_JET_RETRY = 4
# _MAX_NUM_EVENT_RETRY = 4

_MAX_NUM_JET_RETRY   = 8
_MAX_NUM_EVENT_RETRY = 8


def extract_all_parentheses_substrings(s):
    substrings = []
    start_indices = []
    counter = 0

    for i, char in enumerate(s):
        if char == '(':
            if counter == 0:
                start_indices.append(i)
            counter += 1
        elif char == ')':
            counter -= 1
            if counter == 0:
                start_index = start_indices.pop(0)
                substrings.append(s[start_index:i + 1])

    return substrings


def extract_outermost_pair(s):
    # Trim leading and trailing whitespaces and ensure it starts with '(' and ends with ')'
    s = s.strip()

    if not (s.startswith("(") and s.endswith(")")):
        raise ValueError("Input must start with '(' and end with ')'")

    # We will count the open and close parentheses
    open_count = 0
    comma_index = -1  # To store the position of the main separating comma

    # Traverse the string to find the outermost comma
    for i, char in enumerate(s):
        if char == '(':
            open_count += 1
        elif char == ')':
            open_count -= 1
        elif char == ',' and open_count == 1:  # The outermost comma is when open_count == 1
            comma_index = i
            break

    if comma_index == -1:
        raise ValueError("Input doesn't seem to have a valid '(A, B)' format")

    # Extract A and B by splitting at the found comma
    A = s[1:comma_index].strip()   # Everything after '(' up to the comma
    B = s[comma_index + 1:-1].strip()  # Everything after the comma up to ')'

    return A, B


def children_jet_flavors(comb_flavor):

    if len(comb_flavor) < 2:
        print(f"ERROR len of combined flavor is too low {len(comb_flavor)}  {comb_flavor}")

    sub_combs = extract_all_parentheses_substrings(comb_flavor)

    if len(sub_combs) == 0:
        child_A = comb_flavor[0]
        child_B = comb_flavor[1]
    elif len(sub_combs) == 1:
        child_A = sub_combs[0][1:-1]    # the 1:-1 remove the leading and trailing parems
        child_B = str(comb_flavor).replace(sub_combs[0], "")
    elif len(sub_combs) == 2:
        child_A = sub_combs[0][1:-1]
        child_B = sub_combs[1][1:-1]
    else:
        print(f"ERROR comb_flavor is {comb_flavor} sub_combs is {sub_combs} len {len(sub_combs)}")

    return child_A, child_B


def get_splitting_summary(comb_flavor):

    childA, childB = children_jet_flavors(comb_flavor)

    n_b_A = childA.count("b")
    n_j_A = childA.count("j")

    n_b_B = childB.count("b")
    n_j_B = childB.count("j")

    return (n_b_A, n_j_A), (n_b_B, n_j_B)


def get_splitting_name(comb_flavor):

    A_stats, B_stats = get_splitting_summary(comb_flavor)

    nA = A_stats[0] + A_stats[1]
    nB = B_stats[0] + B_stats[1]

    #  X / X
    if nA > 3 and nB > 2:
        return "X/X"

    #  3 / 3
    if nA == 3 and nB == 3:
        return f"3/3"

    # 3/2,  4/2,  and X/2
    if nA > 2 and nB > 1:
        if nA > 4:
            return f"X/{nB}"

        return f"{nA}/{nB}"

    # 4/1, X,1
    if nA > 3 and nB > 0:
        if nA > 4:
            return f"X/{nB}"

        return f"{nA}/{nB}"

    return f"{A_stats[0]}b{A_stats[1]}j/{B_stats[0]}b{B_stats[1]}j"


def get_list_of_combined_jet_types(jets):
    """
      returns a list of all the splitting types that are the results of a combination
        (ie: no b or j )
    """
    all_jet_types = get_list_of_splitting_types(jets)
    splitting_types = []
    for _s in all_jet_types:

        if len(_s) == 1:
            continue

        splitting_types.append(_s)

    splitting_types.sort()
    return splitting_types


def get_list_of_all_sub_splittings(splitting):
    """
      returns a list of all the sub splitting types (including the original)
    """
    if len(splitting) > 1:
        childA, childB = children_jet_flavors(splitting)
        return [splitting] + get_list_of_all_sub_splittings(childA) + get_list_of_all_sub_splittings(childB)

    return []


def get_list_of_ISR_splittings(splitting_types):

    ISR_splittings = []
    for _s in splitting_types:

        if len(_s) == 1:
            continue

        child_A, child_B = children_jet_flavors(_s)

        child_A_nBs = child_A.count("b")
        child_B_nBs = child_B.count("b")

        #
        #  All splittings are ISR unless there is a b in both children
        #
        if (child_A_nBs > 0) and (child_B_nBs > 0):
            continue

        ISR_splittings.append(_s)
    ISR_splittings.sort()
    return ISR_splittings


def get_list_of_splitting_types(splittings):
    unique_splittings = set(ak.flatten(splittings.jet_flavor).to_list())
    unique_splittings_list = list(unique_splittings)
    unique_splittings_list.sort()
    return unique_splittings_list


def get_list_of_splitting_names(splittings):
    unique_splittings = set(ak.flatten(splittings.splitting_name).to_list())
    unique_splittings_list = list(unique_splittings)
    unique_splittings_list.sort()
    return unique_splittings_list


def compute_decluster_variables(clustered_splittings):

    #
    # z-axis
    #
    z_axis      = ak.zip({"x": 0, "y": 0, "z": 1}, with_name="ThreeVector", behavior=vector.behavior,)
    boost_vec_z = ak.zip({"x": 0, "y": 0, "z": clustered_splittings.boostvec.z}, with_name="ThreeVector", behavior=vector.behavior,)

    #
    #  Boost to pz0
    #
    clustered_splittings_pz0        = clustered_splittings.boost(-boost_vec_z)
    clustered_splittings_part_A_pz0 = clustered_splittings.part_A.boost(-boost_vec_z)
    clustered_splittings_part_B_pz0 = clustered_splittings.part_B.boost(-boost_vec_z)

    comb_z_plane_hat = z_axis.cross(clustered_splittings_pz0).unit
    decay_plane_hat = clustered_splittings_part_A_pz0.cross(clustered_splittings_part_B_pz0).unit

    #
    #  Clustering (calc variables to histogram)
    #

    clustered_splittings["zA_num"]    = clustered_splittings_pz0.dot(clustered_splittings_part_A_pz0)
    clustered_splittings["zA"]        = clustered_splittings_pz0.dot(clustered_splittings_part_A_pz0) / (clustered_splittings_pz0.pt**2)
    clustered_splittings["mA"]        = clustered_splittings.part_A.mass
    clustered_splittings["rhoA"]      = clustered_splittings.part_A.mass / clustered_splittings.part_A.pt
    clustered_splittings["abs_eta"]   = np.abs(clustered_splittings.eta)

    clustered_splittings["mB"]        = clustered_splittings.part_B.mass
    clustered_splittings["rhoB"]      = clustered_splittings.part_B.mass / clustered_splittings.part_B.pt

    clustered_splittings["thetaA"]    = np.arccos(clustered_splittings_pz0.unit.dot(clustered_splittings_part_A_pz0.unit))
    clustered_splittings["tan_thetaA"]    = np.tan(np.arccos(clustered_splittings_pz0.unit.dot(clustered_splittings_part_A_pz0.unit)))
    clustered_splittings["decay_phi"] = np.arccos(decay_plane_hat.dot(comb_z_plane_hat))
    clustered_splittings["dr_AB"]     = clustered_splittings.part_A.delta_r(clustered_splittings.part_B)
    clustered_splittings["dpt_AB"]    = clustered_splittings.part_A.pt - (clustered_splittings.pt * clustered_splittings.zA)
    clustered_splittings["rpt_A"]    = clustered_splittings.part_A.pt / clustered_splittings.pt
    clustered_splittings["rpt_B"]    = clustered_splittings.part_B.pt / clustered_splittings.pt
    clustered_splittings["rpt_AB"]    = clustered_splittings.part_B.pt / clustered_splittings.part_A.pt
    clustered_splittings["mass_AB"]     = (clustered_splittings.part_A + clustered_splittings.part_B).mass

    #
    #  The rest of the code Updates the mass in the rotated rest frame
    #

    #
    #  Go to the frame hwere the combined jet is pointing along X (needed to rotate the decay plane with rotateX)
    #
    clustered_splittings_part_A_pz0_phi0  = rotateZ(clustered_splittings_part_A_pz0, -clustered_splittings.phi)
    clustered_splittings_part_B_pz0_phi0  = rotateZ(clustered_splittings_part_B_pz0, -clustered_splittings.phi)

    #
    #    we either need to rotate back by + or - decay phi, figure out which one
    #
    clustered_splittings_part_A_pz0_phi0_dphi0  = rotateX(clustered_splittings_part_A_pz0_phi0, -clustered_splittings.decay_phi)
    clustered_splittings_part_B_pz0_phi0_dphi0  = rotateX(clustered_splittings_part_B_pz0_phi0, -clustered_splittings.decay_phi)
    #decay_plane_dphi0 = clustered_splittings_part_A_pz0_phi0_dphi0.cross(clustered_splittings_part_B_pz0_phi0_dphi0).unit

    clustered_splittings_part_A_pz0_phi0_pdphi0  = rotateX(clustered_splittings_part_A_pz0_phi0, +clustered_splittings.decay_phi)
    clustered_splittings_part_B_pz0_phi0_pdphi0  = rotateX(clustered_splittings_part_B_pz0_phi0, +clustered_splittings.decay_phi)
    decay_plane_pdphi0 = clustered_splittings_part_A_pz0_phi0_pdphi0.cross(clustered_splittings_part_B_pz0_phi0_pdphi0).unit

    pos_decay_phi_mask = np.abs(decay_plane_pdphi0.y - 1) < 0.001
    pos_decay_phi_mask_flat = ak.flatten(pos_decay_phi_mask)

    #
    # Get the pts in the frame we will do de-clustering
    #
    counts = ak.num(clustered_splittings)

    rotated_pt_A = ak.where(
        pos_decay_phi_mask_flat,
        ak.flatten(clustered_splittings_part_A_pz0_phi0_pdphi0.pt),
        ak.flatten(clustered_splittings_part_A_pz0_phi0_dphi0.pt)
    )
    rotated_pt_A = ak.unflatten(rotated_pt_A, counts)

    rotated_pt_B = ak.where(
        pos_decay_phi_mask_flat,
        ak.flatten(clustered_splittings_part_B_pz0_phi0_pdphi0.pt),
        ak.flatten(clustered_splittings_part_B_pz0_phi0_dphi0.pt)
    )
    rotated_pt_B = ak.unflatten(rotated_pt_B, counts)


    #
    #  Update the mass with rho and pt from the rotated frame
    #
    clustered_splittings["mA_rotated"]        = clustered_splittings.rhoA * rotated_pt_A
    clustered_splittings["mB_rotated"]        = clustered_splittings.rhoB * rotated_pt_B

    return


def rotateZ(particles, angle):
    sinT = np.sin(angle)
    cosT = np.cos(angle)
    x_rotated = cosT * particles.x - sinT * particles.y
    y_rotated = sinT * particles.x + cosT * particles.y

    return ak.zip(
        {
            "x": x_rotated,
            "y": y_rotated,
            "z": particles.z,
            "t": particles.t,
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )


def rotateX(particles, angle):
    sinT = np.sin(angle)
    cosT = np.cos(angle)
    y_rotated = cosT * particles.y - sinT * particles.z
    z_rotated = sinT * particles.y + cosT * particles.z

    return ak.zip(
        {
            "x": particles.x,
            "y": y_rotated,
            "z": z_rotated,
            "t": particles.t,
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )


def update_single_jet_mass(p, jet_flavor, rho, btag_string, counts):
    """Update mass for single jets using pt × rho."""
    jet_flavor_flat = ak.flatten(jet_flavor)
    single_jet_mask = (np.char.str_len(jet_flavor_flat) == 1)

    pt_flat   = ak.flatten(p.pt)
    mass_flat = ak.flatten(p.mass)
    rho_flat  = ak.flatten(rho)

    # Use mass = (pt x rho) for single jet clusters "b" or "j" use the mass for others
    p_mass = ak.unflatten(ak.where(single_jet_mask, pt_flat * rho_flat, mass_flat),
                          counts
                          )

    return ak.zip(
        {
            "pt": p.pt,
            "eta": p.eta,
            "phi": p.phi,
            "mass": p_mass,
            "jet_flavor": jet_flavor,
            "btag_string": btag_string,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )


def decluster_combined_jets(input_jet, debug=False):

    n_jets = np.sum(ak.num(input_jet))

    jet_flav_flat = ak.flatten(input_jet.jet_flavor)
    simple_comb_mask = (np.char.str_len(jet_flav_flat) == 2)

    # Build jet_flav_child lists
    jet_flav_child_A = []
    jet_flav_child_B = []

    for _, (flav, is_simple) in enumerate(zip(jet_flav_flat, simple_comb_mask)):
        if is_simple:
            jet_flav_child_A.append(str(flav)[0])
            jet_flav_child_B.append(str(flav)[1])
        else:
            child_A, child_B = children_jet_flavors(flav)
            jet_flav_child_A.append(child_A)
            jet_flav_child_B.append(child_B)

    jet_flavor_A = ak.unflatten(jet_flav_child_A, ak.num(input_jet))
    jet_flavor_B = ak.unflatten(jet_flav_child_B, ak.num(input_jet))

    # Build jet_btag_string lists
    flat_jet_btag_string = ak.flatten(input_jet.btag_string)
    flat_jet_btag_string_A = []
    flat_jet_btag_string_B = []
    for s in flat_jet_btag_string:
        a, b = extract_outermost_pair(str(s))
        flat_jet_btag_string_A.append(a)
        flat_jet_btag_string_B.append(b)

    jet_btag_string_A = ak.unflatten(flat_jet_btag_string_A, ak.num(input_jet))
    jet_btag_string_B = ak.unflatten(flat_jet_btag_string_B, ak.num(input_jet))

    combined_pt = input_jet.pt
    tanThetaA = np.tan(input_jet.thetaA)
    tanThetaB = input_jet.zA / (1 - input_jet.zA) * tanThetaA

    #
    #  pA (in frame with pz=0 phi=0 decay_phi = 0)
    #
    pA_pz0_px = input_jet.zA * combined_pt
    pA_pz0_py = 0
    pA_pz0_pz = - input_jet.zA * combined_pt * tanThetaA
    # pA_mass   = input_jet.rhoA * input_jet.pt * input_jet.zA
    pA_mass   = input_jet.mA
    pA_pz0_E  = np.sqrt(pA_pz0_px**2 + pA_pz0_pz**2 + pA_mass**2)

    pA_pz0_phi0_decayPhi0 = ak.zip(
        {
            "x": pA_pz0_px,
            "y": pA_pz0_py,
            "z": pA_pz0_pz,
            "t": pA_pz0_E,
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )

    pB_pz0_px = (1 - input_jet.zA) * combined_pt
    pB_pz0_py = 0
    pB_pz0_pz = (1 - input_jet.zA) * combined_pt * tanThetaB
    # pB_mass   = input_jet.rhoB * input_jet.pt * (1 - input_jet.zA)
    pB_mass   = input_jet.mB
    pB_pz0_E  = np.sqrt(pB_pz0_px**2 + pB_pz0_pz**2 + pB_mass**2)

    pB_pz0_phi0_decayPhi0 = ak.zip(
        {
            "x": pB_pz0_px,
            "y": pB_pz0_py,
            "z": pB_pz0_pz,
            "t": pB_pz0_E,
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )

    #
    # Do Rotation of the decay plane
    #

    # Pseudo-random number to decide if we rotate by phi or phi + pi
    decay_phi = input_jet.decay_phi + np.pi * ((input_jet.pt % 1) > 0.5)

    pA_pz0_phi0 = rotateX(pA_pz0_phi0_decayPhi0, decay_phi)
    pB_pz0_phi0 = rotateX(pB_pz0_phi0_decayPhi0, decay_phi)

    #
    #  Boost back to jet pZ
    #
    boost_vec_z = ak.zip(
        {
            "x": 0,
            "y": 0,
            "z": input_jet.boostvec.z,
        },
        with_name="ThreeVector",
        behavior=vector.behavior,
    )

    pA_phi0 = pA_pz0_phi0.boost(boost_vec_z)
    pB_phi0 = pB_pz0_phi0.boost(boost_vec_z)

    #
    #  Rotate to jet phi
    #
    pA = rotateZ(pA_phi0, input_jet.phi)
    pB = rotateZ(pB_phi0, input_jet.phi)

    #
    #  Logic to update mass of a single jet "b" or "j" with pt x rho
    #
    counts = ak.num(input_jet)
    pA = update_single_jet_mass(pA, jet_flavor_A, input_jet.rhoA, jet_btag_string_A, counts)
    pB = update_single_jet_mass(pB, jet_flavor_B, input_jet.rhoB, jet_btag_string_B, counts)

    return pA, pB


def decluster_splitting_types(input_jets, splitting_types, input_pdfs, rand_seed, *, b_pt_threshold=40, dr_threshold=0.4, chunk=None, debug=False):

    if debug:
        print(f"{chunk} decluster_splitting_types input rand_seed {rand_seed}\n")

    #
    #  Create a mask for all the jets that need declustered
    #
    input_jets['split_mask'] = False
    for _s in splitting_types:
        jet_flavor_flat  = ak.flatten(input_jets.jet_flavor)
        _split_mask_flat = jet_flavor_flat == _s
        _split_mask = ak.unflatten(_split_mask_flat, ak.num(input_jets.jet_flavor))
        input_jets["split_mask"] = _split_mask | input_jets.split_mask

    #
    #  Save the jets that dont need to be declustered
    #
    unclustered_jets = input_jets[~input_jets.split_mask]

    #
    #  Mask the jets to be declustered
    #
    input_jets_to_decluster = input_jets[input_jets.split_mask]

    #
    #  Need to iterate b/c
    #   - Some unclusterings fail the jet pt and eta
    #   - Some lead to dR too close (Not checked yet!)
    #   - Some of the splittings are recursive (no implemented yet!)
    num_trys = 0

    while ak.any(input_jets_to_decluster):

        if debug:
            print(f"{chunk} decluster_splitting_types num_trys {num_trys}\n")
            print(f"{chunk} decluster_splitting_types splitting_types {splitting_types}\n")

        if debug:
            print(f" (decluster_splitting_types) num_trys {num_trys} ")

        splittings_info = []

        if debug:
            print(f"splittings_types is {splitting_types} num_trys {num_trys}")

        for _s in splitting_types:

            # Pre compute these to save time
            jet_flavor_flat = ak.flatten(input_jets_to_decluster.jet_flavor)
            _s_mask_flat = jet_flavor_flat == _s
            _s_mask = ak.unflatten(_s_mask_flat, ak.num(input_jets_to_decluster.jet_flavor))

            _num_samples   = np.sum(ak.num(input_jets_to_decluster[_s_mask]))
            _indicies = np.where(ak.flatten(_s_mask))
            _indicies_tuple = (_indicies[0].to_list())

            splittings_info.append((get_splitting_name(_s), _num_samples, _indicies_tuple))

        if debug:
            print(f"{chunk} decluster_splitting_types rand_seed {rand_seed}\n")

        #
        #  Sample the PDFs,  add sampled varibales to the jets to be declustered
        #
        sample_PDFs_vs_pT(input_jets_to_decluster, input_pdfs, 11 * num_trys + rand_seed, splittings_info, chunk=chunk)

        #
        #  do the declustering
        #
        declustered_jets_A, declustered_jets_B  = decluster_combined_jets(input_jets_to_decluster, debug=debug)

        #
        #  Check for declustered jets failing kinematic requirements
        #
        # Update to only be bjets
        fail_pt_mask    = (declustered_jets_A.pt < 20) | (declustered_jets_B.pt < 20)

        declustered_jet_A_jet_flavor_flat  = ak.flatten(declustered_jets_A.jet_flavor)
        A_is_b_mask_flat = declustered_jet_A_jet_flavor_flat == "b"
        A_is_b_mask = ak.unflatten(A_is_b_mask_flat, ak.num(declustered_jets_A.jet_flavor))

        declustered_jet_B_jet_flavor_flat  = ak.flatten(declustered_jets_B.jet_flavor)
        B_is_b_mask_flat = declustered_jet_B_jet_flavor_flat == "b"
        B_is_b_mask = ak.unflatten(B_is_b_mask_flat, ak.num(declustered_jets_B.jet_flavor))

        #fail_pt_b_mask  = (A_is_b_mask & (declustered_jets_A.pt < 40) )          | (B_is_b_mask & (declustered_jets_B.pt < 40))
        fail_pt_b_mask  = (A_is_b_mask & (declustered_jets_A.pt < b_pt_threshold) )          | (B_is_b_mask & (declustered_jets_B.pt < b_pt_threshold))
        fail_eta_b_mask = (A_is_b_mask & (np.abs(declustered_jets_A.eta) > 2.5)) | (B_is_b_mask & (np.abs(declustered_jets_B.eta) > 2.5))

        fail_dr_mask  = declustered_jets_A.delta_r(declustered_jets_B) < dr_threshold
        clustering_fail = fail_pt_mask | fail_pt_b_mask | fail_eta_b_mask | fail_dr_mask

        if num_trys > _MAX_NUM_JET_RETRY:
            print(f"Bailing with {np.sum(ak.num(input_jets_to_decluster))}\n")
            clustering_fail = ~(fail_pt_mask | ~fail_pt_mask)  # All False

        #
        #  Save unclustered jets that are OK
        #
        unclustered_jets = ak.concatenate([unclustered_jets, declustered_jets_A[~clustering_fail], declustered_jets_B[~clustering_fail]], axis=1)

        #
        #  Try again with the other jets
        #
        # print(f"Was {np.sum(ak.num(input_jets_decluster))}\n")
        input_jets_to_decluster = input_jets_to_decluster[clustering_fail]
        # print(f"Now {np.sum(ak.num(input_jets_decluster))}\n")
        num_trys += 1

    return unclustered_jets


def make_synthetic_event_core(input_jets, input_pdfs, rand_seed, *, b_pt_threshold=40, dr_threshold=0.4, chunk=None, debug=False):

    if debug:
        print(f"{chunk} make_synthetic_event_core rand_seed {rand_seed}\n")

    #
    #  Get all the different types of splitted needed
    #
    splitting_types = get_list_of_combined_jet_types(input_jets)

    if debug:
        print(f" (make_synthetic_event_core) splitting_types {splitting_types}")

    while len(splitting_types):

        if debug:
            print(f"(make_synthetic_event_core) splitting_types was {splitting_types}")

        input_jets = decluster_splitting_types(input_jets, splitting_types, input_pdfs, rand_seed, b_pt_threshold=b_pt_threshold, dr_threshold=dr_threshold, chunk=chunk, debug=debug)

        splitting_types = get_list_of_combined_jet_types(input_jets)

        if debug:
            print(f"(make_synthetic_event_core) splitting_types is now {splitting_types}")

    if debug:
        print(f" (make_synthetic_event_core) splitting_types now {splitting_types}")

    return input_jets

# No Delta R cut
# def make_synthetic_event(input_jets, input_pdfs, debug=False):
#   return make_synthetic_event_core(input_jets, input_pdfs, debug=debug)


def make_synthetic_event(input_jets, input_pdfs, declustering_rand_seed=66, *, b_pt_threshold=40, dr_threshold=0.4, chunk=None, debug=False):

    if debug:
        print(f"{chunk} make_synthetic_event rand_seed {declustering_rand_seed}\n")

    # Start with all True
    events_to_decluster_mask = np.ones(len(input_jets), dtype=bool)

    n_events = len(input_jets)

    # Get number of expected output jets
    jet_clustering_summary = ["".join(ak.to_list(i)) for i in input_jets.jet_flavor]
    n_declustered_jets_per_event = [s.count('b') + s.count('j') for s in jet_clustering_summary]

    n_total_declustered_jets = np.sum(n_declustered_jets_per_event)

    flat_declustered_pt         = np.zeros(n_total_declustered_jets)
    flat_declustered_eta        = np.zeros(n_total_declustered_jets)
    flat_declustered_phi        = np.zeros(n_total_declustered_jets)
    flat_declustered_mass       = np.zeros(n_total_declustered_jets)
    flat_declustered_jet_flavor = np.full (n_total_declustered_jets, "X")
    flat_declustered_btagScore  = np.full(n_total_declustered_jets, -1.0)

    num_trys = 0

    #
    # Loop until all False
    #
    while np.any(events_to_decluster_mask):

        to_decluster_indicies = np.where(events_to_decluster_mask)[0]

        declustered_events = make_synthetic_event_core(input_jets[to_decluster_indicies], input_pdfs, 7 * num_trys + declustering_rand_seed,
                                                       b_pt_threshold=b_pt_threshold, dr_threshold=dr_threshold, chunk=chunk, debug=debug)

        #
        #  Check the min dr
        #
        delta_r2_matrix = declustered_events.delta_r2(declustered_events[:, None])
        delta_r2_matrix_flat = ak.flatten(delta_r2_matrix)
        delta_r2_matrix_flat_flat = ak.flatten(delta_r2_matrix_flat).to_numpy()
        delta_r2_matrix_flat_flat[delta_r2_matrix_flat_flat == 0] = np.inf
        delta_r2_matrix_flat_masked = ak.unflatten(delta_r2_matrix_flat_flat, ak.num(delta_r2_matrix_flat))
        delta_r2_matrix_masked = ak.unflatten(delta_r2_matrix_flat_masked, ak.num(delta_r2_matrix))

        min_dr2 = ak.min(ak.min(delta_r2_matrix_masked, axis=1), axis=1)

        pass_dr2_mask_local = min_dr2 > (dr_threshold * dr_threshold)

        if num_trys > _MAX_NUM_EVENT_RETRY:
            print(f"Bailing on dR check with {np.sum(events_to_decluster_mask == True)}\n")
            pass_dr2_mask_local = (pass_dr2_mask_local | ~pass_dr2_mask_local)  # All True

        sucessful_deccluster_event_indicies = np.where(pass_dr2_mask_local)

        # which events passed dr_mask
        events_to_update = np.zeros(n_total_declustered_jets, dtype=bool)
        update_indicies_global = to_decluster_indicies[sucessful_deccluster_event_indicies]
        events_to_update[update_indicies_global] = True

        jet_replace_mask = [value for value, count in zip(events_to_update, n_declustered_jets_per_event) for _ in range(count)]

        new_jets_flat = ak.flatten(declustered_events[sucessful_deccluster_event_indicies])

        flat_declustered_pt        [jet_replace_mask]    = new_jets_flat.pt
        flat_declustered_eta       [jet_replace_mask]    = new_jets_flat.eta
        flat_declustered_phi       [jet_replace_mask]    = new_jets_flat.phi
        flat_declustered_mass      [jet_replace_mask]    = new_jets_flat.mass
        flat_declustered_jet_flavor[jet_replace_mask]    = new_jets_flat.jet_flavor
        flat_declustered_btagScore [jet_replace_mask] = [float(str(i)) for i in new_jets_flat.btag_string]
        events_to_decluster_mask[update_indicies_global] = False
        num_trys += 1

    #
    #  Assigning the flavor bit (for writting out the synthetic datasets
    #
    flat_declustered_flavor_bit  = np.full(shape=len(flat_declustered_pt), fill_value=1)
    flat_is_j_mask = flat_declustered_jet_flavor == "j"
    flat_declustered_flavor_bit[flat_is_j_mask] = 0

    newly_declustered_events = ak.zip(
        {
            "pt":         ak.unflatten(flat_declustered_pt,             n_declustered_jets_per_event),
            "eta":        ak.unflatten(flat_declustered_eta,            n_declustered_jets_per_event),
            "phi":        ak.unflatten(flat_declustered_phi,            n_declustered_jets_per_event),
            "mass":       ak.unflatten(flat_declustered_mass,           n_declustered_jets_per_event),
            "jet_flavor": ak.unflatten(flat_declustered_jet_flavor,     n_declustered_jets_per_event),
            "btagScore":  ak.unflatten(flat_declustered_btagScore,      n_declustered_jets_per_event),
            "jet_flavor_bit": ak.unflatten(flat_declustered_flavor_bit, n_declustered_jets_per_event),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )

    return newly_declustered_events


def clean_ISR(clustered_jets, splittings, debug=False):

    all_jet_types = get_list_of_splitting_types(clustered_jets)

    if debug:
        print(f" (clean_ISR) all_jet_types {all_jet_types}")

    ISR_splittings_types = get_list_of_ISR_splittings(all_jet_types)

    if debug:
        print(f" (clean_ISR) ISR_splittings_types {ISR_splittings_types}")

    #
    #  Will need recusion here
    #
    clustered_jets_clean = clustered_jets

    while len(ISR_splittings_types):

        for _isr_splitting in ISR_splittings_types:

            ISR_mask = clustered_jets_clean.jet_flavor == _isr_splitting
            ISR_jets = clustered_jets_clean[ISR_mask]

            ISR_splittings_mask = splittings.jet_flavor == _isr_splitting
            ISR_splittings = splittings[ISR_splittings_mask]

            pairs = ak.cartesian([ISR_jets, ISR_splittings], axis=1, nested=True)
            delta_r_values = pairs[:, "0"].delta_r(pairs[:, "1"])
            closest_indices = ak.argmin(delta_r_values, axis=2)
            match_splitting = ISR_splittings[closest_indices]

            if debug:
                print(f" ISR_jets: {ISR_jets.pt}  {ISR_jets.eta} {ISR_jets.phi} ")
                print(f" match_splitting: {match_splitting.pt}  {match_splitting.eta} {match_splitting.phi} ")
                print(f" ISR_splittings: {ISR_splittings.pt}  {ISR_splittings.eta} {ISR_splittings.phi} ")

            declustered_A = match_splitting.part_A
            declustered_B = match_splitting.part_B

            # To ADd
            #  detclustered_A_jets = decluster(detclustered_A) # recurseive deculstering
            #  detclustered_A_jets = decluster(detclustered_A) # recurseive deculstering

            declustered_ISR_jets = ak.concatenate([declustered_A, declustered_B], axis=1)

            clustered_jets_clean = clustered_jets_clean[~ISR_mask]
            clustered_jets_clean = ak.concatenate([clustered_jets_clean, declustered_ISR_jets], axis=1)

        #
        # Recompute ISR splitting_types
        #
        all_jet_types = get_list_of_splitting_types(clustered_jets_clean)

        if debug:
            print(f" (clean_ISR) all_jet_types now {all_jet_types}")

        ISR_splittings_types = get_list_of_ISR_splittings(all_jet_types)

        if debug:
            print(f" (clean_ISR) ISR_splittings_types now {ISR_splittings_types}")

    return clustered_jets_clean
