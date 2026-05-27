import numpy as np
import awkward as ak
from copy import copy
from coffea.nanoevents.methods import vector
from numba import jit

# anti kt
# dij = min(1 / (part_A['pt']**2), 1 / (part_B['pt']**2)) * delta_r(part_A['eta'], part_A['phi'], part_B['eta'], part_B['phi'])**2 / R**2
# diB = 1 / (part_A['pt'] ** 2)

# C/A
# dij = delta_r(part_A['eta'], part_A['phi'], part_B['eta'], part_B['phi'])**2 / R**2
# diB = R**2

# Speed this up by using avoiding loop...?


# If R = 0 exlusive clustereing
def get_distances(particles, R):
    distances = []
    for i, part_A in enumerate(particles):
        for j, part_B in enumerate(particles):
            if i < j:
                dij = min(part_A.pt**2, part_B.pt**2) * part_A.delta_r(part_B)**2
                if R:
                    dij = dij / R**2
                distances.append((dij, i, j))
        if R:
            diB = part_A.pt ** 2
            distances.append((diB, i, None))
    return distances


# If R = 0 exlusive clustereing
def get_min_indicies(particles, R):
    distances = []
    for i, part_A in enumerate(particles):
        for j, part_B in enumerate(particles):
            if i < j:
                dij = min(part_A.pt**2, part_B.pt**2) * part_A.delta_r(part_B)**2
                # dij = part_A.delta_r(part_B)**2            # KT
                if R:
                    dij = dij / R**2
                distances.append((dij, i, j))
        if R:
            diB = part_A.pt ** 2
            distances.append((diB, i, None))
    # Find the minimum distance
    _, idx_A, idx_B = min(distances)

    return idx_A, idx_B


@jit
def get_min_indicies_numba_core(particles_pt, particles_eta, particles_phi):
    distances = []
    for iA in range(len(particles_pt)):
        for jB in range(len(particles_pt)):
            if iA < jB:

                dphi = particles_phi[iA] - particles_phi[jB]
                if dphi > np.pi:
                    dphi -= 2 * np.pi
                if dphi < -np.pi:
                    dphi += 2 * np.pi

                dij = min(particles_pt[iA]**2, particles_pt[jB]**2) * (np.square(particles_eta[iA] - particles_eta[jB]) + np.square(dphi))

                # dij = part_A.delta_r(part_B)**2            # KT
                distances.append((dij, iA, jB))

    # Find the minimum distance
    _, idx_A, idx_B = min(distances)

    return idx_A, idx_B


def get_min_indicies_numba(particles, R):
    particles_pt  = particles.pt
    particles_eta = particles.eta
    particles_phi = particles.phi

    return get_min_indicies_numba_core(particles.pt, particles.eta, particles.phi)


def distance_matrix_kt(vectors):
    pt1  = ak.values_astype(vectors.pt,  np.float64)
    eta1 = ak.values_astype(vectors.eta, np.float64)
    phi1 = ak.values_astype(vectors.phi, np.float64)

    pt2  = pt1[:, np.newaxis]
    eta2 = eta1[:, np.newaxis]
    phi2 = phi1[:, np.newaxis]

    dphi = np.abs(phi1 - phi2)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    deta = eta1 - eta2

    dr2 = deta**2 + dphi**2

    dij = np.minimum(pt1**2, pt2**2) * dr2

    dij = np.array(dij)
    # Mask to ignore the diagonal elements (where ΔR = 0)
    mask = np.eye(dij.shape[0], dtype=bool)

    # Set the diagonal elements to a large value to ignore them
    dij[mask] = np.inf
    return dij


def get_min_indicies_fast(particles, R):
    distances = []

    dij_matrix = distance_matrix_kt(particles)

    dij_min_per_jet    = np.min(dij_matrix, axis=1)
    dij_argmin_per_jet = np.argmin(dij_matrix, axis=1)

    idx_A = np.argmin(dij_min_per_jet)
    idx_B = dij_argmin_per_jet[idx_A]
    return idx_A, idx_B


def remove_indices(particles, indices_to_remove):
    mask = np.ones(len(particles), dtype=bool)
    mask[indices_to_remove] = False
    return particles[mask]


# Add parenthesis if needed
def comb_jet_flavor(flavor_A, flavor_B):

    # Add Parens if the input is already clustered
    if len(flavor_A) > 1:
        flavor_A = f"({str(flavor_A)})"
    if len(flavor_B) > 1:
        flavor_B = f"({str(flavor_B)})"

    return flavor_A + flavor_B

# Add parenthesis if needed
def comb_jet_btag_string(btag_A, btag_B):

    ## Add Parens if the input is already clustered
    #if len(flavor_A) > 1:
    #    flavor_A = f"({str(flavor_A)})"
    #if len(flavor_B) > 1:
    #    flavor_B = f"({str(flavor_B)})"

    return f"({btag_A},{btag_B})"



def combine_particles(part_A, part_B, *, debug=False):
    # awkward 2.x requires matching behavior types for +; compute sum via LorentzVector (x,y,z,t)
    part_comb = ak.zip(
        {
            "x": part_A.x + part_B.x,
            "y": part_A.y + part_B.y,
            "z": part_A.z + part_B.z,
            "t": part_A.energy + part_B.energy,
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )

    new_part_A = part_A
    new_part_B = part_B

    if debug:
        print(f"(combine_particles) new_part_A {new_part_A}")
        print(f"(combine_particles) new_part_B {new_part_B}")

    # order by complexity
    if len(new_part_B.jet_flavor) > len(new_part_A.jet_flavor):
        new_part_A = part_B
        new_part_B = part_A
        if debug:
            print(f"(combine_particles) swap b/c complexity")

    elif len(new_part_B.jet_flavor) == len(new_part_A.jet_flavor):

        # else order by bjet content
        if new_part_A.jet_flavor.count("b") < new_part_B.jet_flavor.count("b"):
            new_part_A = part_B
            new_part_B = part_A

        # else order by pt
        elif new_part_A.jet_flavor.count("b") == new_part_B.jet_flavor.count("b") and (new_part_A.pt < new_part_B.pt):
            new_part_A = part_B
            new_part_B = part_A

    part_comb_jet_flavor = comb_jet_flavor(new_part_A.jet_flavor, new_part_B.jet_flavor)

    part_comb_btag_string  = comb_jet_btag_string(new_part_A.btag_string, new_part_B.btag_string)


    part_comb_array = ak.zip(
        {
            "pt": [part_comb.pt],
            "eta": [part_comb.eta],
            "phi": [part_comb.phi],
            "mass": [part_comb.mass],
            "jet_flavor": [part_comb_jet_flavor],
            "btag_string": [part_comb_btag_string],
            "part_A": [new_part_A],
            "part_B": [new_part_B],
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )

    return part_comb_array


# Define the kt clustering algorithm
def cluster_bs_core(event_jets, distance_function, *, debug=False):
    clustered_jets = []
    splittings = []

    event_jets["btag_string"] = [[str(round(v,3)) for v in sublist] for sublist in event_jets.btagScore]


    nevents = len(event_jets)

    for iEvent in range(nevents):
        particles = copy(event_jets[iEvent])
        if debug:
            print(particles)
            print(f"iEvent {iEvent}")
            print(f"==============================")
            print(f"nParticles {len(particles)}")
        # Maybe later allow more than 4 bs
        # number_of_unclustered_bs = 4

        splittings.append([])

        while True:    # Break when try to combine more than 2 bs # number_of_unclustered_bs > 2:

            #
            # Calculate the distance measures
            #  R=0 turns off clustering to the beam
            # distances = get_distances(particles, R=0)
            # distances = distance_function(particles, R=0)
            idx_A, idx_B = distance_function(particles, R=0)

            if debug:
                print(f"clustering {idx_A} and {idx_B}")
                print(f"size partilces {len(particles)}")
                print(f"size partilces {len(particles)}")

            part_comb_array = combine_particles(particles[idx_A], particles[idx_B], debug=debug)

            #
            #  Stop if going to combine 3 bs
            #
            if part_comb_array.jet_flavor[0].count("b") > 2:
                if debug:
                    print(f"breaking on {part_comb_array.jet_flavor[0]}")
                break

            if debug:
                print(part_comb_array.jet_flavor)
                print(f"{part_comb_array.jet_flavor}")
            # if debug: print(f"was {number_of_unclustered_bs}")

            # number_of_unclustered_bs += delta_bs(part_comb_array.jet_flavor[0])

            # if debug: print(f"now {number_of_unclustered_bs}")
            splittings[-1].append(part_comb_array[0])

            particles = remove_indices(particles, [idx_A, idx_B])

            particles = ak.concatenate([particles, part_comb_array])
            if debug:
                print(f"size partilces {len(particles)}")

        clustered_jets.append(particles)



    # Create the PtEtaPhiMLorentzVectorArray
    clustered_events = ak.zip(
        {
            "pt":            ak.Array([[v.pt for v in sublist] for sublist in clustered_jets]),
            "eta":           ak.Array([[v.eta for v in sublist] for sublist in clustered_jets]),
            "phi":           ak.Array([[v.phi for v in sublist] for sublist in clustered_jets]),
            "mass":          ak.Array([[v.mass for v in sublist] for sublist in clustered_jets]),
            "jet_flavor":    ak.Array([[v.jet_flavor for v in sublist] for sublist in clustered_jets]),
            "btag_string":   ak.Array([[v.btag_string for v in sublist] for sublist in clustered_jets]),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    )


    # Create the PtEtaPhiMLorentzVectorArray
    splittings_events = ak.zip(
        {
            "pt":   ak.Array([[v.pt   for v in sublist] for sublist in splittings]),
            "eta":  ak.Array([[v.eta  for v in sublist] for sublist in splittings]),
            "phi":  ak.Array([[v.phi  for v in sublist] for sublist in splittings]),
            "mass": ak.Array([[v.mass for v in sublist] for sublist in splittings]),
            "jet_flavor": ak.Array([[v.jet_flavor for v in sublist] for sublist in splittings]),
            "btag_string": ak.Array([[v.btag_string for v in sublist] for sublist in splittings]),
            "part_A": ak.zip(
                {
                    "pt":         ak.Array([[v.part_A.pt   for v in sublist] for sublist in splittings]),
                    "eta":        ak.Array([[v.part_A.eta  for v in sublist] for sublist in splittings]),
                    "phi":        ak.Array([[v.part_A.phi  for v in sublist] for sublist in splittings]),
                    "mass":       ak.Array([[v.part_A.mass for v in sublist] for sublist in splittings]),
                    "jet_flavor": ak.Array([[v.part_A.jet_flavor for v in sublist] for sublist in splittings]),
                    "btag_string": ak.Array([[v.part_A.btag_string for v in sublist] for sublist in splittings]),
                },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior
            ),
            "part_B": ak.zip(
                {
                    "pt":         ak.Array([[v.part_B.pt  for v in sublist] for sublist in splittings]),
                    "eta":        ak.Array([[v.part_B.eta for v in sublist] for sublist in splittings]),
                    "phi":        ak.Array([[v.part_B.phi for v in sublist] for sublist in splittings]),
                    "mass":       ak.Array([[v.part_B.mass for v in sublist] for sublist in splittings]),
                    "jet_flavor": ak.Array([[v.part_B.jet_flavor for v in sublist] for sublist in splittings]),
                    "btag_string": ak.Array([[v.part_B.btag_string for v in sublist] for sublist in splittings]),
                },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior
            ),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    )

    return clustered_events, splittings_events


def cluster_bs(event_jets, *, debug=False):
    return cluster_bs_core(event_jets, get_min_indicies, debug=debug)


def cluster_bs_fast(event_jets, *, debug=False):
    return cluster_bs_core(event_jets, get_min_indicies_fast)


def cluster_bs_numba(event_jets, *, debug=False):
    return cluster_bs_core(event_jets, get_min_indicies_numba)


# Define the kt clustering algorithm
def kt_clustering(event_jets, R, *, debug=False):
    clustered_jets = []

    nevents = len(event_jets)

    for iEvent in range(nevents):
        particles = copy(event_jets[iEvent])

        if debug:
            print(particles)

        clustered_jets.append([])

        if debug:
            print(f"iEvent {iEvent}")

        while len(particles) > 0:

            #
            # Calculate the distance measures
            #
            distances = get_distances(particles, R)

            # Find the minimum distance
            min_dist, idx_A, idx_B = min(distances)

            if idx_B is None:
                # If the minimum distance is diB, declare part_A as a jet
                if debug:
                    print("adding clustered jet")

                clustered_jets[-1].append(copy(particles[idx_A]))

                particles = remove_indices(particles, [idx_A])

                if debug:
                    print(f"size partilces {len(particles)}")

            else:
                if debug:
                    print(f"clustering {idx_A} and {idx_B}")
                    print(f"size partilces {len(particles)}")
                # If the minimum distance is dij, combine particles i and j
                part_A = copy(particles[idx_A])
                part_B = copy(particles[idx_B])

                particles = remove_indices(particles, [idx_A, idx_B])

                if debug:
                    print(f"size partilces {len(particles)}")

                part_comb_array = combine_particles(part_A, part_B)

                particles = ak.concatenate([particles, part_comb_array])
                if debug:
                    print(f"size partilces {len(particles)}")

    # Create the PtEtaPhiMLorentzVectorArray with ndim=2
    clustered_events = ak.zip(
        {
            "pt":         ak.Array([[v.pt for v in sublist] for sublist in clustered_jets]),
            "eta":        ak.Array([[v.eta for v in sublist] for sublist in clustered_jets]),
            "phi":        ak.Array([[v.phi for v in sublist] for sublist in clustered_jets]),
            "mass":       ak.Array([[v.mass for v in sublist] for sublist in clustered_jets]),
            "jet_flavor": ak.Array([[v.jet_flavor for v in sublist] for sublist in clustered_jets]),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    )

    return clustered_events
