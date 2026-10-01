import numpy as np
import awkward as ak
from copy import copy
from coffea.nanoevents.methods import vector
from numba import jit, njit

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


def cluster_bs_reference(event_jets, *, debug=False):
    """The original per-event Python implementation; cluster_bs must reproduce it (tests only)."""
    return cluster_bs_core(event_jets, get_min_indicies, debug=debug)


#
#  cluster_bs: the same algorithm as cluster_bs_core + get_min_indicies, as one numba pass over
#  flat arrays. Per event a node table (leaves = input jets, then one node per merge) and the list
#  of alive nodes in the order cluster_bs_core keeps them (survivors in order, the new node
#  appended), so the argmin over i < j breaks ties exactly as min((dij, i, j)) does. Four-vector
#  conventions follow scikit-hep vector (coffea's behaviors): float64 throughout, so merged nodes
#  agree with cluster_bs_reference to ~1e-7 (its first merges run in float32 scalar math).
#
@njit
def _rectify(dphi):
    return (dphi + np.pi) % (2 * np.pi) - np.pi


@njit
def _wrapped_len(n):
    return n if n == 1 else n + 2


@njit
def _cluster_bs_kernel(offsets, pt, eta, phi, mass, n_b_leaf, len_leaf):
    n_events = len(offsets) - 1
    n_max = 2 * offsets[-1]
    node_pt   = np.empty(n_max)
    node_eta  = np.empty(n_max)
    node_phi  = np.empty(n_max)
    node_mass = np.empty(n_max)
    node_nb   = np.empty(n_max, np.int64)
    node_len  = np.empty(n_max, np.int64)
    node_A    = np.full(n_max, -1, np.int64)    # global node ids of the (ordered) children
    node_B    = np.full(n_max, -1, np.int64)
    node_offsets  = np.zeros(n_events + 1, np.int64)
    alive_ids     = np.empty(offsets[-1], np.int64)
    alive_offsets = np.zeros(n_events + 1, np.int64)

    n_nodes = 0
    n_alive_total = 0
    alive = np.empty(np.max(offsets[1:] - offsets[:-1]) if n_events else 0, np.int64)
    for iE in range(n_events):
        n_jets = offsets[iE + 1] - offsets[iE]
        for k in range(n_jets):
            j = offsets[iE] + k
            node_pt[n_nodes], node_eta[n_nodes], node_phi[n_nodes], node_mass[n_nodes] = pt[j], eta[j], phi[j], mass[j]
            node_nb[n_nodes], node_len[n_nodes] = n_b_leaf[j], len_leaf[j]
            alive[k] = n_nodes
            n_nodes += 1
        n_alive = n_jets

        while n_alive >= 2:
            # argmin of min(pt_A^2, pt_B^2) * dR^2 over i < j in alive order (ties -> smallest i, j)
            best, best_i, best_j = np.inf, -1, -1
            for i in range(n_alive):
                a = alive[i]
                for j in range(i + 1, n_alive):
                    b = alive[j]
                    dr = np.sqrt(_rectify(node_phi[a] - node_phi[b]) ** 2 + (node_eta[a] - node_eta[b]) ** 2)
                    dij = min(node_pt[a] ** 2, node_pt[b] ** 2) * dr ** 2
                    if dij < best:
                        best, best_i, best_j = dij, i, j
            if best_i < 0:
                break
            a, b = alive[best_i], alive[best_j]

            # stop before combining three b's
            if node_nb[a] + node_nb[b] > 2:
                break

            # child order as combine_particles: flavor-string length, then b content, then pt
            if node_len[b] > node_len[a]:
                a, b = b, a
            elif node_len[b] == node_len[a]:
                if node_nb[a] < node_nb[b]:
                    a, b = b, a
                elif node_nb[a] == node_nb[b] and node_pt[a] < node_pt[b]:
                    a, b = b, a

            x = y = z = t = 0.0
            for c in (a, b):
                sh = np.sinh(node_eta[c])
                m = node_mass[c]
                x += node_pt[c] * np.cos(node_phi[c])
                y += node_pt[c] * np.sin(node_phi[c])
                z += node_pt[c] * sh
                t += np.sqrt(max(m * abs(m) + node_pt[c] ** 2 * (1 + sh ** 2), 0.0))
            rho = np.sqrt(x ** 2 + y ** 2)
            tau2 = t ** 2 - (x ** 2 + y ** 2 + z ** 2)
            node_pt[n_nodes]   = rho
            node_eta[n_nodes]  = np.arcsinh(z / rho) if rho > 0 else (np.inf if z > 0 else (-np.inf if z < 0 else 0.0))
            node_phi[n_nodes]  = np.arctan2(y, x)
            node_mass[n_nodes] = np.copysign(np.sqrt(abs(tau2)), tau2)
            node_nb[n_nodes]   = node_nb[a] + node_nb[b]
            node_len[n_nodes]  = _wrapped_len(node_len[a]) + _wrapped_len(node_len[b])
            node_A[n_nodes], node_B[n_nodes] = a, b

            # drop positions best_i < best_j keeping the order, append the new node
            w = 0
            for k in range(n_alive):
                if k != best_i and k != best_j:
                    alive[w] = alive[k]
                    w += 1
            alive[w] = n_nodes
            n_alive = w + 1
            n_nodes += 1

        for k in range(n_alive):
            alive_ids[n_alive_total + k] = alive[k]
        n_alive_total += n_alive
        node_offsets[iE + 1] = n_nodes
        alive_offsets[iE + 1] = n_alive_total

    return (node_pt[:n_nodes], node_eta[:n_nodes], node_phi[:n_nodes], node_mass[:n_nodes],
            node_A[:n_nodes], node_B[:n_nodes], node_offsets, alive_ids[:n_alive_total], alive_offsets)


def _jet_records(fields, counts, **jagged):
    """Jagged PtEtaPhiMLorentzVector records from flat `fields` (+ already-jagged `jagged`)."""
    return ak.zip({**{k: ak.unflatten(v, counts) for k, v in fields.items()}, **jagged},
                  with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)


def cluster_bs(event_jets, *, debug=False):
    """Exclusive kT-like clustering of the jets in each event, merging the closest pair in
    min(pt^2) * dR^2 until the next merge would combine three b's.

    Returns (clustered_jets, splittings) exactly as cluster_bs_reference: the surviving objects
    and, per merge in merge order, the combined object with its ordered children part_A/part_B
    (pt, eta, phi, mass, jet_flavor, btag_string). Adds event_jets["btag_string"] as before."""
    counts = np.asarray(ak.num(event_jets), dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    flat = ak.flatten(event_jets)
    leaf_flavor = ak.to_list(flat.jet_flavor)
    # the leaf b-tag string as cluster_bs_core formats it: str(round(v, 3)) of each score
    leaf_btag = [str(round(v, 3)) for v in ak.to_numpy(flat.btagScore)]
    event_jets["btag_string"] = ak.unflatten(ak.Array(leaf_btag), counts)

    if any(len(f) != 1 for f in leaf_flavor):
        raise ValueError("cluster_bs: input jet_flavor must be single characters ('b'/'j')")
    n_b_leaf = np.fromiter((f == "b" for f in leaf_flavor), dtype=np.int64, count=len(leaf_flavor))
    len_leaf = np.ones(len(leaf_flavor), dtype=np.int64)

    (n_pt, n_eta, n_phi, n_mass, n_A, n_B, node_offsets, alive_ids, alive_offsets) = _cluster_bs_kernel(
        offsets,
        ak.to_numpy(flat.pt).astype(np.float64), ak.to_numpy(flat.eta).astype(np.float64),
        ak.to_numpy(flat.phi).astype(np.float64), ak.to_numpy(flat.mass).astype(np.float64),
        n_b_leaf, len_leaf,
    )

    # flavor / btag strings: leaves are the input jets (node order = jet order within an event),
    # merged nodes always come after their children, so one pass in node order fills them
    n_nodes = len(n_pt)
    node_flavor = [None] * n_nodes
    node_btag = [None] * n_nodes
    is_leaf = np.zeros(n_nodes, dtype=bool)
    leaf_node = np.zeros(n_nodes, dtype=np.int64)
    for iE in range(len(counts)):
        first = node_offsets[iE]
        is_leaf[first:first + counts[iE]] = True
        leaf_node[first:first + counts[iE]] = np.arange(offsets[iE], offsets[iE + 1])
    for k in range(n_nodes):
        if is_leaf[k]:
            node_flavor[k] = leaf_flavor[leaf_node[k]]
            node_btag[k] = leaf_btag[leaf_node[k]]
        else:
            fA, fB = node_flavor[n_A[k]], node_flavor[n_B[k]]
            node_flavor[k] = (fA if len(fA) == 1 else f"({fA})") + (fB if len(fB) == 1 else f"({fB})")
            node_btag[k] = f"({node_btag[n_A[k]]},{node_btag[n_B[k]]})"
    node_flavor = np.array(node_flavor, dtype=object)
    node_btag = np.array(node_btag, dtype=object)

    def node_fields(ids):
        return {"pt": n_pt[ids], "eta": n_eta[ids], "phi": n_phi[ids], "mass": n_mass[ids],
                "jet_flavor": ak.Array(node_flavor[ids].tolist()), "btag_string": ak.Array(node_btag[ids].tolist())}

    clustered_events = _jet_records(node_fields(alive_ids), np.diff(alive_offsets))

    split_ids = np.flatnonzero(~is_leaf)
    split_counts = np.diff(node_offsets) - counts
    splittings_events = _jet_records(
        node_fields(split_ids), split_counts,
        part_A=_jet_records(node_fields(n_A[split_ids]), split_counts),
        part_B=_jet_records(node_fields(n_B[split_ids]), split_counts),
    )

    return clustered_events, splittings_events


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
