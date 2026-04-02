"""
FeynNet ONNX model wrapper for HH→4b analysis.

Wraps an external FeynNet ensemble (k-fold ONNX models) to produce
classification scores and reweighting ratios from awkward-array event data.
"""

import json
import logging
import warnings

import awkward as ak
import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)


def _higgs_cand_flags(event) -> np.ndarray:
    """
    Build one-hot role flags for each canJet.

    Returns
    -------
    flags : ndarray, shape (N, 4, 4)
        Axis 1 = canJet index (0–3, btag-sorted order)
        Axis 2 = role index [h1b1, h1b2, h2b1, h2b2]
        flags[i, j, r] = 1.0 if canJet j plays role r in event i
    """
    n = len(event)
    can_pt = ak.to_numpy(event.canJet.pt).astype("float32")  # (N, 4)

    # pt of each role from the selected quad-jet pairing
    role_pts = np.stack([
        ak.to_numpy(event.quadJet_selected.lead.lead.pt).astype("float32"),  # h1b1
        ak.to_numpy(event.quadJet_selected.lead.subl.pt).astype("float32"),  # h1b2
        ak.to_numpy(event.quadJet_selected.subl.lead.pt).astype("float32"),  # h2b1
        ak.to_numpy(event.quadJet_selected.subl.subl.pt).astype("float32"),  # h2b2
    ], axis=1)  # (N, 4)

    flags = np.zeros((n, 4, 4), dtype="float32")
    claimed = np.zeros((n, 4), dtype=bool)
    for role_idx in range(4):
        role_pt = role_pts[:, role_idx]  # (N,)
        # Find which canJet (by index) has the closest pt to this role
        diff = np.abs(can_pt - role_pt[:, np.newaxis])  # (N, 4)
        diff[claimed] = np.inf  # exclude already-claimed jets
        best_jet = np.argmin(diff, axis=1)  # (N,)
        flags[np.arange(n), best_jet, role_idx] = 1.0
        claimed[np.arange(n), best_jet] = True

    return flags


def _select_forward_jets(event) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Select the two 'forward' (non-b-tagged) jets for each event.

    Selection criteria applied to event.Jet:
      - pt > 30
      - |eta| < 4.7
      - not b-tagged (tagged == False)
      - NOT (pt < 50 AND 2.5 < |eta| < 3.0)  [noisy HF region]

    From the surviving jets, pick the best positive-eta and best negative-eta
    jet (by pt), sort the pair by pt descending, and zero-pad to exactly 2.

    Returns
    -------
    f_pt, f_eta, f_phi, f_mass : ndarray, each shape (N, 2), float32
    """
    # Pad to rectangular shape before converting to numpy (Jet arrays may be jagged)
    nj = ak.max(ak.num(event.Jet.pt)) if ak.num(event.Jet.pt, axis=0) > 0 else 0
    # If already rectangular (fixed-length axis), ak.to_numpy works directly
    try:
        pt = ak.to_numpy(event.Jet.pt).astype("float32")
        eta = ak.to_numpy(event.Jet.eta).astype("float32")
        phi = ak.to_numpy(event.Jet.phi).astype("float32")
        mass = ak.to_numpy(event.Jet.mass).astype("float32")
        tagged = ak.to_numpy(event.Jet.tagged)
    except ValueError:
        # Jagged arrays: pad to max length with sentinel values
        pad_val = 0.0
        pt = ak.fill_none(ak.pad_none(event.Jet.pt, nj, clip=True), pad_val).to_numpy().astype("float32")
        eta = ak.fill_none(ak.pad_none(event.Jet.eta, nj, clip=True), pad_val).to_numpy().astype("float32")
        phi = ak.fill_none(ak.pad_none(event.Jet.phi, nj, clip=True), pad_val).to_numpy().astype("float32")
        mass = ak.fill_none(ak.pad_none(event.Jet.mass, nj, clip=True), pad_val).to_numpy().astype("float32")
        tagged_padded = ak.fill_none(ak.pad_none(event.Jet.tagged, nj, clip=True), True)
        tagged = tagged_padded.to_numpy()

    n, nj = pt.shape

    abs_eta = np.abs(eta)
    sel = (
        (pt > 30)
        & (abs_eta < 4.7)
        & (~tagged)
        & ~((pt < 50) & (abs_eta > 2.5) & (abs_eta < 3.0))
    )

    f_pt = np.zeros((n, 2), dtype="float32")
    f_eta = np.zeros((n, 2), dtype="float32")
    f_phi = np.zeros((n, 2), dtype="float32")
    f_mass = np.zeros((n, 2), dtype="float32")

    for i in range(n):
        mask = sel[i]
        jet_pt = pt[i, mask]
        jet_eta = eta[i, mask]
        jet_phi = phi[i, mask]
        jet_mass = mass[i, mask]

        pos_mask = jet_eta >= 0
        neg_mask = jet_eta < 0

        candidates = []
        if pos_mask.any():
            best = np.argmax(jet_pt[pos_mask])
            idx = np.where(pos_mask)[0][best]
            candidates.append((jet_pt[idx], jet_eta[idx], jet_phi[idx], jet_mass[idx]))
        if neg_mask.any():
            best = np.argmax(jet_pt[neg_mask])
            idx = np.where(neg_mask)[0][best]
            candidates.append((jet_pt[idx], jet_eta[idx], jet_phi[idx], jet_mass[idx]))

        # Sort pair by pt descending
        candidates.sort(key=lambda x: -x[0])
        for slot, (jpt, jeta, jphi, jmass) in enumerate(candidates[:2]):
            f_pt[i, slot] = jpt
            f_eta[i, slot] = jeta
            f_phi[i, slot] = jphi
            f_mass[i, slot] = jmass

    return f_pt, f_eta, f_phi, f_mass


def _compute_min_b_dr(
    f_eta: np.ndarray,
    f_phi: np.ndarray,
    can_eta: np.ndarray,
    can_phi: np.ndarray,
) -> np.ndarray:
    """
    For each forward jet slot, compute the minimum deltaR to any of the 4 canJets.

    Parameters
    ----------
    f_eta, f_phi : (N, 2)
    can_eta, can_phi : (N, 4)

    Returns
    -------
    min_dr : ndarray, shape (N, 2), float32
    """
    # (N, 2, 1) vs (N, 1, 4)
    deta = f_eta[:, :, np.newaxis] - can_eta[:, np.newaxis, :]  # (N, 2, 4)
    dphi = f_phi[:, :, np.newaxis] - can_phi[:, np.newaxis, :]  # (N, 2, 4)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    dr = np.sqrt(deta**2 + dphi**2)
    return dr.min(axis=2).astype("float32")  # (N, 2)


class FeynNetEnsemble:
    """
    Ensemble of k-fold FeynNet ONNX models for HH→4b classification.

    Parameters
    ----------
    paths : list of dict
        Each element has keys ``"path"`` (ONNX model file) and
        ``"preprocess"`` (JSON preprocessing parameters).
    batched : bool
        Whether to run inference in a single batched call (default True).
        Falls back to per-event if the batched call raises RuntimeError or
        produces NaNs.
    """

    def __init__(self, paths: list[dict], batched: bool = True):
        import onnxruntime as ort

        self.sessions = []
        self.preprocessors = []
        for entry in paths:
            session = ort.InferenceSession(
                entry["path"],
                providers=["CPUExecutionProvider"],
            )
            self.sessions.append(session)

            with open(entry["preprocess"], "r") as f:
                prep = json.load(f)
            self.preprocessors.append(prep)

        self.classes = ["ggHH", "qqHH", "ZZ", "ZH", "Background"]
        self.n_folds = len(paths)
        self._batched = batched

    # ------------------------------------------------------------------
    # Input construction
    # ------------------------------------------------------------------

    def _build_inputs(self, event) -> dict[str, np.ndarray]:
        """
        Construct the four model inputs from an awkward event array.

        Returns a dict with keys ``j``, ``j_p4``, ``f``, ``f_p4``.

        Shapes
        ------
        j     : (N, 9, 4)   — candidate jet features
        j_p4  : (N, 4, 4)   — candidate jet 4-vectors (no normalisation)
        f     : (N, 6, 2)   — forward jet features
        f_p4  : (N, 4, 2)   — forward jet 4-vectors (no normalisation)
        """
        n = len(event)

        # --- Candidate jets (re-sort by pt descending) ---
        can_pt_raw = ak.to_numpy(event.canJet.pt).astype("float32")   # (N, 4)
        can_eta = ak.to_numpy(event.canJet.eta).astype("float32")
        can_phi = ak.to_numpy(event.canJet.phi).astype("float32")
        can_mass = ak.to_numpy(event.canJet.mass).astype("float32")

        # Flags before pt-resorting (btag-sorted order)
        flags_orig = _higgs_cand_flags(event)  # (N, 4, 4)

        # pt-sort descending
        pt_order = np.argsort(-can_pt_raw, axis=1)  # (N, 4)
        idx = np.arange(n)[:, np.newaxis]
        can_pt = can_pt_raw[idx, pt_order]
        can_eta = can_eta[idx, pt_order]
        can_phi = can_phi[idx, pt_order]
        can_mass = can_mass[idx, pt_order]

        # Re-index flags to new pt order: flags[i, new_slot, role] = flags_orig[i, old_slot, role]
        flags = flags_orig[np.arange(n)[:, np.newaxis], pt_order, :]  # (N, 4, 4)

        # Candidate jet features (N, n_features, 4)
        j = np.stack([
            np.log(np.clip(can_pt, 1e-6, None)),            # j_log_pt
            np.log(np.clip(can_mass, 1e-6, None)),           # j_log_mass
            can_eta,                                          # j_eta
            np.sin(can_phi),                                  # j_sinphi
            np.cos(can_phi),                                  # j_cosphi
            flags[:, :, 0],                                   # j_h1b1_cand
            flags[:, :, 1],                                   # j_h1b2_cand
            flags[:, :, 2],                                   # j_h2b1_cand
            flags[:, :, 3],                                   # j_h2b2_cand
        ], axis=1).astype("float32")  # (N, 9, 4)

        # Candidate jet 4-vectors — no preprocessing applied
        j_p4 = np.stack([
            can_pt,
            can_mass,
            can_eta,
            can_phi,
        ], axis=1).astype("float32")  # (N, 4, 4)

        # --- Forward jets ---
        f_pt, f_eta, f_phi, f_mass = _select_forward_jets(event)  # each (N, 2)

        min_b_dr = _compute_min_b_dr(f_eta, f_phi, can_eta, can_phi)  # (N, 2)

        f = np.stack([
            np.log(np.where(f_pt > 0, f_pt, 1.0)),          # f_log_pt
            np.log(np.where(f_mass > 0, f_mass, 1.0)),       # f_log_mass
            f_eta,                                             # f_eta
            np.sin(f_phi),                                     # f_sinphi
            np.cos(f_phi),                                     # f_cosphi
            min_b_dr,                                          # f_min_b_dr
        ], axis=1).astype("float32")  # (N, 6, 2)

        f_p4 = np.stack([
            f_pt,
            f_mass,
            f_eta,
            f_phi,
        ], axis=1).astype("float32")  # (N, 4, 2)

        return {"j": j, "j_p4": j_p4, "f": f, "f_p4": f_p4}

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        raw_inputs: dict[str, np.ndarray],
        prep_params: dict,
    ) -> dict[str, np.ndarray]:
        """
        Apply per-feature normalisation: ``(value - median) * norm_factor``.

        Inf values are replaced with ``replace_inf_value`` before normalising.
        """
        result = {}
        for key, arr in raw_inputs.items():
            group = prep_params[key]
            var_names = group["var_names"]
            var_infos = group["var_infos"]
            out = arr.copy()
            for feat_idx, name in enumerate(var_names):
                info = var_infos[name]
                # arr shape: (N, n_features, length)
                col = out[:, feat_idx, :]
                col = np.where(np.isinf(col), info["replace_inf_value"], col)
                col = (col - info["median"]) * info["norm_factor"]
                out[:, feat_idx, :] = col
            result[key] = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def __call__(self, event) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Run FeynNet inference on an event array.

        Parameters
        ----------
        event : ak.Array
            Must have fields ``canJet``, ``quadJet_selected``, ``Jet``, ``event``.

        Returns
        -------
        c_score : ndarray, shape (N, n_classes)
        q_score : ndarray, shape (N, 3)
        """
        n = len(event)
        n_classes = len(self.classes)

        c_score = np.zeros((n, n_classes), dtype="float32")
        q_score = np.zeros((n, 3), dtype="float32")

        # Fold assignment (uint64 → int64 view for modulo)
        event_int = event.event.to_numpy().view(np.int64)
        fold_indices = event_int % self.n_folds

        raw_inputs = self._build_inputs(event)

        for fold_id, (session, prep) in enumerate(
            zip(self.sessions, self.preprocessors)
        ):
            fold_mask = fold_indices == fold_id
            if not fold_mask.any():
                continue

            fold_raw = {k: v[fold_mask] for k, v in raw_inputs.items()}
            fold_inputs = self._preprocess(fold_raw, prep)

            # Build onnxruntime input dict using the model's expected input names
            input_names = prep.get("input_names", list(fold_inputs.keys()))
            ort_inputs = {name: fold_inputs[name] for name in input_names if name in fold_inputs}

            output_names_list = prep.get("output_names", None)

            outputs = None
            if self._batched:
                try:
                    raw_outputs = session.run(output_names_list, ort_inputs)
                    if output_names_list is not None:
                        output_map = dict(zip(output_names_list, raw_outputs))
                        probs = output_map["event_probs"]
                        ratios = output_map["event_reweight"]
                    else:
                        probs = raw_outputs[0]
                        ratios = raw_outputs[2]
                    outputs = raw_outputs
                    if np.any(np.isnan(probs)) or np.any(np.isnan(ratios)):
                        log.warning(
                            "FeynNet fold %d: NaN in batched output, falling back to per-event",
                            fold_id,
                        )
                        outputs = None
                except RuntimeError as exc:
                    log.warning(
                        "FeynNet fold %d: batched inference failed (%s), falling back to per-event",
                        fold_id,
                        exc,
                    )
                    outputs = None

            if outputs is None:
                # Per-event fallback
                batch_size = fold_mask.sum()
                probs_list = []
                ratios_list = []
                for ev_idx in range(batch_size):
                    ev_inputs = {k: v[ev_idx : ev_idx + 1] for k, v in ort_inputs.items()}
                    ev_out = session.run(output_names_list, ev_inputs)
                    if output_names_list is not None:
                        ev_map = dict(zip(output_names_list, ev_out))
                        probs_list.append(ev_map["event_probs"])
                        ratios_list.append(ev_map["event_reweight"])
                    else:
                        probs_list.append(ev_out[0])
                        ratios_list.append(ev_out[2])
                probs = np.concatenate(probs_list, axis=0)
                ratios = np.concatenate(ratios_list, axis=0)

            c_score[fold_mask] = probs
            n_ratio_cols = min(ratios.shape[1], 3)
            q_score[fold_mask, :n_ratio_cols] = ratios[:, :n_ratio_cols]

        return c_score, q_score
