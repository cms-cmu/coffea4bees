from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import awkward as ak
import numpy as np

if TYPE_CHECKING:
    from coffea4bees.analysis.helpers.classifier.HCR import HCREnsemble


def set_ttHbb_SvB_vars(SvBName: str, event: ak.Array):
    """Derive native analysis-level ttHbb fields from friend-tree or model outputs.
    Natively maps pmj (p_multijet), ptt (p_ttbar), pttHbb (p_ttHbb), ps_ttHbb, and tt_vs_mj.
    Does NOT generate fake ps_hh / ps_zh / ps_zz fields.
    """
    sv = getattr(event, SvBName)
    fields = sv.fields

    if "p_ttHbb" in fields:
        pttHbb = sv.p_ttHbb
    elif "pttHbb" in fields:
        pttHbb = sv.pttHbb
    elif "p_sig" in fields:
        pttHbb = sv.p_sig
    elif "ps" in fields:
        pttHbb = sv.ps
    else:
        pttHbb = np.zeros(len(event), dtype=float)

    if "p_multijet" in fields:
        pmj = sv.p_multijet
    elif "pmj" in fields:
        pmj = sv.pmj
    else:
        pmj = np.zeros(len(event), dtype=float)

    if "p_ttbar" in fields:
        ptt = sv.p_ttbar
    elif "ptt" in fields:
        ptt = sv.ptt
    else:
        ptt = np.zeros(len(event), dtype=float)

    ps_ttHbb = ak.nan_to_num(pttHbb / np.maximum(pmj + ptt + pttHbb, 1e-10), nan=0.0)
    tt_vs_mj = ak.nan_to_num(ptt / np.maximum(ptt + pmj, 1e-10), nan=0.0)

    event[SvBName, "pmj"] = pmj
    event[SvBName, "ptt"] = ptt
    event[SvBName, "pttHbb"] = pttHbb
    event[SvBName, "ps"] = ps_ttHbb
    event[SvBName, "ps_ttHbb"] = ps_ttHbb
    event[SvBName, "tt_vs_mj"] = tt_vs_mj


def compute_SvB_ttHbb(events, mask, doCheck=True, **models: HCREnsemble):
    masked_events = events[mask]

    for name, model in models.items():
        if model is None:
            continue

        if name in events.fields:
            events[f"old_{name}"] = events[name]

        # Handle empty mask case
        if len(masked_events) == 0:
            classes = model.classes
            tmp_c_score = np.zeros((0, len(classes)))
            tmp_q_score = np.zeros((0, 3))
        else:
            try:
                if hasattr(masked_events, 'canJet') and len(masked_events.canJet) == 0:
                    logging.warning(f"Model {name}: masked_events has length {len(masked_events)} but empty canJet, using zero arrays")
                    classes = model.classes
                    tmp_c_score = np.zeros((0, len(classes)))
                    tmp_q_score = np.zeros((0, 3))
                elif hasattr(masked_events, 'notCanJet_coffea') and len(ak.flatten(masked_events.notCanJet_coffea.pt, axis=None)) == 0:
                    logging.warning(f"Model {name}: masked_events has empty notCanJet_coffea tensors, using zero arrays")
                    classes = model.classes
                    tmp_c_score = np.zeros((0, len(classes)))
                    tmp_q_score = np.zeros((0, 3))
                else:
                    tmp_c_score, tmp_q_score = model(masked_events)
            except RuntimeError as e:
                if "cannot reshape tensor" in str(e) and "0 elements" in str(e):
                    logging.warning(f"Model {name}: Detected tensor reshape error, creating zero arrays instead. Error: {str(e)}")
                    classes = model.classes
                    tmp_c_score = np.zeros((0, len(classes)))
                    tmp_q_score = np.zeros((0, 3))
                else:
                    raise e

        c_score = np.zeros((len(events), tmp_c_score.shape[1]))
        q_score = np.zeros((len(events), tmp_q_score.shape[1]))

        if tmp_c_score.shape[0] > 0:
            c_score[mask] = tmp_c_score
            q_score[mask] = tmp_q_score

        del tmp_c_score, tmp_q_score

        classes = model.classes
        pmj = c_score[:, classes.index("multijet")]
        ptt = c_score[:, classes.index("ttbar")]
        pttHbb = c_score[:, classes.index("ttHbb")]

        ps = pttHbb / np.maximum(pmj + ptt + pttHbb, 1e-10)
        tt_vs_mj = ptt / np.maximum(ptt + pmj, 1e-10)

        events[name] = ak.zip({
            "pmj": pmj,
            "ptt": ptt,
            "pttHbb": pttHbb,
            "ps": ps,
            "ps_ttHbb": ps,
            "tt_vs_mj": tt_vs_mj,
            "q_1234": q_score[:, 0],
            "q_1324": q_score[:, 1],
            "q_1423": q_score[:, 2],
        })

        if doCheck and f"old_{name}" in events.fields:
            error = ~np.isclose(events[f"old_{name}"].ps, events[name].ps, atol=1e-5, rtol=1e-3)
            if np.any(error):
                delta = np.abs(events[f"old_{name}"].ps - events[name].ps)
                worst = np.max(delta) == delta
                worst_events = events[worst][0]

                logging.warning(f"Error {name}: delta ps {delta[worst]}")
                logging.warning("Worst Event with error:")
                logging.warning(f"event: {worst_events.event} run: {worst_events.run} lumi: {worst_events.luminosityBlock}")
                logging.warning("----------")
                logging.warning("New Event:")
                for field in events[name].fields:
                    logging.warning(f"{field} {events[name][worst][field]}")

                logging.warning("----------")
                logging.warning("Old Event:")
                for field in events[f"old_{name}"].fields:
                    logging.warning(f"{field} {events[f'old_{name}'][worst][field]}")

                logging.warning("----------")

                for field in events[name].fields:
                    logging.warning(f"{field} {events[name][worst][field]}")
