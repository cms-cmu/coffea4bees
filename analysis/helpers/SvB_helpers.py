from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import awkward as ak
import numpy as np
from src.math_tools.random import Squares

if TYPE_CHECKING:
    from coffea4bees.analysis.helpers.classifier.HCR import HCREnsemble

def setSvBVars(SvBName, event):

    event[SvBName, "passMinPs"] = ( (getattr(event, SvBName).pzz > 0.01)
                                    | (getattr(event, SvBName).pzh > 0.01)
                                    | (getattr(event, SvBName).phh > 0.01) )

    event[SvBName, "zz"] = ( getattr(event, SvBName).pzz >  getattr(event, SvBName).pzh ) & (getattr(event, SvBName).pzz > getattr(event, SvBName).phh)

    event[SvBName, "zh"] = ( getattr(event, SvBName).pzh >  getattr(event, SvBName).pzz ) & (getattr(event, SvBName).pzh > getattr(event, SvBName).phh)

    event[SvBName, "hh"] = ( getattr(event, SvBName).phh >= getattr(event, SvBName).pzz ) & (getattr(event, SvBName).phh >= getattr(event, SvBName).pzh)

    event[SvBName, "tt_vs_mj"] = ( getattr(event, SvBName).ptt / (getattr(event, SvBName).ptt + getattr(event, SvBName).pmj) )


    #
    #  Set ps_{bb}
    #
    this_ps_zz = np.full(len(event), -1, dtype=float)
    this_ps_zz[getattr(event, SvBName).zz] = getattr(event, SvBName).ps[ getattr(event, SvBName).zz ]
    this_ps_zz[getattr(event, SvBName).passMinPs == False] = -2
    event[SvBName, "ps_zz"] = this_ps_zz

    this_ps_zh = np.full(len(event), -1, dtype=float)
    this_ps_zh[getattr(event, SvBName).zh] = getattr(event, SvBName).ps[ getattr(event, SvBName).zh ]
    this_ps_zh[getattr(event, SvBName).passMinPs == False] = -2
    event[SvBName, "ps_zh"] = this_ps_zh

    this_ps_hh = np.full(len(event), -1, dtype=float)
    this_ps_hh[getattr(event, SvBName).hh] = getattr(event, SvBName).ps[ getattr(event, SvBName).hh ]
    this_ps_hh[getattr(event, SvBName).passMinPs == False] = -2
    event[SvBName, "ps_hh"] = this_ps_hh

    this_phh_hh = np.full(len(event), -1, dtype=float)
    this_phh_hh[getattr(event, SvBName).hh] = getattr(event, SvBName).phh[ getattr(event, SvBName).hh ]
    this_phh_hh[getattr(event, SvBName).passMinPs == False] = -2
    event[SvBName, "phh_hh"] = this_phh_hh


def compute_SvB(events, mask, doCheck=True, **models: HCREnsemble):
    masked_events = events[mask]

    for name, model in models.items():
        if model is None:
            continue

        if name in events.fields:
            events[f"old_{name}"] = events[name]

        # Handle empty mask case - if no events pass the mask, skip model evaluation
        if len(masked_events) == 0:
            # Create zero arrays with proper shape for all events
            classes = model.classes
            tmp_c_score = np.zeros((0, len(classes)))
            tmp_q_score = np.zeros((0, 3))
        else:
            # Additional checks for tensor compatibility
            try:
                # Check if required fields have proper structure before model evaluation
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
                    # This happens when masked_events appears non-empty but has empty arrays internally
                    logging.warning(f"Model {name}: Detected tensor reshape error with 0 elements, creating zero arrays instead. Error: {str(e)}")
                    classes = model.classes
                    tmp_c_score = np.zeros((0, len(classes)))
                    tmp_q_score = np.zeros((0, 3))
                else:
                    raise e

        c_score = np.zeros((len(events),tmp_c_score.shape[1]))
        q_score = np.zeros((len(events),tmp_q_score.shape[1]))

        # Only assign if we have actual scores to assign
        if tmp_c_score.shape[0] > 0:
            c_score[mask] = tmp_c_score
            q_score[mask] = tmp_q_score

        del tmp_c_score, tmp_q_score

        classes = model.classes
        pmj = c_score[:, classes.index("multijet")]
        ptt = c_score[:, classes.index("ttbar")]
        pzz = c_score[:, classes.index("ZZ")]
        pzh = c_score[:, classes.index("ZH")]
        phh = c_score[:, classes.index("ggF")]
        ps = pzz + pzh + phh
        passMinPs = (pzz > 0.01) | (pzh > 0.01) | (phh > 0.01)

        zz = (pzz > pzh) & (pzz > phh)
        this_ps_zz = np.full(len(events), -1, dtype=float)
        this_ps_zz[ zz ] = ps[zz]
        this_ps_zz[ passMinPs == False ] = -2
        ps_zz = this_ps_zz

        zh = (pzh > pzz) & (pzh > phh)
        this_ps_zh = np.full(len(events), -1, dtype=float)
        this_ps_zh[ zh ] = ps[zh]
        this_ps_zh[ passMinPs == False ] = -2
        ps_zh = this_ps_zh

        hh = (phh > pzz) & (phh > pzh)
        this_ps_hh = np.full(len(events), -1, dtype=float)
        this_ps_hh[ hh ] = ps[hh]
        this_ps_hh[ passMinPs == False ] = -2
        ps_hh = this_ps_hh

        this_phh_hh = np.full(len(events), -1, dtype=float)
        this_phh_hh[ hh ] = phh[hh]
        this_phh_hh[ passMinPs == False ] = -2
        phh_hh = this_phh_hh


        largest_name = np.array(["None", "ZZ", "ZH", "HH"])
        events[name] = ak.zip({
            "pmj": pmj,
            "ptt": ptt,
            "pzz": pzz,
            "pzh": pzh,
            "phh": phh,
            "q_1234": q_score[:, 0],
            "q_1324": q_score[:, 1],
            "q_1423": q_score[:, 2],
            "ps": ps,
            "passMinPs": passMinPs,
            "zz": zz,
            "zh": zh,
            "hh": hh,
            "ps_zz": ps_zz,
            "ps_zh": ps_zh,
            "ps_hh": ps_hh,
            "phh_hh": phh_hh,
            "largest": largest_name[ (passMinPs * ( 1 * zz + 2* zh + 3*hh ) ) ],
            "tt_vs_mj": ( ptt / (ptt + pmj) )
        })

        if doCheck and f"old_{name}" in events.fields:
            error = ~np.isclose(events[f"old_{name}"].ps, events[name].ps, atol=1e-5, rtol=1e-3)
            if np.any(error):
                delta = np.abs(events[f"old_{name}"].ps - events[name].ps)
                worst = np.max(delta) == delta
                worst_events = events[worst][0]

                logging.warning( f"WARNING: Calculated {name} does not agree within tolerance for some events ({np.sum(error)}/{len(error)}) {delta[worst]}" )

                logging.warning("----------")

                for field in events[name].fields:
                    logging.warning(f"{field} {worst_events[name][field]}")

                logging.warning("----------")

                for field in events[name].fields:
                    logging.warning(f"{field} {events[name][worst][field]}")


def compute_SvB_FeynNet(events, mask, **models):
    """Compute FeynNet SvB scores and store in events[name].

    The FeynNet model classes are [ggHH, qqHH, ZZ, ZH, Background].
    Derived fields mirror the structure of compute_SvB for compatibility.

    Args:
        events: full event array (unmasked)
        mask: boolean mask — only masked events are fed to the model
        **models: keyword args mapping field name → FeynNetEnsemble instance
                  e.g. compute_SvB_FeynNet(events, mask, SvB_FeynNet=ensemble)
    """
    masked_events = events[mask]

    for name, model in models.items():
        if model is None:
            continue

        if len(masked_events) == 0:
            c_score = np.zeros((0, len(model.classes)), dtype=np.float32)
            q_score = np.zeros((0, 3), dtype=np.float32)
        else:
            c_score, q_score = model(masked_events)

        c_full = np.zeros((len(events), len(model.classes)), dtype=np.float32)
        q_full = np.zeros((len(events), q_score.shape[1] if q_score.ndim > 1 else 1), dtype=np.float32)
        if c_score.shape[0] > 0:
            c_full[mask] = c_score
            q_full[mask] = q_score if q_score.ndim > 1 else q_score[:, np.newaxis]

        classes = model.classes
        p_ggHH = c_full[:, classes.index("ggHH")]
        p_qqHH = c_full[:, classes.index("qqHH")]
        p_ZZ   = c_full[:, classes.index("ZZ")]
        p_ZH   = c_full[:, classes.index("ZH")]
        p_bkg  = c_full[:, classes.index("Background")]

        p_ggHH_vs_bkg = p_ggHH / (p_ggHH + p_bkg)
        p_qqHH_vs_bkg = p_qqHH / (p_qqHH + p_bkg)
        p_ZZ_vs_bkg   = p_ZZ   / (p_ZZ   + p_bkg)
        p_ZH_vs_bkg   = p_ZH   / (p_ZH   + p_bkg)

        passMinPs = (p_ggHH > 0.01) | (p_qqHH > 0.01) | (p_ZZ > 0.01) | (p_ZH > 0.01)

        p_hh = p_ggHH + p_qqHH  # combined HH score
        zz   = (p_ZZ  > p_ZH)  & (p_ZZ  > p_hh)
        zh   = (p_ZH  > p_ZZ)  & (p_ZH  > p_hh)
        hh   = (p_hh  >= p_ZZ) & (p_hh  >= p_ZH)


        def _ps_gated(input_var, win_mask):
            arr = np.full(len(events), -1, dtype=float)
            arr[~passMinPs] = -2
            arr[win_mask] = input_var[win_mask]
            return arr

        events[name] = ak.zip({
            "p_ggHH_vs_bkg":  _ps_gated(p_ggHH_vs_bkg,  hh),
            "p_qqHH_vs_bkg":  _ps_gated(p_qqHH_vs_bkg,  hh),
            "p_ZZ_vs_bkg":    _ps_gated(p_ZZ_vs_bkg  ,  zz),
            "p_ZH_vs_bkg":    _ps_gated(p_ZH_vs_bkg  ,  zh),
            "p_bkg"  :  p_bkg ,
            "passMinPs": passMinPs,
            "hh":        hh,
            "zz":        zz,
            "zh":        zh,
            "reweight":  q_full[:, 0],
            "tt_vs_mj":  np.zeros(len(events), dtype=np.float32),
        })


def subtract_ttbar_with_SvB(selev, dataset, year):

    #
    # Get reproducible random numbers
    #
    rng = Squares("ttbar_subtraction", dataset, year)
    counter = np.empty((len(selev), 2), dtype=np.uint64)
    counter[:, 0] = np.asarray(selev.event).view(np.uint64)
    counter[:, 1] = np.asarray(selev.run).view(np.uint32)
    counter[:, 1] <<= 32
    counter[:, 1] |= np.asarray(selev.luminosityBlock).view(np.uint32)
    ttbar_rand = rng.uniform(counter, low=0, high=1.0).astype(np.float32)

    return (ttbar_rand > selev.SvB_MA.tt_vs_mj)


def setFvTVars(FvTName, event):
    if "std" not in event.FvT.fields:
        event[FvTName, "std"] = np.ones(len(event))

    if "pt4" not in event.FvT.fields:
        event[FvTName, "pt4"] = np.ones(len(event))
        event[FvTName, "pt3"] = np.ones(len(event))
        event[FvTName, "pd4"] = np.ones(len(event))
        event[FvTName, "pd3"] = np.ones(len(event))

    event[FvTName, "frac_err"]   = getattr(event, FvTName).std / getattr(event, FvTName).FvT
    event[FvTName, "d4_to_t4"]   = getattr(event, FvTName).pt4 / getattr(event, FvTName).pd4
    event[FvTName, "d3_to_t3"]   = getattr(event, FvTName).pt3 / getattr(event, FvTName).pd3
    event[FvTName, "d3_to_t4"]   = getattr(event, FvTName).pt4 / getattr(event, FvTName).pd3






def subtract_ttbar_with_FvT(selev, dataset, year, tt_vs_mj_var="d3_to_t3"):

    #
    # Get reproducible random numbers
    #
    rng = Squares("ttbar_subtraction", dataset, year)
    counter = np.empty((len(selev), 2), dtype=np.uint64)
    counter[:, 0] = np.asarray(selev.event).view(np.uint64)
    counter[:, 1] = np.asarray(selev.run).view(np.uint32)
    counter[:, 1] <<= 32
    counter[:, 1] |= np.asarray(selev.luminosityBlock).view(np.uint32)
    ttbar_rand = rng.uniform(counter, low=0, high=1.0).astype(np.float32)

    return (ttbar_rand > getattr(selev.FvT, tt_vs_mj_var))
