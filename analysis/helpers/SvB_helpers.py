from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import awkward as ak
import numpy as np
from src.math_tools.random import Squares

if TYPE_CHECKING:
    from .classifier.HCR import HCREnsemble

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

        tmp_c_score, tmp_q_score = model(masked_events)

        c_score = np.zeros((len(events),tmp_c_score.shape[1]))
        c_score[mask] = tmp_c_score
        q_score = np.zeros((len(events),tmp_q_score.shape[1]))
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
