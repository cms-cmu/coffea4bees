import awkward as ak
import src.data_formats.awkward as akext
import numpy as np
from src.data_formats.root import Chunk, Friend
from src.storage.eos import PathLike
from src.aktools import has_record, get_field
from src.friendtrees.dump_friend import dump_friend, _build_cutflow

_NAMING = "{path1}/{name}_{uuid}_{start}_{stop}_{path0}"

# TODO: dump trigger weight?


def dump_input_friend(
    events: ak.Array,
    output: PathLike,
    name: str,
    *selections: ak.Array,
    CanJet: str = "canJet",
    NotCanJet: str = "notCanJet",
    weight: str = "weight",
    dump_naming: str = _NAMING,
):
    selection = _build_cutflow(*selections)
    padded = akext.pad.selected()
    data = ak.Array(
        {
            "CanJet": padded(
                ak.zip(
                    {
                        "pt": events[CanJet].pt,
                        "eta": events[CanJet].eta,
                        "phi": events[CanJet].phi,
                        "mass": events[CanJet].mass,
                    }
                ),
                selection,
            ),
            "NotCanJet": padded(
                ak.zip(
                    {
                        "pt": events[NotCanJet].pt,
                        "eta": events[NotCanJet].eta,
                        "phi": events[NotCanJet].phi,
                        "mass": events[NotCanJet].mass,
                        "isSelJet": events[NotCanJet].isSelJet,
                    }
                ),
                selection,
            ),
        }
        | akext.to_numpy(
            padded(
                events[
                    [
                        "ZZSR",
                        "ZHSR",
                        "HHSR",
                        "SR",
                        "SB",
                        "fourTag",
                        "threeTag",
                        "passHLT",
                        "nSelJets",
                        "xbW",
                        "xW",
                    ]
                ],
                selection,
            )
        )
        | {"weight": padded(events[weight], selection)}
    )
    return dump_friend(
        events=events,
        output=output,
        name=name,
        data=data,
        dump_naming=dump_naming,
    )


def dump_JCM_weight(
    events: ak.Array,
    output: PathLike,
    name: str,
    *selections: ak.Array,
    pseudo_tag: str = "pseudoTagWeight",
    dump_naming: str = _NAMING,
):
    if pseudo_tag not in events.fields:
        weight = np.ones(len(selections[0]), dtype=np.float64)
    else:
        selection = _build_cutflow(*selections)
        padded = akext.pad.selected(1)
        weight = padded(events[pseudo_tag], selection)
    return dump_friend(
        events=events,
        output=output,
        name=name,
        data=ak.Array({"pseudoTagWeight": weight}),
        dump_naming=dump_naming,
    )


def dump_FvT_weight(  ### TODO: replace with proper evaluation code
    events: ak.Array,
    output: PathLike,
    name: str,
    *selections: ak.Array,
    FvT_name: tuple[str, ...] = ("FvT", "FvT"),
    dump_naming: str = "{path1}/{name}_{uuid}_{start}_{stop}_{path0}",
):
    if not has_record(events, FvT_name) == FvT_name:
        weight = np.ones(len(selections[0]), dtype=np.float64)
    else:
        selection = _build_cutflow(*selections)
        padded = akext.pad.selected(1)
        weight = padded(get_field(events, FvT_name), selection)
    return dump_friend(
        events=events,
        output=output,
        name=name,
        data=ak.Array({"FvT": weight}),
        dump_naming=dump_naming,
    )


def dump_unsup_friend(
    events: ak.Array,
    output: PathLike,
    name: str,
    *selections: ak.Array,
    dump_naming: str = "{path1}/{name}_{start}_{stop}_{path0}",
):
    data = ak.zip(
        {
            "run": events["run"],
            "event": events["event"],
            "m4j": events["m4j"],
            "leadStM": events["leadStM"],
            "sublStM": events["sublStM"],
            "nSelJets": events["nSelJets"],
            "weight": events["weight"],
            "passHLT": events["passHLT"],
            "lumimask": events["lumimask"],
            "passNoiseFilter": events["passNoiseFilter"],
            "passJetMult": events["passJetMult"],
            "fourTag": events["fourTag"],
            "threeTag": events["threeTag"],
        }
    )
    selection = _build_cutflow(*selections)
    padded = akext.pad.selected()
    data = padded(data, selection)
    return dump_friend(
        events=events,
        output=output,
        name=name,
        data=data,
        dump_naming=dump_naming,
    )


def dump_trigger_weight(
    events: ak.Array,
    output: PathLike,
    name: str,
    *selections: ak.Array,
    dump_naming: str = _NAMING,
):
    data = ak.zip(
        {
            "MC": events["trigWeight"].MC,
            "Data": events["trigWeight"].Data,
        }
    )
    selection = _build_cutflow(*selections)
    padded = akext.pad.selected()
    data = padded(data, selection)
    return dump_friend(
        events=events,
        output=output,
        name=name,
        data=data,
        dump_naming=dump_naming,
    )


def dump_top_reconstruction(
    events: ak.Array,
    output: PathLike,
    name: str,
    *selections: ak.Array,
    dump_naming: str = _NAMING,
):
    data = ak.zip(
        {
            "b_pt": events["top_cand"].b.pt,
            "b_eta": events["top_cand"].b.eta,
            "b_phi": events["top_cand"].b.phi,
            "b_mass": events["top_cand"].b.mass,
            "b_puId": events["top_cand"].b.puId,
            "b_jetId": events["top_cand"].b.jetId,
            "b_btagScore": events["top_cand"].b.btagScore,
            "p_pt": events["top_cand"].p.pt,
            "p_eta": events["top_cand"].p.eta,
            "p_phi": events["top_cand"].p.phi,
            "p_mass": events["top_cand"].p.mass,
            "W_p_pt": events["top_cand"].W.p.pt,
            "W_p_eta": events["top_cand"].W.p.eta,
            "W_p_phi": events["top_cand"].W.p.phi,
            "W_p_mass": events["top_cand"].W.p.mass,
            "W_pW_pt": events["top_cand"].W.pW.pt,
            "W_pW_eta": events["top_cand"].W.pW.eta,
            "W_pW_phi": events["top_cand"].W.pW.phi,
            "W_pW_mass": events["top_cand"].W.pW.mass,
            "W_l_pt": events["top_cand"].W.l.pt,
            "W_l_eta": events["top_cand"].W.l.eta,
            "W_l_phi": events["top_cand"].W.l.phi,
            "W_l_mass": events["top_cand"].W.l.mass,
            "W_l_puId": events["top_cand"].W.l.puId,
            "W_l_jetId": events["top_cand"].W.l.jetId,
            "W_l_btagScore": events["top_cand"].W.l.btagScore,
            "W_j_pt": events["top_cand"].W.j.pt,
            "W_j_eta": events["top_cand"].W.j.eta,
            "W_j_phi": events["top_cand"].W.j.phi,
            "W_j_mass": events["top_cand"].W.j.mass,
            "W_j_puId": events["top_cand"].W.j.puId,
            "W_j_jetId": events["top_cand"].W.j.jetId,
            "W_j_btagScore": events["top_cand"].W.j.btagScore,
            "mbW": events["top_cand"].mbW,
            "xW": events["top_cand"].xW,
            "xt": events["top_cand"].xt,
            "xbW": events["top_cand"].xbW,
            "xWt": events["top_cand"].xWt,
            "xWbW": events["top_cand"].xWbW,
            "rWbW": events["top_cand"].rWbW,
        }
    )
    selection = _build_cutflow(*selections)
    padded = akext.pad.selected()
    data = padded(data, selection)
    return dump_friend(
        events=events,
        output=output,
        name=name,
        data=data,
        dump_naming=dump_naming,
    )


def dump_SvB(
    events: ak.Array,
    output: PathLike,
    name: str,
    *selections: ak.Array,
    dump_naming: str = _NAMING,
):
    data = ak.zip({
        'pmj': events[name].pmj,
        'ptt': events[name].ptt, 
        "pzz": events[name].pzz,
        "pzh": events[name].pzh,
        "phh": events[name].phh,
        "q_1234": events[name].q_1234,
        "q_1324": events[name].q_1324,
        "q_1423": events[name].q_1423,
        "ps": events[name].ps,
        "passMinPs": events[name].passMinPs,
        "zz": events[name].zz,
        "zh": events[name].zh,
        "hh": events[name].hh,
        "ps_zz": events[name].ps_zz,
        "ps_zh": events[name].ps_zh,
        "ps_hh": events[name].ps_hh,
        # "largest": events[name].largest,
        # "weight": events.weight,
        "tt_vs_mj": events[name].tt_vs_mj
        })
    selection = _build_cutflow(*selections)
    padded = akext.pad.selected()
    data = padded(data, selection)
    return dump_friend(
        events=events,
        output=output,
        name=name,
        data=data,
        dump_naming=dump_naming,
    )
