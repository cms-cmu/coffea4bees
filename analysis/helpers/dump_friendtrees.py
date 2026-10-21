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
    threeTag_label: str = "threeTag",
    fourTag_label: str = "fourTag",
    seljet_label: str | list[str] = "nSelJets",
    dump_naming: str = _NAMING,
):
    selection = _build_cutflow(*selections)
    if len(events) == 0:
        n = len(selection)
        canjet_fields = {
            feat: ak.unflatten(np.zeros(0, dtype=np.float32), np.zeros(n, dtype=int))
            for feat in (
                "pt", "eta", "phi", "mass",
                "btagScore", "btagPNetCvB", "btagPNetCvL", "btagPNetQvG", "btagPNetTauVJet",
                "PNetRegPtRawRes", "nSVs", "chHEF", "neHEF", "chEmEF", "neEmEF", "muEF",
                "nConstituents", "area", "rawFactor",
            )
        }
        notcanjet_fields = {
            "pt": ak.unflatten(np.zeros(0, dtype=np.float32), np.zeros(n, dtype=int)),
            "eta": ak.unflatten(np.zeros(0, dtype=np.float32), np.zeros(n, dtype=int)),
            "phi": ak.unflatten(np.zeros(0, dtype=np.float32), np.zeros(n, dtype=int)),
            "mass": ak.unflatten(np.zeros(0, dtype=np.float32), np.zeros(n, dtype=int)),
            "isSelJet": ak.unflatten(np.zeros(0, dtype=bool), np.zeros(n, dtype=int)),
        }
        base_dict = {
            "CanJet": ak.zip(canjet_fields),
            "NotCanJet": ak.zip(notcanjet_fields),
        }
        seljet_labels = [seljet_label] if isinstance(seljet_label, str) else list(seljet_label)
        mass_regions = ["ZZSR", "ZHSR", "HHSR", "SR", "SB"]
        numpy_fields = {
            k: np.zeros(n, dtype=bool) for k in mass_regions + [fourTag_label, threeTag_label, "passHLT"]
        }
        for k in ["xbW", "xW"]:
            numpy_fields[k] = np.zeros(n, dtype=np.float32)
        for k in seljet_labels:
            numpy_fields[k] = np.zeros(n, dtype=np.int32)
        weight_field = {"weight": np.zeros(n, dtype=np.float32)}
        merged_dict = {**base_dict, **numpy_fields, **weight_field}
        data = ak.Array(merged_dict)
        return dump_friend(
            events=events,
            output=output,
            name=name,
            data=data,
            dump_naming=dump_naming,
        )

    padded = akext.pad.selected()
    # Build the dictionary step by step, then pass to ak.Array
    canjet_fields = {
        "pt": events[CanJet].pt,
        "eta": events[CanJet].eta,
        "phi": events[CanJet].phi,
        "mass": events[CanJet].mass,
    }
    # Extra per-jet classifier-input features, included when attached to the
    # candidate jets (see cand_jet_selection). Kept in sync with what is
    # available so Run2/Run3 inputs both dump cleanly.
    for _feat in (
        "btagScore",
        "btagPNetCvB", "btagPNetCvL", "btagPNetQvG", "btagPNetTauVJet",
        "PNetRegPtRawRes", "nSVs",
        "chHEF", "neHEF", "chEmEF", "neEmEF", "muEF",
        "nConstituents", "area", "rawFactor",
    ):
        if _feat in events[CanJet].fields:
            canjet_fields[_feat] = events[CanJet][_feat]
        else:
            canjet_fields[_feat] = ak.zeros_like(events[CanJet].pt)
    base_dict = {
        "CanJet": padded(
            ak.zip(canjet_fields),
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
    # Add numpy fields
    seljet_labels = [seljet_label] if isinstance(seljet_label, str) else list(seljet_label)
    mass_regions = [m for m in ["ZZSR", "ZHSR", "HHSR", "SR", "SB"] if m in events.fields]
    fields = mass_regions + [
        fourTag_label,
        threeTag_label,
        "passHLT",
        "xbW",
        "xW",
    ] + seljet_labels
    numpy_fields = akext.to_numpy(
        padded(
            events[fields],
            selection,
        )
    )
    # Add weight field
    weight_field = {"weight": padded(events[weight], selection)}
    # Merge all dictionaries
    merged_dict = {**base_dict, **numpy_fields, **weight_field}
    data = ak.Array(merged_dict)
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
    selection = _build_cutflow(*selections)
    if len(events) == 0 or pseudo_tag not in events.fields:
        weight = np.ones(len(selection), dtype=np.float64)
    else:
        padded = akext.pad.selected(1)
        weight = padded(events[pseudo_tag], selection)
    return dump_friend(
        events=events,
        output=output,
        name=name,
        data=ak.Array({"pseudoTagWeight": weight}),
        dump_naming=dump_naming,
    )


def dump_isParking(
    events: ak.Array,
    output: PathLike,
    name: str,
    is_parking: ak.Array,
    dump_naming: str = _NAMING,
):
    return dump_friend(
        events=events,
        output=output,
        name=name,
        data=ak.Array({"is_parking": ak.values_astype(is_parking, "bool")}),
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
    selection = _build_cutflow(*selections)
    if len(events) == 0 or not has_record(events, FvT_name) == FvT_name:
        weight = np.ones(len(selection), dtype=np.float64)
    else:
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
    selection = _build_cutflow(*selections)
    data = ak.zip(
        {
            "MC": events["trigWeight"].MC,
            "Data": events["trigWeight"].Data,
        }
    )
    if len(data) == len(selection):
        data = data[selection]
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
    selection = _build_cutflow(*selections)
    if len(events) == 0:
        n = len(selection)
        top_reco_fields = [
            "b_pt", "b_eta", "b_phi", "b_mass", "b_puId", "b_jetId", "b_btagScore",
            "p_pt", "p_eta", "p_phi", "p_mass",
            "W_p_pt", "W_p_eta", "W_p_phi", "W_p_mass",
            "W_pW_pt", "W_pW_eta", "W_pW_phi", "W_pW_mass",
            "W_l_pt", "W_l_eta", "W_l_phi", "W_l_mass", "W_l_puId", "W_l_jetId", "W_l_btagScore",
            "W_j_pt", "W_j_eta", "W_j_phi", "W_j_mass", "W_j_puId", "W_j_jetId", "W_j_btagScore",
            "mbW", "xW", "xt", "xbW", "xWt", "xWbW", "rWbW",
        ]
        data = ak.zip({f: np.zeros(n, dtype=np.float32) for f in top_reco_fields})
        return dump_friend(
            events=events,
            output=output,
            name=name,
            data=data,
            dump_naming=dump_naming,
        )

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
    padded = akext.pad.selected()
    data = padded(data, selection)
    return dump_friend(
        events=events,
        output=output,
        name=name,
        data=data,
        dump_naming=dump_naming,
    )


def dump_SvB_FeynNet(
    events: ak.Array,
    output: PathLike,
    name: str,
    *selections: ak.Array,
    dump_naming: str = _NAMING,
):
    data = ak.zip({
        "p_ggHH_vs_bkg": events[name].p_ggHH_vs_bkg,
        "p_qqHH_vs_bkg": events[name].p_qqHH_vs_bkg,
        "p_ZZ_vs_bkg": events[name].p_ZZ_vs_bkg,
        "p_ZH_vs_bkg": events[name].p_ZH_vs_bkg,
        "p_bkg": events[name].p_bkg,
        "passMinPs": events[name].passMinPs,
        "hh": events[name].hh,
        "zz": events[name].zz,
        "zh": events[name].zh,
        "reweight": events[name].reweight,
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


def dump_SvB(
    events: ak.Array,
    output: PathLike,
    name: str,
    *selections: ak.Array,
    dump_naming: str = _NAMING,
):
    selection = _build_cutflow(*selections)
    if len(events) == 0:
        n = len(selection)
        all_fields = [
            'pmj', 'ptt', 'pzz', 'pzh', 'phh', 'pttHbb',
            'q_1234', 'q_1324', 'q_1423',
            'ps', 'passMinPs', 'zz', 'zh', 'hh',
            'ps_zz', 'ps_zh', 'ps_hh', 'ps_ttHbb',
            'tt_vs_mj'
        ]
        data = ak.zip({f: np.zeros(n, dtype=np.float32) for f in all_fields})
        return dump_friend(
            events=events,
            output=output,
            name=name,
            data=data,
            dump_naming=dump_naming,
        )

    ev_name = events[name]
    fields = ev_name.fields
    data_dict = {}
    for f in [
        'pmj', 'ptt', 'pzz', 'pzh', 'phh', 'pttHbb',
        'q_1234', 'q_1324', 'q_1423',
        'ps', 'passMinPs', 'zz', 'zh', 'hh',
        'ps_zz', 'ps_zh', 'ps_hh', 'ps_ttHbb',
        'tt_vs_mj'
    ]:
        if f in fields:
            data_dict[f] = getattr(ev_name, f)

    data = ak.zip(data_dict)
    padded = akext.pad.selected()
    data = padded(data, selection)
    return dump_friend(
        events=events,
        output=output,
        name=name,
        data=data,
        dump_naming=dump_naming,
    )
