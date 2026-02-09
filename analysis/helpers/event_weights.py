from src.physics.common import apply_btag_sf
from src.physics.event_weights import add_weights as base_add_weights
import correctionlib
import awkward as ak
import numpy as np
import uproot
import logging

def add_weights(
        event,
        dataset: str = None,
        year_label: str = None,
        corrections_metadata: dict = None,
        apply_trigWeight: bool = True,
        friend_trigWeight: callable = None,
        target: callable = None,
        run_systematics: bool = False,
        config : dict = {"do_MC_weights": True, "isTTForMixed": False}
    ):

    weights, list_weight_names = base_add_weights(
        event,
        do_MC_weights=config["do_MC_weights"],
        dataset=dataset,
        year_label=year_label,
        corrections_metadata=corrections_metadata,
        isTTForMixed=config["isTTForMixed"],
        run_systematics=run_systematics
    )

    if apply_trigWeight and config["do_MC_weights"]:

        trigWeight = event.trigWeight if "trigWeight" in event.fields else friend_trigWeight.arrays(target) if friend_trigWeight else logging.error(f"No friend tree for trigWeight found.")

        if run_systematics:
            hlt = ak.where(event.passHLT, 1., 0.) # type: ignore
            weights.add(
                "CMS_bbbb_resolved_ggf_triggerEffSF",
                trigWeight.Data, ##* ak.where(trigWeight.MC != 0, hlt / trigWeight.MC, 1) ### uncomment for new data.
                trigWeight.MC,
                hlt
            )
        else:
            if config["isRun3"]:

                safe_m_total = ak.where(trigWeight.MC > 0, trigWeight.MC, 1.0)
                raw_sf = trigWeight.Data / safe_m_total
                sf = ak.where(trigWeight.MC > 0, raw_sf, 1.0)


                weights.add(
                    "CMS_bbbb_resolved_ggf_triggerEffSF",
                    sf, #trigWeight.Data
                )

            else:
                weights.add(
                    "CMS_bbbb_resolved_ggf_triggerEffSF",
                    trigWeight.Data
                )
        list_weight_names.append('CMS_bbbb_resolved_ggf_triggerEffSF')
        logging.debug( f"trigWeight {weights.partial_weight(include=['CMS_bbbb_resolved_ggf_triggerEffSF'])[:10]}\n" )

    return weights, list_weight_names

def add_pseudotagweights(
    event,
    weights,
    JCM: callable = None,
    # JCM_lowpt: callable = None,
    apply_FvT: bool = False,
    isDataForMixed: bool = False,
    list_weight_names: list = [],
    event_metadata: dict = {},
    year_label: str = None,
    len_event: int = None,
    label3b: str = "threeTag",
    lowpt: bool = False,
):
    """
    Add pseudo-tagging weights to the selected events and update the weights object.

    Args:
        event (awkward.Array): Selected events.
        weights (Weights): Weights object to store the calculated weights.
        JCM (callable, optional): Jet Combinatoric Model function for pseudo-tagging. Defaults to None.
        apply_FvT (bool, optional): Whether to apply FvT weights. Defaults to False.
        JCM_lowpt (callable, optional): Jet Combinatoric Model function for low-pt jets. Defaults to None.
        isDataForMixed (bool, optional): Whether the data is for mixed events. Defaults to False.
        list_weight_names (list, optional): List to store the names of added weights. Defaults to an empty list.
        event_metadata (dict, optional): Metadata for the event. Defaults to an empty dictionary.
        year_label (str, optional): Year label for the dataset. Defaults to None.
        len_event (int, optional): Total number of events. Defaults to None.
        label3b (str, optional): Label for three-tag events. Defaults to "threeTag".

    Returns:
        tuple: Updated weights object and list of weight names.
    """

    # Calculate weight without JCM and FvT
    all_weights = ['genweight', 'CMS_bbbb_resolved_ggf_triggerEffSF', f'CMS_pileup_{year_label}', 'CMS_btag']
    logging.debug( f"noJCM_noFVT partial {weights.partial_weight(include=all_weights)[:10]}" )
    event["weight_noJCM_noFvT"] = weights.partial_weight(include=all_weights)

    if JCM:

        # Calculate pseudo-tagging weights
        selected_jets = event.Jet.selected_lowpt if lowpt else event.Jet.selected
        tagged_loose = event.Jet.tagged_loose_lowpt if lowpt else event.Jet.tagged_loose
        event["Jet_untagged_loose"] = event.Jet[selected_jets & ~tagged_loose]
        pseudoTagWeight = np.ones(len(event), dtype=float)
        nJet_pseudotagged = np.zeros(len(event), dtype=int)

        pseudoTagWeight[event[label3b]], nJet_pseudotagged[event[label3b]] = JCM(
            ak.num(event[event[label3b]]['Jet_untagged_loose'], axis=1),
            event.event[event[label3b]]
        )
        event["nJet_pseudotagged"] = nJet_pseudotagged
        event["pseudoTagWeight"] = pseudoTagWeight
        logging.debug( f"pseudoTagWeight {event.pseudoTagWeight[:10]}\n" )
        logging.debug( f"nJet_pseudotagged {event.nJet_pseudotagged[:10]}\n" )

        # Update number of tagged jets
        tagJet = event.tagJet_lowpt if lowpt else event.tagJet
        tagJet_loose = event.tagJet_loose_lowpt if lowpt else event.tagJet_loose
        nTagJets = ak.num(tagJet, axis=1).to_numpy()
        nTagJets[event[label3b]] = ak.num(tagJet_loose[event[label3b]], axis=1)
        event["nJet_ps_and_tag"] = nJet_pseudotagged + nTagJets

        event["pseudoTagWeight_lowpt"] = np.ones(len(event), dtype=float)  # Initialize dummy lowpt pseudoTagWeight

        # if 'lowpt' in label3b:
        #     event["weight_noJCM_lowpt_noFvT"] = event.weight_noJCM_noFvT * event.pseudoTagWeight

        #     if JCM_lowpt:        print(f"event ids: {event.event[event[label3b]][:20]}")

        #         event["Jet_untagged_loose_lowpt"] = event.Jet[event.Jet.selected_lowpt & ~event.Jet.tagged_loose_lowpt]
        #         pseudoTagWeight_lowpt = np.ones(len(event), dtype=float)
        #         nJet_pseudotagged_lowpt = np.zeros(len(event), dtype=int)

        #         pseudoTagWeight_lowpt[event[label3b]], nJet_pseudotagged_lowpt[event[label3b]] = JCM_lowpt(
        #             ak.num(event[event[label3b]]['Jet_untagged_loose_lowpt'], axis=1),
        #             event.event[event[label3b]]
        #         )
        #         event["nJet_pseudotagged_lowpt"] = nJet_pseudotagged_lowpt
        #         event["pseudoTagWeight_lowpt"] = pseudoTagWeight_lowpt
        #         logging.debug( f"pseudoTagWeight_lowpt {event.pseudoTagWeight_lowpt[:10]}\n" )
        #         logging.debug( f"nJet_pseudotagged_lowpt {event.nJet_pseudotagged_lowpt[:10]}\n" )

        #         # Update number of tagged jets
        #         nTagJets_lowpt = ak.num(event.Jet.tagged_lowpt, axis=1).to_numpy()
        #         nTagJets_lowpt[event[label3b]] = ak.num(event.Jet.tagged_loose_lowpt[event[label3b]], axis=1)
        #         event["nJet_ps_and_tag_lowpt"] = nJet_pseudotagged_lowpt + nTagJets_lowpt

        # Calculate weight without FvT
        weight_noFvT = ak.where(
            event[label3b], 
            event.weight * event.pseudoTagWeight * event.pseudoTagWeight_lowpt, 
            event.weight
        )

        event["weight_noFvT"] = weight_noFvT
        logging.debug( f"weight_noFvT {event.weight_noFvT[:10]}\n" )

        # Apply FvT weights if required
        if apply_FvT:
            if isDataForMixed:
                for _JCM_load, _FvT_name in zip(event_metadata["JCM_loads"], event_metadata["FvT_names"]):
                    event[f"weight_{_FvT_name}"] = np.where(
                        event[label3b],
                        event.weight * getattr(event, f"{_JCM_load}") * getattr(getattr(event, _FvT_name), _FvT_name),
                        event.weight,
                    )

                weight_JCM = np.where(
                    event[label3b],
                    getattr(event, f"{event_metadata['JCM_loads'][0]}"),
                    1.0
                )
                weights.add("JCM", weight_JCM)
                list_weight_names.append("JCM")
                logging.debug( f"JCM {weights.partial_weight(include=['JCM'])[:10]}\n" )

                weight_FvT = np.where(
                    event[label3b],
                    event.FvT.FvT,
                    1.0
                )
                weights.add("FvT", weight_FvT)
                list_weight_names.append("FvT")
                logging.debug( f"FvT {weights.partial_weight(include=['FvT'])[:10]}\n" )

            else:
                weight = np.where(
                    event[label3b],
                    event["pseudoTagWeight"] * event["pseudoTagWeight_lowpt"] * event.FvT.FvT,
                    1.0
                )
                weights.add("FvT", weight)
                list_weight_names.append("FvT")
                logging.debug( f"FvT {weights.partial_weight(include=['FvT'])[:10]}\n" )
        else:
            weight_noFvT = np.copy(event.weight)
            weight_noFvT = np.where(
                event[label3b],
                event["pseudoTagWeight"] * event["pseudoTagWeight_lowpt"],
                1.0
            )
            weights.add("no_FvT", weight_noFvT)
            list_weight_names.append("no_FvT")
            logging.debug( f"no_FvT {weights.partial_weight(include=['no_FvT'])[:10]}\n" )

    return weights, list_weight_names

def add_btagweights( event, weights,
                    list_weight_names: list = [],
                    shift_name: str = None,
                    run_systematics: bool = False,
                    use_prestored_btag_SF: bool = False,
                    corrections_metadata: dict = None,
                    ):

    if use_prestored_btag_SF:
        weights.add( "CMS_btag", event.CMSbtag )
    else:

        sys_value = "central"
        if shift_name and ( 'CMS_scale_j_' in shift_name ):
            if 'Down' in shift_name:
                sys_value = f"down_jes{shift_name.replace('CMS_scale_j_', '').replace('Down', '')}"
            elif 'Up' in shift_name:
                sys_value = f"up_jes{shift_name.replace('CMS_scale_j_', '').replace('Up', '')}"
        logging.debug(f"shift_name: {shift_name}, sys_value: {sys_value}\n\n")

        btag_SF_weights = apply_btag_sf(
            event.selJet_no_bRegCorr,
            sys_value="central",
            correction_file=corrections_metadata["btagSF"],
            btag_uncertainties=corrections_metadata["btag_uncertainties"] if (not shift_name) & run_systematics else None
        )

        if (not shift_name) & run_systematics:
            weights.add_multivariation( f"CMS_btag", btag_SF_weights["btagSF_central"],
                                        corrections_metadata["btag_uncertainties"],
                                        [ var.to_numpy() for name, var in btag_SF_weights.items() if "_up" in name ],
                                        [ var.to_numpy() for name, var in btag_SF_weights.items() if "_down" in name ], )
        else:
            weights.add( "CMS_btag", btag_SF_weights["btagSF_central"] )

    list_weight_names.append(f"CMS_btag")

    return weights, list_weight_names
