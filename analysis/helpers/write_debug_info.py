import awkward as ak

def add_debug_info_to_output(event, processOutput, weights, list_weight_names, analysis_selections):
    # passSR = (selev["quadJet_selected"].SR)
    # passSR = (selev["SR"])

    out_data = {}
    # out_data["SvB"    ] = selev["SvB_MA"].ps[passSR]
    out_data["event"  ] = event["event"]#[passSR]
    out_data["run"    ] = event["run"]  #[passSR]
    out_data["lumisection"] = event["luminosityBlock"]
    out_data["fourTag"    ] = event["fourTag"]  #[passSR]

    out_data["passPreSel"    ] = event["passPreSel"]
    out_data["lumimask"    ] = event["lumimask"]
    out_data["passNoiseFilter"    ] = event["passNoiseFilter"]
    out_data["passHLT"    ] = event["passHLT"]
    out_data["passJetMult"    ] = event["passJetMult"]
    # out_data["passSR"    ] = event["SR"]

    #debug_mask = ((event["event"] == 66688  ) |
    #              (event["event"] == 249987 ) |
    #              (event["event"] == 121603 ) |
    #              (event["event"] == 7816   ) |
    #              (event["event"] == 25353  ) |
    #              (event["event"] == 165389 ) |
    #              (event["event"] == 293138 ) |
    #              (event["event"] == 150164 ) |
    #              (event["event"] == 262806 ) |
    #              (event["event"] == 281111 ) )
    # debug_mask = ((event.event == 110614) & (event.run == 275890) & (event.luminosityBlock == 1))
    # debug_event = event[debug_mask]
    # print(f"debug {debug_event.fourTag} {debug_event.threeTag} {debug_event.nJet_tagged} {debug_event.nJet_tagged_loose} {debug_event.nJet_selected} {debug_event.Jet.tagged} {debug_event.Jet.selected} {debug_event.Jet.btagScore}")
    # print(f"debug {debug_event.passHLT} {debug_event.passJetMult} {debug_event.passPreSel} {debug_event.Jet.pt} {debug_event.Jet.pt_raw} \n\n\n")

    #print(f"\n {event.Jet.pt[event.passJetMult].to_list()[0:5]} \n")

    #out_data["passJetMult_event"  ]    = event["event"][event.passJetMult]
    #out_data["passJetMult_run"    ]    = event["run"][event.passJetMult]
    #out_data["passJetMult_passDiJetMass"    ]    = event["passDiJetMass"][event.passJetMult]
    #out_data["passJetMult_weight" ]    = event["weight"]
    #
    #for _w in list_weight_names:
    #    print(f"adding {_w}\n")
    #    out_data[f"passJetMult_weight_{_w}"] = weights.partial_weight(include=[_w])[analysis_selections]


    for out_k, out_v in out_data.items():
        processOutput[out_k] = {}
        processOutput[out_k][event.metadata['dataset']] = list(out_v)


def add_debug_Run3_data_early(event, processOutput):

    out_data = {}


    #debug_mask = (event.region.SB == True)
    run_SB_mask = (event.run == 355921)

    event_mask = ((event["event"] == 632465280) |
                  (event["event"] == 109801856) |
                  (event["event"] == 136082628) |
                  (event["event"] == 187272057))

    debug_mask = run_SB_mask & event_mask

    print(debug_mask.to_list()[0:10])
    print("Any Early", ak.any(debug_mask), "len", len(debug_mask[debug_mask == True]))

    return



def add_debug_Run3_data_skim(event, processOutput, selection):

    out_data = {}


    #debug_mask = (event.region.SB == True)
    run_SB_mask = (event.run == 355921)

    event_mask = ((event["event"] == 632465280) |
                  (event["event"] == 109801856) |
                  (event["event"] == 136082628) |
                  (event["event"] == 187272057))


    debug_mask = run_SB_mask & event_mask

    #print(debug_mask.to_list()[0:10])
    #print("Any Early", ak.any(debug_mask), "len", len(debug_mask[debug_mask == True]))


    #print(debug_mask, ak.any(debug_mask))
    out_data["event"  ]    = event["event"][debug_mask]
    out_data["run"    ]    = event["run"][debug_mask]
    out_data["fourTag"    ]    = event["fourTag"][debug_mask]
    out_data["threeTag"    ]    = event["threeTag"][debug_mask]
    out_data["passJetMult"    ]    = event["passJetMult"][debug_mask]
    #out_data["passJetMult_lowpt_forskim"    ]    = event["passJetMult_lowpt_forskim"][debug_mask]
    #out_data["passPreSel_lowpt_forskim"    ]    = event["passPreSel_lowpt_forskim"][debug_mask]
    out_data["lumimask"    ]    = event["lumimask"][debug_mask]
    out_data["passNoiseFilter"    ]    = event["passNoiseFilter"][debug_mask]
    out_data["passPreSel"    ]    = event["passPreSel"][debug_mask]
    out_data["passHLT"    ]    = event["passHLT"][debug_mask]
    out_data["selection"    ]    = selection[debug_mask]


    out_data["Jet_pt"    ]    = event.Jet.pt  [debug_mask].to_list()
    out_data["Jet_eta"   ]    = event.Jet.eta [debug_mask].to_list()
    out_data["Jet_phi"   ]    = event.Jet.phi [debug_mask].to_list()
    out_data["Jet_mass"  ]    = event.Jet.mass[debug_mask].to_list()
    out_data["Jet_bTagScore"] = event.Jet.btagScore[debug_mask].to_list()
    out_data["Jet_selected"]  = event.Jet.selected[debug_mask].to_list()
    #out_data["Jet_selected_lowpt_forskim"]  = event.Jet.selected_lowpt_forskim[debug_mask].to_list()
    out_data["Jet_pileup"]    = event.Jet.pileup[debug_mask].to_list()
    out_data["Jet_jetId"]     = event.Jet.jetId[debug_mask].to_list()
    #out_data["Jet_lepton_cleaned"]     = event.Jet.lepton_cleaned[debug_mask].to_list()



    for out_k, out_v in out_data.items():
        processOutput[out_k] = {}
        processOutput[out_k][event.metadata['dataset']] = list(out_v)


    return




def add_debug_Run3_data(event, processOutput):

    out_data = {}


    debug_mask = (event.region.SB == True)

    # run_SB_mask = (event.run == 355921)
    #
    # event_mask = ((event["event"] == 632465280) |
    #               (event["event"] == 109801856) |
    #               (event["event"] == 136082628) |
    #               (event["event"] == 187272057))
    #
    # debug_mask = run_SB_mask & event_mask
    #
    # print(debug_mask.to_list()[0:10])
    # print("Any", ak.any(debug_mask), "len", len(debug_mask[debug_mask == True]))
    # print("passJetMult", event["passJetMult"][debug_mask].to_list()[0:10])
    # print("threeTag", event["threeTag"][debug_mask].to_list()[0:10])
    # print("fourTag", event["fourTag"][debug_mask].to_list()[0:10])
    # print("SB", event.region.SB[debug_mask].to_list()[0:10])
    # print("event", event["event"][debug_mask].to_list()[0:10])
    # print("quadJet rhh", event.quadJet_selected.rhh[debug_mask].to_list()[0:10])
    # print("quadJet selected lead,mass", event.quadJet_selected.lead.mass[debug_mask].to_list()[0:10])
    # print("quadJet selected subl mass", event.quadJet_selected.subl.mass[debug_mask].to_list()[0:10])
    # print("quadJet selected lead,pt", event.quadJet_selected.lead.pt[debug_mask].to_list()[0:10])
    # print("quadJet selected subl pt", event.quadJet_selected.subl.pt[debug_mask].to_list()[0:10])
    # print("dhh", event.quadJet.dhh[debug_mask].to_list()[0:10])
    # print("quadJet lead mass", event.quadJet.lead.mass[debug_mask].to_list()[0:10])
    # print("quadJet subl mass", event.quadJet.subl.mass[debug_mask].to_list()[0:10])
    # print("quadJet lead pt", event.quadJet.lead.pt[debug_mask].to_list()[0:10])
    # print("quadJet subl pt", event.quadJet.subl.pt[debug_mask].to_list()[0:10])
    # print("dhh selected", event.quadJet.selected[debug_mask].to_list()[0:10])
    #
    #
    #
    # print("Jets ",
    #       event.quadJet_selected.lead.lead.pt[debug_mask].to_list()[0:10],
    #       event.quadJet_selected.lead.subl.pt[debug_mask].to_list()[0:10],
    #       event.quadJet_selected.subl.lead.pt[debug_mask].to_list()[0:10],
    #       event.quadJet_selected.subl.subl.pt[debug_mask].to_list()[0:10])
    # print("canJet pt", event.canJet.pt[debug_mask].to_list()[0:10])
    # print("canJet eta", event.canJet.eta[debug_mask].to_list()[0:10])
    # print("canJet phi", event.canJet.phi[debug_mask].to_list()[0:10])
    # print("canJet mass", event.canJet.mass[debug_mask].to_list()[0:10])
    # print("canJet bTagScore", event.canJet.btagScore[debug_mask].to_list()[0:10])





    #print(debug_mask, ak.any(debug_mask))
    out_data["event"  ]    = event["event"][debug_mask]
    out_data["run"    ]    = event["run"][debug_mask]
    out_data["fourTag"    ]    = event["fourTag"][debug_mask]
    out_data["passJetMult"    ]    = event["passJetMult"][debug_mask]
    out_data["passJetMult_event"    ]    = event["event"][event.passJetMult][debug_mask]
    out_data["passJetMult_run"    ]    = event["run"][event.passJetMult][debug_mask]
    out_data["passJetMult_passDiJetMass"    ]    = event["passDiJetMass"][event.passJetMult][debug_mask]

    out_data["canJet_pt"    ] = event.canJet.pt  [debug_mask].to_list()
    out_data["canJet_eta"   ] = event.canJet.eta [debug_mask].to_list()
    out_data["canJet_phi"   ] = event.canJet.phi [debug_mask].to_list()
    out_data["canJet_mass"  ] = event.canJet.mass[debug_mask].to_list()
    out_data["canJet_bTagScore"] = event.canJet.btagScore[debug_mask].to_list()

    out_data["notCanJet_pt"    ]    = event.notCanJet_coffea.pt  [debug_mask].to_list()
    out_data["notCanJet_eta"   ]    = event.notCanJet_coffea.eta [debug_mask].to_list()
    out_data["notCanJet_phi"   ]    = event.notCanJet_coffea.phi [debug_mask].to_list()
    out_data["notCanJet_mass"  ]    = event.notCanJet_coffea.mass[debug_mask].to_list()
    out_data["notCanJet_bTagScore"] = event.notCanJet_coffea.btagScore[debug_mask].to_list()


    for out_k, out_v in out_data.items():
            processOutput[out_k] = {}
            processOutput[out_k][event.metadata['dataset']] = list(out_v)



def add_debug_Run3_declustering(event, jets_for_clustering, declustered_jets, clustered_jets, processOutput):

    out_data = {}

    #debug_mask = event.region > 2
    out_data["event"  ]    = event["event"]
    out_data["run"    ]    = event["run"]

    out_data["selJet_pt"    ] = event.selJet.pt  .to_list()
    out_data["selJet_eta"   ] = event.selJet.eta .to_list()
    out_data["selJet_phi"   ] = event.selJet.phi .to_list()
    out_data["selJet_mass"  ] = event.selJet.mass.to_list()

    out_data["Jet_bCalib_pt"    ] = event.Jet_bCalib.pt  .to_list()
    out_data["Jet_bCalib_pt_raw" ] = event.Jet_bCalib.pt_raw  .to_list()
    out_data["Jet_bCalib_mass_raw" ] = event.Jet_bCalib.mass_raw  .to_list()
    out_data["Jet_bCalib_mass"  ] = event.Jet_bCalib.mass.to_list()
    out_data["Jet_bCalib_rho"  ] = event.Jet_bCalib.rho.to_list()
    out_data["Jet_bCalib_area"  ] = event.Jet_bCalib.area.to_list()
    out_data["Jet_bCalib_eta"  ] = event.Jet_bCalib.eta.to_list()
    out_data["Jet_bCalib_phi"  ] = event.Jet_bCalib.phi.to_list()
    #out_data["Jet_bCalib_pt_gen"  ] = event.Jet_bCalib.pt_gen.to_list()

    out_data["Jet_nonbCalib_pt"    ] = event.Jet_nonbCalib.pt  .to_list()
    out_data["Jet_nonbCalib_pt_raw"    ] = event.Jet_nonbCalib.pt_raw  .to_list()
    out_data["Jet_nonbCalib_mass"  ] = event.Jet_nonbCalib.mass.to_list()


    out_data["selJet_no_bRegCorr_pt"    ] = event.selJet_no_bRegCorr.pt  .to_list()
    out_data["selJet_no_bRegCorr_pt_preCalib"    ] = event.selJet_no_bRegCorr.pt_preCalib  .to_list()
    out_data["selJet_no_bRegCorr_eta"   ] = event.selJet_no_bRegCorr.eta .to_list()
    out_data["selJet_no_bRegCorr_phi"   ] = event.selJet_no_bRegCorr.phi .to_list()
    out_data["selJet_no_bRegCorr_mass"  ] = event.selJet_no_bRegCorr.mass.to_list()
    out_data["selJet_no_bRegCorr_mass_preCalib"  ] = event.selJet_no_bRegCorr.mass_preCalib.to_list()


    out_data["input_jet_pt"    ] = jets_for_clustering.pt  .to_list()
    out_data["input_jet_eta"   ] = jets_for_clustering.eta .to_list()
    out_data["input_jet_phi"   ] = jets_for_clustering.phi .to_list()
    out_data["input_jet_mass"  ] = jets_for_clustering.mass.to_list()

    out_data["output_jet_pt"    ] = declustered_jets.pt  .to_list()
    out_data["output_jet_eta"   ] = declustered_jets.eta .to_list()
    out_data["output_jet_phi"   ] = declustered_jets.phi .to_list()
    out_data["output_jet_mass"  ] = declustered_jets.mass.to_list()

    out_data["clustered_jet_pt"    ] = clustered_jets.pt  .to_list()
    out_data["clustered_jet_eta"   ] = clustered_jets.eta .to_list()
    out_data["clustered_jet_phi"   ] = clustered_jets.phi .to_list()
    out_data["clustered_jet_mass"  ] = clustered_jets.mass.to_list()



    for out_k, out_v in out_data.items():
        processOutput[out_k] = {}
        processOutput[out_k][event.metadata['dataset']] = list(out_v)


def add_debug_info_to_output_clustering_inputs(event, jets_for_clustering, processOutput):

    out_data = {}

    out_data["clusteringInputs_event"  ]    = event["event"][0:10]
    out_data["clusteringInputs_run"    ]    = event["run"][0:10]
    out_data["clusteringInputs_jet_pt"    ] = jets_for_clustering.pt  [0:10].to_list()
    out_data["clusteringInputs_jet_eta"   ] = jets_for_clustering.eta [0:10].to_list()
    out_data["clusteringInputs_jet_phi"   ] = jets_for_clustering.phi [0:10].to_list()
    out_data["clusteringInputs_jet_mass"  ] = jets_for_clustering.mass[0:10].to_list()

    for out_k, out_v in out_data.items():
        processOutput[out_k] = {}
        processOutput[out_k][event.metadata['dataset']] = list(out_v)


def add_debug_info_to_output_clustering_outputs(event, clustered_jets, processOutput):

    out_data = {}

    out_data["clusteringOutputs_event"  ]    = event["event"][0:10]
    out_data["clusteringOutputs_run"    ]    = event["run"][0:10]
    out_data["clusteringOutputs_jet_pt"    ] = clustered_jets.pt  [0:10].to_list()
    out_data["clusteringOutputs_jet_eta"   ] = clustered_jets.eta [0:10].to_list()
    out_data["clusteringOutputs_jet_phi"   ] = clustered_jets.phi [0:10].to_list()
    out_data["clusteringOutputs_jet_mass"  ] = clustered_jets.mass[0:10].to_list()
    out_data["clusteringOutputs_jet_flavor"] = clustered_jets.jet_flavor[0:10].to_list()

    for out_k, out_v in out_data.items():
        processOutput[out_k] = {}
        processOutput[out_k][event.metadata['dataset']] = list(out_v)


def add_debug_info_to_output_declustering_outputs(event, declustered_jets, processOutput):

    out_data = {}

    out_data["declusteredJets_event"  ]      = event["event"][0:10]
    out_data["declusteredJets_run"    ]      = event["run"][0:10]
    out_data["declusteredJets_jet_pt"    ]   = declustered_jets.pt  [0:10].to_list()
    out_data["declusteredJets_jet_eta"   ]   = declustered_jets.eta [0:10].to_list()
    out_data["declusteredJets_jet_phi"   ]   = declustered_jets.phi [0:10].to_list()
    out_data["declusteredJets_jet_mass"  ]   = declustered_jets.mass[0:10].to_list()
    out_data["declusteredJets_jet_flavor"] = declustered_jets.jet_flavor[0:10].to_list()
    for out_k, out_v in out_data.items():
        processOutput[out_k] = {}
        processOutput[out_k][event.metadata['dataset']] = list(out_v)


def debug_three_tag_events(event, processOutput):

    for iEvent in range(5):

        print(f'event num: {event["event"][event.threeTag][iEvent]}')
        print(f'event run: {event["run"][event.threeTag][iEvent]}')
        print(f'event fourTag: {event.fourTag[event.threeTag][iEvent]}')
        print(f'event threeTag: {event.fourTag[event.threeTag][iEvent]}')
        print(f'event jetPt:  {[i for i in event.Jet.pt    [event.threeTag][iEvent]]}')
        print(f'event eta:    {[i for i in event.Jet.eta   [event.threeTag][iEvent]]}')
        print(f'event phi:    {[i for i in event.Jet.phi   [event.threeTag][iEvent]]}')
        print(f'event selected: {[i for i in event.Jet.selected[event.threeTag][iEvent]]}')
        print(f'\t event pileup: {[i for i in event.Jet.pileup[event.threeTag][iEvent]]}')
        print(f'\t\t event puId: {[i for i in event.Jet.puId[event.threeTag][iEvent]]}')
        print(f'\t event jetId: {[i for i in event.Jet.jetId[event.threeTag][iEvent]]}')
        print(f'\t event lepton_cleaned: {[i for i in event.Jet.lepton_cleaned[event.threeTag][iEvent]]}')
        print(f'event tagged: {[i for i in event.Jet.tagged[event.threeTag][iEvent]]}')
        print(f'event btagScore: {[i for i in event.Jet.btagScore[event.threeTag][iEvent]]}')
        print("\n")

    #out_data["passJetMult_event"  ]    = event["event"][event.threeTag]
    #out_data["passJetMult_run"    ]    = event["run"][event.passJetMult]
    #out_data["passJetMult_jet_pt"    ] = event.Jet.pt[event.passJetMult].to_list()
    #out_data["passJetMult_jet_eta"   ] = event.Jet.eta[event.passJetMult].to_list()
    #out_data["passJetMult_jet_phi"   ] = event.Jet.phi[event.passJetMult].to_list()
    #out_data["passJetMult_jet_pu"    ] = event.Jet.pileup[event.passJetMult].to_list()
    #out_data["passJetMult_jet_jetId" ] = event.Jet.jetId[event.passJetMult].to_list()
    #out_data["passJetMult_jet_lep"   ] = event.Jet.lepton_cleaned[event.passJetMult].to_list()



def add_debug_info_for_Hbb_reclustering(event, processOutput):
    # passSR = (selev["quadJet_selected"].SR)
    # passSR = (selev["SR"])

    event_filter = event["fourTag"]

    out_data = {}
    # out_data["SvB"    ] = selev["SvB_MA"].ps[passSR]
    out_data["event"  ] = event["event"]#[passSR]
    out_data["run"    ] = event["run"]  #[passSR]


    out_data[f"leadDiJet_mass"    ] = event.quadJet_selected.lead.mass
    out_data[f"sublDiJet_mass"    ] = event.quadJet_selected.subl.mass

    for v in ["pt","eta","phi","mass"]:
        out_data[f"leadDiJet_leadJet_{v}"    ] = event.quadJet_selected.lead.lead[v]
        out_data[f"leadDiJet_sublJet_{v}"    ] = event.quadJet_selected.lead.subl[v]

        out_data[f"sublDiJet_leadJet_{v}"    ] = event.quadJet_selected.subl.lead[v]
        out_data[f"sublDiJet_sublJet_{v}"    ] = event.quadJet_selected.subl.subl[v]

        out_data[f"otherJet_{v}" ] = event.notCanJet_coffea[v].to_list()

    for out_k, out_v in out_data.items():
        processOutput[out_k] = {}
        processOutput[out_k][event.metadata['dataset']] = list(out_v)


def add_debug_info_for_Boosted_Synthetic(events, processOutput):


    FatJet_fields = ['area', 'eta', 'mass', 'msoftdrop', 'n2b1', 'n3b1', 'particleNetMD_Xbb', 'particleNet_mass', 'phi', 'pt', 'tau1', 'tau2', 'tau3', 'tau4', 'lsf3', 'nConstituents']
    SubJet_fields = ['btagDeepB', 'eta', 'mass', 'n2b1', 'n3b1', 'phi', 'pt', 'rawFactor', 'tau1', 'tau2', 'tau3', 'tau4']

    out_data = {}
    out_data["event"  ] = events["event"]
    out_data["run"    ] = events["run"]

    for _f in FatJet_fields:
        #print(_f, "is", getattr(events.FatJet, _f),"\n")
        #print(_f, "is", type(getattr(events.FatJet, _f)),"\n")
        out_data[f"FatJet_{_f}"    ] = getattr(events.FatJet, _f).to_list()

    for _s in SubJet_fields:
        #print(_s, "subjets:", getattr(events.FatJet.subjets, _s),"\n")
        out_data[f"SubJet_{_s}"    ] = getattr(events.FatJet.subjets, _s).to_list()

    for out_k, out_v in out_data.items():
        processOutput[out_k] = {}
        processOutput[out_k][events.metadata['dataset']] = list(out_v)



def dump_SvB_in_SR(event, processOutput, dataset, key="SvB_in_SR"):
    """add_SvB_in_SR under one namespaced key, as plain lists (accumulate concatenates them across
    chunks): processOutput[key][dataset] = {run, luminosityBlock, event, SvB_MA_ps} for the
    four-tag SR events. Input of workflows/scripts/svb_bootstrap.py (Poisson bootstrap of the SvB
    distribution over the input events)."""
    passSR = ak.to_numpy(ak.fill_none(event["quadJet_selected"].SR & event.fourTag, False))
    sel = event[passSR]
    processOutput[key] = {dataset: {
        "run": ak.to_numpy(sel.run).astype("int64").tolist(),
        "luminosityBlock": ak.to_numpy(sel.luminosityBlock).astype("int64").tolist(),
        "event": ak.to_numpy(sel.event).astype("int64").tolist(),
        "SvB_MA_ps": ak.to_numpy(sel.SvB_MA.ps).astype("float64").tolist(),
    }}


def add_SvB_in_SR(event, processOutput):
    # passSR = (selev["quadJet_selected"].SR)
    passSR = (event["quadJet_selected"].SR & event.fourTag)


    out_data = {}
    # out_data["SvB"    ] = selev["SvB_MA"].ps[passSR]
    out_data["event"  ] = event["event"][passSR]
    out_data["run"    ] = event["run"]  [passSR]
    out_data["luminosityBlock"    ] = event["luminosityBlock"]  [passSR]

    out_data["SvB_MA_ps"] = event["SvB_MA"].ps[passSR]
    #out_data["passPreSel"    ] = event["passJetMult"]
    #out_data["lumimask"    ] = event["lumimask"]
    #out_data["passNoiseFilter"    ] = event["passNoiseFilter"]
    #out_data["passHLT"    ] = event["passHLT"]
    #out_data["passJetMult"    ] = event["passJetMult"]

    #debug_mask = ((event["event"] == 66688  ) |
    #              (event["event"] == 249987 ) |
    #              (event["event"] == 121603 ) |
    #              (event["event"] == 7816   ) |
    #              (event["event"] == 25353  ) |
    #              (event["event"] == 165389 ) |
    #              (event["event"] == 293138 ) |
    #              (event["event"] == 150164 ) |
    #              (event["event"] == 262806 ) |
    #              (event["event"] == 281111 ) )


    #print(f"\n {event.Jet.pt[event.passJetMult].to_list()[0:5]} \n")

    #out_data["passJetMult_event"  ]    = event["event"][event.passJetMult]
    #out_data["passJetMult_run"    ]    = event["run"][event.passJetMult]
    #out_data["passJetMult_passDiJetMass"    ]    = event["passDiJetMass"][event.passJetMult]
    #out_data["passJetMult_weight" ]    = event["weight"]
    #
    #for _w in list_weight_names:
    #    print(f"adding {_w}\n")
    #    out_data[f"passJetMult_weight_{_w}"] = weights.partial_weight(include=[_w])[analysis_selections]


    for out_k, out_v in out_data.items():
        processOutput[out_k] = {}
        processOutput[out_k][event.metadata['dataset']] = list(out_v)



def add_hemi_events(event, processOutput):
    # passSR = (selev["quadJet_selected"].SR)
    passSR = (event["quadJet_selected"].SR & event.fourTag)


    out_data = {}
    # out_data["SvB"    ] = selev["SvB_MA"].ps[passSR]
    out_data["event"  ] = event["event"][passSR]
    out_data["run"    ] = event["run"]  [passSR]
    out_data["luminosityBlock"    ] = event["luminosityBlock"]  [passSR]

    # print(event.h1.run,"\n")
    # print(event.h1.event,"\n")
    # print(event.h2.run,"\n")
    # print(event.h2.event,"\n")

    out_data["h1_run"]   = event.h1.run[passSR]
    out_data["h1_event"] = event.h1.event[passSR]
    out_data["h2_run"]   = event.h2.run[passSR]
    out_data["h2_event"] = event.h2.event[passSR]



    #out_data["passPreSel"    ] = event["passJetMult"]
    #out_data["lumimask"    ] = event["lumimask"]
    #out_data["passNoiseFilter"    ] = event["passNoiseFilter"]
    #out_data["passHLT"    ] = event["passHLT"]
    #out_data["passJetMult"    ] = event["passJetMult"]

    #debug_mask = ((event["event"] == 66688  ) |
    #              (event["event"] == 249987 ) |
    #              (event["event"] == 121603 ) |
    #              (event["event"] == 7816   ) |
    #              (event["event"] == 25353  ) |
    #              (event["event"] == 165389 ) |
    #              (event["event"] == 293138 ) |
    #              (event["event"] == 150164 ) |
    #              (event["event"] == 262806 ) |
    #              (event["event"] == 281111 ) )


    #print(f"\n {event.Jet.pt[event.passJetMult].to_list()[0:5]} \n")

    #out_data["passJetMult_event"  ]    = event["event"][event.passJetMult]
    #out_data["passJetMult_run"    ]    = event["run"][event.passJetMult]
    #out_data["passJetMult_passDiJetMass"    ]    = event["passDiJetMass"][event.passJetMult]
    #out_data["passJetMult_weight" ]    = event["weight"]
    #
    #for _w in list_weight_names:
    #    print(f"adding {_w}\n")
    #    out_data[f"passJetMult_weight_{_w}"] = weights.partial_weight(include=[_w])[analysis_selections]


    for out_k, out_v in out_data.items():
        processOutput[out_k] = {}
        processOutput[out_k][event.metadata['dataset']] = list(out_v)
