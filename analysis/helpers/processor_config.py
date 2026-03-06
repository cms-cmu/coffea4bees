from collections import defaultdict

def processor_config(processName, dataset, event):

    config = defaultdict(lambda : False)

    #
    # Set process type flags
    #
    config["isMC"]     = False if "data"    in processName else True
    config["isPSData"] = True  if "ps_data" in processName else False
    config["isMixedData"]    = not (dataset.find("mix_v") == -1) or not (dataset.find("mix_noTT_v") == -1) or not (dataset.find("mix_pz_v") == -1) or not (dataset.find("mixeddata_all") == -1)
    config["isSignal"] = False if processName.startswith(("data", 'syn', 'TT', 'mix')) else True
    config["isRun3"] = True if "202" in dataset else False

    if config["isMixedData"]:
        config["isMC"] = False

    config["isSyntheticData"]  = not (dataset.find("syn_v") == -1) or not (dataset.find("syn_noTT_v") == -1)
    if config["isSyntheticData"]:
        config["isMC"] = False

    config["isSyntheticMC"]  = not (dataset.find("synthetic_mc") == -1)
    if config["isSyntheticMC"]:
        config["isMC"] = False

    config["isDataForMixed"] = not (dataset.find("data_3b_for_mixed") == -1)
    config["isTTForMixed"]   = not (dataset.find("TTTo") == -1) and not ( dataset.find("_for_mixed") == -1 )


    #
    #  Nominal config (...what we would do for data)
    #
    config["cut_on_lumimask"]         = True
    config["cut_on_HLT_decision"]     = True
    config["do_MC_weights"]           = False
    config["do_jet_calibration"]      = True
    config["do_lepton_jet_cleaning"]  = True
    config["override_selected_with_flavor_bit"]  = False
    config["use_prestored_btag_SF"]  = False
    config["do_jet_veto_maps"]       = False   ## false for run2 until check effect


    if config["isMC"]:
        config["cut_on_lumimask"]     = False
        config["cut_on_HLT_decision"] = False
        config["do_jet_calibration"]  = True
        config["do_MC_weights"]       = True

    if config["isRun3"]:
        config['do_jet_veto_maps'] = False
        config['do_jet_calibration'] = False # Need a better name here (Jet calib is applied in Run3 by default !)
        config["cut_on_HLT_decision"]  = True


    if config["isSyntheticData"]:
        config["do_lepton_jet_cleaning"]  = False
        config["override_selected_with_flavor_bit"]  = False
        config["isPSData"] = True if event.run[0] == 1 else False
        config["do_jet_calibration"]      = False
        config["do_jet_veto_maps"]       = False

    if config["isSyntheticMC"]:
        config["cut_on_lumimask"]         = False
        config["cut_on_HLT_decision"]     = False
        config["do_MC_weights"]           = True
        config["do_jet_calibration"]     = False
        config["do_lepton_jet_cleaning"]  = False
        config["override_selected_with_flavor_bit"]  = False
        config["do_jet_veto_maps"]       = False
        config["use_prestored_btag_SF"]  = True

    if config["isPSData"]:
        config["cut_on_lumimask"]     = False
        config["cut_on_HLT_decision"] = False
        config["do_jet_calibration"]  = False
        config["do_jet_veto_maps"]       = False

    if config["isMixedData"]:
        config["cut_on_lumimask"]     = False
        config["cut_on_HLT_decision"] = False
        config["do_lepton_jet_cleaning"]  = False
        config["do_jet_calibration"]  = False
        config["do_jet_veto_maps"]       = False

    if config["isTTForMixed"]:
        config["cut_on_lumimask"]        = False
        config["cut_on_HLT_decision"]    = False
        config["do_lepton_jet_cleaning"] = False
        config["do_jet_calibration"]     = False
        config["do_jet_veto_maps"]       = False

    if config["isDataForMixed"]:
        config["cut_on_HLT_decision"] = False

        config["do_lepton_jet_cleaning"]  = False
        config["do_jet_calibration"]  = False
        config["do_jet_veto_maps"]       = False

    return config
