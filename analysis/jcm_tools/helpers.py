import numpy as np
from scipy.special import comb
import logging
from coffea.util import load
from copy import copy
from typing import Tuple
from src.plotting.helpers import get_cut_dict

logger = logging.getLogger('JCMHelpers')

def loadROOTHistograms(inputFile: str) -> Tuple:
        h = load(inputFile)["Hists"]

        data4b = h["data4b"]
        data3b = h["data3b"]
        tt4b = h["tt4b"]
        tt3b = h["tt3b"]
        qcd4b = h["qcd4b"]
        qcd3b = h["qcd3b"]
        data4b_nTagJets = h["data4b_nTagJets"]
        tt4b_nTagJets = h["tt4b_nTagJets"]
        qcd3b_nTightTags = h["qcd3b_nTightTags"]

        return data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags



def loadHistograms(inputFile: str, jcm_config: dict, format: str = 'coffea', cfg=None, cut: str = "passPreSel", year: str = "RunII", weightRegion: str = "SB") -> Tuple:
    """
    Load histograms from either ROOT or coffea files.

    Args:
        inputFile: Path to the input file
        format: Format of the input file ('ROOT' or 'coffea')
        cfg: Configuration object (required for coffea format)
        cut: Selection cut to apply (coffea only)
        year: Data-taking year (coffea only)
        weightRegion: Region for weight calculation (coffea only)
        logger: Logger instance

    Returns:
        Tuple of histograms:
        (data4b, data3b, tt4b, tt3b, qcd4b, qcd3b,
        data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags)
    """

    if format == 'ROOT':
        logger.info(f"Loading histograms from ROOT file: {inputFile}")
        return loadROOTHistograms(inputFile)

    elif not format == 'coffea':
        raise ValueError(f"Unsupported format: {format}")

    #
    #  Reading in coffea format
    #
    if cfg is None:
        raise ValueError("Configuration object (cfg) is required for coffea format")

    data4bName = jcm_config.get("data4bName", "data")
    data3bName = jcm_config.get("data3bName", "data")
    subtract3bTT = jcm_config.get("subtract3bTT", True)
    ignoreTT     = jcm_config.get("ignoreTT", False)
    selJets = jcm_config.get("selJets", "selJets_noJCM.n")
    tagJets = jcm_config.get("tagJets", "tagJets_noJCM.n")

    logger.info(f"Loading coffea histograms with cut={cut}, year={year}, weightRegion={weightRegion}")
    cutDict = get_cut_dict(cut, cfg.cutList) if cut is not None else {}

    year_val = sum if year == "RunII" else year
    region_selection = sum if weightRegion in ["sum", sum] else weightRegion

    region_year_dict = {
        "year":   year_val,
        "region": region_selection,
    }

    # Read config
    fourTag_dict =     {"tag": jcm_config.get("taglabel4b",   "fourTag") }
    threeTag_dict =    {"tag": jcm_config.get("taglabel3b",   "threeTag")}
    threeTag_tt_dict = {"tag": jcm_config.get("taglabel3b_tt","threeTag")}

    fourTag_data_dict = {"process": data4bName} | fourTag_dict | region_year_dict | cutDict
    threeTag_data_dict = {"process": data3bName} | threeTag_dict | region_year_dict | cutDict

    ttbar_list = ['TTTo2L2Nu', 'TTToSemiLeptonic', 'TTToHadronic']
    fourTag_ttbar_dict  = {"process": ttbar_list} | fourTag_dict | region_year_dict | cutDict
    threeTag_ttbar_dict = {"process": ttbar_list} | threeTag_tt_dict | region_year_dict | cutDict

    hists_data_4b = None
    hists_data_3b = None
    hists_tt     = None

    for _input_data in cfg.hists:
        if (selJets in _input_data['hists']):

            if (data4bName in _input_data['hists'][selJets].axes["process"]):
                hists_data_4b = _input_data['hists']

            if (data3bName in _input_data['hists'][selJets].axes["process"]):
                hists_data_3b = _input_data['hists']

            if ("TTTo2L2Nu" in _input_data['hists'][selJets].axes["process"]):
                hists_tt = _input_data['hists']



    if hists_data_4b is None:
        raise ValueError(f"Could not find histograms for data4bName={data4bName}")

    if hists_data_3b is None:
        raise ValueError(f"Could not find histograms for data3bName={data3bName}")

    if (not ignoreTT) and (hists_tt is None):
        raise ValueError(f"Could not find histograms for TTTo2L2Nu")



    data4b = hists_data_4b[selJets][fourTag_data_dict]
    data4b_nTagJets = hists_data_4b[tagJets][fourTag_data_dict]

    qcd4b = copy(data4b)

    data3b = hists_data_3b[selJets][threeTag_data_dict]
    data3b_nTagJets_tight = hists_data_3b[tagJets][threeTag_data_dict]

    qcd3b = copy(data3b)
    qcd3b_nTightTags = copy(data3b_nTagJets_tight)

    if ignoreTT:
        return data4b, data3b, None, None, qcd4b, qcd3b, data4b_nTagJets, None, qcd3b_nTightTags


    tt4b = hists_tt[selJets][fourTag_ttbar_dict][sum, :]
    tt4b_nTagJets = hists_tt[tagJets][fourTag_ttbar_dict][sum, :]

    tt3b = hists_tt[selJets][threeTag_ttbar_dict][sum, :]
    tt3b_nTagJets_tight = hists_tt[tagJets][threeTag_ttbar_dict][sum, :]


    qcd4b.view().value = data4b.values() - tt4b.values()
    qcd4b.view().variance = data4b.variances() + tt4b.variances()

    if subtract3bTT:
        print("subtracting tt from 3b to get qcd3b")
        qcd3b.view().value = data3b.values() - tt3b.values()
        qcd3b.view().variance = data3b.variances() + tt3b.variances()

        qcd3b_nTightTags.view().value = data3b_nTagJets_tight.values() - tt3b_nTagJets_tight.values()
        qcd3b_nTightTags.view().variance = data3b_nTagJets_tight.variances() + tt3b_nTagJets_tight.variances()

    return data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags




def data_from_Hist(inputHist, maxBin: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract data arrays from a histogram.

    Args:
        inputHist: Input histogram
        maxBin: Maximum bin to extract

    Returns:
        Tuple of (bin_centers, values, errors)
    """
    x_centers = inputHist.axes[0].centers
    values = inputHist.values()
    errors = np.sqrt(inputHist.variances())

    # Adjust bin centers if needed
    if x_centers[0] == 0.5:
        x_centers = x_centers - 0.5

    return x_centers[0:maxBin], values[0:maxBin], errors[0:maxBin]


def prepHists(data4b, qcd3b, tt4b, data4b_nTagJets, tt4b_nTagJets, lowpt: bool = False) -> None:
    """
    Prepare histograms for the JCM fit by combining different components.

    This modifies the input histograms in-place by setting values for the
    first 4 bins to represent the number of additional tag jets.

    Args:
        data4b: Data 4-tag histogram
        qcd3b: QCD 3-tag histogram
        tt4b: tt 4-tag histogram
        data4b_nTagJets: Data 4-tag tagged jets histogram
        tt4b_nTagJets: tt 4-tag tagged jets histogram
    """

    if lowpt:

        # Put the number of additional tag jets in the first 4 bins of data4b
        data4b_new_values = np.zeros(len(data4b.values()))
        data4b_new_variances = np.zeros(len(data4b.variances()))
        data4b_new_values[0:4] = data4b_nTagJets.values()[1:5]
        data4b_new_values[4:14] = data4b.values()[1:11]
        data4b_new_variances[0:4] = data4b_nTagJets.variances()[1:5]
        data4b_new_variances[4:14] = data4b.variances()[1:11]

        # Do the same for tt4b
        if tt4b is not None:
            tt4b_new_values = np.zeros(len(tt4b.values()))
            tt4b_new_variances = np.zeros(len(tt4b.variances()))
            tt4b_new_values[0:4] = tt4b_nTagJets.values()[1:5]
            tt4b_new_values[4:14] = tt4b.values()[1:11]
            tt4b_new_variances[0:4] = tt4b_nTagJets.variances()[1:5]
            tt4b_new_variances[4:14] = tt4b.variances()[1:11]

        qcd3b_new_values = np.zeros(len(qcd3b.values()))
        qcd3b_new_variances = np.zeros(len(qcd3b.variances()))
        qcd3b_new_values[4:14] = qcd3b.values()[1:11]
        qcd3b_new_variances[4:14] = qcd3b.variances()[1:11]
        qcd3b.view().value = qcd3b_new_values
        qcd3b.view().variance = qcd3b_new_variances

    else:

        # Put the number of additional tag jets in the first 4 bins of data4b
        data4b_new_values = data4b.values()
        data4b_new_variances = data4b.variances()

        data4b_new_values[0:4] = data4b_nTagJets.values()[4:8]
        data4b_new_variances[0:4] = data4b_nTagJets.variances()[4:8]


        # Do the same for tt4b
        if tt4b is not None:
            tt4b_new_values = tt4b.values()
            tt4b_new_variances = tt4b.variances()

            tt4b_new_values[0:4] = tt4b_nTagJets.values()[4:8]
            tt4b_new_variances[0:4] = tt4b_nTagJets.variances()[4:8]

    data4b.view().value = data4b_new_values
    data4b.view().variance = data4b_new_variances

    if tt4b is not None:
        tt4b.view().value = tt4b_new_values
        tt4b.view().variance = tt4b_new_variances
