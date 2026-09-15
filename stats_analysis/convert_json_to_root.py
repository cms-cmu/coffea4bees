import os
import argparse
import logging
import json
import array
import numpy as np

try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    HAS_ROOT = True
except ImportError:
    ROOT = None
    HAS_ROOT = False
try:
    import uproot
    import hist
    HAS_UPROOT = True
except ImportError:
    uproot = None
    hist = None
    HAS_UPROOT = False


def json_to_TH1( coffea_hist, iname, rebin ):
    """docstring for hist_to_root"""

    edges     = coffea_hist['edges']
    centers   = coffea_hist['centers']
    values    = coffea_hist['values']
    variances = coffea_hist['variances']
    underflow_value      = coffea_hist['underflow_value']
    underflow_variance   = coffea_hist['underflow_variance']
    overflow_value       = coffea_hist['overflow_value']  
    overflow_variance    = coffea_hist['overflow_variance']

    # Check if edges are non-uniform (variable binning)
    widths = np.diff(edges)
    if len(edges) > 1 and not np.allclose(widths, widths[0]):
        rHist = ROOT.TH1F(iname, iname, len(edges) - 1, array.array('d', edges))
    else:
        rHist = ROOT.TH1F(iname, iname, len(centers), edges[0], edges[-1])
    rHist.Sumw2()

    rHist.SetBinContent(0, underflow_value)
    rHist.SetBinError(0, ROOT.TMath.Sqrt(underflow_variance))

    for ibin in range(1, len(centers)+1 ):
        rHist.SetBinContent(ibin, values[ibin-1])
        rHist.SetBinError(ibin, ROOT.TMath.Sqrt(variances[ibin-1]))

    rHist.SetBinContent( len(centers)+1, overflow_value)
    rHist.SetBinError( len(centers)+1, ROOT.TMath.Sqrt(overflow_variance))

    if isinstance(rebin, list):
        rHist = rHist.Rebin( len(rebin)-1, f"{iname}_rebinned", array.array('d', rebin))
    else:
        rHist.Rebin( rebin )

    return rHist

def create_root_file(file_to_convert, histos, output_dir):
    logging.info( "in create_root_file")
    coffea_hists = json.load(open(file_to_convert, 'r'))
    logging.info( "leaded coffea_hists")

    root_hists = {}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info( "made dirs")
    output = output_dir + "/" + (file_to_convert.split("/")
                                 [-1].replace(".json", "")) + ".root"

    # Check if the output file exists and delete it if it does
    if os.path.exists(output):
        os.remove(output)
        logging.info(f"Deleted existing file: {output}")

    if HAS_ROOT:
        root_file = ROOT.TFile(output, 'recreate')
        for ih in coffea_hists.keys():
            # if len(histos) > 0 and ((ih in histos) or (ih.replace(".", "_") in histos)):
            for iprocess in coffea_hists[ih].keys():
                for iy in coffea_hists[ih][iprocess].keys():
                    for itag in coffea_hists[ih][iprocess][iy].keys():
                        for iregion in coffea_hists[ih][iprocess][iy][itag].keys():
                            this_hist = json_to_TH1(
                                coffea_hists[ih][iprocess][iy][itag][iregion],
                                ih.replace(".", "_") + "_" + iprocess + "_" + iy + "_" + itag + "_" + iregion,
                                1)
                            print( 'Converting hist', ih, ih.replace(".", "_") + "_" + iprocess + "_" + iy + "_" + itag + "_" + iregion)
                            this_hist.Write()
        root_file.Close()
    elif HAS_UPROOT:
        with uproot.recreate(output) as f_out:
            for ih in coffea_hists.keys():
                for iprocess in coffea_hists[ih].keys():
                    for iy in coffea_hists[ih][iprocess].keys():
                        for itag in coffea_hists[ih][iprocess][iy].keys():
                            for iregion in coffea_hists[ih][iprocess][iy][itag].keys():
                                h_data = coffea_hists[ih][iprocess][iy][itag][iregion]
                                edges = np.array(h_data['edges'], dtype=np.float64)
                                values = np.array(h_data['values'], dtype=np.float64)
                                variances = np.array(h_data['variances'], dtype=np.float64)
                                h = hist.Hist.new.Var(edges, name="h").Weight()
                                h.view().value = values
                                h.view().variance = variances
                                key = ih.replace(".", "_") + "_" + iprocess + "_" + iy + "_" + itag + "_" + iregion
                                print('Converting hist (uproot)', ih, key)
                                f_out[key] = h
    else:
        raise ImportError("Neither ROOT nor uproot is available.")
    logging.info("\n File " + output + " created.")


if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser(
        description='Convert json hist to root TH1F', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output_dir', dest="output_dir",
                        default="./datacards/", help='Output directory.')
    parser.add_argument('--histos', dest="histos", nargs="+",
                        default=[  ], help='List of histograms to convert')
    parser.add_argument('-f', '--file', dest='file_to_convert',
                        default="histos/histAll.json", help="File with coffea hists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("\nRunning with these parameters: ")
    logging.info(args)

    logging.info("Creating root files from json")
    create_root_file(args.file_to_convert, args.histos, args.output_dir)
