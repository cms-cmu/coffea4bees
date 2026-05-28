import os, sys
import argparse
import logging
import json
import tempfile
import numpy as np
from coffea.util import load
import hist

def hist_to_json( coffea_hist ):
    """docstring for hist_to_root"""

    yhist = {
        'edges' : coffea_hist.axes[0].edges.tolist(),
        'centers' : coffea_hist.axes[0].centers.tolist(),
        'values' : coffea_hist.values().tolist(),
        'variances' : coffea_hist.variances().tolist(),
        'underflow_value' : coffea_hist[hist.loc(-np.inf)].value,
        'underflow_variance' : coffea_hist[hist.loc(-np.inf)].variance,
        'overflow_value' : coffea_hist[hist.loc(+np.inf)].value,
        'overflow_variance' : coffea_hist[hist.loc(+np.inf)].variance,
    }


#    ### in case of negative values
#    yhist['values'] = np.where( yhist['values']<0, 0, yhist['values'] ).tolist()
#    yhist['variances'] = np.where( yhist['variances']<0, 0, yhist['variances'] ).tolist()

    return yhist


if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser( description='Convert json hist to root TH1F',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--histos', dest="histos", nargs="+",
                        default=['SvB.ps_zz', 'SvB.ps_zh', 'SvB.ps_hh',
                                 'SvB.ps_zz_fine', 'SvB.ps_zh_fine', 'SvB.ps_hh_fine',
                                 'SvB_MA.ps_zz', 'SvB_MA.ps_zh', 'SvB_MA.ps_hh',
                                 'SvB_MA.ps_zz_fine', 'SvB_MA.ps_zh_fine', 'SvB_MA.ps_hh_fine' ],
                        help='List of histograms to convert')


    parser.add_argument('-o', '--output', dest="output",
                        default="./histos/histAll.json", help='Output file and directory.')
    parser.add_argument('-i', '--input_file', dest='input_file',
                        default="../analysis/hists/histAll.coffea", help="File with coffea hists")
    parser.add_argument('-s', '--syst_file', dest='systematics_file', action='store_true',
                        default=False, help="File contain systematic variations")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"\nRunning with these parameters: {args}")

    codes = {
        'region' : {
            2 : 'other',
            1 : 'SB',
            0 : 'SR'
        },
        'tag' : {
            0 : 'threeTag',
            1 : 'fourTag',
            2 : 'other'
        }
    }

    coffea_hists = load(args.input_file)["hists"]

    json_dict = {}

    if not args.systematics_file:
        for ih in args.histos:
            json_dict[ih] = {}
            for iprocess in coffea_hists[ih].axes[0]:
                json_dict[ih][iprocess] = {}
                for iy in coffea_hists[ih].axes[1]:
                    json_dict[ih][iprocess][iy] = {}
                    for itag in coffea_hists[ih].axes[2]:
                        json_dict[ih][iprocess][iy][itag] = {}
                        for iregion in coffea_hists[ih].axes[3]:
                            this_hist = {
                                'process' : iprocess,
                                'year' : iy,
                                'tag' : itag,
                                'region' : iregion,
                            }
                            logging.info(f"Converting hist {ih} {this_hist}")
                            json_dict[ih][iprocess][iy][itag][iregion] = hist_to_json( coffea_hists[ih][this_hist] )
                            
    else:
        codes = {
            'region' : {
                2 : 'other',
                1 : 'SB',
                0 : 'SR'
            },
            'tag' : {
                1 : 'threeTag',
                0 : 'fourTag',
            }
        }
        for ih in args.histos:
            json_dict[ih] = {}
            for iprocess in coffea_hists[ih].axes[0]:
                json_dict[ih][iprocess] = {}
                for iy in coffea_hists[ih].axes[1]:
                    json_dict[ih][iprocess][iy] = {}
                    for ivar in coffea_hists[ih].axes[2]:
                        json_dict[ih][iprocess][iy][ivar] = {}
                        for itag in coffea_hists[ih].axes[3]:
                            json_dict[ih][iprocess][iy][ivar][itag] = {}
                            for iregion in coffea_hists[ih].axes[4]:
                                this_hist = {
                                    'process' : iprocess,
                                    'year' : iy,
                                    'variation' : ivar,
                                    'tag' : itag,
                                    'region' : iregion,
                                }
                                logging.info(f"Converting hist {ih} {this_hist}")
                                json_dict[ih][iprocess][iy][ivar][itag][iregion] = hist_to_json( coffea_hists[ih][this_hist] )

    logging.info(f"Saving histos in json format in {args.output}")
    output_dir = '/'.join( args.output.split('/')[:-1] )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_dir or '.', suffix='.json.tmp')
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            json.dump(json_dict, f)
        os.replace(tmp_path, args.output)
    except:
        os.unlink(tmp_path)
        raise
