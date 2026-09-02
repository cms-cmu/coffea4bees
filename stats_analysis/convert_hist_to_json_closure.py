import os, sys
import argparse
import logging
import json
import tempfile
import numpy as np
from coffea.util import load
from convert_hist_to_json import hist_to_json



if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser( description='Convert json hist to root TH1F',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hist_key', nargs="+",
                        default=['ps_zz',      'ps_zh',      'ps_hh',
                                 'ps_zz_fine', 'ps_zh_fine', 'ps_hh_fine',
                                 'ps_ttHbb_fine', 'ps', 'm4j'],
                        help='List of histograms to convert')

    parser.add_argument('-o', '--output', dest="output",
                        default=None, help='Output file and directory.')

    parser.add_argument('-i', '--input_file', dest='input_file', nargs='+',
                        default=["../output/histAll.coffea"], help="File(s) with coffea hists")
    parser.add_argument('--scale_mixed', type=float, default=1.0,
                        help="Scale factor to apply to mixed datasets (k-factor)")
    parser.add_argument('--auto_scale_mixed', action="store_true", default=False,
                        help="Automatically normalize mixed data to match total background prediction (Data 3b + TTbar4b 3b) in SR")
    parser.add_argument('--pure_qcd', '--no_ttbar', dest='pure_qcd', action="store_true", default=False,
                        help="Pure QCD closure mode where mixed data is already ttbar-subtracted (do not include or require ttbar)")

    parser.add_argument("--debug", action="store_true")
    #parser.add_argument("--signal", action="store_true")
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


    input_files = args.input_file if isinstance(args.input_file, list) else [args.input_file]
    coffea_hists_list = {}
    for f in input_files:
        logging.info(f"Loading {f}")
        f_hists = load(f)["hists"]
        for k, v in f_hists.items():
            if k not in coffea_hists_list:
                coffea_hists_list[k] = [v]
            else:
                coffea_hists_list[k].append(v)


    #
    # Collect the histogram names
    #
    hists_to_save = []
    for k in coffea_hists_list.keys():
        for _hist_key in args.hist_key:
            if not k.find(_hist_key) == -1:
                hists_to_save.append(k)

    print(hists_to_save)

    # Remove duplicates
    hists_to_save = set(hists_to_save)

    #
    #  Criteria to save
    #
    save_dict = {}
    for sub_sample in range(15):
        save_dict[f"mix_v{sub_sample}"] = [('fourTag','SR')]
        save_dict[f"syn_v{sub_sample}"] = [('fourTag','SR')]
    save_dict["synthetic_data"]             = [('fourTag','SR')]

    save_dict["data_3b_for_mixed"]          = [('threeTag','SR')]
    save_dict["data_3b"]                    = [('threeTag','SR')]
    save_dict["data"]                       = [('threeTag','SR'), ('fourTag','SR')]
    save_dict["TTTo2L2Nu_for_mixed"]        = [('fourTag','SR')]
    save_dict["TTToSemiLeptonic_for_mixed"] = [('fourTag','SR')]
    save_dict["TTToHadronic_for_mixed"]     = [('fourTag','SR')]
    save_dict["TTbar4b_from_d3"]            = [('threeTag','SR'), ('fourTag','SR')]
    save_dict["TTbar3b_from_d3"]            = [('threeTag','SR')]
    save_dict["ZZ4b"]                       = [('fourTag','SR')]
    save_dict["ZH4b"]                       = [('fourTag','SR')]
    save_dict["GluGluToHHTo4B_cHHH1"]       = [('fourTag','SR')]
    save_dict["ttHbb"]                      = [('fourTag','SR')]


    json_dict = {}
    for ih in hists_to_save:
        json_dict[ih] = {}

        eff_scale_mixed = args.scale_mixed
        if args.auto_scale_mixed:
            # Calculate ratio: (Data 3b SR + TTbar4b 3b SR) / Mix_v0 4b SR across all years
            bkg_sr_sum = 0.0
            mix_sr_sum = 0.0
            try:
                for h in coffea_hists_list[ih]:
                    for proc in ['data', 'data_3b_for_mixed', 'data_3b']:
                        if proc in h.axes['process']:
                            for iy in h.axes['year']:
                                sel = {'process': proc, 'year': iy, 'tag': 'threeTag', 'region': 'SR'}
                                for ax in h.axes:
                                    if ax.name.startswith(('pass', 'fail')) and ax.name not in sel:
                                        sel[ax.name] = sum
                                bkg_sr_sum += float(np.sum(h[sel].values()))
                            break
                    if not args.pure_qcd and 'TTbar4b_from_d3' in h.axes['process']:
                        for iy in h.axes['year']:
                            sel = {'process': 'TTbar4b_from_d3', 'year': iy, 'tag': 'threeTag', 'region': 'SR'}
                            for ax in h.axes:
                                if ax.name.startswith(('pass', 'fail')) and ax.name not in sel:
                                    sel[ax.name] = sum
                            bkg_sr_sum += float(np.sum(h[sel].values()))
                    if 'mix_v0' in h.axes['process']:
                        for iy in h.axes['year']:
                            sel = {'process': 'mix_v0', 'year': iy, 'tag': 'fourTag', 'region': 'SR'}
                            for ax in h.axes:
                                if ax.name.startswith(('pass', 'fail')) and ax.name not in sel:
                                    sel[ax.name] = sum
                            mix_sr_sum += float(np.sum(h[sel].values()))
                if mix_sr_sum > 0 and bkg_sr_sum > 0:
                    eff_scale_mixed = bkg_sr_sum / mix_sr_sum
                    logging.info(f"[{ih}] Auto-scale mixed factor: {eff_scale_mixed:.6f} (Bkg SR: {bkg_sr_sum:.1f}, Mix SR: {mix_sr_sum:.1f})")
            except Exception as e:
                logging.warning(f"Failed to auto-calculate scale_mixed for {ih}: {e}")

        for h in coffea_hists_list[ih]:
            for iprocess in h.axes['process']:
                if iprocess not in save_dict:
                    continue
                if iprocess in json_dict[ih]:
                    continue

                json_dict[ih][iprocess] = {}

                for iy in h.axes['year']:
                    json_dict[ih][iprocess][iy] = {}

                    for itag in h.axes['tag']:
                        json_dict[ih][iprocess][iy][itag] = {}

                        for iregion in h.axes['region']:
                            tag_region_pair = (itag, iregion)

                            if tag_region_pair not in save_dict[iprocess]:
                                if args.debug:
                                    print(f"skipping {iprocess} {tag_region_pair}")
                                continue

                            this_hist = {
                                'process' : iprocess,
                                'year' : iy,
                                'tag' : itag,
                                'region' : iregion,
                            }
                            for iaxis in h.axes:
                                if iaxis.name.startswith(('pass', 'fail')) and iaxis.name not in this_hist:
                                    this_hist[iaxis.name] = sum
                            logging.info(f"Converting hist {ih} {this_hist}")
                            h_data = hist_to_json( h[this_hist] )
                            if (iprocess.startswith('mix_') or iprocess.startswith('syn_') or iprocess == 'synthetic_data') and eff_scale_mixed != 1.0:
                                h_data['values'] = [v * eff_scale_mixed for v in h_data['values']]
                                h_data['variances'] = [v * (eff_scale_mixed ** 2) for v in h_data['variances']]
                                h_data['underflow_value'] *= eff_scale_mixed
                                h_data['underflow_variance'] *= (eff_scale_mixed ** 2)
                                h_data['overflow_value'] *= eff_scale_mixed
                                h_data['overflow_variance'] *= (eff_scale_mixed ** 2)
                            json_dict[ih][iprocess][iy][itag][iregion] = h_data

    if args.output is None:
        output = input_files[0].replace(".coffea",".json")
    else:
        output = args.output

    logging.info(f"Saving histos in json format in {output}")
    output_dir = '/'.join( output.split('/')[:-1] )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_dir or '.', suffix='.json.tmp')
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            json.dump(json_dict, f)
        os.replace(tmp_path, output)
    except:
        os.unlink(tmp_path)
        raise
