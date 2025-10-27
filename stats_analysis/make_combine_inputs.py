import os, sys
import ROOT
import argparse
import logging
import json
import yaml
import numpy as np
import pickle
from copy import copy, deepcopy
from convert_json_to_root import json_to_TH1
from make_variable_binning import make_variable_binning

import CombineHarvester.CombineTools.ch as ch
ROOT.gROOT.SetBatch(True)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def make_trigger_syst( json_input, root_output, name, rebin ):

    hData = json_input['nominal']['fourTag']['SR']
    hMC = json_input['CMS_bbbb_resolved_ggf_triggerEffSFUp']['fourTag']['SR']
    nominal = json_input['CMS_bbbb_resolved_ggf_triggerEffSFDown']['fourTag']['SR']

    # Previous way
    # for num, denom, ivar in [ (nominal, hData, 'Up'), (nominal, hMC, 'Down')]:
    #     n_sumw, n_sumw2 = np.array(num['values']), np.array(num['variances'])
    #     d_sumw, d_sumw2 = np.array(denom['values']), np.array(denom['variances'])
    #     variation = np.divide(n_sumw, d_sumw, out=np.ones(len(n_sumw)), where=d_sumw!=0)
    #     htrig = copy(hData)
    #     htrig['values'] *= ratio
    #     htrig['variances'] *= ratio*ratio
    #     root_output[f'CMS_bbbb_resolved_ggf_triggerEffSF{ivar}'] = json_to_TH1( htrig, name+ivar, rebin )
    
    n_MC, n_HLT = np.array(hMC['values']), np.array(nominal['values'])
    variation = 0.5 * (n_MC - n_HLT)

    for direction in ['Up', 'Down']:
        htrig = copy(hData)
        htrig['values'] += variation if (direction == 'Up') else -variation
        htrig['variances'] += 0.25 * (np.array(hMC['variances']) + np.array(nominal['variances']))
        root_output[f'CMS_bbbb_resolved_ggf_triggerEffSF{direction}'] = json_to_TH1(htrig, f"{name}{direction}", rebin)


def create_combine_root_file( file_to_convert,
                                rebin,
                                var,
                                output_dir,
                                systematics_file,
                                bkg_systematics_file,
                                metadata_file='coffea4bees/stats_analysis/metadata/HH4b.yml',
                                mixeddata_file=None,
                                variable_binning=False,
                                stat_only=False ):

    logging.info(f"Reading {metadata_file}")
    metadata = yaml.safe_load(open(metadata_file, 'r'))
    metadata['processes']['all'] = { **metadata['processes']['signal'], **metadata['processes']['background'] }
    logging.info(f"Reading {file_to_convert}")
    with open(file_to_convert, 'r') as f: coffea_hists = json.load(f)
    if systematics_file:
        logging.info(f"Reading {systematics_file}")
        with open(systematics_file, 'r') as f: coffea_hists_syst = json.load(f)
    if bkg_systematics_file and not stat_only:
        logging.info(f"Reading {bkg_systematics_file}")
        with open(bkg_systematics_file, 'rb') as f: bkg_syst_file = pickle.load(f) 
    if mixeddata_file:
        logging.info(f"Reading data from mixeddata file {mixeddata_file} and loading tt and multijet from mixeddata")
        with open(mixeddata_file, 'r') as f: mixeddata = json.load(f)
        with open(f"{os.path.dirname(mixeddata_file)}/histMixedBkg_TT.json", 'r') as f: mixedbkg_tt = json.load(f)
        with open(f"{os.path.dirname(mixeddata_file)}/histMixedBkg_data_3b_for_mixed.json", 'r') as f: mixedBkg_data3b = json.load(f)

    root_hists = {}
    mcSysts, closureSysts = [], []
    for iyear in coffea_hists[var]['data'].keys():
        root_hists[iyear] = {}

        ### For multijets
        root_hists[iyear]['multijet'] = {}
        root_hists[iyear]['multijet']['nominal'] = json_to_TH1(
            coffea_hists[var]['data'][iyear]['threeTag']['SR'], 'multijet_'+iyear+var, rebin )

        for iprocess in coffea_hists[var].keys():

            if iprocess.startswith(('TTTo', 'data')):
                
                if iprocess.startswith('TTTo') and mixeddata_file:
                    coffea_hist = mixedbkg_tt[var][f"{iprocess}_for_mixed"][iyear]['fourTag']['SR']
                else:
                    coffea_hist = coffea_hists[var][iprocess][iyear]['fourTag']['SR']
                root_hists[iyear][iprocess] = json_to_TH1( coffea_hist, 
                                                            f'{iprocess.split("4b")[0]}_{iyear}', rebin )
            else:
                root_hists[iyear][iprocess] = {}
                root_hists[iyear][iprocess]['nominal'] = json_to_TH1(
                    coffea_hists[var][iprocess][iyear]['fourTag']['SR'], iprocess+'_'+iyear, rebin )
                
        if systematics_file:
            for iprocess in metadata['processes']['signal']:
                root_hists[iyear][iprocess] = {}
                if stat_only:
                    root_hists[iyear][iprocess]["nominal"] = json_to_TH1(
                                                        coffea_hists_syst[var][iprocess][iyear]["nominal"]['fourTag']['SR'], 
                                                        f'{iprocess}_nominal_{iyear}', rebin )
                else:
                    for ivar in coffea_hists_syst[var][iprocess][iyear].keys():
                        
                        ## renaming syst
                        if 'prefire' in ivar: namevar = ivar.replace("CMS_prefire", 'CMS_l1_ecal_prefiring')
                        elif 'Absolute' in ivar: namevar = ivar.replace("Absolute", "Abs")
                        elif 'Relative' in ivar: namevar = ivar.replace("Relative", "Rel")
                        elif 'Flavor' in ivar: namevar = ivar.replace("Flavor", "Flav")
                        else: namevar = ivar
                        namevar = namevar.replace('_Up', 'Up').replace('_Down', 'Down')

                        # ### making btagging decorrelated
                        for stat in ['hfstats1', 'hfstats2', 'lfstats1', 'lfstats2']:
                            if stat in namevar:
                                namevar = namevar.replace(stat, f'{stat}_{iyear.replace("UL", "20")}')
                                break

                        ### check for dedicated JESUnc per year, if not conitnue
                        tmpvar = namevar.replace('Up','').replace('Down', '').replace('_postVFP', '').replace('_preVFP', '')
                        if tmpvar not in mcSysts and not 'nominal' in tmpvar: mcSysts.append( tmpvar )
                        tmpvar = ''.join(tmpvar[-2:])
                        if tmpvar.isdigit() and int(tmpvar) != int(iyear[2:4]): continue
                        
                        ### trigger efficiency
                        if 'triggerEffSFUp' in namevar:
                            make_trigger_syst(coffea_hists_syst[var][iprocess][iyear],
                                                root_hists[iyear][iprocess],
                                                f'{iprocess}_{ivar}_{iyear}', rebin)
                        elif 'triggerEffSFDown' in namevar: continue
                        else:
                            root_hists[iyear][iprocess][namevar] = json_to_TH1(
                                                            coffea_hists_syst[var][iprocess][iyear][ivar]['fourTag']['SR'], 
                                                            f'{iprocess}_{ivar}_{iyear}', rebin )
        if 'ggZH4b' in root_hists[iyear].keys():
            for ih in root_hists[iyear]['ggZH4b'].keys():
                root_hists[iyear]['ZH4b'][ih].Add( root_hists[iyear]['ggZH4b'][ih] )
                logging.info(f"Adding ggZH4b to ZH4b in {iyear} {ih}")
            del root_hists[iyear]['ggZH4b']
    
    logging.info("\n Merging UL16_preVFP and UL16_postVFP")
    for iy in list(root_hists.keys()):
        if 'UL16_preVFP' in iy:
            for ip, _ in list(root_hists[iy].items()):
                if isinstance(root_hists[iy][ip], dict):
                    for iv, _ in list(root_hists[iy][ip].items()):
                        root_hists[iy][ip][iv].Add( root_hists[iy.replace('pre', 'post')][ip][iv.replace('preV', 'postV')] )
                else:
                    root_hists[iy][ip].Add( root_hists[iy.replace('pre', 'post')][ip] )
            del root_hists[iy.replace('pre', 'post')]
            root_hists['_'.join(iy.split('_')[:-1])] = root_hists.pop(iy)


    ### renaming histos for final combine inputs
    for iy in list(root_hists.keys()):
        for jy in metadata['bin']:
            if ''.join(iy[-2:]) == ''.join(jy[-2:]):
                root_hists[jy] = root_hists.pop(iy)

    ### checking one-sided signal systematics
    for iy in root_hists.keys():
        for ip in root_hists[iy].keys():
            if ip in metadata['processes']['signal']:
                for iv in root_hists[iy][ip].keys():
                    if ('Up' in iv) or ('nominal' in iv): continue
                    nominal = root_hists[iy][ip]['nominal'] 
                    Up_var = root_hists[iy][ip][iv.replace('Down', 'Up')]
                    Down_var = root_hists[iy][ip][iv]
                    
                    for ibin in range(Up_var.GetNbinsX()):
                        up_bin = Up_var.GetBinContent(ibin+1)
                        down_bin = Down_var.GetBinContent(ibin+1)
                        nom_bin = nominal.GetBinContent(ibin+1)

                        if ((up_bin < nom_bin) and ( down_bin < nom_bin)) or ((up_bin > nom_bin) and ( down_bin > nom_bin)):
                            max_diff = max(abs(up_bin - nom_bin), abs(down_bin - nom_bin))
                            Up_var.SetBinContent(ibin+1, nom_bin + max_diff)
                            Down_var.SetBinContent(ibin+1, nom_bin - max_diff)
                            # print( iv, ibin, up_bin, down_bin, nom_bin, max_diff)

                        if max(up_bin, down_bin) > nom_bin*1.5:
                            tmp_nom_bin = nominal.GetBinContent(ibin)
                            tmp_up_bin = Up_var.GetBinContent(ibin)
                            tmp_down_bin = Down_var.GetBinContent(ibin)

                            Down_var.SetBinContent(ibin+1, nom_bin - (tmp_nom_bin - tmp_down_bin))
                            Up_var.SetBinContent(ibin+1, nom_bin + (tmp_up_bin - tmp_nom_bin))
        
    if mixeddata_file:
        logging.info("\n Using multijet from mixeddata")
        for iy in root_hists:
            root_hists[iy]["multijet"]["nominal"] = json_to_TH1(
                mixedBkg_data3b[var]['data_3b_for_mixed'][iy.split("_")[1]]['threeTag']['SR'], 
                f'multijet_{iy}_{var}', rebin )

    if not stat_only:
        for channel in metadata['bin']:
            for ibin, ivalues in bkg_syst_file.items():
                bkg_name_syst = f"CMS_bbbb_resolved_bkg_datadriven_{ibin.replace('_hh', '').replace('vari', 'variance')}"
                root_hists[channel]['multijet'][bkg_name_syst] = root_hists[channel]['multijet']['nominal'].Clone()
                root_hists[channel]['multijet'][bkg_name_syst].SetName(f'multijet_{bkg_name_syst}')
                for i in range(len(ivalues)):
                    nom_val = root_hists[channel]['multijet'][bkg_name_syst].GetBinContent( i+1 )
                    root_hists[channel]['multijet'][bkg_name_syst].SetBinContent( i+1, nom_val*ivalues[i]  )

        closureSysts = [ i.replace('Up', '') for i in root_hists[next(iter(root_hists))]['multijet'].keys() if i.endswith('Up') ]

    ### renaming histos for final combine inputs
    for channel in root_hists.keys():
        tt_label = metadata['processes']['background']['tt']['label']
        root_hists[channel][tt_label] = root_hists[channel]['data'].Clone()
        root_hists[channel][tt_label].SetName(tt_label)
        root_hists[channel][tt_label].SetTitle(f"{tt_label}_{channel}")
        root_hists[channel][tt_label].Reset()
        for ip, _ in list(root_hists[channel].items()):
            if 'TTTo' in ip:
                root_hists[channel][tt_label].Add( root_hists[channel][ip] )
                del root_hists[channel][ip]
            elif 'data' in ip:
                if mixeddata_file:
                    logging.info(f"Using mixeddata for data_obs")
                    root_hists[channel]['data_obs'] = json_to_TH1(
                        mixeddata[var]['mix_v0'][channel.split("_")[1]]['fourTag']['SR'], 
                        f'data_obs{channel}', rebin )
                else:
                    root_hists[channel]['data_obs'] = root_hists[channel][ip]
                root_hists[channel]['data_obs'].SetName("data_obs")
                root_hists[channel]['data_obs'].SetTitle(f"data_obs_{channel}")
                del root_hists[channel][ip]
            elif ip in metadata['processes']['all'].keys():
                label = metadata['processes']['signal'][ip]['label'] if ip in metadata['processes']['signal'].keys() else metadata['processes']['background'][ip]['label']
                root_hists[channel][label] = deepcopy(root_hists[channel][ip])
                if isinstance(root_hists[channel][label], ROOT.TH1F):
                    root_hists[channel][label].SetName(label)
                    root_hists[channel][label].SetTitle(f'{label}_{channel}')
                else:
                    for ivar, _ in root_hists[channel][label].items():
                        if 'nominal' in ivar:
                            root_hists[channel][label][ivar].SetName(label)
                            root_hists[channel][label][ivar].SetTitle(f'{label}_{channel}')
                        else:
                            tmpivar = ivar.replace("_preVFP", "")
                            root_hists[channel][label][ivar] = root_hists[channel][label][ivar].Clone(f'{label}_{tmpivar}')
                            root_hists[channel][label][ivar].SetTitle(f'{label}_{tmpivar}_{channel}')
                if not ip.startswith(label): del root_hists[channel][ip]
            else: 
                logging.info(f"{ip} not in metadata processes, removing from root file.")
                del root_hists[channel][ip]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = "shapes.root" 
    output = output_dir+"/"+output_file

    root_file = ROOT.TFile(output, 'recreate')

    for channel in root_hists.keys():
        root_file.cd()
        try:
            directory = root_file.Get(channel)
            directory.IsZombie()
        except ReferenceError:
            directory = root_file.mkdir(channel)

        root_file.cd(channel)
        for ih_name, ih in root_hists[channel].items():
            if isinstance(ih, dict):
                for _, ih2 in root_hists[channel][ih_name].items():
                    ih2.Write()
            else:
                ih.Write()
    root_file.Close()

    logging.info("\n File "+output+" created.")

    #### make datacard
    for i, ibin in enumerate(metadata['bin']):

        ibin_label = ibin.split("_")[0]
        cb = ch.CombineHarvester()
        cb.SetVerbosity(3)

        cats = [(i, ibin)]
        cb.AddObservations(['*'], [''], ['13TeV'], ['*'], cats)
        cb.AddProcesses(['*'], [''], ['13TeV'], ['*'], [ metadata['processes']['all'][ibin]['label'] for ibin in metadata['processes']['background'] ], cats, False)
        signals = [ metadata['processes']['all'][ibin]['label'] for ibin in metadata['processes']['signal'] ]
        cb.AddProcesses(['*'], [''], ['13TeV'], ['*'], signals, cats, True)

        if stat_only:
            cb.cp().backgrounds().ExtractShapes( output, '$BIN/$PROCESS', '')
            cb.cp().signals().ExtractShapes( output, '$BIN/$PROCESS', '')
            cb.PrintAll()
            cb.WriteDatacard(f"{output_dir}/datacard_{ibin}.txt", f"{output_dir}/{ibin}_{output_file}")

        else:
            for nuisance in closureSysts:
                cb.cp().process(["multijet"]).AddSyst(cb, nuisance, 'shape', ch.SystMap()(1.0))
            cb.SetGroup("multijet", closureSysts)
            
            btagSysts = []
            othersSysts = []
            psfsrSysts = []
            mtopSysts = []
            for nuisance in mcSysts:
                if ('2016' in nuisance) or ('UL16' in nuisance):
                    nuisance = nuisance.replace('UL16_postVFP', '2016')
                    if ('2016' in ibin):
                        cb.cp().signals().AddSyst(cb, nuisance, 'shape', ch.SystMap('bin')([f'{ibin_label}_2016'],1.0))
                elif ('2017' in nuisance):
                    if('2017' in ibin):
                        cb.cp().signals().AddSyst(cb, nuisance, 'shape', ch.SystMap('bin')([f'{ibin_label}_2017'],1.0))
                elif ('2018' in nuisance):
                    if ('2018' in ibin):
                        cb.cp().signals().AddSyst(cb, nuisance, 'shape', ch.SystMap('bin')([f'{ibin_label}_2018'],1.0))
                else:
                    cb.cp().signals().AddSyst(cb, nuisance, 'shape', ch.SystMap()(1.0))
                if 'btag' in nuisance:
                    btagSysts.append(nuisance)
                elif 'ps_fsr' in nuisance:
                    psfsrSysts.append(nuisance)
                else: othersSysts.append(nuisance)
            cb.SetGroup("ps_fsr", psfsrSysts)
            cb.SetGroup("btag", btagSysts)

            for isyst in metadata['uncertainty']:
                if 'mtop' in isyst: mtopSysts.append(isyst)
                else: othersSysts.append(isyst)
                if ('2016' in isyst):
                    if ('2016' in ibin):
                        cb.cp().signals().AddSyst(cb, isyst, metadata['uncertainty'][isyst]['type'], ch.SystMap('bin')([f'{ibin_label}_2016'],metadata['uncertainty'][isyst]['years'][f'{ibin_label}_2016']))
                elif ('2017' in isyst):
                    if ('2017' in ibin):
                        cb.cp().signals().AddSyst(cb, isyst, metadata['uncertainty'][isyst]['type'], ch.SystMap('bin')([f'{ibin_label}_2017'],metadata['uncertainty'][isyst]['years'][f'{ibin_label}_2017']))
                elif ('2018' in isyst):
                    if ('2018' in ibin):
                        cb.cp().signals().AddSyst(cb, isyst, metadata['uncertainty'][isyst]['type'], ch.SystMap('bin')([f'{ibin_label}_2018'],metadata['uncertainty'][isyst]['years'][f'{ibin_label}_2018']))
                elif ('1718' in isyst):
                    if '2017' in ibin or '2018' in ibin:
                        cb.cp().signals().AddSyst(cb, isyst, metadata['uncertainty'][isyst]['type'], ch.SystMap('bin')
                                            ([ibin],metadata['uncertainty'][isyst]['years'][ibin]))
                else:
                    cb.cp().signals().AddSyst(cb, isyst, metadata['uncertainty'][isyst]['type'], ch.SystMap('bin')
                                            ([ibin], metadata['uncertainty'][isyst]['years'][ibin])
                                            )
            cb.SetGroup("others", othersSysts)
            if ibin_label.startswith('hh'):    
                cb.SetGroup("mtop", mtopSysts)
                cb.SetGroup("signal_norm_xsbr", [
                    'pdf_Higgs_ggHH', 'BR_hbb', 'THU_HH'])
                cb.SetGroup("signal_norm_xs", ['THU_HH', 'pdf_Higgs_ggHH'])

            cb.cp().backgrounds().ExtractShapes(
                output, '$BIN/$PROCESS', '$BIN/$PROCESS_$SYSTEMATIC')
            cb.cp().signals().ExtractShapes(
                output, '$BIN/$PROCESS', '$BIN/$PROCESS_$SYSTEMATIC')
            
            cb.cp().SetAutoMCStats(cb, 0, 1, 1)

            cb.PrintAll()
            cb.WriteDatacard(f"{output_dir}/datacard_{ibin}.txt", f"{output_dir}/{ibin}_{output_file}")



if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser(
        description='Convert json hist to root TH1F', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output_dir', dest="output_dir",
                        default="./datacards/", help='Output directory.')
    parser.add_argument('--var', dest="variable", 
                        default="SvB_MA.ps_hh_fine", help='Variable to make histograms.')
    parser.add_argument('-f', '--file', dest='file_to_convert',
                        default="histos/histAll.json", help="File with coffea hists")
    parser.add_argument('-r', '--rebin', dest='rebin', type=int,
                        default=15, help="Rebin")
    parser.add_argument('--variable_binning', dest='variable_binning', action="store_true",
                        default=False, help="Make variable binning based on the amount of signal. (ran make_variable_binning.py)")
    parser.add_argument('-s', '--syst_file', dest='systematics_file',
                        default='', help="File contain systematic variations")
    parser.add_argument('-b', '--bkg_syst_file', dest='bkg_systematics_file',
                        default='', help="File contain background systematic variations")
    parser.add_argument('--mixeddata_file', dest='mixeddata_file',
                        default='', help="File contain mixeddata")                    
    parser.add_argument('-m', '--metadata', dest='metadata',
                        default='coffea4bees/stats_analysis/metadata/HH4b.yml', help="File contain systematic variations")
    parser.add_argument('--stat_only', dest='stat_only', action="store_true",
                        default=False, help="Create stat only inputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("\nRunning with these parameters: ")
    logging.info(args)

    if args.variable_binning:
        try:
            args.rebin= list(np.loadtxt(f"{os.path.dirname(args.bkg_systematics_file)}/variable_binning_array.txt"))
            logging.info(f"Using variable binning {args.rebin}")
        except:
            logging.info(f"Creating variable binning and using {args.rebin} as threshold for data and signal.")
            args.rebin = list(make_variable_binning(args.file_to_convert, args.variable, args.rebin, None ))

    logging.info("Creating root files for combine")
    create_combine_root_file(
        args.file_to_convert,
        args.rebin,
        args.variable,
        args.output_dir,
        args.systematics_file,
        args.bkg_systematics_file,
        metadata_file=args.metadata,
        mixeddata_file=args.mixeddata_file,
        stat_only=args.stat_only,
    )