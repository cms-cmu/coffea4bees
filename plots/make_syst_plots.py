import os, sys
import ROOT
import cmsstyle as CMS
import argparse
import logging
import yaml
import json

ROOT.gROOT.SetBatch(True)


def variable_to_regular_binning( hist ):
    nbins = hist.GetNbinsX()
    xbins = hist.GetXaxis().GetXbins()
    if xbins.GetSize() == 0:
        return hist.Clone()
    
    new_hist = ROOT.TH1F(hist.GetName() + "_regular", hist.GetTitle(), nbins, 0., 1.)
    for i in range(1, nbins + 1):
        new_hist.SetBinContent(i, hist.GetBinContent(i))
        new_hist.SetBinError(i, hist.GetBinError(i))
    
    return new_hist


if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser(
        description='Takes root hists from combine inputs and make variations', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output_dir', dest="output_dir",
                        default="./plots/", help='Output directory.')
    parser.add_argument('-i', '--inputfile', dest='inputfile', type=str,
                        default="shapes.root", help="File with root hists")
    parser.add_argument('-d', '--datacard', dest='datacard', type=str,
                        default="", help="Datacard created with input file")
    parser.add_argument('-v', '--variable', dest='variable', type=str,
                        default="SvB_MA_ps_hh_fine", help="Variable to plot")
    parser.add_argument('-s', '--signal', dest='signal',
                        default='GluGluToHHTo4B_cHHH1', help="Signal to plot")
    parser.add_argument('-m', '--metadata', dest='metadata',
                        default='coffea4bees/stats_analysis/metadata/HH4b.yml', help="Metadata file")
    # parser.add_argument('-r', '--rebin', dest='rebin', type=int,
    #                     default=15, help="Rebin")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("\nRunning with these parameters: ")
    logging.info(args)
    
    logging.info(f"Reading {args.metadata}")
    metadata = yaml.safe_load(open(args.metadata, 'r'))
    label_mj = metadata['processes']['background']['multijet']['label']
    label_tt = metadata['processes']['background']['tt']['label']
    label_signal = metadata['processes']['signal'][args.signal]['label']

    with open('coffea4bees/stats_analysis/nuisance_names.json', 'r') as f:
        nuisance_names = json.load(f)

    root_hists = ROOT.TFile.Open(args.inputfile)

    nom_data = root_hists.Get(f'{metadata["bin"][0]}/data_obs').Clone()
    nom_data.Reset()
    nom_tt = nom_data.Clone(label_tt)
    nom_mj = nom_data.Clone(label_mj)
    nom_signal = nom_data.Clone(label_signal)
    for ichannel in metadata['bin']:
        nom_data.Add( root_hists.Get(f"{ichannel}/data_obs") )
        nom_tt.Add( root_hists.Get(f"{ichannel}/{label_tt}" ))
        nom_mj.Add( root_hists.Get(f"{ichannel}/{label_mj}") )
        nom_signal.Add( root_hists.Get(f"{ichannel}/{label_signal}") )
    nom_data = variable_to_regular_binning(nom_data)
    nom_tt = variable_to_regular_binning(nom_tt)
    nom_mj = variable_to_regular_binning(nom_mj)
    nom_signal = variable_to_regular_binning(nom_signal)

    if not os.path.exists(f"{args.output_dir}"):
        os.makedirs(f"{args.output_dir}")

    xmax = nom_mj.GetXaxis().GetXmax()
    ymax = nom_mj.GetMaximum()*1.2

    # Create a new TH1 histogram with the nominal + high error bar of the nom_mj
    nom_mj_high_error = nom_mj.Clone("nom_mj_high_error")
    nom_mj_high_error.Reset()
    nom_mj_low_error = nom_mj.Clone("nom_mj_low_error")
    nom_mj_low_error.Reset()
    for i in range(1, nom_mj.GetNbinsX() + 1):
        nom_mj_high_error.SetBinContent(i, (nom_mj.GetBinErrorUp(i)+nom_mj.GetBinContent(i))/nom_mj.GetBinContent(i))
        nom_mj_low_error.SetBinContent(i, (nom_mj.GetBinContent(i)-nom_mj.GetBinErrorUp(i))/nom_mj.GetBinContent(i))

    # Styling
    CMS.SetExtraText("Preliminary")
    iPos = 0
    CMS.SetLumi("")
    CMS.SetEnergy("13")
    CMS.ResetAdditionalInfo()
    bkg_can = CMS.cmsDiCanvas('bkg_can',0,xmax,0,ymax,0.8,1.2,
                                "SvB MA Classifier Regressed P(Signal) | P(HH) is largest",
                                "Events", "Var/Nom",
                                    square=CMS.kSquare, extraSpace=0.05, iPos=iPos)
    bkg_can.cd(1)
    leg = CMS.cmsLeg(0.70, 0.89 - 0.05 * 4, 0.99, 0.89, textSize=0.04)
    tmp_bkg = nom_mj.Clone("tmp_bkg")
    leg.AddEntry( tmp_bkg, f'Multijet', 'l' )
    CMS.cmsDraw( tmp_bkg, 'histe', fstyle=0,  marker=1, alpha=1, lcolor=ROOT.kBlack, fcolor=ROOT.kBlack )
    CMS.GetcmsCanvasHist(bkg_can.cd(1)).GetXaxis().SetTitleSize(0.05)
    CMS.GetcmsCanvasHist(bkg_can.cd(1)).GetYaxis().SetTitleOffset(1.5)
    CMS.GetcmsCanvasHist(bkg_can.cd(1)).GetYaxis().SetTitleSize(0.05)
    
    bkg_can.cd(2)
    CMS.cmsDraw( nom_mj_high_error, 'p', fstyle=0, marker=22, alpha=1, mcolor=ROOT.kRed )
    CMS.cmsDraw( nom_mj_low_error, 'psame', fstyle=0, marker=23, alpha=1, mcolor=ROOT.kBlue )
    ref_line = ROOT.TLine(0, 1, 1, 1)
    CMS.cmsDrawLine(ref_line, lcolor=ROOT.kBlack, lstyle=ROOT.kDotted)
    CMS.GetcmsCanvasHist(bkg_can.cd(2)).GetXaxis().SetTitleOffset(1.5)
    CMS.GetcmsCanvasHist(bkg_can.cd(2)).GetYaxis().SetTitleOffset(0.8)
    CMS.GetcmsCanvasHist(bkg_can.cd(2)).GetYaxis().SetTitleSize(0.09)
    CMS.GetcmsCanvasHist(bkg_can.cd(2)).GetXaxis().SetTitleSize(0.095)

    CMS.SaveCanvas( bkg_can, f"{args.output_dir}/{args.variable}_multijet_only.pdf" )

    # Styling
    CMS.SetExtraText("Preliminary")
    iPos = 0
    CMS.SetLumi("")
    CMS.SetEnergy("13")
    CMS.ResetAdditionalInfo()
    nominal_can = CMS.cmsDiCanvas('nominal_can',0,xmax,0,ymax,0.8,1.2,
                                "SvB MA Classifier Regressed P(Signal) | P(HH) is largest",
                                "Events", 'Data/Pred.',
                                    square=CMS.kSquare, extraSpace=0.05, iPos=iPos)
    nominal_can.cd(1)
    # leg = CMS.cmsLeg(0.81, 0.89 - 0.05 * 7, 0.99, 0.89, textSize=0.04)
    leg = CMS.cmsLeg(0.70, 0.89 - 0.05 * 4, 0.99, 0.89, textSize=0.04)

    stack = ROOT.THStack()
    CMS.cmsDrawStack(stack, leg, {'ttbar': nom_tt, 'Multijet': nom_mj }, data= nom_data, palette=['#85D1FBff', '#FFDF7Fff'] )
    tmp_signal = nom_signal.Clone("tmp_signal")
    tmp_signal.Scale( 100 )
    leg.AddEntry( tmp_signal, f'HH4b (x100)', 'l' )
    CMS.cmsDraw( tmp_signal, 'hist', fstyle=0,  marker=1, alpha=1, lcolor=ROOT.kRed, fcolor=ROOT.kRed )
    CMS.GetcmsCanvasHist(nominal_can.cd(1)).GetYaxis().SetTitleOffset(1.5)
    CMS.GetcmsCanvasHist(nominal_can.cd(1)).GetYaxis().SetTitleSize(0.05)

    nominal_can.cd(2)

    bkg = nom_mj.Clone()
    bkg.Add( nom_tt )
    print(nom_data.GetNbinsX())
    print(bkg.GetNbinsX())
    ratio = ROOT.TGraphAsymmErrors()
    ratio.Divide( nom_data, bkg, 'pois' )
    CMS.cmsDraw( ratio, 'P', mcolor=ROOT.kBlack )

    ref_line = ROOT.TLine(0, 1, 1, 1)
    CMS.cmsDrawLine(ref_line, lcolor=ROOT.kBlack, lstyle=ROOT.kDotted)
    CMS.GetcmsCanvasHist(nominal_can.cd(2)).GetXaxis().SetTitleSize(0.095)
    CMS.GetcmsCanvasHist(nominal_can.cd(2)).GetYaxis().SetTitleSize(0.09)
    CMS.GetcmsCanvasHist(nominal_can.cd(2)).GetXaxis().SetTitleOffset(1.5)
    CMS.GetcmsCanvasHist(nominal_can.cd(2)).GetYaxis().SetTitleOffset(0.8)

    CMS.SaveCanvas( nominal_can, f"{args.output_dir}/{args.variable}_nominal.pdf", close=False )
    CMS.SaveCanvas( nominal_can, f"{args.output_dir}/{args.variable}_nominal.png" )
    
    if args.datacard:
        ## extract systematics from datacard
        list_systs = []
        with open(args.datacard, 'r') as infile:
            for line in infile:
                if ' shape ' in line:
                    list_systs.append(line.split()[0])

        for syst in list_systs:
            print(f"Processing {syst}")
            if 'bkg' in syst: 
                nominal = nom_mj
                label = label_mj
                plot_label = 'Nominal Multijet'
            else: 
                nominal = nom_signal
                label = label_signal
                plot_label = 'Nominal HH4b'

            
            nominal_hist = nominal.Clone(f"{syst}_nom")
            up_hist = nominal.Clone(f"{syst}_up")
            down_hist = nominal.Clone(f"{syst}_down")
            up_hist.Reset()
            down_hist.Reset()

            if syst.endswith(('2016', '2017', '2018')):
                tmp = syst.split('_')[-1]
                nominal_hist = root_hists.Get(f"HH4b_{tmp}/{label_signal}")
                up_hist.Add( root_hists.Get(f"HH4b_{tmp}/{label}_{syst}Up") )
                down_hist.Add( root_hists.Get(f"HH4b_{tmp}/{label}_{syst}Down") )

            else:
                for ichannel in metadata['bin']:
                    up_hist.Add( root_hists.Get(f"{ichannel}/{label}_{syst}Up") )
                    down_hist.Add( root_hists.Get(f"{ichannel}/{label}_{syst}Down") )

            nominal_hist = variable_to_regular_binning(nominal_hist)
            up_hist = variable_to_regular_binning(up_hist)
            down_hist = variable_to_regular_binning(down_hist)

            xmax = nominal_hist.GetXaxis().GetXmax()
            ymax = nominal_hist.GetMaximum()*1.2

            CMS.SetExtraText("Simulation Preliminary")
            iPos = 0
            CMS.SetLumi("")
            CMS.SetEnergy("13")
            CMS.ResetAdditionalInfo()
            syst_can = CMS.cmsDiCanvas(f'{syst}_can',0,xmax,0,ymax,0.8,1.2,
                                        "SvB MA Classifier Regressed P(Signal) | P(HH) is largest",
                                        "Events", 'Var/Nom',
                                            square=CMS.kSquare, extraSpace=0.05, iPos=iPos)
            syst_can.cd(1)
            leg = CMS.cmsLeg(0.35, 0.89 - 0.04 * 4, 0.69, 0.89, textSize=0.04)
            try: leg.SetHeader(nuisance_names[syst] , 'L')
            except: leg.SetHeader(syst , 'L')

            leg.AddEntry( nominal_hist, plot_label, 'lp' )
            CMS.cmsDraw( nominal_hist, 'P', mcolor=ROOT.kBlack )
            leg.AddEntry( up_hist, f'Up', 'lp' )
            CMS.cmsDraw( up_hist, 'hist', fstyle=0, marker=1, alpha=1, lcolor=ROOT.kBlue, fcolor=ROOT.kBlue )
            leg.AddEntry( down_hist, f'Down', 'lp' )
            CMS.cmsDraw( down_hist, 'hist', fstyle=0,  marker=1, alpha=1, lcolor=ROOT.kRed, fcolor=ROOT.kRed )
            CMS.GetcmsCanvasHist(syst_can.cd(1)).GetYaxis().SetTitleOffset(1.5)
            CMS.GetcmsCanvasHist(syst_can.cd(1)).GetYaxis().SetTitleSize(0.05)

            syst_can.cd(2)

            ratio_up = ROOT.TGraphAsymmErrors()
            ratio_up.Divide( nominal_hist.Clone(f"nominalup_{syst}"), up_hist, 'pois' )
            CMS.cmsDraw( ratio_up, 'p', fstyle=0, marker=23, mcolor=ROOT.kBlue, lcolor=0 )
            ratio_dn = ROOT.TGraphAsymmErrors()
            ratio_dn.Divide( nominal_hist.Clone(f"nominaldown_{syst}"), down_hist, 'pois' )
            CMS.cmsDraw( ratio_dn, 'psame', fstyle=0, marker=22, mcolor=ROOT.kRed, lcolor=0 )

            ref_line = ROOT.TLine(0, 1, 1, 1)
            CMS.cmsDrawLine(ref_line, lcolor=ROOT.kBlack, lstyle=ROOT.kDotted)
            CMS.GetcmsCanvasHist(syst_can.cd(2)).GetXaxis().SetTitleSize(0.095)
            CMS.GetcmsCanvasHist(syst_can.cd(2)).GetYaxis().SetTitleSize(0.09)
            CMS.GetcmsCanvasHist(syst_can.cd(2)).GetXaxis().SetTitleOffset(1.5)
            CMS.GetcmsCanvasHist(syst_can.cd(2)).GetYaxis().SetTitleOffset(0.8)

            CMS.SaveCanvas( syst_can, f"{args.output_dir}/{label}_{args.variable}_{syst}.pdf", close=False )
            CMS.SaveCanvas( syst_can, f"{args.output_dir}/{label}_{args.variable}_{syst}.png" )

            del syst_can, ratio_up, ratio_dn, nominal_hist, up_hist, down_hist
