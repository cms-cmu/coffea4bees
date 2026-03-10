import os, sys
import yaml
import hist
import argparse
#matplotlib.use('Agg')
sys.path.insert(0, os.getcwd())
import matplotlib.pyplot as plt
from coffea.util import load
sys.path.insert(0, os.getcwd())
from src.plotting.plots import load_hists
from hist.intervals import ratio_uncertainty
import yaml

def _round(val):
    return round(float(val),1)

def printLine(words):
    for iW in range(len(words)):
        if iW == 0:
            print(f'\t{words[iW]:<30}', end='')
        else:
            if iW % 2 == 0:
                print(f'\t{words[iW]:<10}', end='')
            else:
                print(f'\t\t{words[iW]:<10}', end='')
    print('')

def printCF(procKey, cf4, cf4_unit, cf3, cf3_unit, cf2=None, cf2_unit=None):

    bar = "-"*10
    print('\n')
    print(procKey,':\n')

    cuts_line = ["Cuts","FourTag","","ThreeTag",""]
    if cf2: cuts_line += ["TwoTag",""]
    printLine(cuts_line)

    bar_line = ["",bar,bar,bar,bar]
    if cf2: bar_line += [bar,bar]
    printLine(bar_line)

    weight_line = ["","weighted","(unit weight)","weighted","(unit weight)"]
    if cf2: weight_line += ["weighted","(unit weight)"]
    printLine(weight_line)
    print('\n')

    for cut in cf4.keys():
        counts_line =  [cut,
                        _round(cf4[cut]), _round(cf4_unit[cut]),
                        _round(cf3[cut]), _round(cf3_unit[cut])]
        if cf2:
               counts_line += [ _round(cf2[cut]), _round(cf2_unit[cut])]

        printLine(counts_line)

    print("\n")


def add(thisKey):
    print(f"\tadding {thisKey}")


if __name__ == '__main__':

    era_aliases = {}
    era_aliases["Run2"] = ["UL16_postVFPF" , "UL16_postVFPG" , "UL16_postVFPH" , "UL16_preVFPB" , "UL16_preVFPC" , "UL16_preVFPD" , "UL16_preVFPE" , "UL17C" , "UL17D" , "UL17E" , "UL17F" , "UL18A" , "UL18B" , "UL18C" , "UL18D"]
    era_aliases["Run2MC"] = ["UL16_postVFP" , "UL16_preVFP" , "UL17" ,  "UL18"]
    era_aliases["2022_preEE"] = ["2022_preEEB", "2022_preEEC", "2022_preEED"]
    era_aliases["2022_EE"] = ["2022_EEE", "2022_EEF", "2022_EEG"]
    era_aliases["2023_preBPix"] = ["2023_preBPixA", "2023_preBPixB", "2023_preBPixC", "2023_preBPixD", "2023_preBPixE", "2023_preBPixF"]
    era_aliases["2023_BPix"] = ["2023_BPixD",'2023_BPixE']

    era_aliases["2022_preEEMC"] = ["2022_preEE"]
    era_aliases["2023_preBPixMC"] = ["2023_preBPix"]
    era_aliases["2022_EEMC"] = ["2022_EE"]
    era_aliases["2023_BPixMC"] = ["2023_BPix"]
    era_aliases["Run3MC"] = era_aliases["2022_preEEMC"] + era_aliases["2022_EEMC"] + era_aliases["2023_preBPixMC"] + era_aliases["2023_BPixMC"]

    parser = argparse.ArgumentParser(description='uproot_plots')
    parser.add_argument('-i','--inputFile', default='hists.pkl', help='Input File. Default: hists.pkl')
    parser.add_argument('-p','--process',   default='data', help='Input process. Default: hists.pkl')
    parser.add_argument('-e','--era',   nargs='+', dest='eras', default=era_aliases["Run2"], help='Input process. Default: hists.pkl')
    #parser.add_argument('-d', '--datasets', nargs='+', dest='datasets', , help="Name of dataset to run. Example if more than one: -d HH4b ZZ4b")
    args = parser.parse_args()

    if "Run2" in args.eras:
        if args.process == "data":
            eras = era_aliases["Run2"]
        else:
            eras = era_aliases["Run2MC"]
    else:
        eras = []
        for e in args.eras:
            if e in era_aliases:
                eras += era_aliases[e]
            else:
                eras += args.eras

    eraString = "_".join(eras)
    print(eras)
    print(eraString)
    key = args.process+"_"+eraString

    if not args.inputFile.find(".yml") == -1:
        print("Have yml")
        in_file = yaml.safe_load(open(args.inputFile, 'r'))

        doTwoTag = True
        if not "cutFlowTwoTag" in in_file[key]:
            doTwoTag = False


        cf4 = {}
        cf4_unit = {}
        cf3 = {}
        cf3_unit = {}
        if doTwoTag:
            cf2 = {}
            cf2_unit = {}


        cf4_unit[key] = in_file[key]["cutFlowFourTagUnitWeight"]
        cf3_unit[key] = in_file[key]["cutFlowThreeTagUnitWeight"]
        if doTwoTag:
            cf2_unit[key] = in_file[key]["cutFlowTwoTagUnitWeight"]

        genEventSumw = in_file[key]["sumw"]
        lumi         = in_file[key]["lumi"][0]
        xs           = in_file[key]["xs"][0]
        kFactor      = in_file[key]["kFactor"][0]
        weighted_correction = (lumi * xs * kFactor / genEventSumw)

        cf4[key]      = in_file[key]["cutFlowFourTag"]
        for k, v in cf4[key].items():
            cf4[key][k] *= weighted_correction

        cf3[key]      = in_file[key]["cutFlowThreeTag"]
        for k, v in cf3[key].items():
            cf3[key][k] *= weighted_correction

        if doTwoTag:
            cf2[key]      = in_file[key]["cutFlowTwoTag"]
            for k, v in cf2[key].items():
                cf2[key][k] *= weighted_correction



    else:
        with open(f'{args.inputFile}', 'rb') as hfile:
            print(f"loading {args.inputFile}...")
            hists = load(hfile)
            #load_hists(args.inputFile)

        doTwoTag = True
        if not "cutFlowTwoTag" in hists:
            doTwoTag = False

        cf4      = hists["cutFlowFourTag"]
        cf4_unit = hists["cutFlowFourTagUnitWeight"]
        cf3      = hists["cutFlowThreeTag"]
        cf3_unit = hists["cutFlowThreeTagUnitWeight"]
        if doTwoTag:
            cf2      = hists["cutFlowTwoTag"]
            cf2_unit = hists["cutFlowTwoTagUnitWeight"]



    if key not in cf4:
        print(f"summing {key}...")

        cf4      [key] = {}
        cf4_unit [key] = {}
        cf3      [key] = {}
        cf3_unit [key] = {}

        if doTwoTag:
            cf2      [key] = {}
            cf2_unit [key] = {}

        for e in eras:

            for cut, v in cf4[args.process+"_"+e].items():
                if cut not in cf4[key]:      cf4[key][cut] = 0
                if cut not in cf4_unit[key]: cf4_unit[key][cut] = 0
                if cut not in cf3[key]:      cf3[key][cut] = 0
                if cut not in cf3_unit[key]: cf3_unit[key][cut] = 0

                if doTwoTag:
                    if cut not in cf2[key]:      cf2[key][cut] = 0
                    if cut not in cf2_unit[key]: cf2_unit[key][cut] = 0


                cf4[key][cut]      += cf4[args.process+"_"+e][cut]
                cf4_unit[key][cut] += cf4_unit[args.process+"_"+e][cut]
                cf3[key][cut]      += cf3[args.process+"_"+e][cut]
                cf3_unit[key][cut] += cf3_unit[args.process+"_"+e][cut]
                if doTwoTag:
                    cf2[key][cut]      += cf2[args.process+"_"+e][cut]
                    cf2_unit[key][cut] += cf2_unit[args.process+"_"+e][cut]


    if doTwoTag:
        printCF(key, cf4[key], cf4_unit[key], cf3[key], cf3_unit[key], cf2[key], cf2_unit[key])
    else:
        printCF(key, cf4[key], cf4_unit[key], cf3[key], cf3_unit[key])
