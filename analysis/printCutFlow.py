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
    print(f'\t{words[0]:<30}\t{words[1]:<10}   {words[2]:<10} \t\t {words[3]:<10}\t{words[4]:<10}')

def printCF(procKey, cf4, cf4_unit, cf3, cf3_unit):

    bar = "-"*10
    print('\n')
    print(procKey,':\n')
    printLine(["Cuts","FourTag","","ThreeTag",""])
    printLine(["",bar,bar,bar,bar])
    printLine(["","weighted","(unit weight)","weighted","(unit weight)"])
    print('\n')
    for cut in cf4.keys():
        printLine([cut,_round(cf4[cut]),_round(cf4_unit[cut]),_round(cf3[cut]),_round(cf3_unit[cut])])

    print("\n")


def add(thisKey):
    print(f"\tadding {thisKey}")


if __name__ == '__main__':

    run2_eras_data = ["UL16_postVFPF" , "UL16_postVFPG" , "UL16_postVFPH" , "UL16_preVFPB" , "UL16_preVFPC" , "UL16_preVFPD" , "UL16_preVFPE" , "UL17C" , "UL17D" , "UL17E" , "UL17F" , "UL18A" , "UL18B" , "UL18C" , "UL18D"]
    run2_eras_mc   = ["UL16_postVFP" , "UL16_preVFP" , "UL17" ,  "UL18"]

    parser = argparse.ArgumentParser(description='uproot_plots')
    parser.add_argument('-i','--inputFile', default='hists.pkl', help='Input File. Default: hists.pkl')
    parser.add_argument('-p','--process',   default='data', help='Input process. Default: hists.pkl')
    parser.add_argument('-e','--era',   nargs='+', dest='eras', default=run2_eras_data, help='Input process. Default: hists.pkl')
    #parser.add_argument('-d', '--datasets', nargs='+', dest='datasets', , help="Name of dataset to run. Example if more than one: -d HH4b ZZ4b")
    args = parser.parse_args()

    if "Run2" in args.eras:
        if args.process == "data":
            eras = run2_eras_data
        else:
            eras = run2_eras_mc
    else:
        eras = args.eras

    eraString = "_".join(eras)
    print(eras)
    print(eraString)
    key = args.process+"_"+eraString

    if not args.inputFile.find(".yml") == -1:
        print("Have yml")
        in_file = yaml.safe_load(open(args.inputFile, 'r'))

        cf4 = {}
        cf4_unit = {}
        cf3 = {}
        cf3_unit = {}

        cf4_unit[key] = in_file[key]["cutFlowFourTagUnitWeight"]

        cf3_unit[key] = in_file[key]["cutFlowThreeTagUnitWeight"]

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


    else:
        with open(f'{args.inputFile}', 'rb') as hfile:
            print(f"loading {args.inputFile}...")
            hists = load(hfile)
            #load_hists(args.inputFile)

        cf4      = hists["cutFlowFourTag"]
        cf4_unit = hists["cutFlowFourTagUnitWeight"]
        cf3      = hists["cutFlowThreeTag"]
        cf3_unit = hists["cutFlowThreeTagUnitWeight"]


    if key not in cf4:
        print(f"summing {key}...")

        cf4      [key] = {}
        cf4_unit [key] = {}
        cf3      [key] = {}
        cf3_unit [key] = {}

        for e in eras:

            for cut, v in cf4[args.process+"_"+e].items():
                if cut not in cf4[key]:      cf4[key][cut] = 0
                if cut not in cf4_unit[key]: cf4_unit[key][cut] = 0
                if cut not in cf3[key]:      cf3[key][cut] = 0
                if cut not in cf3_unit[key]: cf3_unit[key][cut] = 0

                cf4[key][cut]      += cf4[args.process+"_"+e][cut]
                cf4_unit[key][cut] += cf4_unit[args.process+"_"+e][cut]
                cf3[key][cut]      += cf3[args.process+"_"+e][cut]
                cf3_unit[key][cut] += cf3_unit[args.process+"_"+e][cut]

    #cutList = hists["cutFlowThreeTagUnitWeight"][key].keys()
    printCF(key, cf4[key], cf4_unit[key], cf3[key], cf3_unit[key])
