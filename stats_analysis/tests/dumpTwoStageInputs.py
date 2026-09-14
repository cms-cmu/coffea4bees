import argparse
import ROOT




def print_counts_yaml(channel, process, hist, mix=None):

    if mix is None:
        outputFile.write(f"{'_'.join([channel,process])}:\n")
    else:
        outputFile.write(f"{'_'.join([mix,channel,process])}:\n")
    outputFile.write(f"    channel:\n")
    outputFile.write(f"        {channel}\n")
    outputFile.write(f"    process:\n")
    outputFile.write(f"        {process}\n")

    if not mix is None:
        outputFile.write(f"    mix:\n")
        outputFile.write(f"        {mix}\n")

    counts = []
    for ibin in range(hist.GetSize()):
        counts.append(hist.GetBinContent(ibin))

    outputFile.write(f"    counts:\n")
    outputFile.write(f"           {counts}\n")
    outputFile.write("\n\n")




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='uproot_plots')
    parser.add_argument('-i','--inputFile', default='../hists_closure_3bDvTMix4bDvT_New.root')
    parser.add_argument('-o','--outputFile', default='../hists_closure_3bDvTMix4bDvT_New.yml')
    parser.add_argument('-c', '--channels', nargs='+', default=['hh', 'ttHbb'], help='Channels to dump')
    parser.add_argument('-m', '--mix_dir', nargs='+', default=None, help='Mix directories to dump')
    args = parser.parse_args()

    outputFile = open(f'{args.outputFile}', 'w')
    inputFile = ROOT.TFile(f"{args.inputFile}","READ")

    channels = args.channels
    procs = ["ttbar", "multijet", "data_obs", "signal"]

    for c in channels:
        for p in procs:
            h = inputFile.Get(f"{c}/{p}")
            if h:
                print(f"{c}/{p}")
                print_counts_yaml(c, p, h)

    mix_dir = args.mix_dir if args.mix_dir is not None else ["3bDvTMix4bDvT_v0", "3bDvTMix4bDvT_v14", "test_phaseE_v0", "test_phaseE_v1"]
    procs_mix = ["ttbar", "multijet", "data_obs"]
    for mix in mix_dir:
        for c in channels:
            for p in procs_mix:
                h = inputFile.Get(f"{mix}/{c}/{p}")
                if h:
                    print(f"{mix}/{c}/{p}")
                    print_counts_yaml(c, p, h, mix=mix)

    outputFile.close()
