import argparse
try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    ROOT = None
    HAS_ROOT = False
try:
    import uproot
    HAS_UPROOT = True
except ImportError:
    uproot = None
    HAS_UPROOT = False

def print_counts_yaml(outputFile, channel, process, counts, mix=None):
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

    channels = args.channels
    procs = ["ttbar", "multijet", "data_obs", "signal"]
    mix_dir = args.mix_dir if args.mix_dir is not None else ["3bDvTMix4bDvT_v0", "3bDvTMix4bDvT_v14", "test_phaseE_v0", "test_phaseE_v1"]
    procs_mix = ["ttbar", "multijet", "data_obs"]

    with open(f'{args.outputFile}', 'w') as outputFile:
        if HAS_ROOT:
            inputFile = ROOT.TFile(f"{args.inputFile}","READ")
            for c in channels:
                for p in procs:
                    h = inputFile.Get(f"{c}/{p}")
                    if h:
                        print(f"{c}/{p}")
                        counts = [h.GetBinContent(ibin) for ibin in range(h.GetSize())]
                        print_counts_yaml(outputFile, c, p, counts)

            for mix in mix_dir:
                for c in channels:
                    for p in procs_mix:
                        h = inputFile.Get(f"{mix}/{c}/{p}")
                        if h:
                            print(f"{mix}/{c}/{p}")
                            counts = [h.GetBinContent(ibin) for ibin in range(h.GetSize())]
                            print_counts_yaml(outputFile, c, p, counts, mix=mix)
        elif HAS_UPROOT:
            with uproot.open(args.inputFile) as inputFile:
                for c in channels:
                    for p in procs:
                        key = f"{c}/{p}"
                        if key in inputFile:
                            print(key)
                            h = inputFile[key]
                            counts = list(h.values(flow=True))
                            print_counts_yaml(outputFile, c, p, counts)

                for mix in mix_dir:
                    for c in channels:
                        for p in procs_mix:
                            key = f"{mix}/{c}/{p}"
                            if key in inputFile:
                                print(key)
                                h = inputFile[key]
                                counts = list(h.values(flow=True))
                                print_counts_yaml(outputFile, c, p, counts, mix=mix)
        else:
            raise ImportError("Neither ROOT nor uproot is available.")
