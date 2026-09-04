import os
import sys
import ROOT
import argparse
from runTwoStageClosure import mkpath, combine_hists, BE, pearsonr, COLORS, fTest, regionName
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.patches import Ellipse
import collections
sys.path.insert(0, os.getcwd())
import src.plotting.ROOTPlotTools as ROOTPlotTools

def print_log(string):
    print(string)
    log_file.write(string+"\n")



def addYears(f, input_file, channel):

    directory = f"{channel}"
    f.mkdir(directory)

    #
    # data_obs
    #
    var_name = args.var

    hist_data_obs = combine_hists(input_file,
                                  f"{var_name}_PROC_YEAR_fourTag_SR",
                                  years=["UL16_preVFP", "UL16_postVFP", "UL17", "UL18"],
                                  procs=[f"data"],
                                  debug=args.debug)

    f.cd(directory)
    hist_data_obs.SetName("data_obs")
    hist_data_obs.Write()

    #
    # multijet
    #
    hist_multijet = combine_hists(input_file,
                                  f"{var_name}_PROC_YEAR_threeTag_SR",
                                  years=["UL16_preVFP", "UL16_postVFP", "UL17", "UL18"],
                                  procs=[f"data"],
                                  debug=args.debug)                                  

    f.cd(directory)
    hist_multijet.SetName("multijet")
    hist_multijet.Write()

    #
    # TTBar
    #
    ttbar_procs = ["TTTo2L2Nu", "TTToHadronic", "TTToSemiLeptonic"]

    hist_ttbar = combine_hists(input_file,
                               f"{var_name}_PROC_YEAR_fourTag_SR",
                               years=["UL16_preVFP", "UL16_postVFP", "UL17", "UL18"],
                               procs=["TTTo2L2Nu", "TTToHadronic", "TTToSemiLeptonic"],
                               debug=args.debug)
                               

    f.cd(directory)
    hist_ttbar.SetName("ttbar")
    hist_ttbar.Write()

    return

class multijetEnsemble:
    def __init__(self, f, channel):

        self.channel = channel
        self.rebin = rebin

        self.output_yml = open(f'{args.outputPath}/{classifier}/rebin{self.rebin}/{args.region}/{self.channel}/0_variance_results.yml', 'w')

        self.data_minus_ttbar = f.Get(f'{self.channel}/ttbar')
        self.data_minus_ttbar.SetName(f'data_minus_ttbar_average_{self.channel}')
        self.data_minus_ttbar.Scale(-1)
        self.data_minus_ttbar.Add( f.Get(f'{self.channel}/data_obs') )
        self.data_minus_ttbar.Rebin(self.rebin)

        self.average = f.Get(f'{self.channel}/multijet')
        self.average.SetName('%s_average_%s' % (self.average.GetName(), self.channel))
        self.models  = [f.Get(f'{self.channel}/multijet').Clone()]

        self.nBins   = self.average.GetSize() - 2  # size includes under/overflow bins


        self.f = f
        self.f.cd(self.channel)

        self.average_rebin = self.average.Clone()
        self.average_rebin.SetName('%s_rebin' % self.average.GetName())
        self.average_rebin.Rebin(self.rebin)

        self.models_rebin = [model.Clone() for model in self.models]

        for model in self.models_rebin:
            model.SetName('%s_rebin' % model.GetName())
        for model in self.models_rebin:
            model.Rebin(self.rebin)
        self.nBins_rebin = self.average_rebin.GetSize() - 2

        self.f.cd(self.channel)
        self.nBins_ensemble = self.nBins_rebin 
        self.bin_width = 1. / self.nBins_rebin
        self.fit_bin_min = int(1 + closure_fit_x_min // self.bin_width)
        print(self.nBins_rebin)
        self.nBins_fit = self.nBins_rebin - int(closure_fit_x_min // self.bin_width)
        print(self.nBins_fit)
        self.multijet_ensemble_average  = ROOT.TH1F('multijet_ensemble_average', '', self.nBins_ensemble, 0.5, 0.5 + self.nBins_ensemble)
        self.multijet_ensemble          = ROOT.TH1F('multijet_ensemble'        , '', self.nBins_ensemble, 0.5, 0.5 + self.nBins_ensemble)
        self.data_minus_ttbar_ensemble  = ROOT.TH1F('data_minus_ttbar_ensemble', '', self.nBins_ensemble, 0.5, 0.5 + self.nBins_ensemble)

        for m in range(1):
            for b in range(self.nBins_rebin):
                local_bin    = 1 + b
                ensemble_bin = 1 + b + m * self.nBins_rebin
                # error = (self.models_rebin[m].GetBinError(local_bin)**2 + (self.average_rebin.GetBinError(local_bin)/nMixes)**2 + (2/nMixes)**2)**0.5
                error = (self.models_rebin[m].GetBinError(local_bin)**2 + (2 / 1)**2)**0.5
                self.multijet_ensemble_average.SetBinContent(ensemble_bin, self.average_rebin.GetBinContent(local_bin))
                self.multijet_ensemble_average.SetBinError  (ensemble_bin, error)

                self.multijet_ensemble        .SetBinContent(ensemble_bin, self.models_rebin[m].GetBinContent(local_bin))
                # self.multijet_ensemble        .SetBinError  (ensemble_bin, self.models_rebin[m].GetBinError  (local_bin))
                self.multijet_ensemble        .SetBinError  (ensemble_bin, 0.0)

                self.data_minus_ttbar_ensemble.SetBinContent(ensemble_bin, self.data_minus_ttbar.GetBinContent(local_bin))
                self.data_minus_ttbar_ensemble.SetBinError  (ensemble_bin, self.data_minus_ttbar.GetBinError  (local_bin))

        self.f.cd(self.channel)
        self.multijet_ensemble_average.Write()
        self.multijet_ensemble        .Write()
        self.data_minus_ttbar_ensemble.Write()

        self.bases = range(0, maxBasisEnsemble + 1, 1)

        #
        # Make kernel for basis orthogonalization
        #
        h = np.array([self.average_rebin.GetBinContent(bin) for bin in range(1, self.nBins_rebin + 1)])
        h_no_rebin = np.array([self.average.GetBinContent(bin) for bin in range(1, self.nBins + 1)])
        h_err = np.array([self.multijet_ensemble_average.GetBinError(bin) for bin in range(1, self.nBins_rebin + 1)])
        # h = np.array([self.average_rebin.GetBinError(bin)+2 for bin in range(1,self.nBins_rebin + 1)])
        self.h = h
        self.h_no_rebin = h_no_rebin
        # Make matrix of initial basis
        B_no_rebin = np.array([[b.Integral(self.average.GetBinLowEdge(bin), self.average.GetXaxis().GetBinUpEdge(bin)) / self.average.GetBinWidth(bin) for bin in range(1, self.nBins + 1)] for b in BE])
        B = np.array([[b.Integral(self.average_rebin.GetBinLowEdge(bin), self.average_rebin.GetXaxis().GetBinUpEdge(bin)) / self.average_rebin.GetBinWidth(bin) for bin in range(1, self.nBins_rebin + 1)] for b in BE])
        self.basis_element = B
        self.basis_element_no_rebin = B_no_rebin

        for basis in self.bases[1:]:
            self.plotBasis('initial', basis)
            self.plotBasis('initial', basis, rebin=False)

        # Subtract off cross correlation from higher order basis elements
        for i in range(1, len(B)):
            c = (B[i - 1] * h**1.0 * B[i - 1]).sum()
            B[i:] = B[i:] - (B[i - 1] * h**1.0 * B[i:]).sum(axis=1, keepdims=True) * B[i - 1] / c  # make each b_i orthogonal to those before it
            c_no_rebin = (B_no_rebin[i - 1] * h_no_rebin**1.0 * B_no_rebin[i - 1]).sum()
            B_no_rebin[i:] = B_no_rebin[i:] - (B_no_rebin[i - 1] * h_no_rebin**1.0 * B_no_rebin[i:]).sum(axis=1, keepdims=True) * B_no_rebin[i - 1] / c_no_rebin  # make each b_i orthogonal to those before it

        for i in range(0, len(B)):
            B[i] = B[i] * np.sign(B[i, -1])  # set all b_i's to be positive for the last bin
            B_no_rebin[i] = B_no_rebin[i] * np.sign(B_no_rebin[i, -1])  # set all b_i's to be positive for the last bin
            c = (B[i] * h**1.0 * B[i]).sum()

        for basis in self.bases[1:]:
            self.plotBasis('diagonalized', basis)
            self.plotBasis('diagonalized', basis, rebin=False)

        # scale dynamic range of each element to 1
        for i in range(1, len(B)):
            d = B[i].max() - B[i].min()
            B[i] = B[i] / d
            d = B_no_rebin[i].max() - B_no_rebin[i].min()
            B_no_rebin[i] = B_no_rebin[i] / d


        for basis in self.bases[1:]:
            self.plotBasis('normalized', basis)
            self.plotBasis('normalized', basis, rebin=False)

        self.fit_result = {}
        self.eigenVars = {}
        self.multijet_TF1, self.multijet_TH1 = {}, {}
        self.pvalue, self.chi2, self.ndf = {}, {}, {}
        self.pulls = {}
        self.pearsonr = {}
        self.ymax = {}
        self.fit_parameters, self.fit_parameters_error = {}, {}
        self.cUp, self.cDown = {}, {}
        # self.fProb = {}
        self.basis = None
        self.exit_message = ['--- None (%s) --- Multijet Ensemble' % self.channel.upper()]
        min_r = 1.0

        for basis in self.bases:

            self.makeFitFunction(basis)
            self.fit(basis)
            self.write_to_yml(basis)

            print(basis)
            self.plotFitResults(basis)
            for i in range(1, basis):
                self.plotFitResults(basis, projection=(i, i + 1))
            self.plotPulls(basis)

            # if abs(self.pearsonr[basis]['total'][0]) < min_r:
            if self.basis is None and abs(self.pearsonr[basis]['total'][1]) > probThreshold:
                min_r = abs(self.pearsonr[basis]['total'][0])
                self.basis = basis  # store first basis to satisfy min threshold. Will be used in closure fits
                self.exit_message = []
                self.exit_message.append('-' * 50)
                self.exit_message.append('%s channel' % self.channel.upper())
                self.exit_message.append('Satisfied adjacent bin de-correlation p-value for multijet ensemble variance at basis %d:' % self.basis)
                self.exit_message.append('>> p-value, r-value = %2.0f%%, %0.2f ' % (100 * self.pearsonr[self.basis]['total'][1], self.pearsonr[self.basis]['total'][0]))
                self.exit_message.append('-' * 50)

        if self.basis is None:
            self.basis = self.bases[ np.argmin([abs(self.pearsonr[basis]['total'][0]) for basis in self.bases]) ]
            self.exit_message = []
            self.exit_message.append('-' * 50)
            self.exit_message.append('%s channel' % self.channel.upper())
            self.exit_message.append('Minimized adjacent bin correlation abs(r) for multijet ensemble variance at basis %d:' % self.basis)
            self.exit_message.append('>> p-value, r-value = %2.0f%%, %0.2f ' % (100 * self.pearsonr[self.basis]['total'][1], self.pearsonr[self.basis]['total'][0]))
            self.exit_message.append('-' * 50)

        self.plotPearson()

    def print_exit_message(self):
        self.output_yml.close()
        for line in self.exit_message:
            print_log(line)

    def makeFitFunction(self, basis):

        def background_UserFunction(xArray, pars):
            ensemble_bin = int(xArray[0])
            m = (ensemble_bin - 1) // self.nBins_rebin
            local_bin = 1 + ((ensemble_bin - 1) % self.nBins_rebin)
            model = self.models[m]

            l, u = self.average_rebin.GetBinLowEdge(local_bin), self.average_rebin.GetXaxis().GetBinUpEdge(local_bin)

            p = 1.0
            for BE_idx in range(basis + 1):
                par_idx = m * (basis + 1) + BE_idx
                p += pars[par_idx] * self.basis_element[BE_idx][local_bin - 1]

            return p * self.multijet_ensemble.GetBinContent(ensemble_bin)

        self.f.cd(self.channel)
        self.pycallable = background_UserFunction
        self.multijet_TF1[basis] = ROOT.TF1 ('multijet_ensemble_TF1_basis%d' % basis, self.pycallable, 0.5, 0.5 + self.nBins_ensemble, 1 * (basis + 1))
        self.multijet_TH1[basis] = ROOT.TH1F('multijet_ensemble_TH1_basis%d' % basis, '', self.nBins_ensemble, 0.5, 0.5 + self.nBins_ensemble)

        for m in range(1):
            for o in range(basis + 1):
                self.multijet_TF1[basis].SetParName  (m * (basis + 1) + o, 'v%d c_%d' % (m, o))
                self.multijet_TF1[basis].SetParameter(m * (basis + 1) + o, 0.0)

    def getEigenvariations(self, basis=None, debug=False):
        if basis is None:
            basis = self.basis
        n = basis + 1

        if n == 1:
            self.eigenVars[basis] = [np.array([[self.multijet_TF1[basis].GetParError(m * n)]]) for m in range(1)]
            return

        cov = [ROOT.TMatrixD(n, n) for m in range(1)]
        cor = [ROOT.TMatrixD(n, n) for m in range(1)]

        for m in range(1):
            for i in range(n):
                for j in range(n):  # full fit is block diagonal in nMixes blocks since there is no correlation between fit parameters of different multijet models
                    cov[m][i][j] = self.fit_result[basis].CovMatrix  (m * n + i, m * n + j)
                    cor[m][i][j] = self.fit_result[basis].Correlation(m * n + i, m * n + j)

        if debug:
            for m in range(1):
                print('Covariance Matrix:', m)
                cov[m].Print()
                print('Correlation Matrix:', m)
                cor[m].Print()

        eigenVal = [ROOT.TVectorD(n) for m in range(1)]
        eigenVec = [cov[m].EigenVectors(eigenVal[m]) for m in range(1)]

        for m in range(1):
            # define relative sign of eigen-basis such that the first coordinate is always positive
            for j in range(n):
                if eigenVec[m][0][j] >= 0:
                    continue
                for i in range(n):
                    eigenVec[m][i][j] *= -1

            if debug:
                print('Eigenvectors (columns)', m)
                eigenVec[m].Print()
                print('Eigenvalues', m)
                eigenVal[m].Print()

        self.eigenVars[basis] = [np.zeros((n, n), dtype=float) for m in range(1)]
        for m in range(1):
            for i in range(n):
                for j in range(n):
                    self.eigenVars[basis][m][i, j] = eigenVec[m][i][j] * eigenVal[m][j]**0.5

        if debug:
            for m in range(1):
                print('Eigenvariations', m)
                for j in range(n):
                    print(j, self.eigenVars[basis][m][:, j])

    def getParameterDistribution(self, basis):
        n = basis + 1
        parMean    = np.array([0 for i in range(n)], dtype=float)
        parMeanErr = np.array([0 for i in range(n)], dtype=float)
        parMean2   = np.array([0 for i in range(n)], dtype=float)
        for m in range(1):
            parMean    += self.fit_parameters[basis][m]    / 1
            parMean2   += self.fit_parameters[basis][m]**2 / 1
            parMeanErr += self.fit_parameters_error[basis][m] / 1
        var = parMean2 - parMean**2
        parStd  = var**0.5
        #parStd *= 1 / (1 - 1)  # bessel's correction https://en.wikipedia.org/wiki/Bessel's_correction
        print('Parameter Mean:', parMean)
        print('Parameter  Std:', parStd)

        for i in range(n):
            # cUp   =  ( (abs(parMean[i])+parStd[i])**2 + parMeanErr[i]**2 )**0.5 # * n**0.5 # add scaling term so that 1 sigma corresponds to quadrature sum over i of (abs(parMean[i])+parStd[i])
            cUp   =  abs(parMean[i]) + parStd[i]

            if cUp < 1e-4:
                print("Normalizing errors")
                cUp += 0.01

            cDown = -cUp
            try:
                self.cUp  [basis].append( cUp )
                self.cDown[basis].append( cDown )
            except KeyError:
                self.cUp  [basis] = [cUp  ]
                self.cDown[basis] = [cDown]

    def fit(self, basis):
        # print(self.multijet_TF1[basis])
        # print(type(self.multijet_TF1[basis]))
        # self.multijet_ensemble_average.Fit(self.multijet_TF1[basis], 'N0SQ')
        self.fit_result[basis] = self.multijet_ensemble_average.Fit(self.multijet_TF1[basis], 'N0SQ')
        self.getEigenvariations(basis)
        self.pvalue[basis], self.chi2[basis], self.ndf[basis] = self.multijet_TF1[basis].GetProb(), self.multijet_TF1[basis].GetChisquare(), self.multijet_TF1[basis].GetNDF()
        print("=" * 50)
        print('Fit multijet ensemble %s at basis %d' % (self.channel, basis))
        print('chi2/ndf = %3.2f/%3d = %2.2f' % (self.chi2[basis], self.ndf[basis], self.chi2[basis] / self.ndf[basis]))
        print(' p-value = %0.2f' % self.pvalue[basis])

        self.ymax[basis] = self.multijet_TF1[basis].GetMaximum(1, self.nBins_ensemble)
        self.fit_parameters[basis], self.fit_parameters_error[basis] = [], []
        n = basis + 1

        for m in range(1):
            self.fit_parameters      [basis].append( np.array([self.multijet_TF1[basis].GetParameter(m * n + o) for o in range(basis + 1)]) )
            self.fit_parameters_error[basis].append( np.array([self.multijet_TF1[basis].GetParError (m * n + o) for o in range(basis + 1)]) )
        self.getParameterDistribution(basis)

        for _bin in range(1, self.nBins_ensemble + 1):
            self.multijet_TH1[basis].SetBinContent(_bin, self.multijet_TF1[basis].Eval(_bin))
            # self.multijet_TH1[basis].SetBinError  (_bin, self.multijet_ensemble.GetBinError(bin))
            self.multijet_TH1[basis].SetBinError  (_bin, 0.0)

        pulls = []
        bins = range(self.fit_bin_min, self.nBins_ensemble + 1)
        for _bin in bins:
            error = self.multijet_ensemble_average.GetBinError(_bin)
            pull = (self.multijet_TF1[basis].Eval(_bin) - self.multijet_ensemble_average.GetBinContent(_bin)) / error if error > 0 else 0
            pulls.append(pull)
        self.pulls[basis] = np.array(pulls)

        # check bin to bin correlations using pearson R test
        xs = np.array([self.pulls[basis][m * self.nBins_fit  : (m + 1) * self.nBins_fit - 1] for m in range(1)])
        ys = np.array([self.pulls[basis][m * self.nBins_fit + 1: (m + 1) * self.nBins_fit  ] for m in range(1)])
        # x1s = np.array([self.pulls[basis][m * self.nBins_fit  : (m + 1) * self.nBins_fit-1] for m in range(1)])
        # y1s = np.array([self.pulls[basis][m * self.nBins_fit + 1: (m + 1) * self.nBins_fit  ] for m in range(1)])
        # x2s = np.array([self.pulls[basis][m * self.nBins_fit  : (m + 1) * self.nBins_fit - 2] for m in range(1)])
        # y2s = np.array([self.pulls[basis][m * self.nBins_fit+2: (m + 1) * self.nBins_fit  ] for m in range(1)])
        # x3s = np.array([self.pulls[basis][m * self.nBins_fit  : (m + 1) * self.nBins_fit-3] for m in range(1)])
        # y3s = np.array([self.pulls[basis][m * self.nBins_fit+3: (m + 1) * self.nBins_fit  ] for m in range(1)])
        # xs, ys = np.concatenate((x1s,x2s,x3s), axis=1), np.concatenate((y1s,y2s,y3s), axis=1)
        x, y = xs.flatten(), ys.flatten()
        r, p = pearsonr(x, y, n=len(x) - 1 * (basis + 1))

        self.pearsonr[basis] = {'total': (r, p),
                                'mixes': [pearsonr(xs[m], ys[m], n=len(xs[m]) - basis - 1) for m in range(1)]}

        print('-------------------------')
        print('>> r, prob = %0.2f, %0.2f' % self.pearsonr[basis]['total'])
        print('-------------------------')
        # n = x.shape[0] - 1*(basis + 1)
        # dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)
        # p_manual = 2*dist.cdf(-abs(r))
        # print('manual R p-value: n, p = %d, %f' % (n,p_manual))
        # raw_input()
        self.f.cd(self.channel)
        self.multijet_TH1[basis].Write()

    def write_to_yml(self, basis):
        self.output_yml.write(str(basis) + ":\n")

        write_pairs = [("chi2", self.chi2[basis]), ("ndf", self.ndf[basis]), ("pvalue", self.pvalue[basis]),
                       ("pearson_r", self.pearsonr[basis]['total'][0]), ("pearson_pvalue", self.pearsonr[basis]['total'][0]),
                       ("variance", self.cUp[basis])]

        for wp in write_pairs:
            self.output_yml.write(" " * 4 + f"{wp[0]}:\n")
            self.output_yml.write(" " * 8 + f"{str(wp[1])}\n")

    def plotBasis(self, name, basis, rebin=True):
        fig, (ax) = plt.subplots(nrows=1)
        if rebin:
            x = [self.average_rebin.GetBinCenter(_bin) for _bin in range(1, self.nBins_rebin + 1)]
        else:
            x = [self.average.GetBinCenter(_bin) for _bin in range(1, self.nBins + 1)]
        xlim = [0, 1]
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_title('%s Multiplicitive Basis (%s)' % (name[0].upper() + name[1:], self.channel.upper()))

        ax.plot(xlim, [0, 0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        if rebin:
            for i, y in enumerate(self.basis_element[:basis + 1]):
                ax.plot(x, y, label='b$_{%i}$' % i, linewidth=1)
        else:
            for i, y in enumerate(self.basis_element_no_rebin[:basis + 1]):
                ax.plot(x, y, label='b$_{%i}$' % i, linewidth=1)

        # if name == 'normalized':
        #     ax.plot(x, self.basis_signal[basis]*10, label=r'Spurious Signal ($\times 10$)', linewidth=1)
        # else:
        #     ax.plot(x, self.basis_signal[basis],    label=r'Spurious Signal',               linewidth=1)

        ax.set_xlabel('P(Signal)')
        ax.set_ylabel('Multijet Scale')

        ax.legend(fontsize='small', loc='best')

        rebin_name = '' if rebin else '_no_rebin'
        if type(self.rebin) is list:
            figname = f'{args.outputPath}/{classifier}/variable_rebin/{args.region}/{self.channel}/{name}_basis{rebin_name}{basis}.pdf'
        else:
            figname = f'{args.outputPath}/{classifier}/rebin{self.rebin}/{args.region}/{self.channel}/{name}_basis{rebin_name}{basis}.pdf'

        # print('fig.savefig( '+figname+' )')
        plt.tight_layout()
        fig.savefig( figname )
        plt.close(fig)

        fig, (ax) = plt.subplots(nrows=1)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_xticks(np.arange(0, 1.1, 0.1))

        ax.set_title('%s Additive Basis (%s)' % (name[0].upper() + name[1:], self.channel.upper()))

        ax.plot(xlim, [0, 0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        if rebin:
            for i, y in enumerate(self.basis_element[:basis + 1]):
                ax.plot(x, y * self.h, label='b$_{%i}$' % i, linewidth=1)
        else:
            for i, y in enumerate(self.basis_element_no_rebin[:basis + 1]):
                ax.plot(x, y * self.h_no_rebin, label='b$_{%i}$' % i, linewidth=1)

        # if name == 'normalized':
        #     ax.plot(x, self.basis_signal[basis] * self.h*100, label=r'Spurious Signal ($\times 100$)', linewidth=1)
        # else:
        #     ax.plot(x, self.basis_signal[basis] * self.h,     label=r'Spurious Signal',                linewidth=1)

        ax.set_xlabel('P(Signal)')
        ax.set_ylabel('Events')

        ax.legend(fontsize='small', loc='best')

        if type(self.rebin) is list:
            figname = f'{args.outputPath}/{classifier}/variable_rebin/{args.region}/{self.channel}/{name}_additive_basis{rebin_name}{basis}.pdf'
        else:
            figname = f'{args.outputPath}/{classifier}/rebin{self.rebin}/{args.region}/{self.channel}/{name}_additive_basis{rebin_name}{basis}.pdf'
        # print('fig.savefig( '+figname+' )')
        plt.tight_layout()
        fig.savefig( figname )
        plt.close(fig)

    def plotPearson(self):
        fig, (ax) = plt.subplots(nrows=1)
        # ax.set_ylim(0.001,1)
        # plt.yscale('log')
        x = np.array(sorted(self.pearsonr.keys())) + 1
        ax.set_ylim(-1, 1)
        ax.set_xticks(x)
        xlim = [x[0] - 0.5, x[-1] + 0.5]
        ax.set_xlim(xlim[0], xlim[1])
        ax.plot(xlim, [0, 0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)

        r = np.array([self.pearsonr[o]['total'][0] for o in x - 1])
        p = np.array([self.pearsonr[o]['total'][1] for o in x - 1])
        ax.set_title('Multijet Model Variance Fits (%s)' % self.channel.upper())
        ax.plot(x, r, label='Combined', color='k', linewidth=2)
        ax.plot(x, p, label='p-value',  color='r', linewidth=2)

        if self.basis is not None:
            ax.plot([self.basis + 1, self.basis + 1], [-1, 1], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
            ax.scatter(self.basis + 1, p[x == (self.basis + 1)], color='k', marker='*', s=100, zorder=10)

        for m in range(1):
            r = [self.pearsonr[o]['mixes'][m][0] for o in x - 1]
            # p = [self.pearsonr[o]['mixes'][m][1] for o in x]
            label = 'v$_{%d}$' % m
            ax.plot(x, r, color=COLORS[m], linewidth=1, alpha=0.5, label=label)  # underscore tells pyplot to not show this in the legend
            # ax.plot(x, r, color=colors[m], linewidth=1, alpha=0.3, linestyle='dotted', label='_'+label)#underscore tells pyplot to not show this in the legend
            # ax.plot(x, p, color=colors[m], linewidth=1, alpha=0.3, label=label)

        ax.plot(xlim, [probThreshold, probThreshold], color='r', alpha=0.5, linestyle='--', linewidth=0.5)

        ax.set_xlabel('Parameters')
        ax.set_ylabel('Adjacent Bin Pearson Correlation (r) and p-value')
        plt.legend(fontsize='small', loc='best')

        if type(self.rebin) is list:
            name = f'{args.outputPath}/{classifier}/variable_rebin/{args.region}/{self.channel}/0_variance_pearsonr_multijet_variance.pdf'
        else:
            name = f'{args.outputPath}/{classifier}/rebin{self.rebin}/{args.region}/{self.channel}/0_variance_pearsonr_multijet_variance.pdf'
        # print('fig.savefig( ' + name+' )')
        plt.tight_layout()
        fig.savefig( name )
        plt.close(fig)

    def plotFitResults(self, basis, projection=(0, 1)):
        n = basis + 1
        if n > 1:
            dims = tuple(list(projection) + [d for d in range(n) if d not in projection])
        else:
            dims = (0, 1)

        # plot fit parameters
        x, y, s, c = [], [], [], []
        for m in range(1):
            x.append( 100 * self.fit_parameters[basis][m][dims[0]] )
            if n == 1:
                y.append( 0 )
            if n > 1:
                y.append( 100 * self.fit_parameters[basis][m][dims[1]] )
            if n > 2:
                c.append( 100 * self.fit_parameters[basis][m][dims[2]] )
            if n > 3:
                s.append( 100 * self.fit_parameters[basis][m][dims[3]] )

        x = np.array(x)
        y = np.array(y)

        kwargs = {'lw': 0.5,
                  'marker': 'o',
                  'edgecolors': 'k',
                  's': 8,
                  'c': 'k',
                  'zorder': 2,
                  }
        if n > 2:
            kwargs['c'] = c
            kwargs['cmap'] = 'BuPu'
        if n > 3:
            s = np.array(s)
            smin = s.min()
            smax = s.max()
            srange = smax - smin

            if s.max():
                s = s - s.min()  # shift so that min is at zero
                s = s / s.max()  # scale so that max is 1
            s = (s + 5.0 / 25) * 25  # shift and scale so that min is 5.0 and max is 25+5.0
            print(f"s is {s}")

            kwargs['s'] = s

        fig, (ax) = plt.subplots(nrows=1, figsize=(7, 6)) if n > 2 else plt.subplots(nrows=1, figsize=(6, 6))
        ax.set_aspect(1)
        ax.set_title('Multijet Model Variance Fits (%s)' % self.channel.upper())
        ax.set_xlabel('c$_' + str(dims[0]) + '$ (\%)')
        ax.set_ylabel('c$_' + str(dims[1]) + '$ (\%)')

        xlim, ylim = [-8, 8], [-8, 8]
        ax.plot(xlim, [0, 0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot([0, 0], ylim, color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        xticks = np.arange(-6, 8, 2)
        yticks = np.arange(-6, 8, 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        if n > 1:
            # draw 1\sigma ellipse
            ellipse = Ellipse((0, 0),
                              width =100 * (self.cUp[basis][dims[0]] - self.cDown[basis][dims[0]]),
                              height=100 * (self.cUp[basis][dims[1]] - self.cDown[basis][dims[1]]),
                              facecolor = 'none',
                              edgecolor = 'b',  # CMURED,
                              linestyle = '-',
                              linewidth = 0.75,
                              zorder=1,
                              )
            ax.add_patch(ellipse)

        bbox = dict(boxstyle='round', facecolor='w', alpha=0.8, linewidth=0, pad=0)
        if n > 2:
            # draw range bars for other priors
            for i, d in enumerate(dims[2:]):
                thisx = xlim[-1] - 0.5 * (n - 2) + 0.5 * i
                up, down = self.cUp[basis][d], self.cDown[basis][d]
                ax.quiver(thisx, 0, 0, 100 * up,   color='b', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
                ax.quiver(thisx, 0, 0, 100 * down, color='b', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

                ax.annotate('c$_{%d}$' % (d), [thisx, 100 * down - 0.5], ha='center', va='center', bbox=bbox)

        maxr = np.zeros((2, len(x)), dtype=float)
        minr = np.zeros((2, len(x)), dtype=float)
        if n > 1:
            # generate a ton of random points on a hypersphere in dim=n so surface is dim=n - 1.
            points  = np.random.randn(n, min(100 * (n - 1), 10**7))  # random points in a hypercube
            points /= np.linalg.norm(points, axis=0)  # normalize them to the hypersphere surface

            # for each model, find the point which maximizes the change in c_0**2 + c_1**2
            for m in range(1):
                plane = np.matmul( self.eigenVars[basis][m][dims[:2], :], points )
                r2 = plane[0]**2
                if n > 1:
                    r2 += plane[1]**2

                maxr[:, m] = plane[:, r2 == r2.max()].T[0]

                # construct orthogonal unit vector to maxr
                minrvec = np.copy(maxr[::-1, m])
                minrvec[0] *= -1
                minrvec /= np.linalg.norm(minrvec)

                # find maxr along minrvec to get minr
                dr2 = np.matmul( minrvec, plane )**2
                # minr[:,m] = plane[:,dr2 == dr2.max()].T[0]#this guy is the ~right length but might be slightly off orthogonal
                minr[:, m] = minrvec * dr2.max()**0.5  # this guy is the ~right length and is orthogonal by construction
        else:
            for m in range(1):
                maxr[0, m] = self.eigenVars[basis][m][dims[0]]

        minr *= 100
        maxr *= 100

        # print(maxr)
        # print(minr)
        ax.quiver(x, y,  maxr[0],  maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -maxr[0], -maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

        ax.quiver(x, y,  minr[0],  minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -minr[0], -minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

        plt.scatter(x, y, **kwargs)
        plt.tight_layout()

        for m in range(1):
            x_offset, y_offset = (maxr[0, m] + minr[0, m]) / 2, (maxr[1, m] + minr[1, m]) / 2
            ax.annotate('v$_{%d}$' % m, (x[m] + x_offset, y[m] + y_offset), bbox=bbox)

        if n > 2:
            plt.colorbar(label='c$_' + str(dims[2]) + '$ (\%)')  # , cax=cax)
            plt.subplots_adjust(right=1)

        if n > 3:
            l1 = plt.scatter([], [], s=(0.0 / 3 + 10.0 / 30) * 30, lw=1, edgecolors='black', facecolors='none')
            l2 = plt.scatter([], [], s=(1.0 / 3 + 10.0 / 30) * 30, lw=1, edgecolors='black', facecolors='none')
            l3 = plt.scatter([], [], s=(2.0 / 3 + 10.0 / 30) * 30, lw=1, edgecolors='black', facecolors='none')
            l4 = plt.scatter([], [], s=(3.0 / 3 + 10.0 / 30) * 30, lw=1, edgecolors='black', facecolors='none')

            handles = [l1,
                       l2,
                       l3,
                       l4]

            labels = ['%0.2f' % smin,
                      '%0.2f' % (smin + srange * 1.0 / 3),
                      '%0.2f' % (smin + srange * 2.0 / 3),
                      '%0.2f' % smax]

            leg = plt.legend(handles, labels,
                             ncol=1,
                             fontsize='medium',
                             loc='best',
                             title='c$_' + str(dims[3]) + '$ (\%)',
                             scatterpoints=1)

        projection = '_'.join([str(d) for d in projection])
        if type(self.rebin) is list:
            name = f'{args.outputPath}/{classifier}/variable_rebin/{args.region}/{self.channel}/0_variance_parameters_basis{basis}_projection_{projection}.pdf'
        else:
            name = f'{args.outputPath}/{classifier}/rebin{self.rebin}/{args.region}/{self.channel}/0_variance_parameters_basis{basis}_projection_{projection}.pdf'
        # print('fig.savefig( ' + name+' )')
        try:
            print(name)
            fig.savefig( name )
            plt.close(fig)
        except IndexError:
            print('Weird index error...')

    def plotPulls(self, basis):
        n = basis + 1

        xs = np.array([self.pulls[basis][m * self.nBins_fit    :(m + 1) * self.nBins_fit - 1] for m in range(1)])
        ys = np.array([self.pulls[basis][m * self.nBins_fit + 1:(m + 1) * self.nBins_fit    ] for m in range(1)])
        # x1s = np.array([self.pulls[basis][m * self.nBins_fit  :(m + 1) * self.nBins_fit-1] for m in range(1)])
        # y1s = np.array([self.pulls[basis][m * self.nBins_fit + 1:(m + 1) * self.nBins_fit  ] for m in range(1)])
        # x2s = np.array([self.pulls[basis][m * self.nBins_fit  :(m + 1) * self.nBins_fit - 2] for m in range(1)])
        # y2s = np.array([self.pulls[basis][m * self.nBins_fit+2:(m + 1) * self.nBins_fit  ] for m in range(1)])
        # x3s = np.array([self.pulls[basis][m * self.nBins_fit  :(m + 1) * self.nBins_fit-3] for m in range(1)])
        # y3s = np.array([self.pulls[basis][m * self.nBins_fit+3:(m + 1) * self.nBins_fit  ] for m in range(1)])
        # xs, ys = np.concatenate((x1s,x2s,x3s), axis=1), np.concatenate((y1s,y2s,y3s), axis=1)

        kwargs = {'lw': 0.5,
                  'marker': 'o',
                  'edgecolors': 'k',
                  's': 8,
                  'c': 'k',
                  'zorder': 2,
                  }

        fig, (ax) = plt.subplots(nrows=1, figsize=(6, 6))
        ax.set_aspect(1)
        ax.set_title('Adjacent Bin Pulls (%s, %d parameters)' % (self.channel.upper(), basis + 1))
        ax.set_xlabel('Bin$_{i}$, Pull')
        ax.set_ylabel('Bin$_{i + 1}$ Pull')
        # ax.set_xlabel('Bin$_{2i}$, Pull')
        # ax.set_ylabel('Bin$_{2i + 1}$ Pull')

        # xlim, ylim = list(ax.get_xlim()), list(ax.get_ylim())
        # lim_max = max(int(max([abs(lim) for lim in xlim+ylim])), 1)
        lim_max = 1.5 * max(abs(xs).max(), abs(ys).max())
        xlim, ylim = [-lim_max, lim_max], [-lim_max, lim_max]
        # xlim, ylim = [-5,5], [-5,5]
        ax.plot(xlim, [0, 0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot([0, 0], ylim, color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # xticks = np.arange(-int(lim_max) + 1, int(lim_max), 1)
        # yticks = np.arange(-int(lim_max) + 1, int(lim_max), 1)
        # ax.set_xticks(xticks)
        # ax.set_yticks(yticks)

        for m in range(1):
            # r, p = scipy.stats.pearsonr(xs[m], ys[m])
            (r, p) = self.pearsonr[basis]['mixes'][m]
            kwargs['label'] = 'v$_{%d}$, r=%0.2f (%2.0f%s)' % (m, r, p * 100, '\%')
            kwargs['c'] = COLORS[m]
            plt.scatter(xs[m], ys[m], **kwargs)
        plt.tight_layout()

        # x, y = xs.flatten(), ys.flatten()
        # r, p = scipy.stats.pearsonr(x, y)
        (r, p) = self.pearsonr[basis]['total']

        plt.legend(fontsize='small', loc='upper left', ncol=2, title='Overall r=%0.2f (%2.0f%s)' % (r, p * 100, '\%'))

        if type(self.rebin) is list:
            name = f'{args.outputPath}/{classifier}/variable_rebin/{args.region}/{self.channel}/0_variance_pull_correlation_basis{basis}.pdf'
        else:
            name = f'{args.outputPath}/{classifier}/rebin{self.rebin}/{args.region}/{self.channel}/0_variance_pull_correlation_basis{basis}.pdf'
        # print('fig.savefig( ' + name+' )')
        fig.savefig( name )
        plt.close(fig)

    def plotFit(self, basis):
        samples = collections.OrderedDict()
        samples[bias_file_out] = collections.OrderedDict()
        # samples[bias_file_out]['%s/data_minus_ttbar_ensemble' % self.channel] = {
        #     'label' : '#LTMixed Data#GT - #lower[0.10]{t#bar{t}}',
        #     'legend': 1,
        #     'ratioDrawOptions' : 'P ex0',
        #     'isData' : True,
        #     'ratio' : 'numer A',
        #     'color' : 'ROOT.kBlack'}
        samples[bias_file_out]['%s/multijet_ensemble_average' % self.channel] = {
            'label' : '#LTMultijet Model#GT',
            'legend': 2,
            'isData' : True,
            'ratio' : 'denom A',
            'color' : 'ROOT.kBlack'}
        samples[bias_file_out]['%s/multijet_ensemble' % self.channel] = {
            'label' : 'Multijet Models',
            'legend': 3,
            'stack' : 1,
            'ratio' : 'numer A',
            'color' : 'ROOT.kYellow'}
        samples[bias_file_out]['%s/multijet_ensemble_TH1_basis%d' % (self.channel, basis)] = {
            'label' : 'Fit (%d parameter%s)' % (basis + 1, 's' if basis else ''),
            'legend': 4,
            'ratio' : 'numer A',
            'color' : 'ROOT.kBlue'}

        xTitle = 'P(Signal) #(Bin) + #(Bins)#(Mix) #cbar P(%s) is largest' % (self.channel.upper())

        parameters = {'titleLeft'   : '#bf{CMS} Internal',
                      'titleCenter' : regionName[args.region],
                      'titleRight'  : 'Pass #DeltaR(j,j)',
                      'maxDigits'   : 4,
                      'drawLines'   : [[self.nBins_rebin * m + 0.5,  0, self.nBins_rebin * m + 0.5, self.ymax[0] * 1.1] for m in range(1, 1 + 1)],
                      'ratioErrors': False,
                      'ratio'     : 'significance',  # True,
                      'rMin'      : -3,  # 0.9,
                      'rMax'      :  3,  # 1.1,
                      'rTitle'    : 'Pulls',  # 'Data / Bkgd.',
                      # 'ratioErrors': True,
                      # 'ratio'      : True,
                      # 'rMin'       : 0.9,
                      # 'rMax'       : 1.1,
                      # 'rTitle'     : 'Model / Average',
                      'xTitle'    : xTitle,
                      'yTitle'    : 'Events',
                      'yMax'      : self.ymax[0] * 1.6,  # *ymaxScale, # make room to show fit parameters
                      'xleg'      : [0.13, 0.13 + 0.4],
                      'legendSubText' : ['#bf{Adjacent Bin Pull Correlation:}',
                                         'r = %1.2f' % (self.pearsonr[basis]['total'][0]),
                                         'p-value = %2.0f%%' % (self.pearsonr[basis]['total'][1] * 100),],
                      'lstLocation' : 'right',
                      'rPadFraction': 0.5,
                      'outputName': '0_variance_multijet_ensemble_basis%d' % (basis)}

        parameters['ratioLines'] = [[self.nBins_rebin * m + 0.5, parameters['rMin'], self.nBins_rebin * m + 0.5, parameters['rMax']] for m in range(1, 1 + 1)]

        if type(self.rebin) is list:
            parameters['outputDir'] = f'{args.outputPath}/{classifier}/variable_rebin/{args.region}/{self.channel}/'
        else:
            parameters['outputDir'] = f'{args.outputPath}/{classifier}/rebin{self.rebin}/{args.region}/{self.channel}/'

        # print('make ',parameters['outputDir']+parameters['outputName']+'.pdf')
        ROOTPlotTools.plot(samples, parameters, debug=False)



class closure:
    def __init__(self, f, channel, multijet):
        self.channel = channel
        self.rebin = rebin
        self.multijet = multijet
        self.ttbar = f.Get('%s/ttbar' % self.channel)
        self.ttbar.SetName('%s_average_%s' % (self.ttbar.GetName(), self.channel))
        self.data_obs = f.Get(f'{channel}/data_obs')
        self.data_obs.SetName('%s_average_%s' % (self.data_obs.GetName(), self.channel))
        self.nBins = self.data_obs.GetSize() - 2  # GetSize includes under/overflow bins

        self.output_yml = open(f'{args.outputPath}/{classifier}/rebin{self.rebin}/{args.region}/{self.channel}/1_bias_results.yml', 'w')


        self.f = f
        self.f.cd(self.channel)

        self.ttbar_rebin = self.ttbar.Clone()
        self.ttbar_rebin.SetName('%s_rebin' % self.ttbar.GetName())
        self.ttbar_rebin.Rebin(self.rebin)
        self.data_obs_rebin = self.data_obs.Clone()
        self.data_obs_rebin.SetName('%s_rebin' % self.data_obs.GetName())
        self.data_obs_rebin.Rebin(self.rebin)
        self.nBins_rebin = self.data_obs_rebin.GetSize() - 2

        self.bin_width = 1. / self.nBins_rebin
        self.fit_x_min = 0.5 + closure_fit_x_min / self.bin_width

        self.basis_element = self.multijet.basis_element
        self.basis_element_no_rebin = self.multijet.basis_element_no_rebin

        self.f.cd(self.channel)
        # self.bases = range(self.multijet.basis, maxBasisClosure + 1, 2)
        self.bases = range(-1, maxBasisClosure + 1)
        max_basis = max(self.bases[-1], self.multijet.basis)
        self.nBins_closure = self.nBins_rebin + max_basis + 1  # add bins for multijet shape priors
        self.multijet_closure = ROOT.TH1F('multijet_closure', '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)
        self.ttbar_closure    = ROOT.TH1F('ttbar_closure',    '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)
        self.data_obs_closure = ROOT.TH1F('data_obs_closure', '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)

        for _bin in range(1, self.nBins_rebin + 1):
            self.multijet_closure.SetBinContent(_bin, self.multijet.average_rebin.GetBinContent(_bin))
            self.ttbar_closure   .SetBinContent(_bin, self.ttbar_rebin           .GetBinContent(_bin))
            self.data_obs_closure.SetBinContent(_bin, self.data_obs_rebin        .GetBinContent(_bin))

            self.multijet_closure.SetBinError  (_bin, 0.0)
            self.ttbar_closure   .SetBinError  (_bin, 0.0)
            # self.signal_closure  .SetBinError  (_bin, 0.0)
            # self.multijet_closure.SetBinError  (_bin, self.multijet.average_rebin.GetBinError(_bin))
            # self.ttbar_closure   .SetBinError  (_bin, self.ttbar_rebin           .GetBinError(_bin))

            error = (self.data_obs_rebin.GetBinError(_bin)**2 + self.ttbar_rebin.GetBinError(_bin)**2 + self.multijet.average_rebin.GetBinError(_bin)**2 + (2.0 / 1)**2)**0.5  # adding 2 in quadrature improves gaussian approx of poisson errors
            self.data_obs_closure.SetBinError  (_bin, error)

        for _bin in range(self.nBins_rebin + 1, self.nBins_closure + 1):
            self.data_obs_closure.SetBinError  (_bin, 1.0)

        self.f.cd(self.channel)
        self.multijet_closure.Write()
        self.ttbar_closure   .Write()
        self.data_obs_closure.Write()

        self.fit_result = {}
        self.eigenVars = {}
        self.closure_TF1, self.closure_TH1 = {}, {}
        self.pvalue, self.chi2, self.ndf = {}, {}, {}
        self.ymax = {}
        self.fit_parameters, self.fit_parameters_error = {}, {}
        self.cUp, self.cDown = {}, {}
        self.fProb = {-1: np.nan}
        self.basis = None
        self.exit_message = ['--- NONE (%s) ---' % self.channel.upper()]

        for basis in self.bases:
            self.makeFitFunction(basis)
            self.fit(basis)
            self.write_to_yml(basis)

            # self.writeClosureResults(basis)
            self.plotFitResults(basis)
            max_basis = max(basis, self.multijet.basis)
            for j in range(1, max_basis):
                self.plotFitResults(basis, projection=(j, j + 1))


        for i, basis in enumerate(self.bases[:-1]):
            next_basis = self.bases[i + 1]
            print('fit f-test basis', next_basis)
            # self.fProb[next_basis] = 0.5
            self.fProb[next_basis] = fTest(self.chi2[basis], self.chi2[next_basis], self.ndf[basis], self.ndf[next_basis])

            if self.basis is None and (self.pvalue[basis] > probThreshold) and (self.fProb[next_basis] < 0.95):
                self.exit_message = []
                print(self.pvalue)
                print(self.fProb)
                self.basis = basis  # store first basis to satisfy min threshold. Will be used in closure fits
                self.exit_message.append('-' * 50)
                self.exit_message.append('%s channel' % self.channel.upper())
                self.exit_message.append('Satisfied goodness of fit and f-test')
                self.exit_message.append('>> %d, %d basis elements (variance, bias)' % (self.multijet.basis, self.basis))
                self.exit_message.append('>> p-value, f-test = %2.0f%%, %2.0f%% with %d basis elements (p-value above threshold and f-test prefers this fit over previous)' % (100 * self.pvalue[basis], 100 * self.fProb[basis], basis))
                self.exit_message.append('>> p-value, f-test = %2.0f%%, %2.0f%% with %d basis elements (f-test does not prefer this over previous fit at greater than 95%%)' % (100 * self.pvalue[next_basis], 100 * self.fProb[next_basis], next_basis))
                self.exit_message.append('-' * 50)

        self.plotPValues()
        if self.basis is None:
            self.basis = self.bases[-1]

        self.writeClosureResults(self.basis)

    def write_to_yml(self, basis):
        self.output_yml.write(str(basis) + ":\n")

        n = max(self.multijet.basis, basis) + 1
        nConstrained = max(self.multijet.basis - basis, 0)
        nUnconstrained = n - nConstrained

        write_pairs = [("chi2", self.chi2[basis]), ("ndf", self.ndf[basis]), ("pvalue", self.pvalue[basis]),
                       ("nConstrained", nConstrained), ("nUnconstrained", nUnconstrained), ("expected_ndfs", self.nBins_rebin - nUnconstrained),
                       ("variance", self.cUp[basis])]

        for wp in write_pairs:
            self.output_yml.write(" " * 4 + f"{wp[0]}:\n")
            if type(wp[1]) is dict:
                self.output_yml.write(" " * 8 + f"{list(wp[1].values())}\n")
            else:
                self.output_yml.write(" " * 8 + f"{str(wp[1])}\n")

    def makeFitFunction(self, basis):

        max_basis = max(basis, self.multijet.basis)

        def background_UserFunction(xArray, pars):
            this_bin = int(xArray[0])

            if this_bin > self.nBins_rebin:
                BE_idx = this_bin - self.nBins_rebin - 1
                
                BE_idx += basis + 1  # only apply priors to higher order terms
                if BE_idx > self.multijet.basis:
                    return 0.0

                # use variance priors
                BE_coefficient = pars[BE_idx]
                if BE_coefficient > 0:
                    # return -BE_coefficient/(0.2581988897471611*abs(self.multijet.cUp  [self.multijet.basis][BE_idx]))
                    return -BE_coefficient / abs(self.multijet.cUp  [self.multijet.basis][BE_idx])
                else:
                    # return -BE_coefficient/(0.2581988897471611*abs(self.multijet.cDown[self.multijet.basis][BE_idx]))
                    return -BE_coefficient / abs(self.multijet.cDown[self.multijet.basis][BE_idx])

            # in distribution: evaluate basis elements times multijet
            p = 1.0
            n = max_basis + 1
            for BE_idx in range(n):
                p += pars[BE_idx] * self.basis_element[BE_idx][this_bin - 1]

            mj = self.multijet.average_rebin.GetBinContent(this_bin)
            background = p * mj + self.ttbar_rebin.GetBinContent(this_bin)
            # spuriousSignal = pars[n] * mj * self.multijet.basis_signal[max_basis][bin - 1]

            return background 

        self.f.cd(self.channel)
        n = max_basis + 1
        self.pycallable = background_UserFunction
        self.closure_TF1[basis] = ROOT.TF1 ('closure_TF1_basis%d' % basis, self.pycallable, 0.5, 0.5 + self.nBins_closure, n + 1)  # +1 for spurious signal
        self.closure_TH1[basis] = ROOT.TH1F('closure_TH1_basis%d' % basis,  '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)

        # for o in range(max(basis, self.multijet.basis)+1):
        for b in range(n):
            self.closure_TF1[basis].SetParName  (b, 'c_%d' % b)
            self.closure_TF1[basis].SetParameter(b, 0.0)
        self.closure_TF1[basis].SetParName  (n, 'spurious signal')
        self.closure_TF1[basis].FixParameter(n, 0)

    def getEigenvariations(self, basis, debug=False):
        n = max(self.multijet.basis, basis) + 1


        if n == 1:
            self.eigenVars[basis] = np.array([self.closure_TF1[basis].GetParError(0)])
            return

        cov = ROOT.TMatrixD(n, n)
        cor = ROOT.TMatrixD(n, n)
        fit_result = self.fit_result[basis]
        for i in range(n):
            for j in range(n):
                cov[i][j] = fit_result.CovMatrix  (i, j)
                cor[i][j] = fit_result.Correlation(i, j)

        if debug:
            print('Covariance Matrix:')
            cov.Print()
            print('Correlation Matrix:')
            cor.Print()

        eigenVal = ROOT.TVectorD(n)
        eigenVec = cov.EigenVectors(eigenVal)

        # define relative sign of eigen-basis such that the first coordinate is always positive
        for j in range(n):
            if eigenVec[0][j] >= 0:
                continue
            for i in range(n):
                eigenVec[i][j] *= -1

        if debug:
            print('Eigenvectors (columns)')
            eigenVec.Print()
            print('Eigenvalues')
            eigenVal.Print()

        eigenVars = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                eigenVars[i, j] = eigenVec[i][j] * eigenVal[j]**0.5

        self.eigenVars[basis] = eigenVars

        if debug:
            print('Eigenvariations')
            for j in range(n):
                print(j, self.eigenVars[basis][:, j])

    def getParameterDistribution(self, basis):

        n = max(self.multijet.basis, basis) + 1
        self.cUp[basis], self.cDown[basis] = {}, {}
        for i in range(n):
            if i <= self.multijet.basis:  # use variance and bias
                cUp = (self.multijet.cUp[self.multijet.basis][i]**2 + self.fit_parameters[basis][i]**2 + self.fit_parameters_error[basis][i]**2)**0.5
            else:  # only have bias
                cUp = (self.fit_parameters[basis][i]**2 + self.fit_parameters_error[basis][i]**2)**0.5
            cDown = -cUp
            self.cUp  [basis][i] = cUp
            self.cDown[basis][i] = cDown

    def fit(self, basis):
        n = max(self.multijet.basis, basis) + 1
        nConstrained = max(self.multijet.basis - basis, 0)
        nUnconstrained = n - nConstrained
        fit_x_max = self.nBins_rebin + 0.5 + nConstrained

        print("=" * 50)
        self.fit_result[basis] = self.data_obs_closure.Fit(self.closure_TF1[basis], 'N0S', '', self.fit_x_min, fit_x_max)
        self.getEigenvariations(basis)
        self.pvalue[basis], self.chi2[basis], self.ndf[basis] = self.closure_TF1[basis].GetProb(), self.closure_TF1[basis].GetChisquare(), self.closure_TF1[basis].GetNDF()
        print('Fit closure %s with %d basis elements. x_range = (%f, %f)' % (self.channel, basis, self.fit_x_min, fit_x_max))
        print('chi2/ndf = %3.2f/%3d = %2.2f' % (self.chi2[basis], self.ndf[basis], self.chi2[basis] / self.ndf[basis]))
        print(' p-value = %0.2f' % self.pvalue[basis])
        print('nConstrained', nConstrained)
        print('nUnonstrained', nUnconstrained)
        print('expected ndfs = ', self.nBins_rebin - nUnconstrained)

        self.ymax[basis] = self.closure_TF1[basis].GetMaximum(1, self.nBins_closure)
        # self.ymax[basis] = max(self.closure_TF1[basis].GetMaximum(1,self.nBins_closure), 100 * self.signal.GetMaximum())
        self.fit_parameters[basis], self.fit_parameters_error[basis] = [], []
        self.fit_parameters      [basis] = np.array([self.closure_TF1[basis].GetParameter(b) for b in range(n)])
        self.fit_parameters_error[basis] = np.array([self.closure_TF1[basis].GetParError (b) for b in range(n)])
        self.getParameterDistribution(basis)

        for _bin in range(1, self.nBins_closure + 1):
            self.closure_TH1[basis].SetBinContent(_bin, self.closure_TF1[basis].Eval(_bin))
            # self.closure_TH1[basis].SetBinError  (_bin, self.data_obs_closure.GetBinError(_bin))
            self.closure_TH1[basis].SetBinError  (_bin, 0.0)

        self.f.cd(self.channel)
        self.closure_TH1[basis].Write()


    def writeClosureResults(self, basis=None):
        systematics = {}

        if basis is None:
            basis = self.basis

        max_basis = max(self.multijet.basis, basis)
        nBEs = max_basis + 1
        # closureResults = 'ZZ4b/nTupleAnalysis/combine/closureResults_%s_%s.pkl' % (classifier,self.channel)
        # closureResults = 'closureResults_%s_%s.pkl' % (classifier,self.channel)
        # closureResultsRoot = ROOT.TFile(closureResults.replace('.txt', '.root'), 'RECREATE')
        # closureResultsFile = open(closureResults, 'w')
        print('Write Closure Results File: \n>> %s' % (bias_file_out_pkl))
        for i in range(nBEs):
            nuissance = 'basis%i' % i
            print(i, "vs", len(self.multijet.cUp  [self.multijet.basis]) )

            if i < len(self.multijet.cUp  [self.multijet.basis]):
                cUp_vari   = self.multijet.cUp  [self.multijet.basis][i]
                cDown_vari = self.multijet.cDown[self.multijet.basis][i]
            else:
                cUp_vari = 0
                cDown_vari = 0

            cUp   = self.cUp  [basis][i]
            cDown = self.cDown[basis][i]
            cUp_bias   = 0
            cDown_bias = 0
            if cUp != cUp_vari:  # break into variance and bias terms
                cUp_bias   =  (cUp**2 - cUp_vari**2)**0.5
            if cDown != cDown_vari:
                cDown_bias = -(cDown**2 - cDown_vari**2)**0.5

            systematics['%s_vari_%sUp' % (nuissance, self.channel)] = 1 + cUp_vari * self.basis_element[i]
            systematics['%s_vari_%sDown' % (nuissance, self.channel)] = 1 + cDown_vari * self.basis_element[i]

            if cUp_bias:
                systematics['%s_bias_%sUp' % (nuissance, self.channel)] = 1 + cUp_bias * self.basis_element[i]
            if cDown_bias:
                systematics['%s_bias_%sDown' % (nuissance, self.channel)] = 1 + cDown_bias * self.basis_element[i]


        #     # if self.spuriousSignalError[basis] < abs(self.spuriousSignal[basis]):
        #     #     print('WARNING: Spurious Signal for channel %s is inconsistent with zero: (%f, %f)' % (self.channel, ssDown, ssUp), end='')
        #     #     ssUp   = max([abs(ssUp), abs(ssDown)])
        #     #     ssDown = -ssUp
        #     #     print(' -> (%f, %f)'%(ssDown, ssUp))
        #     SS_string  = ', '.join('%7.4f'%SS_i for SS_i in self.multijet.basis_signal[max_basis] * 10)
        #     systUp     = '1 + (%9.6f)*np.array([%s])'%(ssUp/10,   SS_string)
        #     systDown   = '1 + (%9.6f)*np.array([%s])'%(ssDown/10, SS_string)
        #     systUp     = '%s_%sUp   %s'%(nuissance, channel, systUp)
        #     systDown   = '%s_%sDown %s'%(nuissance, channel, systDown)
        #     print(systUp)
        #     print(systDown)
        #     closureResultsFile.write(systUp+'\n')
        #     closureResultsFile.write(systDown + '\n')
        # closureResultsFile.close()

        with open(bias_file_out_pkl, 'wb') as sfile:
            pickle.dump(systematics, sfile, protocol=1)

    def plotFitResults(self, basis, projection=(0, 1)):
        max_basis = max(self.multijet.basis, basis)
        n = max_basis + 1

        d_ss = n

        if n > 1:
            dims = tuple(list(projection) + [d for d in range(n) if d not in projection])
        else:
            dims = (0, 1)

        labels = ['c$_' + str(d) + '$' for d in dims]


        # plot fit parameters
        x, y, s, c = [], [], [], []
        parameters = self.fit_parameters[basis]
        x.append( parameters[dims[0]] * (1 if dims[0] == d_ss else 100) )
        if n == 1:
            y.append( 0 )
        if n > 1:
            y.append( parameters[dims[1]] * (1 if dims[1] == d_ss else 100) )
        if n > 2:
            c.append( parameters[dims[2]] * (1 if dims[2] == d_ss else 100) )
        if n > 3:
            s.append( parameters[dims[3]] * (1 if dims[3] == d_ss else 100) )

        x = np.array(x)
        y = np.array(y)

        kwargs = {'lw': 0.5,
                  'marker': 'o',
                  'edgecolors': 'k',
                  's': 8,
                  'c': 'k',
                  'zorder': 2,
                  }

        fig, (ax) = plt.subplots(nrows=1, figsize=(6, 6))
        ax.set_aspect(1)
        ax.set_title('Multijet Model Bias Fit (%s)' % self.channel.upper())

        ax.set_xlabel(labels[0] + ('' if dims[0] == d_ss else ' (\%)'))
        ax.set_ylabel(labels[1] + ('' if dims[1] == d_ss else ' (\%)'))

        xlim, ylim = [-10, 10], [-10, 10]
        ax.plot(xlim, [0, 0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot([0, 0], ylim, color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        xticks = np.arange(xlim[0] + 2, xlim[1], 2)
        yticks = np.arange(ylim[0] + 2, ylim[1], 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        if n > 1:
            # draw 1\sigma ellipses
            try:
                width = self.multijet.cUp[self.multijet.basis][dims[0]] - self.multijet.cDown[self.multijet.basis][dims[0]]
            except IndexError:
                width = 0.0  # no variance for this basis element
            try:
                height = self.multijet.cUp[self.multijet.basis][dims[1]] - self.multijet.cDown[self.multijet.basis][dims[1]]
            except IndexError:
                height = 0.0

            ellipse_self_consistency = Ellipse((0, 0),
                                               width =100 * width,
                                               height=100 * height,
                                               facecolor = 'none',
                                               edgecolor = 'b',  # CMURED,
                                               linestyle = '-',
                                               linewidth = 0.75,
                                               zorder=1,
                                               )

            ax.add_patch(ellipse_self_consistency)

            ellipse_closure = Ellipse((0, 0),
                                      width =100 * (self.cUp[basis][dims[0]] - self.cDown[basis][dims[0]]),
                                      height=100 * (self.cUp[basis][dims[1]] - self.cDown[basis][dims[1]]),
                                      facecolor='none',
                                      edgecolor='r',  # CMURED,
                                      linestyle='-',
                                      linewidth=0.75,
                                      zorder=1,
                                      )

            ax.add_patch(ellipse_closure)

        bbox = dict(boxstyle='round', facecolor='w', alpha=0.8, linewidth=0)
        if n > 2:
            # draw range bars for other priors
            for i, d in enumerate(dims[2:]):
                up, down = self.cUp[basis][d], self.cDown[basis][d]
                thisx = xlim[-1] - 0.5 * (n - 2) + 0.5 * i
                ax.quiver(thisx, 0, 0, 100 * up,   color='r', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
                ax.quiver(thisx, 0, 0, 100 * down, color='r', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

                ax.annotate(labels[i + 2], [thisx, 100 * down - 0.5], ha='center', va='center', bbox=bbox)

            for i, d in enumerate(dims[2:]):
                try:
                    up, down = self.multijet.cUp[self.multijet.basis][d], self.multijet.cDown[self.multijet.basis][d]
                    thisx = xlim[-1] - 0.5 * (n - 2) + 0.5 * i
                    ax.quiver(thisx, 0, 0, 100 * up,   color='b', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
                    ax.quiver(thisx, 0, 0, 100 * down, color='b', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
                except IndexError:
                    pass  # there is no variance term for this basis

        maxr = np.zeros((2, len(x)), dtype=float)
        minr = np.zeros((2, len(x)), dtype=float)

        if n > 1:
            # generate a ton of random points on a hypersphere in dim=n so surface is dim=n - 1.
            points  = np.random.randn(n, min(100 * (n - 1), 10**7))  # random points in a hypercube
            points /= np.linalg.norm(points, axis=0)  # normalize them to the hypersphere surface

            # find the point which maximizes the change in c_0**2 + c_1**2
            for i in range(len(x)):
                eigenVars = self.eigenVars[basis]
                plane = np.matmul( eigenVars[dims[0:2], :], points )
                r2 = plane[0]**2
                if n > 1:
                    r2 += plane[1]**2

                maxr[:, i] = plane[:, r2 == r2.max()].T[0]

                # construct orthogonal unit vector to maxr
                minrvec = np.copy(maxr[::-1, i])
                minrvec[0] *= -1
                minrvec /= np.linalg.norm(minrvec)

                # find maxr along minrvec to get minr
                dr2 = np.matmul( minrvec, plane )**2
                # minr[:, i] = plane[:,dr2==dr2.max()].T[0]#this guy is the ~right length but might be slightly off orthogonal
                minr[:, i] = minrvec * dr2.max()**0.5  # this guy is the ~right length and is orthogonal by construction
        else:
            for i in range(len(x)):
                maxr[0, i] = self.eigenVars[basis][dims[0]]

        if dims[0] != d_ss:
            minr[0] *= 100
            maxr[0] *= 100
        if dims[1] != d_ss:
            minr[1] *= 100
            maxr[1] *= 100

        # print(maxr)
        # print(minr)
        ax.quiver(x, y,  maxr[0],  maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -maxr[0], -maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

        ax.quiver(x, y,  minr[0],  minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -minr[0], -minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

        plt.scatter(x, y, **kwargs)

        if n > 2:
            for i in range(len(x)):
                label = '\n'.join(['%s = %2.1f%s' % (labels[dims.index(d)], parameters[d] * (1 if d == d_ss else 100), '' if d == d_ss else '\%') for d in dims[2:]])
                # xy = np.array([x[i],y[i]])
                # xy = [xy+minr[:,i]+maxr[:,i],
                #       xy+minr[:,i]-maxr[:,i],
                #       xy-minr[:,i]+maxr[:,i],
                #       xy-minr[:,i]-maxr[:,i]]
                # xy = max(xy, key=lambda p: p[0]**2+p[1]**2)
                # if xy[0]>0:
                #     horizontalalignment = 'left'
                # else:
                #     horizontalalignment = 'right'
                # if xy[1]>0:
                #     verticalalignment = 'bottom'
                # else:
                #     verticalalignment = 'top'
                xy = [xlim[-1] - 3, ylim[-1] - 1]
                ax.annotate(label, xy,  # label,
                            ha='left', va='top',
                            bbox=bbox)

        projection = '_'.join([str(d) for d in projection])

        if type(self.rebin) is list:
            name = f'{args.outputPath}/{classifier}/variable_rebin/{args.region}/{self.channel}/1_bais_parameters_basis{basis}_projection_{projection}.pdf'
        else:
            name = f'{args.outputPath}/{classifier}/rebin{self.rebin}/{args.region}/{self.channel}/1_bias_parameters_basis{basis}_projection_{projection}.pdf'

        # print('fig.savefig( ' + name+' )')
        plt.tight_layout()
        fig.savefig( name )
        plt.close(fig)

    def plotPValues(self):
        fig, (ax) = plt.subplots(nrows=1)
        x = np.array(sorted(self.pvalue.keys())) + 1
        ax.set_ylim(0, 1)
        xlim = [x[0] - 0.5, x[-1] + .5]
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_xticks(x)
        # plt.yscale('log')

        y = [self.pvalue[i - 1] for i in x]
        ax.set_title('Multijet Model Bias Fit (%s)' % self.channel.upper())
        ax.plot(xlim, [probThreshold, probThreshold], color='b', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot(xlim, [0.95, 0.95],                   color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot(x, y, label='p-value', color='b', linewidth=2)
        if self.basis is not None:
            ax.plot([self.basis + 1, self.basis + 1], [0, 1], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
            ax.scatter(self.basis + 1, self.pvalue[self.basis], color='k', marker='*', s=100, zorder=10)

        x = np.array(sorted(self.fProb.keys())) + 1
        y = [self.fProb[i - 1] for i in x]
        marker = '' if len(x) > 1 else 'o'
        ax.plot(x, y, label='f-test', color='k', linewidth=2, marker=marker)

        ax.set_xlabel('Unconstrained Parameters')
        ax.set_ylabel('Fit p-value')
        ax.legend(loc='upper left', fontsize='small')

        if type(self.rebin) is list:
            name = f'{args.outputPath}/{classifier}/variable_rebin/{args.region}/{self.channel}/1_bias_pvalues.pdf'
        else:
            name = f'{args.outputPath}/{classifier}/rebin{self.rebin}/{args.region}/{self.channel}/1_bias_pvalues.pdf'

        # print('fig.savefig( ' + name+' )')
        plt.tight_layout()
        fig.savefig( name )
        plt.close(fig)

    def plotMix(self, mix):
        samples = collections.OrderedDict()
        samples[bias_file_out] = collections.OrderedDict()
        if type(mix) is int:
            samples[bias_file_out][f'{self.channel}/data_obs'] = {
                'label' : f'Mixed Data Set {mix}, {lumi}/fb',
                'legend': 1,
                'isData' : True,
                # 'drawOptions': 'P ex0',
                'ratio' : 'numer A',
                'color' : 'ROOT.kBlack'}
        else:
            samples[bias_file_out][f'{self.channel}/data_obs'] = {
                'label' : f'#LTMixed Data#GT {lumi}/fb',
                'legend': 1,
                'isData' : True,
                # 'drawOptions': 'P ex0',
                'ratio' : 'numer A',
                'color' : 'ROOT.kBlack'}
            # samples[bias_file_out]['%s/ratio_c0_up'%(self.channel)] = {
            #     'pad': 'rPad',
            #     'drawOptions': 'HIST',
            #     'color' : 'ROOT.kYellow'}
        if type(mix) is int:
            samples[bias_file_out][f'{self.channel}/multijet'] = {
                'label' : 'Multijet Model %d' % mix,
                'legend': 2,
                'stack' : 3,
                'ratio' : 'denom A',
                'color' : 'ROOT.kYellow'}
        else:
            samples[bias_file_out]['%s/multijet' % self.channel] = {
                'label' : '#LTMultijet#GT',
                'legend': 2,
                'stack' : 3,
                'ratio' : 'denom A',
                'color' : 'ROOT.kYellow'}
        samples[bias_file_out]['%s/ttbar' % self.channel] = {
            'label' : '#lower[0.10]{t#bar{t}}',
            'legend': 3,
            'stack' : 2,
            'ratio' : 'denom A',
            'color' : 'ROOT.kAzure-9'}

        xTitle = f'{classifier} P(Signal) #cbar P({self.channel.upper()}) is largest'

        parameters = {'titleLeft'   : '#bf{CMS} Internal',
                      'titleCenter' : regionName[args.region],
                      'titleRight'  : 'Pass #DeltaR(j,j)',
                      'maxDigits'   : 4,
                      'ratioErrors': True,
                      'ratio'     : True,
                      'rMin'      : 0.9,
                      'rMax'      : 1.1,
                      'rebin'     : self.rebin,
                      'rTitle'    : 'Data / Bkgd.',
                      'xTitle'    : xTitle,
                      'yTitle'    : 'Events',
                      'yMax'      : 1.4 * (self.ymax[0]),  # *ymaxScale, # make room to show fit parameters
                      # 'xleg'      : [0.13, 0.13 + 0.5] if 'SR' in region else ,
                      #  'legendSubText' : ['#bf{Fit:}',
                      #                     '#chi^{2}/DoF = %2.1f/%d = %1.2f'%(self.chi2[basis],self.ndf[basis],self.chi2[basis]/self.ndf[basis]),
                      #                     'p-value = %2.0f%%'%(self.pvalue[basis] * 100),
                      #                     ],
                      'lstLocation' : 'right',
                      'outputName': 'mix_%s' % (str(mix))}

        if type(self.rebin) is list:
            parameters['outputDir'] = f'{args.outputPath}/{classifier}/variable_rebin/{args.region}/{self.channel}/'
        else:
            parameters['outputDir'] = f'{args.outputPath}/{classifier}/rebin{self.rebin}/{args.region}/{self.channel}/'

        # print('make ',parameters['outputDir'] + parameters['outputName']+'.pdf')
        ROOTPlotTools.plot(samples, parameters, debug=False)

    def plotFit(self, basis):
        samples = collections.OrderedDict()
        samples[bias_file_out] = collections.OrderedDict()
        samples[bias_file_out]['%s/data_obs_closure' % self.channel] = {
            'label' : f'#LTMixed Data#GT {lumi}/fb',
            'legend': 1,
            'isData' : True,
            # 'ratioDrawOptions': 'P ex0',
            'ratio' : 'numer A',
            'color' : 'ROOT.kBlack'}
        samples[bias_file_out]['%s/multijet_closure' % self.channel] = {
            'label' : '#LTMultijet#GT',
            'legend': 2,
            'stack' : 3,
            'ratio' : 'denom A',
            'color' : 'ROOT.kYellow'}
        samples[bias_file_out]['%s/ttbar_closure' % self.channel] = {
            'label' : '#lower[0.10]{t#bar{t}}',
            'legend': 3,
            'stack' : 2,
            'ratio' : 'denom A',
            'color' : 'ROOT.kAzure-9'}
        samples[bias_file_out]['%s/closure_TH1_basis%d' % (self.channel, basis)] = {
            'label' : 'Fit (%d unconstrained parameter%s)' % (basis + 1, 's' if basis else ''),
            'legend': 4,
            'ratio': 'denom A',
            'color' : 'ROOT.kRed'}

        xTitle = f'{classifier} P(Signal) Bin #cbar P({self.channel.upper()}) is largest'

        ymaxScale = 1.4  # + max(0, (basis - 2)/4.0)

        parameters = {'titleLeft'   : '#bf{CMS} Internal',
                      'titleCenter' : regionName[args.region],
                      'titleRight'  : 'Pass #DeltaR(j,j)',
                      'maxDigits'   : 4,
                      'drawLines'   : [[self.fit_x_min,          0, self.fit_x_min,         self.ymax[0] / 2],
                                       [self.nBins_rebin + 0.5,  0, self.nBins_rebin + 0.5, self.ymax[0] / 2]],
                      'ratioErrors': False,
                      'ratio'     : 'significance',  # True,
                      'rMin'      : -5,  # 0.9,
                      'rMax'      : 5,  # 1.1,
                      'rTitle'    : 'Pulls',  # 'Data / Bkgd.',
                      # 'ratioErrors': True,
                      # 'ratio'      : True,
                      # 'rMin'       : 0.9,
                      # 'rMax'       : 1.1,
                      # 'rTitle'     : 'Model / Average',
                      'xTitle'    : xTitle,
                      'yTitle'    : 'Events',
                      'yMax'      : self.ymax[0] * ymaxScale,   # *ymaxScale, # make room to show fit parameters
                      'xleg'      : [0.13, 0.13 + 0.42],
                      'lstLocation': 'right',
                      'outputName': '%s_basis%d' % ('1_bias', basis)}

        n = max(self.multijet.basis, basis) + 1
        parameters['legendSubText'] = ['#bf{Fit:}',
                                       '#chi^{2}/DoF = %2.1f/%d = %1.2f' % (self.chi2[basis], self.ndf[basis], self.chi2[basis] / self.ndf[basis]),
                                       'p-value = %2.0f%%' % (self.pvalue[basis] * 100)]
        for i in range(n):
            parameters['legendSubText'] += ['#color[%d]{#font[82]{c_{%i} =%4.1f%% : %3.1f}#sigma}' % (4 if i > basis else 2, i, self.fit_parameters[basis][i] * 100, abs(self.fit_parameters[basis][i]) / self.fit_parameters_error[basis][i])]

        parameters['ratioLines'] = [[self.fit_x_min,         parameters['rMin'], self.fit_x_min,         parameters['rMax']],
                                    [self.nBins_rebin + 0.5, parameters['rMin'], self.nBins_rebin + 0.5, parameters['rMax']]]
        # parameters['xMax'] = self.nBins_rebin + self.multijet.basis + 1.5 if not plotSpuriousSignal else self.nBins_rebin+basis + 1.5
        parameters['xMax'] = self.nBins_rebin + 0.5 + max(self.multijet.basis - basis, 0)

        if type(self.rebin) is list:
            parameters['outputDir'] = f'{args.outputPath}/{classifier}/variable_rebin/{args.region}/{self.channel}/'
        else:
            parameters['outputDir'] = f'{args.outputPath}/{classifier}/rebin{self.rebin}/{args.region}/{self.channel}/'

        # print(f'make {parameters["outputDir"]}{parameters["outputName"]}.pdf')
        ROOTPlotTools.plot(samples, parameters, debug=False)

    def print_exit_message(self):
        self.output_yml.close()
        for line in self.exit_message:
            print_log(line)


        
        
def run():

    f = ROOT.TFile(bias_file_out, 'UPDATE')

    #
    # make multijet ensembles and perform fits
    #
    multijetEnsembles = {}
    if type(rebin) is list:
        mkpath(f'{args.outputPath}/{classifier}/variable_rebin/{args.region}/{channel}')
    else:
        mkpath(f'{args.outputPath}/{classifier}/rebin{rebin}/{args.region}/{channel}')

    multijetEnsembles[channel] = multijetEnsemble(f, channel)

    #
    # run closure fits using average multijet model
    #
    closures = {}
    closures[channel] = closure(f, channel, multijetEnsembles[channel])

    #
    # close input file and make plots
    #
    f.Close()

    for basis in multijetEnsembles[channel].bases:
        multijetEnsembles[channel].plotFit(basis)
    for basis in closures[channel].bases:
        closures[channel].plotFit(basis)

    for m in range(1):
        closures[channel].plotMix(m)
    closures[channel].plotMix('ave')

    multijetEnsembles[channel].print_exit_message()
    closures[channel].print_exit_message()



def prepInput():

    #
    # Read inputs
    #
    input_file = ROOT.TFile(args.input_file, 'READ')

    #
    # Make output
    #
    f = ROOT.TFile(bias_file_out, 'RECREATE')

    addYears(f, input_file, channel=channel)

    f.Close()





    


    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='make JCM weights', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--debug',                 action="store_true")
    parser.add_argument('-l', '--lumi',                 dest="lumi",          default="133",    help="Luminosity for MC normalization: units [pb]")
    parser.add_argument('--classifier', help="SvB or SvB_MA")
    parser.add_argument('--region', default="SR", help="SR or SB")
    parser.add_argument('--input_file',default="output/histAll.root")
    parser.add_argument('--var', default="SvB_MA_ps_hh", help="SvB_MA_ps_XX or SvB_MA_ps_XX_fine")
    parser.add_argument('--rebin', default=1)
    parser.add_argument('--outputPath', default="stats_analysis/fitBiasData")
    parser.add_argument('--do_CI',   action="store_true")
    parser.add_argument('--bkg_syst_file',  default="stats_analysis/closureFits/ULHH/3bDvTMix4bDvT/SvB_MA/rebin8/SR/zz/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_zz_rebin8.pkl/", help="File contain background systematic variations")
    args = parser.parse_args()


    #
    #  Parse channel
    #
    if not args.var.find("hh") == -1:
        channel = "hh"
    elif not args.var.find("zh") == -1:
        channel = "zh"
    elif not args.var.find("zz") == -1:
        channel = "zz"
    else:
        print(f"ERROR cannot parse chanel from {args.var}")
        sys.exit(-1)

    #
    #  Parse classifier
    #
    if not args.var.find("SvB_MA") == -1:
        classifier = "SvB_MA"
    elif not args.var.find("SvB") == -1:
        classifier = "SvB"
    else:
        print(f"ERROR cannot parse classifier from {args.var}")
        sys.exit(-1)

    rebin = int(args.rebin)

    #mkpath(f'{args.outputPath}')
    mkpath(f'{args.outputPath}/{classifier}/rebin{rebin}/{args.region}/{channel}')

    bias_file_out = f"{args.outputPath}/{classifier}/rebin{rebin}/{args.region}/{channel}/hists_bias_{args.var}_rebin{rebin}.root"
    bias_file_out_pkl = bias_file_out.replace("root", "pkl")
    bias_file_out_log = bias_file_out.replace("root", "log")

    log_file = open(bias_file_out_log,"w")

    print_log(f"\nRunning with channel {channel} and rebin {rebin}")
    print_log(f"   creating:\n")
    print_log(f"\t{bias_file_out}")
    print_log(f"\t{bias_file_out_log}")
    print_log(f"\t{bias_file_out_pkl}")

    print_log(f"\nInputs are ")
    print_log(f"\t input_file {args.input_file}")


    lumi = args.lumi

    #
    #  Settings
    #
    closure_fit_x_min = 0  # 0.01
    maxBasisEnsemble  = 5
    maxBasisClosure   = 5

    if not args.do_CI:
        plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    probThreshold = 0.05  # 0.045500263896 #0.682689492137 # 1sigma
    
    prepInput()

    run()
    
#    if args.bkg_syst_file:
#        print_log(f"Reading {args.bkg_syst_file}")
#        bkg_syst_file = pickle.load(open(args.bkg_syst_file, 'rb'))
#
#        for iy in ['UL16', 'UL17', 'UL18']:
#            for ibin, ivalues in bkg_syst_file.items():
#                print(ibin, ivalues)
#                root_hists[f"{channel}_{iy}"]['mj'][ibin] = root_hists[f"{channel}_{iy}"]['mj']['nominal'].Clone()
#                root_hists[f"{channel}_{iy}"]['mj'][ibin].SetName(f'mj_{ibin}')
#                for i in range(len(ivalues)):
#                    nom_val = root_hists[f"{channel}_{iy}"]['mj'][ibin].GetBinContent( i+1 )
#                    root_hists[f"{channel}_{iy}"]['mj'][ibin].SetBinContent( i+1, nom_val*ivalues[i]  )
