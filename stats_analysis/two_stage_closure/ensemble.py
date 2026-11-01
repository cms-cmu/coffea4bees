from __future__ import print_function
import os
import sys
import collections
import array
import numpy as np

from . import config
from .bases import BE
from .metrics import pearsonr
from .io import print_log

class multijetEnsemble:
    
    def __init__(self, f, channel):
        ROOT = config.ROOT
        args = config.args
        rebin = config.rebin
        rebin_histogram = config.rebin_histogram
        output_dir = config.output_dir
        mixes = config.mixes
        nMixes = config.nMixes
        closure_fit_x_min = config.closure_fit_x_min
        maxBasisEnsemble = config.maxBasisEnsemble
        probThreshold = config.probThreshold

        self.channel = channel
        self.rebin = rebin

        self.output_yml = open(f'{output_dir}/0_variance_results.yml', 'w')

        if getattr(args, 'unify_background', False):
            self.data_minus_ttbar = f.Get(f'{self.channel}/data_obs').Clone(f'data_obs_average_{self.channel}')
            self.data_minus_ttbar.SetDirectory(0)
            if isinstance(self.rebin, array.array):
                self.data_minus_ttbar = rebin_histogram(self.data_minus_ttbar, self.rebin)
            else:
                self.data_minus_ttbar.Rebin(self.rebin)
        else:
            self.data_minus_ttbar = f.Get(f'{self.channel}/ttbar').Clone(f'data_minus_ttbar_average_{self.channel}')
            self.data_minus_ttbar.SetDirectory(0)
            self.data_minus_ttbar.Scale(-1)
            h_data_obs = f.Get(f'{self.channel}/data_obs')
            if self.data_minus_ttbar.GetNbinsX() != h_data_obs.GetNbinsX():
                raise ValueError(
                    f"Binning mismatch in multijetEnsemble for {self.channel}: "
                    f"ttbar has {self.data_minus_ttbar.GetNbinsX()} bins, but data_obs has {h_data_obs.GetNbinsX()} bins!"
                )
            self.data_minus_ttbar.Add(h_data_obs)
            if isinstance(self.rebin, array.array):
                self.data_minus_ttbar = rebin_histogram(self.data_minus_ttbar, self.rebin)
            else:
                self.data_minus_ttbar.Rebin(self.rebin)

        self.average = f.Get(f'{self.channel}/multijet')
        self.average.SetName('%s_average_%s' % (self.average.GetName(), self.channel))
        self.models  = [f.Get('%s/%s/multijet' % (mix, self.channel)) for mix in mixes]
        for m, model in enumerate(self.models):
            model.SetName('%s_%s_%s' % (model.GetName(), mixes[m], self.channel))
        self.nBins   = self.average.GetSize() - 2

        print(f"Reading {self.channel}/signal")
        h_sig = f.Get('%s/signal' % self.channel)
        if h_sig and not h_sig.IsZombie():
            self.signal = h_sig.Clone(f'signal_rebin_{self.channel}')
            self.signal.SetDirectory(0)
            if isinstance(self.rebin, array.array):
                self.signal = rebin_histogram(self.signal, self.rebin)
            else:
                self.signal.Rebin(self.rebin)
        else:
            self.signal = None

        self.f = f
        self.f.cd(self.channel)

        self.average_rebin = self.average.Clone()
        self.average_rebin.SetName('%s_rebin' % self.average.GetName())
        if isinstance(self.rebin, array.array):
            self.average_rebin = rebin_histogram(self.average_rebin, self.rebin)
        else:
            self.average_rebin.Rebin(self.rebin)

        self.models_rebin = []

        for imodel in [model.Clone() for model in self.models]:
            if isinstance(self.rebin, array.array):
                imodel = rebin_histogram(imodel, self.rebin)
            else:
                imodel.SetName('%s_rebin' % model.GetName())
                imodel.Rebin(self.rebin)
            self.models_rebin.append(imodel)

        if getattr(args, 'match_normalization', False):
            int_target = self.data_minus_ttbar.Integral()
            int_mj = self.average_rebin.Integral()
            if int_mj > 0:
                scale_mj = int_target / int_mj
                print_log(f"[INFO] Matching multijet normalization to data - ttbar: scale = {scale_mj:.6f} "
                          f"(target = {int_target:.2f}, original multijet = {int_mj:.2f})")
                self.average.Scale(scale_mj)
                self.average_rebin.Scale(scale_mj)
                for m in self.models:
                    m.Scale(scale_mj)
                for mr in self.models_rebin:
                    mr.Scale(scale_mj)

        self.nBins_rebin = self.average_rebin.GetSize() - 2

        self.f.cd(self.channel)
        self.nBins_ensemble = self.nBins_rebin * nMixes
        self.bin_width = 1. / self.nBins_rebin
        if getattr(config, 'skip_first_bin', False) or getattr(args, 'skip_first_bin', False):
            self.fit_bin_min = 2
            self.nBins_fit = self.nBins_rebin - 1
        else:
            self.fit_bin_min = int(1 + closure_fit_x_min // self.bin_width)
            self.nBins_fit = self.nBins_rebin - int(closure_fit_x_min // self.bin_width)
        self.multijet_ensemble_average  = ROOT.TH1F('multijet_ensemble_average', '', self.nBins_ensemble, 0.5, 0.5 + self.nBins_ensemble)
        self.multijet_ensemble          = ROOT.TH1F('multijet_ensemble'        , '', self.nBins_ensemble, 0.5, 0.5 + self.nBins_ensemble)
        self.data_minus_ttbar_ensemble  = ROOT.TH1F('data_minus_ttbar_ensemble', '', self.nBins_ensemble, 0.5, 0.5 + self.nBins_ensemble)

        for m in range(nMixes):
            for b in range(self.nBins_rebin):
                local_bin    = 1 + b
                ensemble_bin = 1 + b + m * self.nBins_rebin
                error = (self.models_rebin[m].GetBinError(local_bin)**2 + (2 / nMixes)**2)**0.5
                self.multijet_ensemble_average.SetBinContent(ensemble_bin, self.average_rebin.GetBinContent(local_bin))
                self.multijet_ensemble_average.SetBinError  (ensemble_bin, error)
                self.multijet_ensemble.SetBinContent(ensemble_bin, self.models_rebin[m].GetBinContent(local_bin))
                self.multijet_ensemble.SetBinError(ensemble_bin, 0.0)
                self.data_minus_ttbar_ensemble.SetBinContent(ensemble_bin, self.data_minus_ttbar.GetBinContent(local_bin))
                self.data_minus_ttbar_ensemble.SetBinError  (ensemble_bin, self.data_minus_ttbar.GetBinError  (local_bin))

        self.f.cd(self.channel)
        self.multijet_ensemble_average.Write()
        self.multijet_ensemble        .Write()
        self.data_minus_ttbar_ensemble.Write()

        self.bases = range(0, maxBasisEnsemble + 1, 1)

        # Make kernel for basis orthogonalization
        h = np.array([self.average_rebin.GetBinContent(bin) for bin in range(1, self.nBins_rebin + 1)])
        h_no_rebin = np.array([self.average.GetBinContent(bin) for bin in range(1, self.nBins + 1)])
        self.h = h
        self.h_no_rebin = h_no_rebin

        # Make matrix of initial basis
        B_no_rebin = np.array([[b.Integral(self.average.GetBinLowEdge(bin), self.average.GetXaxis().GetBinUpEdge(bin)) / self.average.GetBinWidth(bin) for bin in range(1, self.nBins + 1)] for b in BE])
        B = np.array([[b.Integral(self.average_rebin.GetBinLowEdge(bin), self.average_rebin.GetXaxis().GetBinUpEdge(bin)) / self.average_rebin.GetBinWidth(bin) for bin in range(1, self.nBins_rebin + 1)] for b in BE])
        S = np.array([[self.signal.GetBinContent(bin) for bin in range(1, self.nBins_rebin + 1)]])
        S = np.where(h > 0, S / h, 0.0)
        smax = S.max()
        if smax > 0:
            S = S / smax
        S = S.repeat(len(BE), axis=0)
        self.basis_element = B
        self.basis_signal  = S
        self.basis_element_no_rebin = B_no_rebin

        for basis in self.bases[1:]:
            self.plotBasis('initial', basis)
            self.plotBasis('initial', basis, rebin=False)

        # Subtract off cross correlation from higher order basis elements
        for i in range(1, len(B)):
            c = (B[i - 1] * h**1.0 * B[i - 1]).sum()
            B[i:] = B[i:] - (B[i - 1] * h**1.0 * B[i:]).sum(axis=1, keepdims=True) * B[i - 1] / c
            c_no_rebin = (B_no_rebin[i - 1] * h_no_rebin**1.0 * B_no_rebin[i - 1]).sum()
            B_no_rebin[i:] = B_no_rebin[i:] - (B_no_rebin[i - 1] * h_no_rebin**1.0 * B_no_rebin[i:]).sum(axis=1, keepdims=True) * B_no_rebin[i - 1] / c_no_rebin

        for i in range(0, len(B)):
            B[i] = B[i] * np.sign(B[i, -1])
            B_no_rebin[i] = B_no_rebin[i] * np.sign(B_no_rebin[i, -1])
            c = (B[i] * h**1.0 * B[i]).sum()
            S[i:] = S[i:] - (B[i] * h**1.0 * S[i:]).sum(axis=1, keepdims=True) * B[i] / c

        for basis in self.bases[1:]:
            self.plotBasis('diagonalized', basis)
            self.plotBasis('diagonalized', basis, rebin=False)

        for i in range(1, len(B)):
            d = B[i].max() - B[i].min()
            if d > 1e-12:
                B[i] = B[i] / d
            d = B_no_rebin[i].max() - B_no_rebin[i].min()
            if d > 1e-12:
                B_no_rebin[i] = B_no_rebin[i] / d

        for i in range(len(S)):
            S[i] = S[i] / S[i, -1] * self.signal.GetBinContent(self.nBins_rebin) / h[-1]

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
        self.basis = None
        self.passed = False
        self.exit_message = ['--- None (%s) --- Multijet Ensemble' % self.channel.upper()]
        min_r = 1.0

        for basis in self.bases:
            self.makeFitFunction(basis)
            self.fit(basis)
            self.write_to_yml(basis)

            self.plotFitResults(basis)
            for i in range(1, basis):
                self.plotFitResults(basis, projection=(i, i + 1))
            self.plotPulls(basis)

            if self.basis is None and abs(self.pearsonr[basis]['total'][1]) > probThreshold:
                min_r = abs(self.pearsonr[basis]['total'][0])
                self.basis = basis
                self.passed = True
                self.exit_message = []
                self.exit_message.append('-' * 50)
                self.exit_message.append('%s channel' % self.channel.upper())
                self.exit_message.append('Satisfied adjacent bin de-correlation p-value for multijet ensemble variance at basis %d:' % self.basis)
                self.exit_message.append('>> p-value, r-value = %2.0f%%, %0.2f ' % (100 * self.pearsonr[self.basis]['total'][1], self.pearsonr[self.basis]['total'][0]))
                self.exit_message.append('-' * 50)

        if self.basis is None:
            self.passed = False
            self.basis = self.bases[np.argmin([abs(self.pearsonr[basis]['total'][0]) for basis in self.bases])]
            self.exit_message = []
            self.exit_message.append('-' * 50)
            self.exit_message.append('WARNING: %s channel multijet ensemble variance did not satisfy p-value > %0.1f%%' % (self.channel.upper(), 100 * probThreshold))
            self.exit_message.append('Minimized adjacent bin correlation abs(r) for multijet ensemble variance at basis %d:' % self.basis)
            self.exit_message.append('>> p-value, r-value = %2.0f%%, %0.2f ' % (100 * self.pearsonr[self.basis]['total'][1], self.pearsonr[self.basis]['total'][0]))
            self.exit_message.append('-' * 50)

        self.plotPearson()

    def print_exit_message(self):
        self.output_yml.close()
        for line in self.exit_message:
            print_log(line)

    def makeFitFunction(self, basis):
        ROOT = config.ROOT
        nMixes = config.nMixes

        def background_UserFunction(xArray, pars):
            ensemble_bin = int(xArray[0])
            m = (ensemble_bin - 1) // self.nBins_rebin
            local_bin = 1 + ((ensemble_bin - 1) % self.nBins_rebin)

            p = 1.0
            for BE_idx in range(basis + 1):
                par_idx = m * (basis + 1) + BE_idx
                p += pars[par_idx] * self.basis_element[BE_idx][local_bin - 1]

            return p * self.multijet_ensemble.GetBinContent(ensemble_bin)

        self.f.cd(self.channel)
        self.pycallable = background_UserFunction
        self.multijet_TF1[basis] = ROOT.TF1('multijet_ensemble_TF1_basis%d' % basis, self.pycallable, 0.5, 0.5 + self.nBins_ensemble, nMixes * (basis + 1))
        self.multijet_TH1[basis] = ROOT.TH1F('multijet_ensemble_TH1_basis%d' % basis, '', self.nBins_ensemble, 0.5, 0.5 + self.nBins_ensemble)

        for m in range(nMixes):
            for o in range(basis + 1):
                self.multijet_TF1[basis].SetParName(m * (basis + 1) + o, 'v%d c_%d' % (m, o))
                self.multijet_TF1[basis].SetParameter(m * (basis + 1) + o, 0.0)

    def getEigenvariations(self, basis=None, debug=False):
        ROOT = config.ROOT
        nMixes = config.nMixes
        if basis is None:
            basis = self.basis
        n = basis + 1

        if n == 1:
            self.eigenVars[basis] = [np.array([[self.multijet_TF1[basis].GetParError(m * n)]]) for m in range(nMixes)]
            return

        cov = [ROOT.TMatrixD(n, n) for m in range(nMixes)]
        cor = [ROOT.TMatrixD(n, n) for m in range(nMixes)]

        for m in range(nMixes):
            for i in range(n):
                for j in range(n):
                    cov[m][i][j] = self.fit_result[basis].CovMatrix(m * n + i, m * n + j)
                    cor[m][i][j] = self.fit_result[basis].Correlation(m * n + i, m * n + j)

        if debug:
            for m in range(nMixes):
                print('Covariance Matrix:', m)
                cov[m].Print()
                print('Correlation Matrix:', m)
                cor[m].Print()

        eigenVal = [ROOT.TVectorD(n) for m in range(nMixes)]
        eigenVec = [cov[m].EigenVectors(eigenVal[m]) for m in range(nMixes)]

        for m in range(nMixes):
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

        self.eigenVars[basis] = [np.zeros((n, n), dtype=float) for m in range(nMixes)]
        for m in range(nMixes):
            for i in range(n):
                for j in range(n):
                    self.eigenVars[basis][m][i, j] = eigenVec[m][i][j] * (eigenVal[m][j]**0.5)

    def getParameterDistribution(self, basis):
        nMixes = config.nMixes
        n = basis + 1
        params = np.array([self.fit_parameters[basis][m] for m in range(nMixes)])
        params_err = np.array([self.fit_parameters_error[basis][m] for m in range(nMixes)])
        parMean = np.mean(params, axis=0)
        parMeanErr = np.mean(params_err, axis=0)
        if nMixes > 1:
            parStd = np.std(params, axis=0, ddof=1)
        else:
            parStd = np.zeros_like(parMean)
        print('Parameter Mean:', parMean)
        print('Parameter  Std:', parStd)

        for i in range(n):
            cUp   = abs(parMean[i]) + parStd[i]
            cDown = -cUp
            try:
                self.cUp[basis].append(cUp)
                self.cDown[basis].append(cDown)
            except KeyError:
                self.cUp[basis] = [cUp]
                self.cDown[basis] = [cDown]

    def fit(self, basis):
        nMixes = config.nMixes
        self.fit_result[basis] = self.multijet_ensemble_average.Fit(self.multijet_TF1[basis], 'N0SQ')
        self.getEigenvariations(basis)
        self.pvalue[basis], self.chi2[basis], self.ndf[basis] = self.multijet_TF1[basis].GetProb(), self.multijet_TF1[basis].GetChisquare(), self.multijet_TF1[basis].GetNDF()
        print("=" * 50)
        print('Fit multijet ensemble %s at basis %d' % (self.channel, basis))
        print('chi2/ndf = %3.2f/%3d = %2.2f' % (self.chi2[basis], self.ndf[basis], self.chi2[basis] / self.ndf[basis] if self.ndf[basis] > 0 else 0.0))
        print(' p-value = %0.2f' % self.pvalue[basis])

        self.ymax[basis] = self.multijet_TF1[basis].GetMaximum(1, self.nBins_ensemble)
        self.fit_parameters[basis], self.fit_parameters_error[basis] = [], []
        n = basis + 1

        for m in range(nMixes):
            self.fit_parameters[basis].append(np.array([self.multijet_TF1[basis].GetParameter(m * n + o) for o in range(basis + 1)]))
            self.fit_parameters_error[basis].append(np.array([self.multijet_TF1[basis].GetParError(m * n + o) for o in range(basis + 1)]))
        self.getParameterDistribution(basis)

        for _bin in range(1, self.nBins_ensemble + 1):
            self.multijet_TH1[basis].SetBinContent(_bin, self.multijet_TF1[basis].Eval(_bin))
            self.multijet_TH1[basis].SetBinError(_bin, 0.0)

        pulls = []
        for m in range(nMixes):
            for b in range(self.fit_bin_min, self.nBins_rebin + 1):
                _bin = m * self.nBins_rebin + b
                error = self.multijet_ensemble_average.GetBinError(_bin)
                pull = (self.multijet_TF1[basis].Eval(_bin) - self.multijet_ensemble_average.GetBinContent(_bin)) / error if error > 0 else 0
                pulls.append(pull)
        self.pulls[basis] = np.array(pulls)

        xs = np.array([self.pulls[basis][m * self.nBins_fit : (m + 1) * self.nBins_fit - 1] for m in range(nMixes)])
        ys = np.array([self.pulls[basis][m * self.nBins_fit + 1: (m + 1) * self.nBins_fit] for m in range(nMixes)])

        x, y = xs.flatten(), ys.flatten()
        r, p = pearsonr(x, y, n=len(x) - nMixes * (basis + 1))

        self.pearsonr[basis] = {'total': (r, p),
                                'mixes': [pearsonr(xs[m], ys[m], n=len(xs[m]) - basis - 1) for m in range(nMixes)]}

        print('-------------------------')
        print('>> r, prob = %0.2f, %0.2f' % self.pearsonr[basis]['total'])
        print('-------------------------')
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
        plt = config.plt
        args = config.args
        output_dir = config.output_dir
        if not config.HAS_MPL:
            return

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

        ax.set_xlabel('P(Signal)')
        ax.set_ylabel('Multijet Scale')
        ax.legend(fontsize='small', loc='best')

        rebin_name = '' if rebin else '_no_rebin'
        basis_diag_dir = f"{output_dir}/basis_diagnostics"
        os.makedirs(basis_diag_dir, exist_ok=True)

        plt.tight_layout()
        fig.savefig(f"{basis_diag_dir}/{name}_basis{rebin_name}{basis}.png")
        if getattr(args, 'save_all_formats', False):
            fig.savefig(f"{basis_diag_dir}/{name}_basis{rebin_name}{basis}.pdf")
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

        ax.set_xlabel('P(Signal)')
        ax.set_ylabel('Events')
        ax.legend(fontsize='small', loc='best')

        plt.tight_layout()
        fig.savefig(f"{basis_diag_dir}/{name}_additive_basis{rebin_name}{basis}.png")
        if getattr(args, 'save_all_formats', False):
            fig.savefig(f"{basis_diag_dir}/{name}_additive_basis{rebin_name}{basis}.pdf")
        plt.close(fig)

    def plotPearson(self):
        plt = config.plt
        args = config.args
        output_dir = config.output_dir
        nMixes = config.nMixes
        probThreshold = config.probThreshold
        COLORS = config.COLORS
        if not config.HAS_MPL:
            return

        fig, (ax) = plt.subplots(nrows=1)
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

        for m in range(nMixes):
            r = [self.pearsonr[o]['mixes'][m][0] for o in x - 1]
            label = 'v$_{%d}$' % m
            ax.plot(x, r, color=COLORS[m % len(COLORS)], linewidth=1, alpha=0.5, label=label)

        ax.plot(xlim, [probThreshold, probThreshold], color='r', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Adjacent Bin Pearson Correlation (r) and p-value')
        plt.legend(fontsize='small', loc='best')

        plt.tight_layout()
        fig.savefig(f"{output_dir}/0_variance_pearsonr_multijet_variance.png")
        if getattr(args, 'save_all_formats', False):
            fig.savefig(f"{output_dir}/0_variance_pearsonr_multijet_variance.pdf")
        plt.close(fig)

    def plotFitResults(self, basis, projection=(0, 1)):
        plt = config.plt
        args = config.args
        output_dir = config.output_dir
        nMixes = config.nMixes
        Ellipse = config.Ellipse
        if not config.HAS_MPL or nMixes <= 1:
            return

        n = basis + 1
        if n > 1:
            dims = tuple(list(projection) + [d for d in range(n) if d not in projection])
        else:
            dims = (0, 1)

        x, y, s, c = [], [], [], []
        for m in range(nMixes):
            x.append(100 * self.fit_parameters[basis][m][dims[0]])
            if n == 1:
                y.append(0)
            if n > 1:
                y.append(100 * self.fit_parameters[basis][m][dims[1]])
            if n > 2:
                c.append(100 * self.fit_parameters[basis][m][dims[2]])
            if n > 3:
                s.append(100 * self.fit_parameters[basis][m][dims[3]])

        x = np.array(x)
        y = np.array(y)

        kwargs = {'lw': 0.5, 'marker': 'o', 'edgecolors': 'k', 's': 8, 'c': 'k', 'zorder': 2}
        if n > 2:
            kwargs['c'] = c
            kwargs['cmap'] = 'BuPu'
        if n > 3:
            s = np.array(s)
            smin = s.min()
            smax = s.max()
            srange = smax - smin
            s = s - s.min()
            s = s / (s.max() if s.max() > 0 else 1.0)
            s = (s + 5.0 / 25) * 25
            kwargs['s'] = s

        fig, (ax) = plt.subplots(nrows=1, figsize=(7, 6)) if n > 2 else plt.subplots(nrows=1, figsize=(6, 6))
        ax.set_aspect(1)
        ax.set_title('Multijet Model Variance Fits (%s)' % self.channel.upper())
        ax.set_xlabel('c$_' + str(dims[0]) + r'$ (\%)')
        ax.set_ylabel('c$_' + str(dims[1]) + r'$ (\%)')

        xlim, ylim = [-8, 8], [-8, 8]
        ax.plot(xlim, [0, 0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot([0, 0], ylim, color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        xticks = np.arange(-6, 8, 2)
        yticks = np.arange(-6, 8, 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        if n > 1 and Ellipse is not None:
            ellipse = Ellipse((0, 0),
                              width =100 * (self.cUp[basis][dims[0]] - self.cDown[basis][dims[0]]),
                              height=100 * (self.cUp[basis][dims[1]] - self.cDown[basis][dims[1]]),
                              facecolor='none',
                              edgecolor='b',
                              linestyle='-',
                              linewidth=0.75,
                              zorder=1)
            ax.add_patch(ellipse)

        bbox = dict(boxstyle='round', facecolor='w', alpha=0.8, linewidth=0, pad=0)
        if n > 2:
            for i, d in enumerate(dims[2:]):
                thisx = xlim[-1] - 0.5 * (n - 2) + 0.5 * i
                up, down = self.cUp[basis][d], self.cDown[basis][d]
                ax.quiver(thisx, 0, 0, 100 * up, color='b', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
                ax.quiver(thisx, 0, 0, 100 * down, color='b', scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
                ax.annotate('c$_{%d}$' % (d), [thisx, 100 * down - 0.5], ha='center', va='center', bbox=bbox)

        maxr = np.zeros((2, len(x)), dtype=float)
        minr = np.zeros((2, len(x)), dtype=float)
        if n > 1:
            points = np.random.randn(n, min(100 * (n - 1), 10**7))
            points /= np.linalg.norm(points, axis=0)

            for m in range(nMixes):
                plane = np.matmul(self.eigenVars[basis][m][dims[:2], :], points)
                r2 = plane[0]**2
                if n > 1:
                    r2 += plane[1]**2
                maxr[:, m] = plane[:, r2 == r2.max()].T[0]

                minrvec = np.copy(maxr[::-1, m])
                minrvec[0] *= -1
                minrvec /= np.linalg.norm(minrvec)

                dr2 = np.matmul(minrvec, plane)**2
                minr[:, m] = minrvec * (dr2.max()**0.5)
        else:
            for m in range(nMixes):
                maxr[0, m] = self.eigenVars[basis][m][dims[0]]

        minr *= 100
        maxr *= 100

        ax.quiver(x, y,  maxr[0],  maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -maxr[0], -maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y,  minr[0],  minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -minr[0], -minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

        plt.scatter(x, y, **kwargs)
        plt.tight_layout()

        for m in range(nMixes):
            x_offset, y_offset = (maxr[0, m] + minr[0, m]) / 2, (maxr[1, m] + minr[1, m]) / 2
            ax.annotate('v$_{%d}$' % m, (x[m] + x_offset, y[m] + y_offset), bbox=bbox)

        if n > 2:
            plt.colorbar(label='c$_' + str(dims[2]) + r'$ (\%)')
            plt.subplots_adjust(right=1)

        if n > 3:
            l1 = plt.scatter([], [], s=(0.0 / 3 + 10.0 / 30) * 30, lw=1, edgecolors='black', facecolors='none')
            l2 = plt.scatter([], [], s=(1.0 / 3 + 10.0 / 30) * 30, lw=1, edgecolors='black', facecolors='none')
            l3 = plt.scatter([], [], s=(2.0 / 3 + 10.0 / 30) * 30, lw=1, edgecolors='black', facecolors='none')
            l4 = plt.scatter([], [], s=(3.0 / 3 + 10.0 / 30) * 30, lw=1, edgecolors='black', facecolors='none')

            handles = [l1, l2, l3, l4]
            labels = ['%0.2f' % smin, '%0.2f' % (smin + srange * 1.0 / 3), '%0.2f' % (smin + srange * 2.0 / 3), '%0.2f' % smax]
            plt.legend(handles, labels, ncol=1, fontsize='medium', loc='best', title='c$_' + str(dims[3]) + r'$ (\%)', scatterpoints=1)

        proj_str = '_'.join([str(d) for d in projection])
        try:
            fig.savefig(f"{output_dir}/0_variance_parameters_basis{basis}_projection_{proj_str}.png")
            if getattr(args, 'save_all_formats', False):
                fig.savefig(f"{output_dir}/0_variance_parameters_basis{basis}_projection_{proj_str}.pdf")
            plt.close(fig)
        except IndexError:
            print('Weird index error...')

    def plotPulls(self, basis):
        plt = config.plt
        args = config.args
        output_dir = config.output_dir
        nMixes = config.nMixes
        COLORS = config.COLORS
        if not config.HAS_MPL:
            return

        xs = np.array([self.pulls[basis][m * self.nBins_fit : (m + 1) * self.nBins_fit - 1] for m in range(nMixes)])
        ys = np.array([self.pulls[basis][m * self.nBins_fit + 1: (m + 1) * self.nBins_fit] for m in range(nMixes)])

        kwargs = {'lw': 0.5, 'marker': 'o', 'edgecolors': 'k', 's': 8, 'c': 'k', 'zorder': 2}

        fig, (ax) = plt.subplots(nrows=1, figsize=(6, 6))
        ax.set_aspect(1)
        ax.set_title('Adjacent Bin Pulls (%s, %d parameters)' % (self.channel.upper(), basis + 1))
        ax.set_xlabel('Bin$_{i}$, Pull')
        ax.set_ylabel('Bin$_{i + 1}$ Pull')

        lim_max = 1.5 * max(abs(xs).max(), abs(ys).max()) if len(xs) > 0 and xs.size > 0 else 5.0
        xlim, ylim = [-lim_max, lim_max], [-lim_max, lim_max]
        ax.plot(xlim, [0, 0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot([0, 0], ylim, color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        for m in range(nMixes):
            (r, p) = self.pearsonr[basis]['mixes'][m]
            kwargs['label'] = 'v$_{%d}$, r=%0.2f (%2.0f%s)' % (m, r, p * 100, r'\%')
            kwargs['c'] = COLORS[m % len(COLORS)]
            plt.scatter(xs[m], ys[m], **kwargs)
        plt.tight_layout()

        (r, p) = self.pearsonr[basis]['total']
        plt.legend(fontsize='small', loc='upper left', ncol=2, title='Overall r=%0.2f (%2.0f%s)' % (r, p * 100, r'\%'))
        fig.savefig(f'{output_dir}/0_variance_pull_correlation_basis{basis}.png')
        if getattr(args, 'save_all_formats', False):
            fig.savefig(f'{output_dir}/0_variance_pull_correlation_basis{basis}.pdf')
        plt.close(fig)

    def plotFit(self, basis):
        ROOTPlotTools = config.ROOTPlotTools
        args = config.args
        closure_file_out = config.closure_file_out
        color_multijet = config.color_multijet
        lumi = config.lumi
        regionName = config.regionName
        classifier = config.classifier
        nMixes = config.nMixes
        output_dir = config.output_dir

        samples = collections.OrderedDict()
        samples[closure_file_out] = collections.OrderedDict()
        samples[closure_file_out]['%s/multijet_ensemble_average' % self.channel] = {
            'label' : '#LTMultijet Model#GT' if not getattr(args, 'unify_background', False) else '#LTBackground Model#GT',
            'legend': 2,
            'isData' : True,
            'ratio' : 'denom A',
            'color' : 'ROOT.kBlack'}
        samples[closure_file_out]['%s/multijet_ensemble' % self.channel] = {
            'label' : 'Multijet Models' if not getattr(args, 'unify_background', False) else 'Background Models',
            'legend': 3,
            'stack' : 1,
            'ratio' : 'numer A',
            'color' : color_multijet}
        samples[closure_file_out]['%s/multijet_ensemble_TH1_basis%d' % (self.channel, basis)] = {
            'label' : 'Fit (%d parameter%s)' % (basis + 1, 's' if basis else ''),
            'legend': 4,
            'ratio' : 'numer A',
            'color' : 'ROOT.kBlue'}

        lumi_title = f"{lumi} fb^{{-1}} (13 TeV)"
        region_title = 'SR' if args.region == 'SR' else regionName.get(args.region, args.region)
        classifier_name = classifier.replace('_', ' ')
        xTitle = f'{classifier_name} P(Signal) #(Bin) + #(Bins)#(Mix) #cbar P({self.channel.upper()}) is largest'

        parameters = {'titleLeft'   : '#bf{CMS} #it{Internal}',
                      'titleCenter' : region_title,
                      'titleRight'  : lumi_title,
                      'canvasSize'  : [800, 667],
                      'maxDigits'   : 4,
                      'drawLines'   : [[self.nBins_rebin * m + 0.5, 0, self.nBins_rebin * m + 0.5, self.ymax[0] * 1.1] for m in range(1, nMixes + 1)],
                      'ratioErrors' : False,
                      'ratio'       : 'significance',
                      'rMin'        : -3,
                      'rMax'        :  3,
                      'rTitle'      : 'Pulls',
                      'xTitle'      : xTitle,
                      'yTitle'      : 'Events',
                      'logY'        : True,
                      'yMax'        : self.ymax[0] * 35.0,
                      'xTitleOffset': 0.95,
                      'xleg'        : [0.13, 0.13 + 0.40],
                      'yleg'        : [0.74, 0.90],
                      'legendSubText': ['#bf{Adjacent Bin Pull Correlation:}',
                                         'r = %1.2f' % (self.pearsonr[basis]['total'][0]),
                                         'p-value = %2.0f%%' % (self.pearsonr[basis]['total'][1] * 100),],
                      'lstLocation' : 'right',
                      'lstx'        : 0.56,
                      'lsty'        : 0.89,
                      'lst_yspace'  : 0.036,
                      'lst_textsize': 0.028,
                      'rPadFraction': 0.5,
                      'outputName'  : '0_variance_multijet_ensemble_basis%d' % (basis),
                      'save_all_formats': getattr(args, 'save_all_formats', False)}

        parameters['ratioLines'] = [[self.nBins_rebin * m + 0.5, parameters['rMin'], self.nBins_rebin * m + 0.5, parameters['rMax']] for m in range(1, nMixes + 1)]
        parameters['outputDir'] = output_dir

        ROOTPlotTools.plot(samples, parameters, debug=False)
