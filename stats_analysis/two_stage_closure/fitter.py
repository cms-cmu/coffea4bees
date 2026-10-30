from __future__ import print_function
import os
import sys
import pickle
import collections
import array
import numpy as np

from . import config
from .metrics import fTest
from .io import print_log
from .plotting import plotMix

class closure:
    
    def __init__(self, f, channel, multijet):
        ROOT = config.ROOT
        args = config.args
        rebin = config.rebin
        rebin_histogram = config.rebin_histogram
        output_dir = config.output_dir
        mixes = config.mixes
        nMixes = config.nMixes
        closure_fit_x_min = config.closure_fit_x_min
        maxBasisClosure = config.maxBasisClosure
        probThreshold = config.probThreshold
        closure_file_out = config.closure_file_out

        self.channel = channel
        self.rebin = rebin
        self.multijet = multijet
        self.ttbar = f.Get('%s/ttbar' % self.channel)
        self.ttbar.SetName('%s_average_%s' % (self.ttbar.GetName(), self.channel))
        self.data_obs = f.Get('%s/data_obs' % self.channel).Clone()
        self.data_obs.SetDirectory(0)
        self.data_obs.SetName('%s_average_%s' % (self.data_obs.GetName(), self.channel))
        self.nBins = self.data_obs.GetSize() - 2

        self.output_yml = open(f'{output_dir}/1_bias_results.yml', 'w')

        self.doSpuriousSignal = False
        self.spuriousSignal = {}
        self.spuriousSignalError = {}
        self.closure_ss_zero_TH1 = {}
        self.closure_ss_TH1 = {}
        self.signal_orthogonal_TH1 = {}

        self.f = f
        self.f.cd(self.channel)

        self.ttbar_rebin = self.ttbar.Clone()
        self.ttbar_rebin.SetName('%s_rebin' % self.ttbar.GetName())
        if isinstance(self.rebin, array.array):
            self.ttbar_rebin = rebin_histogram(self.ttbar_rebin, self.rebin)
        else:
            self.ttbar_rebin.Rebin(self.rebin)
        self.data_obs_rebin = self.data_obs.Clone()
        self.data_obs_rebin.SetName('%s_rebin' % self.data_obs.GetName())
        if isinstance(self.rebin, array.array):
            self.data_obs_rebin = rebin_histogram(self.data_obs_rebin, self.rebin)
        else:
            self.data_obs_rebin.Rebin(self.rebin)
        self.nBins_rebin = self.data_obs_rebin.GetSize() - 2
        if self.nBins_rebin != self.multijet.nBins_rebin:
            raise ValueError(
                f"FATAL: Inconsistent binning between variance and bias tests for channel {self.channel}! "
                f"Variance test has {self.multijet.nBins_rebin} bins (from multijet.average_rebin), "
                f"while bias test has {self.nBins_rebin} bins (from data_obs_rebin). "
                f"Ensure base histograms in {closure_file_out} have identical number of bins!"
            )

        # Option 1: Account for un-averaged static ttbar_PSData across subsamples
        # Because ttbar_PSData is static across all pseudo-experiments, its statistical variance
        # was artificially divided by nMixes when data_obs was averaged via TH1::Add and Scale(1/nMixes).
        # We restore the full single-realization ttbar variance: sigma_new^2 = sigma_data^2 + (1 - 1/nMixes)*sigma_ttbar^2
        if not getattr(args, 'pure_qcd', False) and nMixes > 1 and self.ttbar_rebin is not None:
            print_log(f"[INFO] Option 1 (Static ttbar correction): Restoring un-averaged ttbar uncertainty for {nMixes} subsamples")
            for _bin in range(1, self.nBins_rebin + 1):
                err_data = self.data_obs_rebin.GetBinError(_bin)
                err_tt = self.ttbar_rebin.GetBinError(_bin)
                corr_var = (1.0 - 1.0 / nMixes) * (err_tt ** 2)
                new_err = (err_data ** 2 + corr_var) ** 0.5
                self.data_obs_rebin.SetBinError(_bin, new_err)

        self.bin_width = 1. / self.nBins_rebin
        self.fit_x_min = 0.5 + closure_fit_x_min / self.bin_width

        self.basis_element = self.multijet.basis_element
        self.basis_element_no_rebin = self.multijet.basis_element_no_rebin

        self.f.cd(self.channel)
        self.bases = range(-1, maxBasisClosure + 1)
        max_basis = max(self.bases[-1], self.multijet.basis)
        self.nBins_closure = self.nBins_rebin + max_basis + 1
        self.multijet_closure = ROOT.TH1F('multijet_closure', '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)
        self.ttbar_closure    = ROOT.TH1F('ttbar_closure',    '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)
        self.data_obs_closure = ROOT.TH1F('data_obs_closure', '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)
        self.signal_closure   = ROOT.TH1F('signal_closure',   '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)

        sem_diff_sq = np.zeros(self.nBins_rebin)
        diff_by_mix = []
        if getattr(args, 'include_ensemble_variance', False) and len(self.multijet.models_rebin) > 1:
            for m_idx, mix in enumerate(mixes):
                h_d = f.Get(f'{mix}/{self.channel}/data_obs')
                if h_d and not h_d.IsZombie() and m_idx < len(self.multijet.models_rebin):
                    h_d_r = h_d.Clone(f'{h_d.GetName()}_rebin_closure_eval')
                    h_d_r.SetDirectory(0)
                    if isinstance(self.rebin, array.array):
                        h_d_r = rebin_histogram(h_d_r, self.rebin)
                    else:
                        h_d_r.Rebin(self.rebin)
                    m_mr = self.multijet.models_rebin[m_idx]
                    diff_arr = [h_d_r.GetBinContent(b) - m_mr.GetBinContent(b) for b in range(1, self.nBins_rebin + 1)]
                    diff_by_mix.append(diff_arr)
            if len(diff_by_mix) > 1:
                diff_by_mix_arr = np.array(diff_by_mix)
                sem_diff_sq = np.var(diff_by_mix_arr, axis=0, ddof=1) / len(diff_by_mix)
                print_log(f"[INFO] Option B: Computed ensemble variance across {len(diff_by_mix)} pseudo-experiments.")
                for b_i in range(self.nBins_rebin):
                    print_log(f"       Bin {b_i+1}: SEM(diff) = {np.sqrt(sem_diff_sq[b_i]):.2f} (std = {np.sqrt(sem_diff_sq[b_i] * len(diff_by_mix)):.2f})")

        for _bin in range(1, self.nBins_rebin + 1):
            self.multijet_closure.SetBinContent(_bin, self.multijet.average_rebin.GetBinContent(_bin))
            if getattr(args, 'unify_background', False):
                self.ttbar_closure.SetBinContent(_bin, 0.0)
            else:
                self.ttbar_closure.SetBinContent(_bin, self.ttbar_rebin.GetBinContent(_bin))
            self.signal_closure  .SetBinContent(_bin, self.multijet.signal.GetBinContent(_bin))
            self.data_obs_closure.SetBinContent(_bin, self.data_obs_rebin.GetBinContent(_bin))

            self.multijet_closure.SetBinError(_bin, 0.0)
            self.ttbar_closure   .SetBinError(_bin, 0.0)
            self.signal_closure  .SetBinError(_bin, self.multijet.signal.GetBinError(_bin))
            if getattr(args, 'unify_background', False):
                error = (self.data_obs_rebin.GetBinError(_bin)**2 + self.multijet.average_rebin.GetBinError(_bin)**2 + (2.0 / nMixes)**2)**0.5
            else:
                error = (self.data_obs_rebin.GetBinError(_bin)**2 + self.ttbar_rebin.GetBinError(_bin)**2 + self.multijet.average_rebin.GetBinError(_bin)**2 + (2.0 / nMixes)**2)**0.5
            if getattr(args, 'include_ensemble_variance', False) and len(diff_by_mix) > 1:
                error = (error**2 + sem_diff_sq[_bin - 1])**0.5
            self.data_obs_closure.SetBinError(_bin, error)

        for _bin in range(self.nBins_rebin + 1, self.nBins_closure + 1):
            self.data_obs_closure.SetBinError(_bin, 1.0)

        self.f.cd(self.channel)
        self.multijet_closure.Write()
        self.ttbar_closure   .Write()
        self.data_obs_closure.Write()
        self.signal_closure  .Write()

        self.multijet_binned = ROOT.TH1F('multijet_binned', '', self.nBins_rebin, 0.5, 0.5 + self.nBins_rebin)
        self.ttbar_binned    = ROOT.TH1F('ttbar_binned',    '', self.nBins_rebin, 0.5, 0.5 + self.nBins_rebin)
        self.data_obs_binned = ROOT.TH1F('data_obs_binned', '', self.nBins_rebin, 0.5, 0.5 + self.nBins_rebin)
        self.signal_binned   = ROOT.TH1F('signal_binned',   '', self.nBins_rebin, 0.5, 0.5 + self.nBins_rebin)

        for _bin in range(1, self.nBins_rebin + 1):
            self.multijet_binned.SetBinContent(_bin, self.multijet.average_rebin.GetBinContent(_bin))
            self.multijet_binned.SetBinError  (_bin, self.multijet.average_rebin.GetBinError(_bin))
            if getattr(args, 'unify_background', False):
                self.ttbar_binned.SetBinContent(_bin, 0.0)
                self.ttbar_binned.SetBinError  (_bin, 0.0)
            else:
                self.ttbar_binned.SetBinContent(_bin, self.ttbar_rebin.GetBinContent(_bin))
                self.ttbar_binned.SetBinError  (_bin, self.ttbar_rebin.GetBinError(_bin))
            self.signal_binned  .SetBinContent(_bin, self.multijet.signal.GetBinContent(_bin))
            self.signal_binned  .SetBinError  (_bin, self.multijet.signal.GetBinError(_bin))
            self.data_obs_binned.SetBinContent(_bin, self.data_obs_rebin.GetBinContent(_bin))
            self.data_obs_binned.SetBinError  (_bin, self.data_obs_rebin.GetBinError(_bin))

        self.f.cd(self.channel)
        self.multijet_binned.Write()
        self.ttbar_binned.Write()
        self.data_obs_binned.Write()
        self.signal_binned.Write()

        for m, mix in enumerate(mixes):
            h_data_mix = self.f.Get(f'{mix}/{self.channel}/data_obs')
            if isinstance(self.rebin, array.array):
                h_data_mix = rebin_histogram(h_data_mix, self.rebin)
            elif int(self.rebin) > 1:
                h_data_mix = h_data_mix.Clone()
                h_data_mix.Rebin(int(self.rebin))
            h_mj_mix = self.multijet.models_rebin[m]

            h_d_b = ROOT.TH1F('data_obs_binned', '', self.nBins_rebin, 0.5, 0.5 + self.nBins_rebin)
            h_m_b = ROOT.TH1F('multijet_binned', '', self.nBins_rebin, 0.5, 0.5 + self.nBins_rebin)
            for _bin in range(1, self.nBins_rebin + 1):
                h_d_b.SetBinContent(_bin, h_data_mix.GetBinContent(_bin))
                h_d_b.SetBinError(_bin, h_data_mix.GetBinError(_bin))
                h_m_b.SetBinContent(_bin, h_mj_mix.GetBinContent(_bin))
                h_m_b.SetBinError(_bin, h_mj_mix.GetBinError(_bin))

            self.f.cd(f'{mix}/{self.channel}')
            h_d_b.Write()
            h_m_b.Write()

        self.fit_result = {}
        self.fit_result_ss = {}
        self.eigenVars = {}
        self.eigenVars_ss = {}
        self.closure_TF1, self.closure_TH1 = {}, {}
        self.pvalue, self.chi2, self.ndf = {}, {}, {}
        self.pvalue_ss, self.chi2_ss, self.ndf_ss = {}, {}, {}
        self.chi2_ss_zero, self.ndf_ss_zero = {}, {}
        self.ymax = {}
        self.fit_parameters, self.fit_parameters_error = {}, {}
        self.fit_parameters_ss, self.fit_parameters_error_ss = {}, {}
        self.cUp, self.cDown = {}, {}
        self.fProb = {-1: np.nan}
        self.fProb_ss = {}
        self.basis = None
        self.passed = False
        self.exit_message = ['--- NONE (%s) ---' % self.channel.upper()]

        for basis in self.bases:
            self.makeFitFunction(basis)
            self.fit(basis)
            self.write_to_yml(basis)
            self.fitSpuriousSignal(basis)
            self.plotFitResults(basis)
            max_basis = max(basis, self.multijet.basis)
            for j in range(1, max_basis):
                self.plotFitResults(basis, projection=(j, j + 1))

        for i, basis in enumerate(self.bases[:-1]):
            next_basis = self.bases[i + 1]
            print('fit f-test basis', next_basis)
            self.fProb[next_basis] = fTest(self.chi2[basis], self.chi2[next_basis], self.ndf[basis], self.ndf[next_basis])

            if self.basis is None and (self.pvalue[basis] > probThreshold) and (self.fProb[next_basis] < 0.95):
                self.passed = True
                self.exit_message = []
                print(self.pvalue)
                print(self.fProb)
                self.basis = basis
                self.exit_message.append('-' * 50)
                self.exit_message.append('%s channel' % self.channel.upper())
                self.exit_message.append('Satisfied goodness of fit and f-test')
                self.exit_message.append('>> %d, %d basis elements (variance, bias)' % (self.multijet.basis, self.basis))
                self.exit_message.append('>> p-value, f-test = %2.0f%%, %2.0f%% with %d basis elements (p-value above threshold and f-test prefers this fit over previous)' % (100 * self.pvalue[basis], 100 * self.fProb[basis], basis))
                self.exit_message.append('>> p-value, f-test = %2.0f%%, %2.0f%% with %d basis elements (f-test does not prefer this over previous fit at greater than 95%%)' % (100 * self.pvalue[next_basis], 100 * self.fProb[next_basis], next_basis))
                if self.fProb_ss[basis] < 0.95:
                    self.exit_message.append('>> SS f-test = %2.0f%%. Do not need to include spurious signal systematic :)' % (100 * self.fProb_ss[basis]))
                else:
                    self.exit_message.append('>> SS f-test = %2.0f%%! STRONG EVIDENCE FOR SPURIOUS SIGNAL SYSTEMATIC' % (100 * self.fProb_ss[basis]))
                self.exit_message.append('-' * 50)

        self.plotPValues()

        if self.basis is None:
            self.passed = False
            for i, b in enumerate(self.bases[:-1]):
                next_b = self.bases[i + 1]
                if self.fProb[next_b] < 0.95:
                    self.basis = b
                    break
            if self.basis is None:
                self.basis = self.bases[-1]

            self.exit_message = []
            self.exit_message.append('=' * 60)
            self.exit_message.append(f'ERROR: Closure bias test FAILED for channel {self.channel.upper()}!')
            self.exit_message.append(
                f'No basis order in range [{self.bases[0]}, {self.bases[-1]}] satisfied both '
                f'goodness-of-fit (p-value > {100 * probThreshold:.1f}%) and F-test (< 95%).'
            )
            self.exit_message.append(f'Goodness-of-fit threshold required: p-value > {100 * probThreshold:.1f}%')
            self.exit_message.append('Summary of fit results by basis:')
            for b in self.bases:
                f_str = f"{100 * self.fProb[b]:.1f}%" if (b in self.fProb and not np.isnan(self.fProb[b])) else "N/A"
                self.exit_message.append(
                    f'  >> Basis {b:2d}: chi2 = {self.chi2[b]:.2f}, ndf = {self.ndf[b]:2d}, '
                    f'p-value = {self.pvalue[b]:.2e}, F-test prob = {f_str}'
                )
            if getattr(args, 'ignore_failures', False):
                self.exit_message.append(f'WARNING: Continuing with fallback basis {self.basis} (--ignore_failures was specified).')
            self.exit_message.append('=' * 60)
        else:
            self.passed = True

        if self.passed or getattr(args, 'ignore_failures', False):
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
        ROOT = config.ROOT
        args = config.args
        nMixes = config.nMixes
        max_basis = max(basis, self.multijet.basis)

        def background_UserFunction(xArray, pars):
            this_bin = int(xArray[0])

            if this_bin > self.nBins_rebin:
                BE_idx = this_bin - self.nBins_rebin - 1
                if self.doSpuriousSignal:
                    if BE_idx > max_basis:
                        return 0.0
                    BE_coefficient = pars[BE_idx]
                    # prior_scale = 1.0 if getattr(args, 'no_sqrt_n_priors', False) else (nMixes**0.5)
                    prior_scale = 1.0
                    sigma_up = abs(self.cUp[basis][BE_idx]) / prior_scale
                    sigma_down = abs(self.cDown[basis][BE_idx]) / prior_scale
                    if BE_coefficient > 0:
                        return -BE_coefficient / sigma_up
                    else:
                        return -BE_coefficient / sigma_down

                BE_idx += basis + 1
                if BE_idx > self.multijet.basis:
                    return 0.0

                BE_coefficient = pars[BE_idx]
                # prior_scale = 1.0 if getattr(args, 'no_sqrt_n_priors', False) else (nMixes**0.5)
                prior_scale = 1.0
                sigma_up = abs(self.multijet.cUp[self.multijet.basis][BE_idx]) / prior_scale
                sigma_down = abs(self.multijet.cDown[self.multijet.basis][BE_idx]) / prior_scale
                if BE_coefficient > 0:
                    return -BE_coefficient / sigma_up
                else:
                    return -BE_coefficient / sigma_down

            p = 1.0
            n = max_basis + 1
            for BE_idx in range(n):
                p += pars[BE_idx] * self.basis_element[BE_idx][this_bin - 1]

            mj = self.multijet.average_rebin.GetBinContent(this_bin)
            if getattr(args, 'unify_background', False):
                background = p * mj
            else:
                background = p * mj + self.ttbar_rebin.GetBinContent(this_bin)
            spuriousSignal = pars[n] * self.multijet.signal.GetBinContent(this_bin)

            return background + spuriousSignal

        self.f.cd(self.channel)
        n = max_basis + 1
        self.pycallable = background_UserFunction
        self.closure_TF1[basis] = ROOT.TF1('closure_TF1_basis%d' % basis, self.pycallable, 0.5, 0.5 + self.nBins_closure, n + 1)
        self.closure_TH1[basis] = ROOT.TH1F('closure_TH1_basis%d' % basis, '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)
        self.closure_ss_zero_TH1[basis]   = ROOT.TH1F('closure_ss_zero_TH1_basis%d' % basis, '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)
        self.closure_ss_TH1[basis]        = ROOT.TH1F('closure_ss_TH1_basis%d' % basis, '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)
        self.signal_orthogonal_TH1[basis] = ROOT.TH1F('signal_orthogonal_TH1_basis%d' % basis, '', self.nBins_closure, 0.5, 0.5 + self.nBins_closure)

        for b in range(n):
            self.closure_TF1[basis].SetParName(b, 'c_%d' % b)
            self.closure_TF1[basis].SetParameter(b, 0.0)
        self.closure_TF1[basis].SetParName(n, 'spurious signal')
        self.closure_TF1[basis].FixParameter(n, 0)

    def getEigenvariations(self, basis, doSpuriousSignal=False, debug=False):
        ROOT = config.ROOT
        n = max(self.multijet.basis, basis) + 1
        if doSpuriousSignal:
            n += 1

        if n == 1:
            self.eigenVars[basis] = np.array([self.closure_TF1[basis].GetParError(0)])
            return

        cov = ROOT.TMatrixD(n, n)
        cor = ROOT.TMatrixD(n, n)
        fit_result = self.fit_result[basis] if not doSpuriousSignal else self.fit_result_ss[basis]
        for i in range(n):
            for j in range(n):
                cov[i][j] = fit_result.CovMatrix(i, j)
                cor[i][j] = fit_result.Correlation(i, j)

        if debug:
            print('Covariance Matrix:')
            cov.Print()
            print('Correlation Matrix:')
            cor.Print()

        eigenVal = ROOT.TVectorD(n)
        eigenVec = cov.EigenVectors(eigenVal)

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
                eigenVars[i, j] = eigenVec[i][j] * (eigenVal[j]**0.5)
        if not doSpuriousSignal:
            self.eigenVars[basis] = eigenVars
        else:
            self.eigenVars_ss[basis] = eigenVars

        if debug:
            print('Eigenvariations')
            for j in range(n):
                print(j, self.eigenVars[basis][:, j])

    def getParameterDistribution(self, basis):
        n = max(self.multijet.basis, basis) + 1
        self.cUp[basis], self.cDown[basis] = {}, {}
        for i in range(n):
            if i <= self.multijet.basis:
                cUp = (self.multijet.cUp[self.multijet.basis][i]**2 + self.fit_parameters[basis][i]**2 + self.fit_parameters_error[basis][i]**2)**0.5
            else:
                cUp = (self.fit_parameters[basis][i]**2 + self.fit_parameters_error[basis][i]**2)**0.5
            cDown = -cUp
            self.cUp[basis][i] = cUp
            self.cDown[basis][i] = cDown

    def fit(self, basis):
        args = config.args
        nMixes = config.nMixes
        n = max(self.multijet.basis, basis) + 1
        nConstrained = max(self.multijet.basis - basis, 0)
        nUnconstrained = n - nConstrained
        fit_x_max = self.nBins_rebin + 0.5 + nConstrained

        print("=" * 50)
        self.fit_result[basis] = self.data_obs_closure.Fit(self.closure_TF1[basis], 'N0S', '', self.fit_x_min, fit_x_max)
        self.getEigenvariations(basis)
        self.pvalue[basis], self.chi2[basis], self.ndf[basis] = self.closure_TF1[basis].GetProb(), self.closure_TF1[basis].GetChisquare(), self.closure_TF1[basis].GetNDF()
        print('Fit closure %s with %d basis elements. x_range = (%f, %f)' % (self.channel, basis, self.fit_x_min, fit_x_max))
        print('chi2/ndf = %3.2f/%3d = %2.2f' % (self.chi2[basis], self.ndf[basis], self.chi2[basis] / self.ndf[basis] if self.ndf[basis] > 0 else 0.0))
        print(' p-value = %0.2f' % self.pvalue[basis])
        print('nConstrained', nConstrained)
        print('nUnonstrained', nUnconstrained)
        print('expected ndfs = ', self.nBins_rebin - nUnconstrained)

        self.ymax[basis] = self.closure_TF1[basis].GetMaximum(1, self.nBins_closure)
        self.fit_parameters[basis], self.fit_parameters_error[basis] = [], []
        self.fit_parameters[basis] = np.array([self.closure_TF1[basis].GetParameter(b) for b in range(n)])
        self.fit_parameters_error[basis] = np.array([self.closure_TF1[basis].GetParError(b) for b in range(n)])
        self.getParameterDistribution(basis)

        for _bin in range(1, self.nBins_rebin + 1):
            self.closure_TH1[basis].SetBinContent(_bin, self.closure_TF1[basis].Eval(_bin))
            self.closure_TH1[basis].SetBinError(_bin, 0.0)

        for i in range(n):
            _bin = self.nBins_rebin + 1 + i
            c_val = self.fit_parameters[basis][i]
            if i > basis:
                # prior_scale = 1.0 if getattr(args, 'no_sqrt_n_priors', False) else (nMixes**0.5)
                prior_scale = 1.0
                sigma_prior = (abs(self.multijet.cUp[self.multijet.basis][i]) if c_val > 0 else abs(self.multijet.cDown[self.multijet.basis][i])) / prior_scale
                val = -c_val / sigma_prior if sigma_prior > 0 else (-c_val / self.fit_parameters_error[basis][i] if self.fit_parameters_error[basis][i] > 0 else 0.0)
            else:
                val = -c_val / self.fit_parameters_error[basis][i] if self.fit_parameters_error[basis][i] > 0 else 0.0
            self.closure_TH1[basis].SetBinContent(_bin, val)
            self.closure_TH1[basis].SetBinError(_bin, 0.0)

        for _bin in range(self.nBins_rebin + 1 + n, self.nBins_closure + 1):
            self.closure_TH1[basis].SetBinContent(_bin, 0.0)
            self.closure_TH1[basis].SetBinError(_bin, 0.0)

        self.f.cd(self.channel)
        self.closure_TH1[basis].Write()

    def fitSpuriousSignal(self, basis):
        max_basis = max(self.multijet.basis, basis)
        n = max_basis + 1
        self.closure_TF1[basis].FixParameter(n, 0)
        self.data_obs_closure.Fit(self.closure_TF1[basis], 'N0', '', self.fit_x_min, self.nBins_rebin + 0.5 + n)
        self.chi2_ss_zero[basis], self.ndf_ss_zero[basis] = self.closure_TF1[basis].GetChisquare(), self.closure_TF1[basis].GetNDF()
        for _bin in range(1, self.nBins_closure + 1):
            self.closure_ss_zero_TH1[basis].SetBinContent(_bin, self.closure_TF1[basis].Eval(_bin))
            self.closure_ss_zero_TH1[basis].SetBinError(_bin, 0.0)
        self.f.cd(self.channel)
        self.closure_ss_zero_TH1[basis].Write()

        self.closure_TF1[basis].SetParameter(n, 0)
        self.closure_TF1[basis].SetParLimits(n, -10, 10)
        self.fit_result_ss[basis] = self.data_obs_closure.Fit(self.closure_TF1[basis], 'N0S', '', self.fit_x_min, self.nBins_rebin + 0.5 + n)
        self.getEigenvariations(basis, doSpuriousSignal=True)
        self.spuriousSignal[basis]      = self.closure_TF1[basis].GetParameter(n)
        self.spuriousSignalError[basis] = self.closure_TF1[basis].GetParError(n)

        self.pvalue_ss[basis], self.chi2_ss[basis], self.ndf_ss[basis] = self.closure_TF1[basis].GetProb(), self.closure_TF1[basis].GetChisquare(), self.closure_TF1[basis].GetNDF()
        print('Fit spurious signal %s with %d basis elements' % (self.channel, basis))
        print('chi2/ndf = %3.2f/%3d = %2.2f' % (self.chi2_ss[basis], self.ndf_ss[basis], self.chi2_ss[basis] / self.ndf_ss[basis] if self.ndf_ss[basis] > 0 else 0.0))
        print(' p-value = %0.2f' % self.pvalue_ss[basis])

        print('SS f-test basis', basis)
        self.fProb_ss[basis] = fTest(self.chi2_ss_zero[basis], self.chi2_ss[basis], self.ndf_ss_zero[basis], self.ndf_ss[basis])

        self.fit_parameters_ss[basis] = np.array([self.closure_TF1[basis].GetParameter(b) for b in range(n + 1)])
        self.fit_parameters_error_ss[basis] = np.array([self.closure_TF1[basis].GetParError(b) for b in range(n + 1)])

        for _bin in range(1, self.nBins_closure + 1):
            self.closure_ss_TH1[basis].SetBinContent(_bin, self.closure_TF1[basis].Eval(_bin))
            self.signal_orthogonal_TH1[basis].SetBinContent(_bin, self.multijet.basis_signal[max_basis][_bin - 1] * self.multijet.average_rebin.GetBinContent(_bin) if _bin <= self.nBins_rebin else 0.0)
            self.closure_ss_TH1[basis].SetBinError(_bin, 0.0)
            self.signal_orthogonal_TH1[basis].SetBinError(_bin, 0.0)
        self.f.cd(self.channel)
        self.closure_ss_TH1[basis].Write()
        self.signal_orthogonal_TH1[basis].Write()

        self.closure_TF1[basis].FixParameter(n, 0)
        self.doSpuriousSignal = False
        print('spurious signal = %2.2f +/- %f' % (self.spuriousSignal[basis], self.spuriousSignalError[basis]))

    def writeClosureResults(self, basis=None):
        closure_file_out_pkl = config.closure_file_out_pkl
        channel = config.channel
        systematics = {}

        if basis is None:
            basis = self.basis

        max_basis = max(self.multijet.basis, basis)
        nBEs = max_basis + 1
        print('Write Closure Results File: \n>> %s' % (closure_file_out_pkl))
        for i in range(nBEs):
            nuissance = 'basis%i' % i
            print(i, "vs", len(self.multijet.cUp[self.multijet.basis]))

            if i < len(self.multijet.cUp[self.multijet.basis]):
                cUp_vari   = self.multijet.cUp[self.multijet.basis][i]
                cDown_vari = self.multijet.cDown[self.multijet.basis][i]
            else:
                cUp_vari = 0
                cDown_vari = 0

            cUp   = self.cUp[basis][i]
            cDown = self.cDown[basis][i]
            cUp_bias   = 0
            cDown_bias = 0
            if cUp != cUp_vari:
                cUp_bias   =  (cUp**2 - cUp_vari**2)**0.5
            if cDown != cDown_vari:
                cDown_bias = -(cDown**2 - cDown_vari**2)**0.5

            if cUp_vari or cDown_vari:
                systematics['%s_vari_%sUp' % (nuissance, self.channel)] = 1 + cUp_vari * self.basis_element[i]
                systematics['%s_vari_%sDown' % (nuissance, self.channel)] = 1 + cDown_vari * self.basis_element[i]

            if cUp_bias:
                systematics['%s_bias_%sUp' % (nuissance, self.channel)] = 1 + cUp_bias * self.basis_element[i]
            if cDown_bias:
                systematics['%s_bias_%sDown' % (nuissance, self.channel)] = 1 + cDown_bias * self.basis_element[i]

        if self.fProb_ss.get(basis, 0.0) >= 0.95:
            nuissance = 'spurious_signal'
            ssUp   = self.spuriousSignal[basis] + self.spuriousSignalError[basis]
            ssDown = self.spuriousSignal[basis] - self.spuriousSignalError[basis]
            systematics['%s_%sUp'  % (nuissance, channel)] = 1 + ssUp * self.multijet.basis_signal[max_basis]
            systematics['%s_%sDown' % (nuissance, channel)] = 1 + ssDown * self.multijet.basis_signal[max_basis]

        with open(closure_file_out_pkl, 'wb') as sfile:
            pickle.dump(systematics, sfile, protocol=1)

    def plotFitResults(self, basis, projection=(0, 1), doSpuriousSignal=False):
        plt = config.plt
        args = config.args
        output_dir = config.output_dir
        Ellipse = config.Ellipse
        if not config.HAS_MPL:
            return

        max_basis = max(self.multijet.basis, basis)
        n = max_basis + 1
        d_ss = n

        if doSpuriousSignal:
            n += 1

        if n > 1:
            dims = tuple(list(projection) + [d for d in range(n) if d not in projection])
        else:
            dims = (0, 1)

        labels = ['c$_' + str(d) + '$' for d in dims]
        if doSpuriousSignal:
            labels[dims.index(n - 1)] = r'$\zeta$'

        x, y, s, c = [], [], [], []
        parameters = self.fit_parameters[basis] if not doSpuriousSignal else self.fit_parameters_ss[basis]
        x.append(parameters[dims[0]] * (1 if dims[0] == d_ss else 100))
        if n == 1:
            y.append(0)
        if n > 1:
            y.append(parameters[dims[1]] * (1 if dims[1] == d_ss else 100))
        if n > 2:
            c.append(parameters[dims[2]] * (1 if dims[2] == d_ss else 100))
        if n > 3:
            s.append(parameters[dims[3]] * (1 if dims[3] == d_ss else 100))

        x = np.array(x)
        y = np.array(y)

        kwargs = {'lw': 0.5, 'marker': 'o', 'edgecolors': 'k', 's': 8, 'c': 'k', 'zorder': 2}

        fig, (ax) = plt.subplots(nrows=1, figsize=(6, 6))
        ax.set_aspect(1)
        if not doSpuriousSignal:
            ax.set_title('Multijet Model Bias Fit (%s)' % self.channel.upper())
        else:
            ax.set_title('Multijet Model Spurious Signal Fit (%s)' % self.channel.upper())

        ax.set_xlabel(labels[0] + ('' if dims[0] == d_ss else r' (\%)'))
        ax.set_ylabel(labels[1] + ('' if dims[1] == d_ss else r' (\%)'))

        xlim, ylim = [-10, 10], [-10, 10]
        ax.plot(xlim, [0, 0], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot([0, 0], ylim, color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        xticks = np.arange(xlim[0] + 2, xlim[1], 2)
        yticks = np.arange(ylim[0] + 2, ylim[1], 2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        if n > 1 and not doSpuriousSignal and Ellipse is not None:
            try:
                width = self.multijet.cUp[self.multijet.basis][dims[0]] - self.multijet.cDown[self.multijet.basis][dims[0]]
            except IndexError:
                width = 0.0
            try:
                height = self.multijet.cUp[self.multijet.basis][dims[1]] - self.multijet.cDown[self.multijet.basis][dims[1]]
            except IndexError:
                height = 0.0

            ellipse_self_consistency = Ellipse((0, 0),
                                               width =100 * width,
                                               height=100 * height,
                                               facecolor='none',
                                               edgecolor='b',
                                               linestyle='-',
                                               linewidth=0.75,
                                               zorder=1)
            ax.add_patch(ellipse_self_consistency)

            ellipse_closure = Ellipse((0, 0),
                                      width =100 * (self.cUp[basis][dims[0]] - self.cDown[basis][dims[0]]),
                                      height=100 * (self.cUp[basis][dims[1]] - self.cDown[basis][dims[1]]),
                                      facecolor='none',
                                      edgecolor='r',
                                      linestyle='-',
                                      linewidth=0.75,
                                      zorder=1)
            ax.add_patch(ellipse_closure)

        bbox = dict(boxstyle='round', facecolor='w', alpha=0.8, linewidth=0)
        if n > 2 and not doSpuriousSignal:
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
                    pass

        maxr = np.zeros((2, len(x)), dtype=float)
        minr = np.zeros((2, len(x)), dtype=float)

        if n > 1:
            points  = np.random.randn(n, min(100 * (n - 1), 10**7))
            points /= np.linalg.norm(points, axis=0)

            for i in range(len(x)):
                eigenVars = self.eigenVars[basis] if not doSpuriousSignal else self.eigenVars_ss[basis]
                plane = np.matmul(eigenVars[dims[0:2], :], points)
                r2 = plane[0]**2
                if n > 1:
                    r2 += plane[1]**2
                maxr[:, i] = plane[:, r2 == r2.max()].T[0]

                minrvec = np.copy(maxr[::-1, i])
                minrvec[0] *= -1
                minrvec /= np.linalg.norm(minrvec)

                dr2 = np.matmul(minrvec, plane)**2
                minr[:, i] = minrvec * (dr2.max()**0.5)
        else:
            for i in range(len(x)):
                maxr[0, i] = self.eigenVars[basis][dims[0]]

        if dims[0] != d_ss:
            minr[0] *= 100
            maxr[0] *= 100
        if dims[1] != d_ss:
            minr[1] *= 100
            maxr[1] *= 100

        ax.quiver(x, y,  maxr[0],  maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -maxr[0], -maxr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y,  minr[0],  minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)
        ax.quiver(x, y, -minr[0], -minr[1], scale_units='xy', angles='xy', scale=1, width=0.002, headlength=0, headaxislength=0, zorder=2)

        plt.scatter(x, y, **kwargs)

        if n > 2:
            for i in range(len(x)):
                label = '\n'.join(['%s = %2.1f%s' % (labels[dims.index(d)], parameters[d] * (1 if d == d_ss else 100), '' if d == d_ss else r'\%') for d in dims[2:]])
                xy = [xlim[-1] - 3, ylim[-1] - 1]
                ax.annotate(label, xy, ha='left', va='top', bbox=bbox)

        projection_str = '_'.join([str(d) for d in projection])
        if not doSpuriousSignal:
            name = f'{output_dir}/1_basis_parameters_basis{basis}_projection_{projection_str}.pdf'
        else:
            name = f'{output_dir}/2_spurious_signal_parameters_basis{basis}_projection_{projection_str}.pdf'

        plt.tight_layout()
        fig.savefig(name.replace('.pdf', '.png'))
        if getattr(args, 'save_all_formats', False):
            fig.savefig(name)
        plt.close(fig)

    def plotPValues(self):
        plt = config.plt
        args = config.args
        output_dir = config.output_dir
        probThreshold = config.probThreshold
        if not config.HAS_MPL:
            return

        fig, (ax) = plt.subplots(nrows=1)
        x = np.array(sorted(self.pvalue.keys())) + 1
        ax.set_ylim(0, 1)
        xlim = [x[0] - 0.5, x[-1] + .5]
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_xticks(x)

        y = [self.pvalue[i - 1] for i in x]
        ax.set_title('Multijet Model Bias Fit (%s)' % self.channel.upper())
        ax.plot(xlim, [probThreshold, probThreshold], color='b', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot(xlim, [0.95, 0.95], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.plot(x, y, label='p-value', color='b', linewidth=2)
        if self.basis is not None:
            ax.plot([self.basis + 1, self.basis + 1], [0, 1], color='k', alpha=0.5, linestyle='--', linewidth=0.5)
            ax.scatter(self.basis + 1, self.pvalue[self.basis], color='k', marker='*', s=100, zorder=10)

        x = np.array(sorted(self.fProb.keys())) + 1
        y = [self.fProb[i - 1] for i in x]
        marker = '' if len(x) > 1 else 'o'
        ax.plot(x, y, label='f-test', color='k', linewidth=2, marker=marker)

        x = np.array(sorted(self.fProb_ss.keys())) + 1
        y = [self.fProb_ss[i - 1] for i in x]
        marker = '' if len(x) > 1 else 'o'
        ax.plot(x, y, label='f-test Spurious Signal', color='k', linestyle='--', linewidth=2, marker=marker)

        ax.set_xlabel('Unconstrained Parameters')
        ax.set_ylabel('Fit p-value')
        ax.legend(loc='upper left', fontsize='small')

        plt.tight_layout()
        fig.savefig(f'{output_dir}/1_bias_pvalues.png')
        if getattr(args, 'save_all_formats', False):
            fig.savefig(f'{output_dir}/1_bias_pvalues.pdf')
        plt.close(fig)

    def plotMix(self, mix):
        ymax = self.ymax[0] if (hasattr(self, 'ymax') and 0 in self.ymax) else None
        plotMix(mix, self.channel, ymax=ymax)

    def plotFit(self, basis, plotSpuriousSignal=False):
        ROOTPlotTools = config.ROOTPlotTools
        args = config.args
        closure_file_out = config.closure_file_out
        color_TTbar = config.color_TTbar
        color_multijet = config.color_multijet
        lumi = config.lumi
        regionName = config.regionName
        classifier = config.classifier
        nMixes = config.nMixes
        output_dir = config.output_dir

        samples = collections.OrderedDict()
        samples[closure_file_out] = collections.OrderedDict()
        samples[closure_file_out]['%s/data_obs_closure' % self.channel] = {
            'label' : '#LTMixed Data#GT',
            'legend': 1,
            'isData' : True,
            'ratio' : 'numer A',
            'color' : 'ROOT.kBlack'}
        if not getattr(args, 'pure_qcd', False) and not getattr(args, 'unify_background', False):
            samples[closure_file_out]['%s/ttbar_closure' % self.channel] = {
                'label' : '#lower[0.10]{t#bar{t}}',
                'legend': 3,
                'stack' : 1,
                'ratio' : 'denom A',
                'color' : color_TTbar}
        samples[closure_file_out]['%s/multijet_closure' % self.channel] = {
            'label' : '#LTMultijet#GT' if not getattr(args, 'unify_background', False) else '#LTBackground#GT',
            'legend': 2,
            'stack' : 2,
            'ratio' : 'denom A',
            'color' : color_multijet}

        if not plotSpuriousSignal:
            samples[closure_file_out]['%s/closure_TH1_basis%d' % (self.channel, basis)] = {
                'label' : 'Fit (%d unconstrained parameter%s)' % (basis + 1, 's' if basis else ''),
                'legend': 4,
                'ratio': 'denom A',
                'color' : 'ROOT.kRed'}
        if plotSpuriousSignal:
            samples[closure_file_out]['%s/closure_ss_zero_TH1_basis%d' % (self.channel, basis)] = {
                'label' : 'Fit #zeta=0',
                'legend': 5,
                'ratio': 'denom A',
                'color' : 'ROOT.kGreen+3'}
            samples[closure_file_out]['%s/closure_ss_TH1_basis%d' % (self.channel, basis)] = {
                'label' : 'Fit #zeta=%1.1f#pm%1.1f' % (self.spuriousSignal[basis], self.spuriousSignalError[basis]),
                'legend': 6,
                'ratio': 'denom A',
                'color' : 'ROOT.kViolet'}
            sig_scale = getattr(args, 'signal_scale', None)
            if sig_scale is None:
                sig_scale = 1.0 if self.channel in ['ttHbb', 'tth'] else 100.0
            if sig_scale == 1.0:
                sig_label = 't#bar{t}H' if self.channel in ['ttHbb', 'tth'] else 'ZZ+ZH+HH'
            else:
                sig_label = f't#bar{{t}}H(#times{sig_scale:g})' if self.channel in ['ttHbb', 'tth'] else f'ZZ+ZH+HH(#times{sig_scale:g})'
            samples[closure_file_out][f'{self.channel}/signal_closure'] = {
                'label' : sig_label,
                'legend': 7,
                'weight': sig_scale,
                'color' : 'ROOT.kViolet+7'}

        ymaxScale = 50.0
        if plotSpuriousSignal:
            ymaxScale = 120.0
        lumi_title = f"{lumi} fb^{{-1}} (13 TeV)"
        region_title = 'SR' if args.region == 'SR' else regionName.get(args.region, args.region)
        classifier_name = classifier.replace('_', ' ')
        xTitle = f'{classifier_name} Classifier Regressed P(Signal) Bin'

        parameters = {'titleLeft'   : '#bf{CMS} #it{Internal}',
                      'titleCenter' : region_title,
                      'titleRight'  : lumi_title,
                      'canvasSize'  : [800, 667],
                      'maxDigits'   : 4,
                      'drawLines'   : [[self.fit_x_min, 0, self.fit_x_min, self.ymax[0] / 2],
                                       [self.nBins_rebin + 0.5, 0, self.nBins_rebin + 0.5, self.ymax[0] / 2]],
                      'ratioErrors' : False,
                      'ratio'       : 'significance',
                      'rMin'        : -5,
                      'rMax'        : 5,
                      'rTitle'      : 'Pulls',
                      'xTitle'      : xTitle,
                      'yTitle'      : 'Events',
                      'logY'        : True,
                      'yMax'        : self.ymax[0] * ymaxScale,
                      'xleg'        : [0.13, 0.13 + 0.40],
                      'yleg'        : [0.73, 0.90],
                      'lstLocation' : 'right',
                      'lstx'        : 0.56,
                      'lsty'        : 0.89,
                      'lst_yspace'  : 0.033,
                      'lst_textsize': 0.026,
                      'outputName'  : '%s_basis%d' % ('2_spurious_signal' if plotSpuriousSignal else '1_bias', basis),
                      'save_all_formats': getattr(args, 'save_all_formats', False)}

        n = max(self.multijet.basis, basis) + 1
        if plotSpuriousSignal:
            parameters['legendSubText'] = ['#bf{Spurious Signal Fit:}',
                                           '#chi^{2}/DoF = %2.1f/%d = %1.2f (#zeta=0)' % (self.chi2_ss_zero[basis], self.ndf_ss_zero[basis], self.chi2_ss_zero[basis] / self.ndf_ss_zero[basis] if self.ndf_ss_zero[basis] > 0 else 0.0),
                                           '#chi^{2}/DoF = %2.1f/%d = %1.2f' % (self.chi2_ss[basis], self.ndf_ss[basis], self.chi2_ss[basis] / self.ndf_ss[basis] if self.ndf_ss[basis] > 0 else 0.0),
                                           'p-value = %2.0f%% (f-test = %2.0f%%)' % (self.pvalue_ss[basis] * 100, self.fProb_ss[basis] * 100)]
            for i in range(n):
                c_val = self.fit_parameters_ss[basis][i]
                # prior_scale = 1.0 if getattr(args, 'no_sqrt_n_priors', False) else (nMixes**0.5)
                prior_scale = 1.0
                sigma_prior = (abs(self.cUp[basis][i]) if c_val > 0 else abs(self.cDown[basis][i])) / prior_scale
                sig_val = abs(c_val) / sigma_prior if sigma_prior > 0 else (abs(c_val) / self.fit_parameters_error_ss[basis][i] if self.fit_parameters_error_ss[basis][i] > 0 else 0.0)
                parameters['legendSubText'] += ['#font[82]{c_{%i} =%4.1f%% : %3.1f}#sigma' % (i, c_val * 100, sig_val)]
        else:
            parameters['legendSubText'] = ['#bf{Fit:}',
                                           '#chi^{2}/DoF = %2.1f/%d = %1.2f' % (self.chi2[basis], self.ndf[basis], self.chi2[basis] / self.ndf[basis] if self.ndf[basis] > 0 else 0.0),
                                           'p-value = %2.0f%%' % (self.pvalue[basis] * 100)]
            for i in range(n):
                c_val = self.fit_parameters[basis][i]
                if i > basis:
                    # prior_scale = 1.0 if getattr(args, 'no_sqrt_n_priors', False) else (nMixes**0.5)
                    prior_scale = 1.0
                    prior_sigma = (abs(self.multijet.cUp[self.multijet.basis][i]) if c_val > 0 else abs(self.multijet.cDown[self.multijet.basis][i])) / prior_scale
                    sig_val = abs(c_val) / prior_sigma if prior_sigma > 0 else (abs(c_val) / self.fit_parameters_error[basis][i] if self.fit_parameters_error[basis][i] > 0 else 0.0)
                    parameters['legendSubText'] += ['#color[4]{#font[82]{c_{%i} =%4.1f%% : %3.1f}#sigma}' % (i, c_val * 100, sig_val)]
                else:
                    sig_val = abs(c_val) / self.fit_parameters_error[basis][i] if self.fit_parameters_error[basis][i] > 0 else 0.0
                    parameters['legendSubText'] += ['#color[2]{#font[82]{c_{%i} =%4.1f%% : %3.1f}#sigma}' % (i, c_val * 100, sig_val)]

        parameters['ratioLines'] = [[self.fit_x_min, parameters['rMin'], self.fit_x_min, parameters['rMax']],
                                    [self.nBins_rebin + 0.5, parameters['rMin'], self.nBins_rebin + 0.5, parameters['rMax']]]
        parameters['xMax'] = self.nBins_rebin + 0.5 + max(self.multijet.basis, basis) + 1
        parameters['outputDir'] = output_dir

        ROOTPlotTools.plot(samples, parameters, debug=False)

    def print_exit_message(self):
        self.output_yml.close()
        for line in self.exit_message:
            print_log(line)
