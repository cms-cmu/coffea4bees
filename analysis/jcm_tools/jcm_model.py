#!/usr/bin/env python3
"""
Jet Combinatoric Model (JCM) Tools

This module provides the core functionality for the Jet Combinatoric Model
used in HH→4b analysis to model the combinatorial background from 3-tag events.
It contains the model parameter and fitting classes, along with helper functions
for data manipulation and model evaluation.
"""

from scipy.special import comb
import numpy as np
from coffea.util import load
import logging
from src.plotting.helpers import get_cut_dict
import src.plotting.iPlot_config as cfg
import hist
from copy import copy, deepcopy
from scipy.optimize import curve_fit, minimize
import scipy.stats
from typing import List, Tuple, Dict, Optional, Union, Any

# Set up module logger
logger = logging.getLogger('JCMTools')


class jetCombinatoricModel:
    """
    Main class for the Jet Combinatoric Model (JCM).

    The JCM is used to reweight the 3-tag multijet events to model
    the 4-tag multijet background. The model fits the jet multiplicity
    distribution to determine the weights.

    Attributes:
        parameters (List[Dict[str, Any]]): List of all model parameters
        pseudoTagProb (Dict[str, Any]): Probability of a light jet to be tagged
        pairEnhancement (Dict[str, Any]): Enhancement for even number of tags
        pairEnhancementDecay (Dict[str, Any]): Decay parameter for pair enhancement
        threeTightTagFraction (Dict[str, Any]): Normalizing parameter
        tt4b_nTagJets (np.ndarray): Number of tagged jets in tt 4-tag events
        tt4b_nTagJets_errors (np.ndarray): Errors on tt 4-tag tagged jets
        qcd3b (np.ndarray): QCD 3-tag events
        qcd3b_errors (np.ndarray): Errors on QCD 3-tag events
        tt4b (np.ndarray): tt 4-tag events
        fit_parameters (List[Dict[str, Any]]): Parameters used in the fit (unfixed only)
        default_parameters (List[float]): Default values for fit parameters
        parameters_lower_bounds (List[float]): Lower bounds for fit parameters
        parameters_upper_bounds (List[float]): Upper bounds for fit parameters
        nParameters (int): Number of parameters in the model
        fit_chi2 (float): Chi-squared of the fit
        fit_ndf (int): Number of degrees of freedom in the fit
        fit_prob (float): P-value of the fit
        fit_errs (np.ndarray): Error matrix from the fit
    """

    def __init__(self, *, tt4b_nTagJets: np.ndarray, tt4b_nTagJets_errors: np.ndarray,
                 qcd3b: np.ndarray, qcd3b_errors: np.ndarray, tt4b: np.ndarray, lowpt_mode: bool = False):
        """
        Initialize the JCM model.

        Args:
            tt4b_nTagJets: Number of tagged jets in tt 4-tag events (or lowpt tags in lowpt mode)
            tt4b_nTagJets_errors: Errors on tt 4-tag tagged jets
            qcd3b: QCD 3-tag events (or 3-tag + 0 lowpt-tag in lowpt mode)
            qcd3b_errors: Errors on QCD 3-tag events
            tt4b: tt 4-tag events (or 3-tag + lowpt in lowpt mode)
            lowpt_mode: Whether to use lowpt-tag mode (default: False)
        """
        self.lowpt_mode = lowpt_mode
        # Initialize model parameters with reasonable defaults and bounds
        self.pseudoTagProb = {
            "name": "pseudoTagProb", "value": None, "error": None, "percentError": None,
            "index": 0, "lowerLimit": 0, "upperLimit": 1, "default": 0.05, "fix": None
        }
        self.pairEnhancement = {
            "name": "pairEnhancement", "value": None, "error": None, "percentError": None,
            "index": 1, "lowerLimit": 0, "upperLimit": 3, "default": 1.0, "fix": None
        }
        self.pairEnhancementDecay = {
            "name": "pairEnhancementDecay", "value": None, "error": None, "percentError": None,
            "index": 2, "lowerLimit": 0.1, "upperLimit": 100, "default": 0.7, "fix": None
        }
        self.tt4bSF = {
            "name": "tt4bSF", "value": None, "error": None, "percentError": None,
            "index": 3, "lowerLimit": 0, "upperLimit": 10, "default": 1.66, "fix": None
        }
        self.threeTightTagFraction = {
            "name": "threeTightTagFraction", "value": None, "error": None, "percentError": None,
            "index": 4, "lowerLimit": 0, "upperLimit": 1000000, "default": 1000, "fix": None
        }


        self.parameters = [
            self.pseudoTagProb,
            self.pairEnhancement,
            self.pairEnhancementDecay,
            self.tt4bSF,
            self.threeTightTagFraction
        ]

        # Store input data
        self.tt4b_nTagJets = tt4b_nTagJets
        self.tt4b_nTagJets_errors = tt4b_nTagJets_errors
        self.qcd3b = qcd3b
        self.qcd3b_errors = qcd3b_errors
        self.tt4b = tt4b

        # Setup fit parameters
        self.default_parameters = []
        self.fit_parameters = []
        self.parameters_lower_bounds = []
        self.parameters_upper_bounds = []

        for p in self.parameters:
            self.fit_parameters.append(p)
            self.parameters_lower_bounds.append(p["lowerLimit"])
            self.parameters_upper_bounds.append(p["upperLimit"])
            self.default_parameters.append(p["default"])

        self.nParameters = len(self.parameters)

        # These will be set during fitting
        self.fit_chi2 = None
        self.fit_ndf = None
        self.fit_prob = None
        self.fit_errs = None

        # Function to use in fitting - will be set when fixing parameters
        self.bkgd_func_njet_constrained = None

    def dump(self) -> None:
        """Print all parameter values and their status."""
        for parameter in self.parameters:
            if parameter["value"] is not None and parameter["error"] is not None:
                parameter["percentError"] = parameter["error"] / parameter["value"] * 100 if parameter["value"] else 0
                logger.info(f"{parameter['name']}: {parameter['value']:.6f} +/- {parameter['error']:.6f} ({parameter['percentError']:.2f}%)")
            elif parameter["value"] is not None:
                logger.info(f"{parameter['name']}: {parameter['value']:.6f} (Fixed)")
            else:
                logger.info(f"{parameter['name']}: Not yet fitted or fixed")

    def fixParameters(self, names: List[str], values: List[float]) -> None:
        """
        Fix parameters to specified values.

        Args:
            names: List of parameter names to fix
            values: List of values to fix the parameters to
        """
        for ip, p in enumerate(self.parameters):
            for _iname, _name in enumerate(names):
                if p["name"] == _name:
                    logger.info(f"Fixing {_name} to {values[_iname]}")
                    p["fix"] = values[_iname]
                    p["value"] = values[_iname] # Also set the value when fixing
                    p["error"] = 0 # Error is 0 for fixed parameters

    def _reset_fit_parameters(self) -> None:
        """Reset the fit parameters after fixing some parameters."""
        self.fit_parameters = []
        self.default_parameters = []
        self.parameters_lower_bounds = []
        self.parameters_upper_bounds = []

        for p in self.parameters:
            if p["fix"] is not None:
                continue
            self.fit_parameters.append(p)
            self.default_parameters.append(p["default"])
            self.parameters_lower_bounds.append(p["lowerLimit"])
            self.parameters_upper_bounds.append(p["upperLimit"])

    def fixParameter_combination(self, params_to_fix: Dict[str, float]) -> None:
        """
        Fix multiple parameters at once with specific values, and set up the constrained function.
        """
        # Extract names and values for fixParameters
        names = list(params_to_fix.keys())
        values = list(params_to_fix.values())

        # Fix the specified parameters
        self.fixParameters(names, values)

        # Reset fit parameters
        self._reset_fit_parameters()

        # Determine which parameters are still free
        free_params = [p for p in self.parameters if p["fix"] is None]
        free_param_indices = [p["index"] for p in free_params]

        # Set up the background function with fixed parameters
        f, e, d, tt4bSF, norm  = 0.05, 0.0, 1.0, 1.0, 1.0

        # Update with fixed values
        if "pseudoTagProb" in params_to_fix:
            f = params_to_fix["pseudoTagProb"]
        if "pairEnhancement" in params_to_fix:
            e = params_to_fix["pairEnhancement"]
        if "pairEnhancementDecay" in params_to_fix:
            d = params_to_fix["pairEnhancementDecay"]
        if "tt4bSF" in params_to_fix:
            tt4bSF = params_to_fix["tt4bSF"]
        if "threeTightTagFraction" in params_to_fix:
            norm = params_to_fix["threeTightTagFraction"]

        # IMPORTANT: Capture lowpt_mode for the lambda
        lowpt_mode = self.lowpt_mode

        # Create lambda functions that pass lowpt parameter
        if len(free_params) == 1 and free_params[0]["name"] == "pseudoTagProb":
            self.bkgd_func_njet_constrained = lambda x, f_val, debug=False: self.bkgd_func_njet(x, f_val, e, d, tt4bSF, norm, debug, lowpt=lowpt_mode)
        elif len(free_params) == 2 and 0 in free_param_indices and 1 in free_param_indices:
            self.bkgd_func_njet_constrained = lambda x, f_val, e_val, debug=False: self.bkgd_func_njet(x, f_val, e_val, d, tt4bSF, norm, debug, lowpt=lowpt_mode)
        elif len(free_params) == 3 and 0 in free_param_indices and 1 in free_param_indices and 2 in free_param_indices:
            self.bkgd_func_njet_constrained = lambda x, f_val, e_val, d_val, debug=False: self.bkgd_func_njet(x, f_val, e_val, d_val, tt4bSF, norm, debug, lowpt=lowpt_mode)
        elif len(free_params) == 4 and 0 in free_param_indices and 1 in free_param_indices and 2 in free_param_indices and 3 in free_param_indices:
            self.bkgd_func_njet_constrained = lambda x, f_val, e_val, d_val, tt4bSF_val, debug=False: self.bkgd_func_njet(x, f_val, e_val, d_val, tt4bSF_val, norm, debug, lowpt=lowpt_mode)
        else:
            def create_constrained_func():
                def constrained_func(x, *args, debug=False):
                    full_params = [f, e, d, tt4bSF, norm ]
                    for i, param in enumerate(free_params):
                        full_params[param["index"]] = args[i]
                    return self.bkgd_func_njet(x, *full_params, debug=debug, lowpt=lowpt_mode)
                return constrained_func

            self.bkgd_func_njet_constrained = create_constrained_func()

        logger.info(f"Fixed parameters: {', '.join([f'{n}={v}' for n, v in params_to_fix.items()])}")
        logger.info(f"Free parameters: {', '.join([p['name'] for p in free_params])}")
        logger.info(f"Lowpt mode: {lowpt_mode}")

    def bkgd_func_njet(self, x: np.ndarray, f: float, e: float, d: float,
                       tt4bSF: float, norm: float, debug: bool = False, lowpt: bool = False) -> np.ndarray:
        """Background model function for jet multiplicity."""
        nj = x.astype(int)
        output = np.zeros(len(x))

        # Add the n-tag component
        if lowpt:
            # For lowpt: bins 0-3 represent 1, 2, 3, 4 lowpt tags
            # We need to call nTagPred for these tag numbers
            nTags_to_predict = np.array([1, 2, 3, 4])  # The actual tag numbers we want
            nTags_pred_result = self.nTagPred(nTags_to_predict, [f, e, d, tt4bSF, norm  ], lowpt=True)["values"]
            # nTags_pred_result[1] = prediction for 1 lowpt tag
            # nTags_pred_result[2] = prediction for 2 lowpt tags, etc.
            output[0] = nTags_pred_result[1]  # 1 lowpt tag
            output[1] = nTags_pred_result[2]  # 2 lowpt tags
            output[2] = nTags_pred_result[3]  # 3 lowpt tags
            output[3] = nTags_pred_result[4]  # 4 lowpt tags

            if debug:
                logger.debug(f"Lowpt tag predictions: bin 0 (1 tag)={output[0]:.2f}, " +
                            f"bin 1 (2 tags)={output[1]:.2f}, bin 2 (3 tags)={output[2]:.2f}, " +
                            f"bin 3 (4 tags)={output[3]:.2f}")
        else:
            # For standard: bins 0-3 represent 4, 5, 6, 7 tags
            nTags = nj + 4
            nTags_pred_result = self.nTagPred(nTags, [f, e, d, tt4bSF, norm  ], lowpt=False)["values"]
            output[0:4] = nTags_pred_result[4:8]

        if debug:
            logger.debug(f"After tag component: {output[:10]}")

        # Add jet multiplicity component
        if lowpt:
            # For lowpt: bins 4+ represent lowpt jet multiplicity
            # bin 4 = 1 lowpt jet, bin 5 = 2 lowpt jets, etc.
            for ibin, this_nj in enumerate(nj):
                if this_nj < 4:  # Skip the tag bins
                    continue

                # this_nj is the bin index: 4, 5, 6, ...
                # Actual number of lowpt jets: this_nj - 3 (bin 4 = 1 jet)
                nLowptJets = this_nj - 3

                # Get probability that these lowpt jets produce ≥1 lowpt tag
                w = np.sum(self.getPseudoTagProbs(nLowptJets, f, e, d, norm, lowpt=True)[1:])

                # qcd3b[this_nj] contains events with (this_nj - 3) lowpt jets
                # e.g., qcd3b[4] = events with 1 lowpt jet
                if self.tt4b is not None:

                    if this_nj < len(self.qcd3b) and this_nj < len(self.tt4b):
                        output[this_nj] += w * self.qcd3b[this_nj] + tt4bSF * self.tt4b[this_nj]

                        if debug:
                            logger.debug(f"Bin {this_nj} (nLowptJets={nLowptJets}): " +
                                    f"w={w:.4f}, qcd3b={self.qcd3b[this_nj]:.2f}, " +
                                    f"contribution={w * self.qcd3b[this_nj]:.2f}")
                else:
                    if this_nj < len(self.qcd3b):
                        output[this_nj] += w * self.qcd3b[this_nj]

                        if debug:
                            logger.debug(f"Bin {this_nj} (nLowptJets={nLowptJets}): " +
                                    f"w={w:.4f}, qcd3b={self.qcd3b[this_nj]:.2f}, " +
                                    f"contribution={w * self.qcd3b[this_nj]:.2f}")

        else:
            # Standard mode: bins 4+ represent jet multiplicity 4, 5, 6, ...
            for ibin, this_nj in enumerate(nj):
                if this_nj < 4:
                    continue

                w = np.sum(self.getPseudoTagProbs(this_nj, f, e, d, norm, lowpt=False)[1:])
                if self.tt4b is not None:
                    if this_nj < len(self.qcd3b) and this_nj < len(self.tt4b):
                        output[this_nj] += w * self.qcd3b[this_nj] + tt4bSF * self.tt4b[this_nj]
                else:
                    if this_nj < len(self.qcd3b):
                        output[this_nj] += w * self.qcd3b[this_nj]

        if debug:
            logger.debug(f"Final output: {output[:10]}")

        return output

    def getPseudoTagProbs(self, nj: int, f: float, e: float = 0.0, d: float = 1.0,
                          norm: float = 1.0, lowpt: bool = False) -> np.ndarray:
        """
        Calculate the pseudo-tag probabilities for a given jet multiplicity.

        Standard mode (lowpt=False):
            nj: Number of jets (≥4)
            Returns: Probability that N light jets (nj-3) become tags
            Pair enhancement when total tags (3 + N) is even

        Lowpt mode (lowpt=True):
            nj: Number of lowpt jets (≥1)
            Returns: Probability that N lowpt jets become lowpt tags
            Enhancement compounds with each lowpt tag: factor (1 + e/N^d)^k for k tags

        Args:
            nj: Number of jets (standard mode) or lowpt jets (lowpt mode)
            f: Pseudo-tag probability
            e: Pair enhancement factor
            d: Pair enhancement decay parameter
            norm: Normalization factor
            lowpt: Whether using lowpt mode

        Returns:
            Array of probabilities for each number of pseudo-tags
        """
        if lowpt:
            # Lowpt mode: nj is number of lowpt jets, all can potentially be tagged
            nLowptJets = nj
            nPseudoTagProb = np.zeros(nLowptJets + 1)

            for nLowptTags in range(0, nLowptJets + 1):
                nNotTagged = nLowptJets - nLowptTags

                # Combinatorial probability
                w_npt = norm * comb(nLowptJets, nLowptTags, exact=True) * f**nLowptTags * (1 - f)**nNotTagged

                # Apply pair enhancement that compounds with each lowpt tag.
                # Each additional lowpt tag multiplies the weight by (1 + e/N^d),
                # so the full enhancement for k lowpt tags is (1 + e/N^d)^k.
                # This naturally captures the growing correlation between lowpt jets:
                # soft b-jets from the same gluon->bb pair are increasingly likely
                # to all be tagged together.  k=0 gives factor 1 (no change).
                if nLowptTags > 0:
                    w_npt *= (1 + e / nLowptJets**d) ** nLowptTags

                logger.debug(f"lowpt mode: nLowptTags: {nLowptTags}, " +
                           f"w_npt: {w_npt:.6f}, enhancement: {(1 + e / nLowptJets**d) ** nLowptTags if nLowptTags > 0 else 1.0:.4f}")
                nPseudoTagProb[nLowptTags] += w_npt
        else:
            # Standard mode: original implementation
            nbt = 3  # Number of required b-tags
            nlt = nj - nbt  # Number of selected untagged jets ("light" jets)

            if nlt < 0:
                raise ValueError(f"Invalid nj={nj} for standard mode (must be ≥ 3)")

            nPseudoTagProb = np.zeros(nlt + 1)

            for npt in range(0, nlt + 1):   # npt is the number of pseudoTags in this combination
                nt = nbt + npt
                nnt = nlt - npt  # Number of not tagged

                # (ways to choose npt pseudoTags from nlt light jets) * pseudoTagProb^npt * (1-pseudoTagProb)^{nlt-npt}
                w_npt = norm * comb(nlt, npt, exact=True) * f**npt * (1 - f)**nnt

                # Apply pair enhancement for even number of tags
                if (nt % 2) == 0:
                    w_npt *= 1 + e / nlt**d

                logger.debug(f"standard mode: npt: {npt}, w_npt: {w_npt}, nt: {nt}, nlt: {nlt}")
                nPseudoTagProb[npt] += w_npt

        return nPseudoTagProb

    def fit(self, bin_centers: np.ndarray, bin_values: np.ndarray,
            bin_errors: np.ndarray, scipy_optimize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the fit of the JCM model to data.

        Args:
            bin_centers: Bin centers (jet multiplicities)
            bin_values: Bin values (event counts)
            bin_errors: Bin errors
            scipy_optimize: Whether to use scipy.optimize.minimize instead of curve_fit

        Returns:
            Tuple of (residuals, pulls)
        """
        if self.bkgd_func_njet_constrained is None:
            raise ValueError("Constrained function is not set. Call fixParameter_* first.")

        logger.info(f"Fitting with {len(self.fit_parameters)} free parameters")

        # Do the fit
        if scipy_optimize:
            # Define the objective function (sum of squared residuals)
            def objective_function(params):
                model_values = self.bkgd_func_njet_constrained(bin_centers, *params)
                residuals = (bin_values - model_values) / bin_errors
                return np.sum(residuals**2)

            # Perform the minimization
            try:
                result = minimize(
                    objective_function,
                    self.default_parameters,
                    bounds=list(zip(self.parameters_lower_bounds, self.parameters_upper_bounds)),
                    method='L-BFGS-B',  # Change to another minimizer if needed
                    options={'maxiter': 5000}
                )

                # Extract the optimized parameters
                popt = result.x

                # Extract the covariance matrix and compute errors
                if hasattr(result, 'hess_inv'):
                    try:
                        if hasattr(result.hess_inv, 'todense'):
                            errs = np.array(result.hess_inv.todense())
                        else:
                            errs = np.array(result.hess_inv)
                    except Exception as e:
                        logger.warning(f"Error converting Hessian: {e}")
                        errs = np.eye(len(popt)) * 0.001  # Fallback
                else:
                    errs = np.eye(len(popt)) * 0.001  # Fallback
                    logger.warning("Hessian not available, using default errors")
            except Exception as e:
                logger.error(f"Minimization failed: {e}")
                raise ValueError(f"Fit failed: {str(e)}")
        else:
            # Use curve_fit which provides the covariance matrix directly
            try:
                popt, errs = curve_fit(
                    self.bkgd_func_njet_constrained,
                    bin_centers,
                    bin_values,
                    self.default_parameters,
                    sigma=bin_errors,
                    bounds=(self.parameters_lower_bounds, self.parameters_upper_bounds),
                )
            except Exception as e:
                logger.error(f"Curve fit failed: {e}")
                raise ValueError(f"Fit failed: {str(e)}")

        # Store the fit error matrix
        self.fit_errs = errs

        # Calculate parameter errors from the covariance matrix diagonal
        sigma_p1 = []
        for i in range(len(popt)):
            try:
                sigma_p1.append(np.sqrt(np.abs(errs[i][i])))
            except (IndexError, ValueError) as e:
                logger.warning(f"Error calculating parameter error: {e}")
                sigma_p1.append(0.001)  # Default error

        # Update parameter values and errors. Iterate over self.fit_parameters
        # (the free-only list, which is in the same order popt was built from)
        # so popt[i] always lines up with fit_parameters[i] regardless of where
        # the gaps fall. Iterating over self.parameters with parameter["index"]
        # only worked when the free params happened to occupy indices [0..N-1].
        for popt_idx, parameter in enumerate(self.fit_parameters):
            parameter["value"] = popt[popt_idx]
            parameter["error"] = sigma_p1[popt_idx]

        # Fixed parameters already have value=fix from fixParameters(); ensure
        # error is zeroed for completeness.
        for parameter in self.parameters:
            if parameter["fix"] is not None:
                parameter["error"] = 0

        # Calculate fit quality metrics
        self.fit_chi2 = np.sum(
            (self.bkgd_func_njet_constrained(bin_centers, *popt) - bin_values)**2 / bin_errors**2
        )
        self.fit_ndf = len(bin_values) - len(popt)
        self.fit_prob = scipy.stats.chi2.sf(self.fit_chi2, self.fit_ndf)

        # Calculate residuals and pulls
        residuals = bin_values - self.bkgd_func_njet_constrained(bin_centers, *popt)
        pulls = residuals / bin_errors

        logger.info(f"Fit completed: chi^2/ndf = {self.fit_chi2:.2f}/{self.fit_ndf} = " +
                   f"{self.fit_chi2/self.fit_ndf:.2f}, p-value = {self.fit_prob:.6f}")

        return residuals, pulls

    def nJetPred_values(self, n: np.ndarray) -> np.ndarray:
        """
        Get predicted values for jet multiplicity using current fit parameters.

        Args:
            n: Array of jet multiplicities

        Returns:
            Array of predicted values
        """
        if self.bkgd_func_njet_constrained is None:
            raise ValueError("Constrained function is not set. Call fixParameter_* first.")

        param_values = [p["value"] for p in self.fit_parameters]
        if None in param_values:
            raise ValueError("One or more parameters have no value. Run fit() first.")

        return self.bkgd_func_njet_constrained(n, *param_values)

    def nTagPred(self, n: np.ndarray, par: Optional[List[float]] = None, lowpt: bool = False) -> Dict[str, np.ndarray]:
        """Get predicted values for the number of tagged jets."""
        if par is None:

            par = []
            for p in self.parameters:
                if p in self.fit_parameters:
                    par.append(p["value"])
                else:
                    par.append(p["fix"])
            logger.debug(f"Using parameters: {par}")


        # Initialize output with proper size
        if self.tt4b is not None:
            max_size = max(len(self.tt4b_nTagJets), len(self.qcd3b), max(n) + 1 if len(n) > 0 else 15)
        else:
            max_size = max(len(self.qcd3b), max(n) + 1 if len(n) > 0 else 15)
        output = np.zeros(max_size)

        # Copy tt4b baseline (this is the ttbar contribution)
        if self.tt4b_nTagJets is not None:
            output[:len(self.tt4b_nTagJets)] = par[3] * self.tt4b_nTagJets

        if lowpt:
            # Lowpt mode: n contains tag numbers [1, 2, 3, 4, ...]
            # We need to predict how many events have each number of lowpt tags
            # by summing over all lowpt jet multiplicities

            for ibin, this_nTag in enumerate(n):
                if this_nTag < 1:  # Skip 0-tag (not relevant for lowpt)
                    continue

                # Sum over all possible lowpt jet multiplicities
                for nLowptJets in range(this_nTag, 11):  # Need at least this_nTag jets to get this_nTag tags
                    nPseudoTagProb = self.getPseudoTagProbs(nLowptJets,
                                                            par[self.pseudoTagProb         ["index"]],
                                                            par[self.pairEnhancement       ["index"]],
                                                            par[self.pairEnhancementDecay  ["index"]],
                                                            par[self.threeTightTagFraction ["index"]],
                                                            lowpt=True)

                    # nPseudoTagProb[this_nTag] = probability of getting exactly this_nTag lowpt tags
                    # from nLowptJets lowpt jets

                    # qcd3b[nLowptJets + 3] contains events with nLowptJets (offset by 3 for histogram)
                    hist_bin = nLowptJets + 3
                    if this_nTag < len(nPseudoTagProb) and hist_bin < len(self.qcd3b):
                        contribution = nPseudoTagProb[this_nTag] * self.qcd3b[hist_bin]
                        output[this_nTag] += contribution

                        logger.debug(f"nTag={this_nTag}, nLowptJets={nLowptJets}, " +
                                f"prob={nPseudoTagProb[this_nTag]:.6f}, " +
                                f"qcd3b[{hist_bin}]={self.qcd3b[hist_bin]:.2f}, " +
                                f"contrib={contribution:.4f}")

                # Also add 0-tag events to output[0] if needed
                if this_nTag == 0:
                    for nLowptJets in range(1, 11):
                        nPseudoTagProb = self.getPseudoTagProbs(nLowptJets,
                                                                par[self.pseudoTagProb        ["index"]],
                                                                par[self.pairEnhancement      ["index"]],
                                                                par[self.pairEnhancementDecay ["index"]],
                                                                par[self.threeTightTagFraction["index"]],
                                                                lowpt=True)

                        hist_bin = nLowptJets + 3
                        if hist_bin < len(self.qcd3b):
                            output[0] += nPseudoTagProb[0] * self.qcd3b[hist_bin]
        else:
            # Standard mode: original implementation
            for ibin, this_nTag in enumerate(n):
                for nj in range(4, 14):
                    nPseudoTagProb = self.getPseudoTagProbs(nj,
                                                            par[self.pseudoTagProb        ["index"]],
                                                            par[self.pairEnhancement      ["index"]],
                                                            par[self.pairEnhancementDecay ["index"]],
                                                            par[self.threeTightTagFraction["index"]],
                                                            lowpt=False)

                    if this_nTag >= 4:
                        pseudotag_idx = this_nTag - 3
                        if pseudotag_idx < len(nPseudoTagProb) and nj < len(self.qcd3b):
                            output[this_nTag] += nPseudoTagProb[pseudotag_idx] * self.qcd3b[nj]

                    # 3-tag bin
                    if nj < len(self.qcd3b):
                        output[3] += nPseudoTagProb[0] * self.qcd3b[nj]

        logger.debug(f"nTagPred output: {output[:10]}")
        return {"values": np.array(output, float), "errors": np.array(output**0.5, float)}

    def getCombinatoricWeightList(self, lowpt: bool = False) -> Tuple[List[float], List[float]]:
        """
        Get the list of combinatoric weights for jet multiplicities.

        Args:
            lowpt: Whether using lowpt selection (affects jet multiplicity range)

        Returns:
            Tuple of (output_weights, zerotag_output_weights)
            For lowpt: weights for jet multiplicity 1-12
            For standard: weights for jet multiplicity 4-15
        """
        output_weights, zerotag_output_weights = [], []

        params = []
        for p in self.parameters:
            if p in self.fit_parameters:
                params.append(p["value"])
            else:
                params.append(p["fix"])


        # Calculate weights for appropriate jet multiplicity range
        if lowpt:
            # For lowpt: jet multiplicity 1 through 12
            jet_range = range(1, 13)
        else:
            # For standard: jet multiplicity 4 through 15
            jet_range = range(4, 16)

        for nj in jet_range:
            nj_pseudoTagProbs = self.getPseudoTagProbs(nj,
                                                       params[self.pseudoTagProb        ["index"]],
                                                       params[self.pairEnhancement      ["index"]],
                                                       params[self.pairEnhancementDecay ["index"]],
                                                       params[self.threeTightTagFraction["index"]],
                                                       lowpt=lowpt)

            zerotag_output_weights.append(nj_pseudoTagProbs[0])
            output_weights.append(np.sum(nj_pseudoTagProbs[1:]))

        logger.info(f"Output weights ({'lowpt' if lowpt else 'standard'}): {output_weights}")

        return output_weights, zerotag_output_weights
