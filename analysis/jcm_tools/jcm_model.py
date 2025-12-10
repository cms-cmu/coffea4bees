#!/usr/bin/env python3
"""
Jet Combinatoric Model (JCM) Tools

This module provides the core functionality for the Jet Combinatoric Model
used in HH→4b analysis to model the combinatorial background from 3-tag events.
It contains the model parameter and fitting classes, along with helper functions
for data manipulation and model evaluation.

Author: Coffea4bees team
"""

import numpy as np
import logging
from scipy.special import comb
from scipy.optimize import curve_fit, minimize
import scipy.stats
from copy import copy
from typing import List, Tuple, Dict, Optional

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
                 qcd3b: np.ndarray, qcd3b_errors: np.ndarray, tt4b: np.ndarray, nbt: int = 3):
        """
        Initialize the JCM model.
        
        Args:
            tt4b_nTagJets: Number of tagged jets in tt 4-tag events
            tt4b_nTagJets_errors: Errors on tt 4-tag tagged jets
            qcd3b: QCD 3-tag events 
            qcd3b_errors: Errors on QCD 3-tag events
            tt4b: tt 4-tag events
            nbt: Number of baseline b-tags (default: 3 for standard, 0 for lowpt)
        """
        # Store input data
        self.tt4b_nTagJets = tt4b_nTagJets
        self.tt4b_nTagJets_errors = tt4b_nTagJets_errors
        self.qcd3b = qcd3b
        self.qcd3b_errors = qcd3b_errors
        self.tt4b = tt4b
        self.nbt = nbt

        # Initialize model parameters using configuration
        param_config = [
            ("pseudoTagProb", 0, 1, 0.05),
            ("pairEnhancement", 0, 3, 1.0),
            ("pairEnhancementDecay", 0.1, 100, 0.7),
            ("threeTightTagFraction", 0, 1000000, 1000),
        ]
        
        self.parameters = []
        for i, (name, lower, upper, default) in enumerate(param_config):
            param = {
                "name": name, "value": None, "error": None, "percentError": None,
                "index": i, "lowerLimit": lower, "upperLimit": upper, 
                "default": default, "fix": None
            }
            self.parameters.append(param)
            setattr(self, name, param)  # Create named attributes

        self.nParameters = len(self.parameters)
        self._reset_fit_parameters()
        
        # These will be set during fitting
        self.fit_chi2 = None
        self.fit_ndf = None
        self.fit_prob = None
        self.fit_errs = None

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
        """Fix parameters to specified values."""
        for name, val in zip(names, values):
            for p in self.parameters:
                if p["name"] == name:
                    p["fix"] = val
                    p["value"] = val
                    p["error"] = 0
                    break

    def _reset_fit_parameters(self) -> None:
        """Update fit parameters to only include unfixed ones."""
        self.fit_parameters = [p for p in self.parameters if p["fix"] is None]
        self.default_parameters = [p["default"] for p in self.fit_parameters]
        self.parameters_lower_bounds = [p["lowerLimit"] for p in self.fit_parameters]
        self.parameters_upper_bounds = [p["upperLimit"] for p in self.fit_parameters]

    def _get_current_parameters(self) -> List[float]:
        """Get current parameter values for model evaluation."""
        params = []
        for p in self.parameters:
            if p["fix"] is not None:
                params.append(p["fix"])
            elif p["value"] is not None:
                params.append(p["value"])
            else:
                raise ValueError(f"Parameter {p['name']} has no value. Run fit() first.")
        return params

    def _bkgd_func_njet_wrapper(self, x, *free_params, debug=False):
        """Wrapper that combines free fit parameters with fixed parameter values."""
        # Build the full parameter list by combining free and fixed values
        full_params = []
        free_idx = 0
        for param in self.parameters:
            if param["fix"] is not None:
                full_params.append(param["fix"])
            else:
                full_params.append(free_params[free_idx])
                free_idx += 1
        return self.bkgd_func_njet(x, *full_params, debug=debug)
    
    def fixParameter_combination(self, params_to_fix: Dict[str, float]) -> None:
        """
        Fix multiple parameters at once with specific values.
        
        Args:
            params_to_fix: Dictionary of parameter names and values to fix
                           e.g. {"threeTightTagFraction": 0.5, "pairEnhancement": 0.0}
        """
        # Fix the specified parameters
        self.fixParameters(list(params_to_fix.keys()), list(params_to_fix.values()))
        self._reset_fit_parameters()
        
        # Set up the constrained function
        self.bkgd_func_njet_constrained = self._bkgd_func_njet_wrapper
        
        logger.info(f"Fixed parameters: {', '.join([f'{n}={v}' for n, v in params_to_fix.items()])}")
        logger.info(f"Free parameters: {', '.join([p['name'] for p in self.fit_parameters])}")


    def bkgd_func_njet(self, x: np.ndarray, f: float, e: float, d: float, 
                       norm: float, debug: bool = False) -> np.ndarray:
        """
        Background model function for jet multiplicity.
        
        Args:
            x: Jet multiplicity bins
            f: pseudoTagProb parameter
            e: pairEnhancement parameter
            d: pairEnhancementDecay parameter
            norm: threeTightTagFraction parameter
            debug: Whether to print debug information
            
        Returns:
            Predicted values for each bin
        """
        nj = x.astype(int)
        output = np.zeros(len(x))

        # Add the n-tag component
        nTags = nj + 4
        nTags_pred_result = self.nTagPred(nTags, [f, e, d, norm])["values"]
        output[0:4] = nTags_pred_result[4:8]
        
        if debug:
            logger.debug(f"bkgd_func_njet output initial: {output}")

        # Add jet multiplicity component
        for ibin, this_nj in enumerate(nj):
            if this_nj < 4:
                continue

            w = np.sum(self.getPseudoTagProbs(this_nj, f, e, d, norm)[1:])
            output[this_nj] += w * self.qcd3b[this_nj] + self.tt4b[this_nj]

        if debug:
            logger.debug(f"bkgd_func_njet output final: {output}")
            
        return output

    def getPseudoTagProbs(self, nj: int, f: float, e: float = 0.0, d: float = 1.0, 
                        norm: float = 1.0) -> np.ndarray:
        """
        Calculate the pseudo-tag probabilities for a given jet multiplicity.
        
        Args:
            nj: Number of jets
            f: Pseudo-tag probability 
            e: Pair enhancement factor
            d: Pair enhancement decay parameter
            norm: Normalization factor
            
        Returns:
            Array of probabilities for each number of pseudo-tags
        """
        nbt = self.nbt  # Number of baseline b-tags
        nlt = nj - nbt  # Number of selected untagged jets ("light" jets)
        nPseudoTagProb = np.zeros(nlt + 1)

        for npt in range(0, nlt + 1):   # npt is the number of pseudoTags in this combination
            nt = nbt + npt
            nnt = nlt - npt  # Number of not tagged

            # (ways to choose npt pseudoTags from nlt light jets) * pseudoTagProb^nlt * (1-pseudoTagProb)^{nlt-npt}
            w_npt = norm * comb(nlt, npt, exact=True) * f**npt * (1 - f)**nnt

            # Apply pair enhancement for even number of tags
            if (nt % 2) == 0:
                w_npt *= 1 + e / nlt**d

            logger.debug(f"npt: {npt}, w_npt: {w_npt}, nt: {nt}, nlt: {nlt}")
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
        try:
            if scipy_optimize:
                # Define the objective function (sum of squared residuals)
                def objective_function(params):
                    model_values = self.bkgd_func_njet_constrained(bin_centers, *params)
                    return np.sum(((bin_values - model_values) / bin_errors)**2)
                
                result = minimize(
                    objective_function,
                    self.default_parameters,
                    bounds=list(zip(self.parameters_lower_bounds, self.parameters_upper_bounds)),
                    method='L-BFGS-B',
                    options={'maxiter': 5000}
                )
                popt = result.x
                
                # Get covariance from Hessian if available
                if hasattr(result, 'hess_inv'):
                    errs = np.array(result.hess_inv.todense() if hasattr(result.hess_inv, 'todense') else result.hess_inv)
                else:
                    errs = np.eye(len(popt)) * 0.001
                    logger.warning("Hessian not available, using default errors")
            else:
                # Use curve_fit which provides the covariance matrix directly
                popt, errs = curve_fit(
                    self.bkgd_func_njet_constrained,
                    bin_centers,
                    bin_values,
                    self.default_parameters,
                    sigma=bin_errors,
                    bounds=(self.parameters_lower_bounds, self.parameters_upper_bounds),
                )
        except Exception as e:
            logger.error(f"Fit failed: {e}")
            raise ValueError(f"Fit failed: {str(e)}")
            
        # Store the fit error matrix and calculate parameter errors
        self.fit_errs = errs
        sigma_p1 = [np.sqrt(np.abs(errs[i][i])) if i < len(errs) else 0.001 for i in range(len(popt))]

        # Update parameter values and errors
        for parameter in self.parameters:
            if parameter["fix"] is not None:
                parameter["value"] = parameter["fix"]
                parameter["error"] = 0
            elif parameter["index"] < len(popt):
                idx = parameter["index"]
                parameter["value"] = popt[idx]
                parameter["error"] = sigma_p1[idx]
        
        # Refresh fit_parameters list
        self.fit_parameters = [p for p in self.parameters if p["fix"] is None]

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
    
    def nTagPred(self, n: np.ndarray, par: Optional[List[float]] = None) -> np.ndarray:
        """
        Get predicted values for the number of tagged jets using current fit parameters.
        
        Args:
            n: Array of number of tags
            par: Optional parameter values. If None, uses current fitted values.
            
        Returns:
            Array of predicted values
        """
        if par is None:
            par = self._get_current_parameters()
            logger.info(f"Using parameters: {par}")

        output = copy(self.tt4b_nTagJets)
        f, e, d, norm = par

        for ibin, this_nTag in enumerate(n):
            for nj in range(this_nTag, 14):
                nPseudoTagProb = self.getPseudoTagProbs(nj, f, e, d, norm)
                logger.debug(f"nj: {nj}, this_nTag: {this_nTag}, nPseudoTagProb: {nPseudoTagProb}")
                output[ibin + 4] += nPseudoTagProb[this_nTag - 3] * self.qcd3b[nj]
                output[3] += nPseudoTagProb[0] * self.qcd3b[nj]
        
        logger.debug(f"output: {output}")
        return { "values": np.array(output, float), "errors": np.array(output**0.5, float) }

    def getCombinatoricWeightList(self) -> Tuple[List[float], List[float]]:
        """
        Get the list of combinatoric weights for jet multiplicities 4-15.
        
        Returns:
            Tuple of (output_weights, zerotag_output_weights) for each jet multiplicity
        """
        params = self._get_current_parameters()
        output_weights, zerotag_output_weights = [], []
        
        # Calculate weights for jet multiplicity 4 through 15
        for nj in range(4, 16):
            nj_pseudoTagProbs = self.getPseudoTagProbs(nj, *params)
            zerotag_output_weights.append(nj_pseudoTagProbs[0])
            output_weights.append(np.sum(nj_pseudoTagProbs[1:]))
        
        logger.info(f"Output weights: {output_weights}")
        return output_weights, zerotag_output_weights