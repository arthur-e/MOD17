'''
Calibration of MOD17 against a representative, global eddy covariance (EC)
flux tower network. The model calibration is based on Markov-Chain Monte
Carlo (MCMC). Example use:

    python calibration.py tune-gpp --pft=1

The general calibration protocol used here involves:

1. Check how well the chain(s) are mixing by running short:
`python calibration.py tune-gpp 1 --draws=2000`
2. If any chain is "sticky," run a short chain while tuning the jump scale:
`python calibration.py tune-gpp 1 --tune=scaling --draws=2000`
3. Using the trace plot from Step (2) as a reference, experiment with
different jump scales to try and achieve the same (optimal) mixing when
tuning on `lambda` (default) instead, e.g.:
`python calibration.py tune-gpp 1 --scaling=1e-2 --draws=2000`
4. When the right jump scale is found, run a chain at the desired length.

Once a good mixture is obtained, it is necessary to prune the samples to
eliminate autocorrelation, e.g., in Python:

    sampler = MOD17StochasticSampler(...)
    sampler.plot_autocorr(burn = 1000, thin = 10)
    trace = sampler.get_trace(burn = 1000, thin = 10)

A thinned posterior can be exported from the command line:

```py
$ python calibration.py export-posterior output.csv --burn=1000 --thin=10
```

References:

    Madani, N., Kimball, J. S., & Running, S. W. (2017).
      "Improving global gross primary productivity estimates by computing
      optimum light use efficiencies using flux tower data."
      Journal of Geophysical Research: Biogeosciences, 122(11), 2939–2951.
'''

import datetime
import json
import os
import warnings
import numpy as np
import h5py
import arviz as az
import pymc as pm
import pytensor.tensor as pt
import mod17
from functools import partial
from multiprocessing import get_context
from numbers import Number
from typing import Callable, Sequence
from pathlib import Path
from matplotlib import pyplot
from scipy import signal
from scipy.stats import gmean
from mod17 import MOD17, PFT_VALID
from mod17.utils import pft_dominant, restore_bplut, write_bplut, rmsd

MOD17_DIR = os.path.dirname(mod17.__file__)
# This matplotlib setting prevents labels from overplotting
pyplot.rcParams['figure.constrained_layout.use'] = True


class BlackBoxLikelihood(pt.Op):
    '''
    A custom Theano operator that calculates the "likelihood" of model
    parameters; it takes a vector of values (the parameters that define our
    model) and returns a single "scalar" value (the log-likelihood).

    Parameters
    ----------
    model : Callable
        An arbitrary "black box" function that takes two arguments: the
        model parameters ("params") and the forcing data ("x")
    observed : numpy.ndarray
        The "observed" data that our log-likelihood function takes in
    x : numpy.ndarray or None
        The forcing data (input drivers) that our model requires, or None
        if no driver data are required
    weights : Sequence or None
        Optional sequence of weights applied to the model residuals (as in
        weighted least squares)
    objective : str
        Name of the objective (or "loss") function to use, one of
        ('rmsd', 'gaussian', 'kge'); defaults to "rmsd"
    constraints : Sequence or None
        Sequence of one or more Callables (function) that return a competing
        value of the objective function (e.g., an RMSE). If there is more than
        one Callable, they are each called and the largest value is retained.
        If the (final) return value is greater than the value of the original
        objective function, than that value is returned instead. This is a way
        to tell the sampler that certain conditions are associated, e.g., with
        very high RMSE. Each Callable should take one argument: a vector of
        model predictions.
    '''
    itypes = [pt.dvector] # Expects a vector of parameter values when called
    otypes = [pt.dscalar] # Outputs a single scalar value (the log likelihood)

    def __init__(
            self, model: Callable, observed: Sequence, x: Sequence = None,
            weights: Sequence = None, objective: str = 'rmsd',
            constraints: Sequence = None):
        '''
        Initialise the Op with various things that our log-likelihood function
        requires. The observed data ("observed") and drivers ("x") must be
        stored on the instance so the Theano Op can work seamlessly.
        '''
        self.model = model
        self.observed = observed
        self.x = x
        self.weights = weights
        self.constraints = constraints
        if objective.lower() in ('rmsd', 'rmse'):
            self._loglik = self.loglik
        elif objective.lower() in ('nrmsd', 'nrmse'):
            self._loglik = self.loglik_norm
        elif objective.lower() in ('weighted_nrmsd', 'weighted_nrmse'):
            self._loglik = self.loglik_norm_weighted
        elif objective.lower() == 'gaussian':
            self._loglik = self.loglik_gaussian
        elif objective.lower() == 'kge':
            self._loglik = self.loglik_kge
        else:
            raise ValueError('Unknown "objective" function specified')

    def loglik(
            self, params: Sequence, observed: Sequence,
            x: Sequence = None) -> Number:
        '''
        Pseudo-log likelihood, based on the root-mean squared deviation
        (RMSD). The sign of the RMSD is forced to be negative so as to allow
        for maximization of this objective function.

        Parameters
        ----------
        params : Sequence
            One or more model parameters
        observed : Sequence
            The observed values
        x : Sequence or None
            Input driver data

        Returns
        -------
        Number
            The (negative) root-mean squared deviation (RMSD) between the
            predicted and observed values
        '''
        predicted = self.model(params, *x)
        if self.weights is not None:
            result = -np.sqrt(
                np.nanmean(((predicted - observed) * self.weights) ** 2))
        else:
            result = -np.sqrt(np.nanmean(((predicted - observed)) ** 2))
        if self.constraints is not None:
            if len(self.constraints) > 0:
                constrained_result = np.max(np.array([
                    func(predicted) for func in self.constraints
                ]))
                return np.max([result, constrained_result])
        return result

    def loglik_norm(
            self, params: Sequence, observed: Sequence,
            x: Sequence = None) -> Number:
        '''
        Pseudo-log likelihood, based on the normalized root-mean squared
        deviation (nRMSD, %). The sign of then RMSD is forced to be negative
        so as to allow for maximization of this objective function.

        Parameters
        ----------
        params : Sequence
            One or more model parameters
        observed : Sequence
            The observed values
        x : Sequence or None
            Input driver data

        Returns
        -------
        Number
            The (negative) root-mean squared deviation (RMSD) between the
            predicted and observed values
        '''
        predicted = self.model(params, *x)
        if self.weights is not None:
            result = -np.sqrt(
                np.nanmean(((predicted - observed) * self.weights) ** 2))
        else:
            result = -np.sqrt(np.nanmean(((predicted - observed)) ** 2))
        # Normalize RMSE by the range of the observed
        result = 100 * (result / (np.nanmax(observed) - np.nanmin(observed)))
        if self.constraints is not None:
            if len(self.constraints) > 0:
                constrained_result = np.max(np.array([
                    func(predicted) for func in self.constraints
                ]))
                # Geometric mean requires positive numbers
                return -gmean([-result, -constrained_result])
        return result

    def loglik_norm_weighted(
            self, params: Sequence, observed: Sequence,
            x: Sequence = None) -> Number:
        '''
        Pseudo-log likelihood, based on the (weighted) normalized root-mean
        squared deviation (nRMSD, %). The sign of then RMSD is forced to be
        negative so as to allow for maximization of this objective function.
        If constraints are not provided, the result is the same as
        `loglik_norm()`; otherwise, a weight on the constraint cost is
        applied, and the weight is assumed to be the last parameter of the
        input `params` Sequence.

        Parameters
        ----------
        params : Sequence
            One or more model parameters
        observed : Sequence
            The observed values
        x : Sequence or None
            Input driver data

        Returns
        -------
        Number
            The (negative) root-mean squared deviation (RMSD) between the
            predicted and observed values
        '''
        predicted = self.model(params[:-1], *x)
        if self.weights is not None:
            result = -np.sqrt(
                np.nanmean(((predicted - observed) * self.weights) ** 2))
        else:
            result = -np.sqrt(np.nanmean(((predicted - observed)) ** 2))
        # Normalize RMSE by the range of the observed
        result = 100 * (result / (np.nanmax(observed) - np.nanmin(observed)))
        if self.constraints is not None:
            if len(self.constraints) > 0:
                weight = params[-1] # The weight to put on joint constraints
                constrained_result = np.max(np.array([
                    func(predicted) for func in self.constraints
                ]))
                return -result + (weight * -constrained_result)
        return result

    def loglik_gaussian(
            self, params: Sequence, observed: Sequence,
            x: Sequence = None) -> Number:
        '''
        Gaussian log-likelihood, assuming independent, identically distributed
        observations.

        Parameters
        ----------
        params : Sequence
            One or more model parameters
        observed : Sequence
            The observed values
        x : Sequence or None
            Input driver data

        Returns
        -------
        Number
            The (negative) log-likelihood
        '''
        predicted = self.model(params, *x)
        sigma = params[-1]
        # Gaussian log-likelihood;
        # -\frac{N}{2}\,\mathrm{log}(2\pi\hat{\sigma}^2)
        #   - \frac{1}{2\hat{\sigma}^2} \sum (\hat{y} - y)^2
        return -0.5 * np.log(2 * np.pi * sigma**2) - (0.5 / sigma**2) *\
            np.nansum((predicted - observed)**2)

    def loglik_kge(
            self, params: Sequence, observed: Sequence,
            x: Sequence = None) -> Number:
        r'''
        Kling-Gupta efficiency. The best possible score is 1 and valid
        scores may be any smaller number (including negative numbers).

        $$
        KGE = 1 - \sqrt{(r - 1)^2 + (\alpha - 1)^2 + (\beta - 1)^2}
        $$

        Parameters
        ----------
        params : Sequence
            One or more model parameters
        observed : Sequence
            The observed values
        x : Sequence or None
            Input driver data

        Returns
        -------
        Number
            The Kling-Gupta efficiency
        '''
        predicted = self.model(params, *x)
        mask = np.isnan(observed)
        r = np.corrcoef(predicted[~mask], observed[~mask])[0, 1]
        alpha = np.nanstd(predicted) / np.nanstd(observed)
        beta = np.nanmean(predicted) / np.nanmean(observed)
        return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    def perform(self, node, inputs, outputs):
        '''
        The method that is used when calling the Op.

        Parameters
        ----------
        node
        inputs : Sequence
        outputs : Sequence
        '''
        (params,) = inputs
        logl = self._loglik(params, self.observed, self.x)
        outputs[0][0] = np.array(logl) # Output the log-likelihood


class AbstractSampler(object):
    '''
    Generic algorithm for fitting a model to data based on observed values
    similar to what we can produce with our model. Not intended to be called
    directly.
    '''

    def get_posterior(self, thin: int = 1) -> np.ndarray:
        '''
        Returns a stacked posterior array, with optional thinning, combining
        all the chains together.

        Parameters
        ----------
        thin : int

        Returns
        -------
        numpy.ndarray
        '''
        trace = az.from_netcdf(self.backend)
        return np.stack([ # i.e., get every ith element, each chain
            trace['posterior'][p].values[:,::thin].ravel()
            for p in self.required_parameters[self.name]
        ], axis = -1)

    def get_trace(
            self, thin: int = None, burn: int = None
        ) -> az.data.inference_data.InferenceData:
        '''
        Extracts the trace from the backend data store.

        Parameters
        ----------
        thin : int
            Thinning rate
        burn : int
            The burn-in (i.e., first N samples to discard)
        '''
        trace = az.from_netcdf(self.backend)
        if thin is None and burn is None:
            return trace
        return trace.sel(draw = slice(burn, None, thin))

    def plot_autocorr(
            self, thin: int = None, burn: int = None, title = None, **kwargs):
        '''
        Auto-correlation plot for an MCMC sample.

        Parameters
        ----------
        thin : int
            Thinning rate
        burn : int
            The burn-in (i.e., first N samples to discard)
        **kwargs
            Additional keyword arguments to `arviz.plot_autocorr()`.
        '''
        assert os.path.exists(self.backend),\
            'Could not find file backend!'
        trace = az.from_netcdf(self.backend)
        kwargs.setdefault('combined', True)
        if thin is None:
            az.plot_autocorr(trace, **kwargs)
        else:
            burn = 0 if burn is None else burn
            try:
                az.plot_autocorr(
                    trace.sel(draw = slice(burn, None, thin))['posterior'],
                    **kwargs)
            except ZeroDivisionError:
                raise ValueError('Cannot burn that many simples; reduce the burn-in')
        if title is not None:
            pyplot.title(title)
        pyplot.show()

    def plot_forest(self, **kwargs):
        '''
        Forest plot for an MCMC sample.

        In particular:

        - `hdi_prob`: A float indicating the highest density interval (HDF) to
            plot
        '''
        assert os.path.exists(self.backend),\
            'Could not find file backend!'
        trace = az.from_netcdf(self.backend)
        az.plot_forest(trace, **kwargs)
        pyplot.show()

    def plot_pair(self, **kwargs):
        '''
        Paired variables plot for an MCMC sample.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to `arviz.plot_pair()`.
        '''
        assert os.path.exists(self.backend),\
            'Could not find file backend!'
        trace = az.from_netcdf(self.backend)
        az.plot_pair(trace, **kwargs)
        pyplot.show()

    def plot_partial_score(
            self, observed: Sequence, drivers: Sequence, fit: dict = None):
        '''
        Plots the "partial effect" of a single parameter: the score of the
        model at that parameter's current value against a sweep of possible
        parameter values. All other parameters are held fixed at the best-fit
        values.

        Parameters
        ----------
        observed : Sequence
            Sequence of observed values that will be used to calibrate the model;
            i.e., model is scored by how close its predicted values are to the
            observed values
        drivers : list or tuple
            Sequence of driver datasets to be supplied, in order, to the
            model's run function
        fit : dict or None
            The best-fit parameter values used for those parameters that are
            fixed
        '''
        trace = az.from_netcdf(self.backend)
        if fit is None:
            # Mean of posterior are "best fit" values
            fit = trace['posterior'].mean()
        fit_params = list(filter(
            lambda p: p in fit, self.required_parameters[self.name]))
        # The NPP model depends on constants not included in the fit
        constants = []
        if self.name == 'NPP':
            constants = [
                self.params[p]
                for p in ['LUE_max', 'tmin0', 'tmin1', 'vpd0', 'vpd1']
            ]
        n = len(fit_params)
        nrow = n
        ncol = 1
        if n > 4:
            nrow = 2
            ncol = n - (n // 2)
        fig, axes = pyplot.subplots(
            nrow, ncol, figsize = (n * 2, n), sharey = True)
        i = 0
        for j in range(nrow):
            for k in range(ncol):
                if i >= n:
                    break
                free_param = fit_params[i]
                fixed = np.stack([
                    fit[p].values for p in fit_params
                ])[np.newaxis,:].repeat(30, axis = 0)
                sweep = np.linspace(
                    trace['posterior'][free_param].min(),
                    trace['posterior'][free_param].max(), num = 30)
                fixed[:,i] = sweep
                # Need to concatenate GPP parameters at begining of fixed
                scores = -np.array(self.score_posterior(
                    observed, drivers, [
                        [*constants, *f] for f in fixed.tolist()
                    ]))
                axes[j,k].plot(sweep, scores, 'k-')
                axes[j,k].axvline(
                    fit[free_param], color = 'red', linestyle = 'dashed',
                    label = 'Posterior Mean')
                axes[j,k].set_xlabel(free_param)
                axes[j,k].set_title(free_param)
                axes[j,k].legend()
                i += 1
        # Delete the last empty subplot
        if n % 2 != 0:
            fig.delaxes(axes.flatten()[-1])
        axes[0, 0].set_ylabel('Score')
        pyplot.tight_layout()
        pyplot.show()

    def plot_posterior(self, **kwargs):
        '''
        Plots the posterior distribution for an MCMC sample.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to `arviz.plot_posterior()`.
        '''
        assert os.path.exists(self.backend),\
            'Could not find file backend!'
        trace = az.from_netcdf(self.backend)
        az.plot_posterior(trace, **kwargs)
        pyplot.show()

    def score_posterior(
            self, observed: Sequence, drivers: Sequence, posterior: Sequence,
            method: str = 'rmsd') -> Number:
        '''
        Returns a goodness-of-fit score based on the existing calibration.

        Parameters
        ----------
        observed : Sequence
            Sequence of observed values that will be used to calibrate the model;
            i.e., model is scored by how close its predicted values are to the
            observed values
        drivers : list or tuple
            Sequence of driver datasets to be supplied, in order, to the
            model's run function
        posterior : list or tuple
            Sequence of posterior parameter sets (i.e., nested sequence); each
            nested sequence will be scored
        method : str
            The method for generating a goodness-of-git score
            (Default: "rmsd")

        Returns
        -------
        float
        '''
        if method != 'rmsd':
            raise NotImplementedError('"method" must be one of: "rmsd"')
        score_func = partial(
            rmsd, func = self.model, observed = observed, drivers = drivers)
        with get_context('spawn').Pool() as pool:
            scores = pool.map(score_func, posterior)
        return scores


class StochasticSampler(AbstractSampler):
    '''
    A Markov Chain-Monte Carlo (MCMC) sampler for an arbitrary model. The
    specific sampler used is the Differential Evolution (DE) MCMC algorithm
    described by Ter Braak (2008), though the implementation is specific to
    the PyMC3 library.

    NOTE: The `model` (a function) should be named "_name" where "name" is
    some uniquely identifiable model name. This helps `StochasticSampler.run()`
    to find the correct compiler for the model. The compiler function should
    be named `compiled_name_model()` (where "name" is the model name) and be
    defined on a subclass of `StochasticSampler`.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters
    model : Callable
        The function to call (with driver data and parameters); this function
        should take driver data as positional arguments and the model
        parameters as a `*Sequence`; it should require no external state.
    observed : Sequence
        Sequence of observed values that will be used to calibrate the model;
        i.e., model is scored by how close its predicted values are to the
        observed values
    params_dict : dict or None
        Dictionary of model parameters, to be used as initial values and as
        the basis for constructing a new dictionary of optimized parameters
    backend : str or None
        Path to a NetCDF4 file backend (Default: None)
    weights : Sequence or None
        Optional sequence of weights applied to the model residuals (as in
        weighted least squares)
    constraints : Sequence or None
        Sequence of one or more Callables (function) that return a competing
        value of the objective function (e.g., an RMSE). If there is more than
        one Callable, they are each called and the largest value is retained.
        If the (final) return value is greater than the value of the original
        objective function, than that value is returned instead. This is a way
        to tell the sampler that certain conditions are associated, e.g., with
        very high RMSE. Each Callable should take one argument: a vector of
        model predictions.
    '''
    def __init__(
            self, config: dict, model: Callable, params_dict: dict = None,
            backend: str = None, weights: Sequence = None,
            model_name: str = None, constraints: Sequence = None):
        self.backend = backend
        self.config = config
        self.constraints = constraints
        self.model = model
        if hasattr(model, '__name__'):
            self.name = model.__name__.strip('_').upper() # "_gpp" = "GPP"
        else:
            self.name = model_name
        self.params = params_dict
        # Set the model's prior distribution assumptions and any fixed values
        self.fixed = dict()
        self.prior = dict()
        for key in self.required_parameters[self.name]:
            # NOTE: This is the default only for LUE_max; other parameters,
            #   with Uniform distributions, don't use anything here
            self.prior.setdefault(key, {
                'mu': params_dict[key],
                'sigma': 2e-4
            })
        self.weights = weights
        assert os.path.exists(os.path.dirname(backend))

    def run(
            self, observed: Sequence, drivers: Sequence,
            draws = 1000, chains = 3, tune = 'lambda', scaling: float = 1e-3,
            prior: dict = dict(), fixed: dict = dict(),
            check_shape: bool = False, save_fig: bool = False,
            show_fig: bool = True, var_names: Sequence = None) -> None:
        '''
        Fits the model using DE-MCMCz approach. `tune="lambda"` (default) is
        recommended; lambda is related to the scale of the jumps learned from
        other chains, but epsilon ("scaling") controls the scale directly.
        Using a larger value for `scaling` (Default: 1e-3) will produce larger
        jumps and may directly address "sticky" chains.

        Parameters
        ----------
        observed : Sequence
            The observed data the model will be calibrated against
        drivers : list or tuple
            Sequence of driver datasets to be supplied, in order, to the
            model's run function
        draws : int
            Number of samples to draw (on each chain); defaults to 1000
        chains : int
            Number of chains; defaults to 3
        tune : str or None
            Which hyperparameter to tune: Defaults to 'lambda', but can also
            be 'scaling' or None.
        scaling : float
            Initial scale factor for epsilon (Default: 1e-3)
        prior : dict
            Dictionary of parameters and their prior values;
            should be of the form `{parameter: value}`
        fixed : dict
            Dictionary of parameters for which a fixed value should be used;
            should be of the form `{parameter: value}`
        check_shape : bool
            True to require that input driver datasets have the same shape as
            the observed values (Default: False)
        save_fig : bool
            True to save figures to files instead of showing them
            (Default: False)
        show_fig: bool
            True to show the trace plot at the end of a run (Default: True)
        var_names : Sequence
            One or more variable names to show in the plot
        '''
        assert not check_shape or drivers[0].shape == observed.shape,\
            'Driver data should have the same shape as the "observed" data'
        assert len(drivers) == len(self.required_drivers[self.name]),\
            'Did not receive expected number of driver datasets!'
        assert tune in ('lambda', 'scaling') or tune is None
        self.fixed.update(fixed) # Update parameters with fixed values
        self.prior.update(prior) # Update prior assumptions
        # Generate an initial goodness-of-fit score
        predicted = self.model([
            self.params[p] for p in self.required_parameters[self.name]
        ], *drivers)
        if self.weights is not None:
            score = np.sqrt(
                np.nanmean(((predicted - observed) * self.weights) ** 2))
        else:
            score = np.sqrt(np.nanmean(((predicted - observed)) ** 2))
        print('-- RMSD at the initial point: %.3f' % score)
        print('Compiling model...')
        try:
            compiler = getattr(self, 'compile_%s_model' % self.name.lower())
        except AttributeError:
            raise AttributeError('''Could not find a compiler for model named
            "%s"; make sure that a function "compile_%s_model()" is defined on
             this class''' % (model_name, model_name))
        with compiler(observed, drivers) as model:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                step_func = pm.DEMetropolisZ(tune = tune, scaling = scaling)
                trace = pm.sample(
                    draws = draws, step = step_func, cores = chains,
                    chains = chains, idata_kwargs = {'log_likelihood': True})
            if self.backend is not None:
                print('Writing results to file...')
                trace.to_netcdf(self.backend)
            if var_names is None:
                az.plot_trace(trace, var_names = ['~log_likelihood'])
            else:
                az.plot_trace(trace, var_names = var_names)
            if save_fig:
                pyplot.savefig('.'.join(self.backend.split('.')[:-1]) + '.png')
            elif show_fig:
                pyplot.show()


class MOD17StochasticSampler(StochasticSampler):
    '''
    A Markov Chain-Monte Carlo (MCMC) sampler for MOD17. The specific sampler
    used is the Differential Evolution (DE) MCMC algorithm described by
    Ter Braak (2008), though the implementation is specific to the PyMC3
    library.

    Considerations:

    1. Tower GPP is censored when values are < 0 or when APAR is
        < 0.1 MJ m-2 d-1.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters
    model : Callable
        The function to call (with driver data and parameters); this function
        should take driver data as positional arguments and the model
        parameters as a `*Sequence`; it should require no external state.
    observed : Sequence
        Sequence of observed values that will be used to calibrate the model;
        i.e., model is scored by how close its predicted values are to the
        observed values
    params_dict : dict or None
        Dictionary of model parameters, to be used as initial values and as
        the basis for constructing a new dictionary of optimized parameters
    backend : str or None
        Path to a NetCDF4 file backend (Default: None)
    weights : Sequence or None
        Optional sequence of weights applied to the model residuals (as in
        weighted least squares)
    '''
    # NOTE: This is different than for mod17.MOD17 because we haven't yet
    #   figured out how the respiration terms are calculated
    required_parameters = {
        'GPP': ['LUE_max', 'tmin0', 'tmin1', 'vpd0', 'vpd1'],
        'NPP': MOD17.required_parameters
    }
    required_drivers = {
        'GPP': ['fPAR', 'Tmin', 'VPD', 'PAR'],
        'NPP': ['fPAR', 'Tmin', 'VPD', 'PAR', 'LAI', 'Tmean', 'years']
    }

    def compile_gpp_model(
            self, observed: Sequence, drivers: Sequence) -> pm.Model:
        '''
        Creates a new GPP model based on the prior distribution. Model can be
        re-compiled multiple times, e.g., for cross validation.

        Parameters
        ----------
        observed : Sequence
            Sequence of observed values that will be used to calibrate the model;
            i.e., model is scored by how close its predicted values are to the
            observed values
        drivers : list or tuple
            Sequence of driver datasets to be supplied, in order, to the
            model's run function

        Returns
        -------
        pm.Model
        '''
        # Define the objective/ likelihood function
        log_likelihood = BlackBoxLikelihood(
            self.model, observed, x = drivers, weights = self.weights,
            objective = self.config['optimization']['objective'].lower())
        # With this context manager, "all PyMC3 objects introduced in the indented
        #   code block...are added to the model behind the scenes."
        with pm.Model() as model:
            # (Stochstic) Priors for unknown model parameters
            LUE_max = pm.TruncatedNormal('LUE_max', **self.prior['LUE_max'])
            # NOTE: tmin0, vpd0 are fixed at Collection 6.1 values
            tmin0 = self.params['tmin0']
            tmin1 = pm.Uniform('tmin1', **self.prior['tmin1'])
            # NOTE: Upper bound on `vpd1` is set by the maximum observed VPD
            vpd0 = self.params['vpd0']
            vpd1 = pm.Uniform('vpd1',
                lower = self.prior['vpd1']['lower'],
                upper = drivers[2].max().round(0))
            # Convert model parameters to a tensor vector
            params_list = [LUE_max, tmin0, tmin1, vpd0, vpd1]
            params = pt.as_tensor_variable(params_list)
            # Key step: Define the log-likelihood as an added potential
            pm.Potential('likelihood', log_likelihood(params))
        return model

    def compile_npp_model(
            self, observed: Sequence, drivers: Sequence) -> pm.Model:
        '''
        Creates a new NPP model based on the prior distribution. Model can be
        re-compiled multiple times, e.g., for cross validation.

        Parameters
        ----------
        observed : Sequence
            Sequence of observed values that will be used to calibrate the model;
            i.e., model is scored by how close its predicted values are to the
            observed values
        drivers : list or tuple
            Sequence of driver datasets to be supplied, in order, to the
            model's run function

        Returns
        -------
        pm.Model
        '''
        # Define the objective/ likelihood function
        log_likelihood = BlackBoxLikelihood(
            self.model, observed, x = drivers, weights = self.weights,
            objective = self.config['optimization']['objective'].lower())
        # With this context manager, "all PyMC3 objects introduced in the indented
        #   code block...are added to the model behind the scenes."
        with pm.Model() as model:
            # Setting GPP parameters that are known--EXCEPT tmin1
            LUE_max = self.params['LUE_max']
            tmin0   = self.params['tmin0']
            tmin1   = self.params['tmin1']
            vpd0    = self.params['vpd0']
            vpd1    = self.params['vpd1']
            # SLA fixed at prior mean
            SLA = np.exp(self.prior['SLA']['mu'])
            # Allometry ratios prescribe narrow range around Collection 6.1 values
            froot_leaf_ratio = pm.Triangular(
                'froot_leaf_ratio', **self.prior['froot_leaf_ratio'])
            # (Stochstic) Priors for unknown model parameters
            Q10_froot = pm.TruncatedNormal(
                'Q10_froot', **self.prior['Q10_froot'], **self.bounds['Q10'])
            leaf_mr_base = pm.LogNormal(
                'leaf_mr_base', **self.prior['leaf_mr_base'])
            froot_mr_base = pm.LogNormal(
                'froot_mr_base', **self.prior['froot_mr_base'])
            # For GRS and CRO, livewood mass and respiration are zero
            if np.equal(list(self.prior['livewood_mr_base'].values()), [0, 0]).all():
                livewood_leaf_ratio = 0
                livewood_mr_base = 0
                Q10_livewood = 0
            else:
                livewood_leaf_ratio = pm.Triangular(
                    'livewood_leaf_ratio', **self.prior['livewood_leaf_ratio'])
                livewood_mr_base = pm.LogNormal(
                    'livewood_mr_base', **self.prior['livewood_mr_base'])
                Q10_livewood = pm.TruncatedNormal(
                    'Q10_livewood', **self.prior['Q10_livewood'],
                    **self.bounds['Q10'])
            # Convert model parameters to a tensor vector
            params_list = [
                LUE_max, tmin0, tmin1, vpd0, vpd1, SLA,
                Q10_livewood, Q10_froot, froot_leaf_ratio, livewood_leaf_ratio,
                leaf_mr_base, froot_mr_base, livewood_mr_base
            ]
            params = pt.as_tensor_variable(params_list)
            # Key step: Define the log-likelihood as an added potential
            pm.Potential('likelihood', log_likelihood(params))
        return model


class CalibrationAPI(object):
    '''
    Convenience class for calibrating the MOD17 GPP and NPP models. Meant to
    be used with `fire.Fire()`.
    '''
    def __init__(self, config = None):
        config_file = config
        if config_file is None:
            config_file = os.path.join(
                MOD17_DIR, 'data/MOD17_calibration_config.json')
        with open(config_file, 'r') as file:
            self.config = json.load(file)
        self.hdf5 = self.config['data']['file']

    def _clean(self, raw: Sequence, drivers: Sequence, protocol: str = 'GPP'):
        'Cleans up data values according to a prescribed protocol'
        if protocol == 'GPP':
            # Filter out observed GPP values when GPP is negative or when
            #   APAR < 0.1 g C m-2 day-1
            apar = drivers['fPAR'] * drivers['PAR']
            return np.where(
                apar < 0.1, np.nan, np.where(raw < 0, np.nan, raw))

    def _filter(self, raw: Sequence, size: int):
        'Apply a smoothing filter with zero phase offset'
        if size > 1:
            window = np.ones(size) / size
            return np.apply_along_axis(
                lambda x: signal.filtfilt(window, np.ones(1), x), 0, raw)
        return raw # Or, revert to the raw data

    def clean_observed(
            self, raw: Sequence, drivers: Sequence, driver_labels: Sequence,
            protocol: str = 'GPP', filter_length: int = 2) -> Sequence:
        '''
        Cleans observed tower flux data according to a prescribed protocol.

        - For GPP data: Removes observations where GPP < 0 or where APAR is
            < 0.1 MJ m-2 day-1

        Parameters
        ----------
        raw : Sequence
        drivers : Sequence
        driver_labels : Sequence
        protocol : str
        filter_length : int
            The window size for the smoothing filter, applied to the observed
            data

        Returns
        -------
        Sequence
        '''
        if protocol != 'GPP':
            raise NotImplementedError('"protocol" must be one of: "GPP"')
        # Read in the observed data and apply smoothing filter
        obs = self._filter(raw, filter_length)
        obs = self._clean(obs, dict([
            (driver_labels[i], data)
            for i, data in enumerate(drivers)
        ]), protocol = 'GPP')
        return obs

    def export_bplut(
            self, model: str, output_path: str, thin: int = 10,
            burn: int = 1000):
        '''
        Export the BPLUT using the posterior mean from the MCMC sampler. NOTE:
        The posterior mean is usually not the best estimate for poorly
        identified parameters.

        Parameters
        ----------
        model : str
            The name of the model ("GPP" or "NPP")
        output_path : str
            The output CSV file path
        thin : int
            Thinning rate
        burn : int
            The burn-in (i.e., first N samples to discard)
        '''
        params_dict = restore_bplut(self.config['BPLUT'][model])
        bplut = params_dict.copy()
        # Filter the parameters to just those for the PFT of interest
        for pft in PFT_VALID:
            backend = self.config['optimization']['backend_template'] %\
                (model, pft)
            params = dict([(k, v[pft]) for k, v in params_dict.items()])
            sampler = MOD17StochasticSampler(
                self.config, getattr(MOD17, '_%s' % model.lower()), params,
                backend = backend)
            trace = sampler.get_trace()
            fit = trace.sel(
                draw = slice(burn, None, thin))['posterior'].mean()
            for each in MOD17.required_parameters:
                try:
                    bplut[each][pft] = float(fit[each])
                except KeyError:
                    continue
        write_bplut(bplut, output_path)

    def export_posterior(
            self, model: str, param: str, output_path: str, thin: int = 10,
            burn: int = 1000, k_folds: int = 1):
        '''
        Exports posterior distribution for a parameter, for each PFT to HDF5.

        Parameters
        ----------
        model : str
            The name of the model ("GPP" or "NPP")
        param : str
            The model parameter to export
        output_path : str
            The output HDF5 file path
        thin : int
            Thinning rate
        burn : int
            The burn-in (i.e., first N samples to discard)
        k_folds : int
            The number of k-folds used in cross-calibration/validation;
            if more than one (default), the folds for each PFT will be
            combined into a single HDF5 file
        '''
        params_dict = restore_bplut(self.config['BPLUT'][model])
        bplut = params_dict.copy()
        # Filter the parameters to just those for the PFT of interest
        post = []
        for pft in PFT_VALID:
            params = dict([(k, v[pft]) for k, v in params_dict.items()])
            backend = self.config['optimization']['backend_template'] %\
                (model, pft)
            post_by_fold = []
            for fold in range(1, k_folds + 1):
                if k_folds > 1:
                    backend = self.config['optimization']['backend_template'] %\
                        (f'{model}-k{fold}', pft)
                sampler = MOD17StochasticSampler(
                    self.config, getattr(MOD17, '_%s' % model.lower()), params,
                    backend = backend)
                trace = sampler.get_trace()
                fit = trace.sel(draw = slice(burn, None, thin))['posterior']
                num_samples = fit.sizes['chain'] * fit.sizes['draw']
                if param in fit:
                    post_by_fold.append(
                        az.extract_dataset(fit, combined = True)[param].values)
                else:
                    # In case there is, e.g., a parameter that takes on a
                    #   constant value for a specific PFT
                    if k_folds > 1:
                        post_by_fold.append(np.ones((1, num_samples)) * np.nan)
                    else:
                        post_by_fold.append(np.ones(num_samples) * np.nan)
            if k_folds > 1:
                post.append(np.vstack(post_by_fold))
            else:
                post.extend(post_by_fold)
        # If not every PFT's posterior has the same number of samples (e.g.,
        #   when one set of chains was run longer than another)...
        if not all([p.shape == post[0].shape for p in post]):
            max_len = max([p.shape for p in post])[0]
            # ...Reshape all posteriors to match the greatest sample size
            post = [
                np.pad(
                    p.astype(np.float32), (0, max_len - p.size),
                    mode = 'constant', constant_values = (np.nan,))
                for p in post
            ]
        with h5py.File(output_path, 'a') as hdf:
            post = np.stack(post)
            ts = datetime.date.today().strftime('%Y-%m-%d') # Today's date
            dataset = hdf.create_dataset(
                f'{param}_posterior', post.shape, np.float32, post)
            dataset.attrs['description'] = 'CalibrationAPI.export_posterior() on {ts}'

    def export_likely_posterior(
            self, model: str, param: str, output_path: str, thin: int = 10,
            burn: int = 1000, ptile: int = 95):
        '''
        Exports posterior distribution for a parameter based on likelihood

        Parameters
        ----------
        model : str
            The name of the model ("GPP" or "NPP")
        param : str
            The model parameter to export
        output_path : str
            The output HDF5 file path
        thin : int
            Thinning rate
        burn : int
            The burn-in (i.e., first N samples to discard)
        ptile : int
            The percentile cutoff for likelihood; only samples at or above
            this likelihood cutoff will be included
        '''
        params_dict = restore_bplut(self.config['BPLUT'][model])
        bplut = params_dict.copy()
        # Filter the parameters to just those for the PFT of interest
        post = []
        likelihood = []
        for pft in PFT_VALID:
            backend = self.config['optimization']['backend_template'] % (model, pft)
            params = dict([(k, v[pft]) for k, v in params_dict.items()])
            sampler = MOD17StochasticSampler(
                self.config, getattr(MOD17, '_%s' % model.lower()), params,
                backend = backend)
            trace = sampler.get_trace()
            fit = trace.sel(draw = slice(burn, None, thin))
            # Find the likelihood value associated with the cutoff percentile
            ll = az.extract_dataset(
                fit, combined = True)['log_likelihood'].values
            values = az.extract_dataset(fit, combined = True)[param].values
            cutoff = np.percentile(ll, ptile)
            post.append(values[ll >= cutoff])
            likelihood.append(ll[ll >= cutoff])
        with h5py.File(output_path, 'a') as hdf:
            n = max([len(p) for p in post])
            # Make sure all arrays are the same size
            post = np.stack([
                np.concatenate((p, np.full((n - len(p),), np.nan)))
                for p in post
            ])
            likelihood = np.stack([
                np.concatenate((p, np.full((n - len(p),), np.nan)))
                for p in likelihood
            ])
            ts = datetime.date.today().strftime('%Y-%m-%d') # Today's date
            dataset = hdf.create_dataset(
                f'{param}_posterior', post.shape, np.float32, post)
            dataset.attrs['description'] = 'CalibrationAPI.export_likely_posterior() on {ts}'
            dataset = hdf.create_dataset(
                f'{param}_likelihood', likelihood.shape, np.float32, likelihood)

    def tune_gpp(
            self, pft: int, plot_trace: bool = False, ipdb: bool = False,
            save_fig: bool = False, **kwargs):
        '''
        Run the MOD17 GPP calibration.

        Parameters
        ----------
        pft : int
            The Plant Functional Type (PFT) to calibrate
        plot_trace : bool
            True to plot the trace for a previous calibration run; this will
            also NOT start a new calibration (Default: False)
        ipdb : bool
            True to drop the user into an ipdb prompt, prior to and instead of
            running calibration
        save_fig : bool
            True to save figures to files instead of showing them
            (Default: False)
        **kwargs
            Additional keyword arguments passed to
            `MOD17StochasticSampler.run()`
        '''
        assert pft in PFT_VALID, f'Invalid PFT: {pft}'
        # Pass configuration parameters to MOD17StochasticSampler.run()
        for key in ('chains', 'draws', 'tune', 'scaling'):
            if key in self.config['optimization'].keys() and not key in kwargs.keys():
                kwargs[key] = self.config['optimization'][key]
        # Filter the parameters to just those for the PFT of interest
        params_dict = restore_bplut(self.config['BPLUT']['GPP'])
        # Load blacklisted sites (if any)
        blacklist = self.config['data']['sites_blacklisted']
        params_dict = dict([(k, v[pft]) for k, v in params_dict.items()])
        model = MOD17(params_dict)
        objective = self.config['optimization']['objective'].lower()

        print('Loading driver datasets...')
        with h5py.File(self.hdf5, 'r') as hdf:
            sites = hdf['FLUXNET/site_id'][:]
            if hasattr(sites[0], 'decode'):
                sites = list(map(lambda x: x.decode('utf-8'), sites))
            # Get dominant PFT
            pft_map = pft_dominant(hdf['state/PFT'][:], site_list = sites)
            # Blacklist various sites
            pft_mask = np.logical_and(pft_map == pft, ~np.in1d(sites, blacklist))
            # NOTE: Converting from Kelvin to Celsius
            tday = hdf['MERRA2/T10M_daytime'][:][:,pft_mask] - 273.15
            qv10m = hdf['MERRA2/QV10M_daytime'][:][:,pft_mask]
            ps = hdf['MERRA2/PS_daytime'][:][:,pft_mask]
            drivers = [ # fPAR, Tmin, VPD, PAR
                hdf['MODIS/MOD15A2HGF_fPAR_interp'][:][:,pft_mask],
                hdf['MERRA2/Tmin'][:][:,pft_mask] - 273.15,
                MOD17.vpd(qv10m, ps, tday),
                MOD17.par(hdf['MERRA2/SWGDN'][:][:,pft_mask]),
            ]
            # Convert fPAR from (%) to [0,1]
            drivers[0] = np.nanmean(drivers[0], axis = -1) / 100
            # If RMSE is used, then we want to pay attention to weighting
            weights = None
            if objective in ('rmsd', 'rmse'):
                weights = hdf['weights'][pft_mask][np.newaxis,:]\
                    .repeat(tday.shape[0], axis = 0)
            # Check that driver data do not contain NaNs
            for d, each in enumerate(drivers):
                name = ('fPAR', 'Tmin', 'VPD', 'PAR')[d]
                assert not np.isnan(each).any(),\
                    f'Driver dataset "{name}" contains NaNs'
            tower_gpp = hdf['FLUXNET/GPP'][:][:,pft_mask]
            # Read the validation mask; mask out observations that are
            #   reserved for validation
            print('Masking out validation data...')
            mask = hdf['FLUXNET/validation_mask'][pft]
            tower_gpp[mask] = np.nan

        # Clean observations, then mask out driver data where the are no
        #   observations
        tower_gpp = self.clean_observed(
            tower_gpp, drivers, MOD17StochasticSampler.required_drivers['GPP'],
            protocol = 'GPP')
        if weights is not None:
            weights = weights[~np.isnan(tower_gpp)]
        for d, _ in enumerate(drivers):
            drivers[d] = drivers[d][~np.isnan(tower_gpp)]
        tower_gpp = tower_gpp[~np.isnan(tower_gpp)]

        print('Initializing sampler...')
        backend = self.config['optimization']['backend_template'] % ('GPP', pft)
        sampler = MOD17StochasticSampler(
            self.config, MOD17._gpp, params_dict, backend = backend,
            weights = weights)
        if plot_trace or ipdb:
            if ipdb:
                import ipdb
                ipdb.set_trace()
            trace = sampler.get_trace()
            az.plot_trace(trace, var_names = MOD17.required_parameters[0:5])
            pyplot.show()
            return
        # Get (informative) priors for just those parameters that have them
        with open(self.config['optimization']['prior'], 'r') as file:
            prior = json.load(file)
        prior_params = filter(
            lambda p: p in prior.keys(), sampler.required_parameters['GPP'])
        prior = dict([
            (p, {'mu': prior[p]['mu'][pft], 'sigma': prior[p]['sigma'][pft]})
            for p in prior_params
        ])
        sampler.run(
            tower_gpp, drivers, prior = prior, save_fig = save_fig, **kwargs)

    def tune_npp(
            self, pft: int, plot_trace: bool = False, ipdb: bool = False,
            save_fig: bool = False, climatology = False,
            cutoff: Number = 2385, k_folds: int = 1, **kwargs):
        r'''
        Run the MOD17 NPP calibration. If k-folds cross-validation is used,
        the model is calibrated on $k$ random subsets of the data and a
        series of file is created, e.g., as:

            MOD17_NPP_calibration_PFT1.h5
            MOD17_NPP-k1_calibration_PFT1.nc4
            MOD17_NPP-k2_calibration_PFT1.nc4
            ...

        Where each `.nc4` file is a standard `arviz` backend and the `.h5`
        indicates which indices from the NPP observations vector, after
        removing NaNs, were excluded (i.e., the indices of the test data).

        Parameters
        ----------
        pft : int
            The Plant Functional Type (PFT) to calibrate
        plot_trace : bool
            True to display the trace plot ONLY and not run calibration
            (Default: False)
        ipdb : bool
            True to drop into an interactive Python debugger (`ipdb`) after
            loading an existing trace (Default: False)
        save_fig : bool
            True to save the post-calibration trace plot to a file instead of
            displaying it (Default: False)
        climatology : bool
            True to use a MERRA-2 climatology (and look for it in the drivers
            file), i.e., use `MERRA2_climatology` group instead of
            `surface_met_MERRA2` group (Default: False)
        cutoff : Number
            Maximum value of observed NPP (g C m-2 year-1); values above this
            cutoff will be discarded and not used in calibration
            (Default: 2385)
        k_folds : int
            Number of folds to use in k-folds cross-validation; defaults to
            k=1, i.e., no cross-validation is performed.
        **kwargs
            Additional keyword arguments passed to
            `MOD17StochasticSampler.run()`
        '''
        assert pft in PFT_VALID, f'Invalid PFT: {pft}'
        prefix = 'MERRA2_climatology' if climatology else 'surface_met_MERRA2'
        params_dict = restore_bplut(self.config['BPLUT']['NPP'])
        # Filter the parameters to just those for the PFT of interest
        params_dict = dict([(k, v[pft]) for k, v in params_dict.items()])
        model = MOD17(params_dict)
        kwargs.update({'var_names': [
            '~LUE_max', '~tmin0', '~tmin1', '~vpd0', '~vpd1', '~log_likelihood'
        ]})
        # Pass configuration parameters to MOD17StochasticSampler.run()
        for key in ('chains', 'draws', 'tune', 'scaling'):
            if key in self.config['optimization'].keys():
                kwargs[key] = self.config['optimization'][key]
        print('Loading driver datasets...')
        with h5py.File(self.hdf5, 'r') as hdf:
            # NOTE: This is only recorded at the site-level; no need to
            #   determine modal PFT across subgrid
            pft_map = hdf['NPP/PFT'][:]
            # Leave out sites where there is no fPAR (and no LAI) data
            fpar = hdf['NPP/MOD15A2H_fPAR_clim_filt'][:]
            mask = np.logical_and(
                    pft_map == pft, ~np.isnan(np.nanmean(fpar, axis = -1))\
                .all(axis = 0))
            # NOTE: Converting from Kelvin to Celsius
            tday = hdf[f'NPP/{prefix}/T10M_daytime'][:][:,mask] - 273.15
            qv10m = hdf[f'NPP/{prefix}/QV10M_daytime'][:][:,mask]
            ps = hdf[f'NPP/{prefix}/PS_daytime'][:][:,mask]
            drivers = [ # fPAR, Tmin, VPD, PAR, LAI, Tmean, years
                hdf['NPP/MOD15A2H_fPAR_clim_filt'][:][:,mask],
                hdf[f'NPP/{prefix}/Tmin'][:][:,mask]  - 273.15,
                MOD17.vpd(qv10m, ps, tday),
                MOD17.par(hdf[f'NPP/{prefix}/SWGDN'][:][:,mask]),
                hdf['NPP/MOD15A2H_LAI_clim_filt'][:][:,mask],
                hdf[f'NPP/{prefix}/T10M'][:][:,mask] - 273.15,
                np.full(ps.shape, 1) # i.e., A 365-day climatological year ("Year 1")
            ]
            observed_npp = hdf['NPP/NPP_total'][:][mask]
        if cutoff is not None:
            observed_npp[observed_npp > cutoff] = np.nan
        # Set negative VPD to zero
        drivers[2] = np.where(drivers[2] < 0, 0, drivers[2])
        # Convert fPAR from (%) to [0,1] and re-scale LAI; reshape fPAR, LAI
        drivers[0] = np.nanmean(drivers[0], axis = -1) * 0.01
        drivers[4] = np.nanmean(drivers[4], axis = -1) * 0.1
        # TODO Mask out driver data where the are no observations
        for d, _ in enumerate(drivers):
            drivers[d] = drivers[d][:,~np.isnan(observed_npp)]
        observed_npp = observed_npp[~np.isnan(observed_npp)]

        if k_folds > 1:
            # Back-up the original (complete) datasets
            _drivers = [d.copy() for d in drivers]
            _observed_npp = observed_npp.copy()
            # Randomize the indices of the NPP data
            indices = np.arange(0, observed_npp.size)
            np.random.shuffle(indices)
            # Get the starting and ending index of each fold
            fold_idx = np.array([indices.size // k_folds] * k_folds) * np.arange(0, k_folds)
            fold_idx = list(map(list, zip(fold_idx, fold_idx + indices.size // k_folds)))
            # Ensure that the entire dataset is used
            fold_idx[-1][-1] = indices.max()
            idx_test = [indices[start:end] for start, end in fold_idx]

        # Loop over each fold (or the entire dataset, if num. folds == 1)
        for k, fold in enumerate(range(1, k_folds + 1)):
            backend = self.config['optimization']['backend_template'] % ('NPP', pft)
            if k_folds > 1 and fold == 1:
                # Create an HDF5 file with the same name as the (original)
                #   netCDF4 back-end, store the test indices
                with h5py.File(backend.replace('nc4', 'h5'), 'w') as hdf:
                    out = list(idx_test)
                    size = indices.size // k_folds
                    try:
                        out = np.stack(out)
                    except ValueError:
                        size = max((o.size for o in out))
                        for i in range(0, len(out)):
                            out[i] = np.concatenate((out[i], [np.nan] * (size - out[i].size)))
                    hdf.create_dataset(
                        'test_indices', (k_folds, size), np.uint32, np.stack(out))
                # Restore the original NPP dataset
                observed_npp = _observed_npp.copy()
                # Set to NaN all the test indices
                idx = idx_test[k]
                observed_npp[idx] = np.nan
                # Same for drivers, after restoring from the original
                drivers = [d.copy()[:,~np.isnan(observed_npp)] for d in _drivers]
                observed_npp = observed_npp[~np.isnan(observed_npp)]
            # Use a different naming scheme for the backend
            if k_folds > 1:
                backend = self.config['optimization']['backend_template'] % (f'NPP-k{fold}', pft)

            print('Initializing sampler...')
            sampler = MOD17StochasticSampler(
                self.config, MOD17._npp, params_dict, backend = backend,
                model_name = 'NPP')
            if plot_trace or ipdb:
                if ipdb:
                    import ipdb
                    ipdb.set_trace()
                trace = sampler.get_trace()
                az.plot_trace(
                    trace, var_names = MOD17.required_parameters[5:])
                pyplot.show()
                return
            # Get (informative) priors for just those parameters that have them
            with open(self.config['optimization']['prior'], 'r') as file:
                prior = json.load(file)
            prior_params = filter(
                lambda p: p in prior.keys(), sampler.required_parameters['NPP'])
            prior = dict([
                (p, prior[p]) for p in prior_params
            ])
            for key in prior.keys():
                # And make sure to subset to the chosen PFT!
                for arg in prior[key].keys():
                    prior[key][arg] = prior[key][arg][pft]
            sampler.run( # Only show the trace plot if not using k-folds
                observed_npp, drivers, prior = prior, save_fig = save_fig,
                show_fig = (k_folds == 1), **kwargs)


if __name__ == '__main__':
    import fire
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fire.Fire(CalibrationAPI)
