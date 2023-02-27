'''
Calibration of MOD17 against a representative, global eddy covariance (EC)
flux tower network. The model calibration is based on Markov-Chain Monte
Carlo (MCMC). This module is for calibrating using VIIRS reflectance, fPAR,
and LAI data, specifically.
'''

import datetime
import json
import os
import warnings
import numpy as np
import h5py
import arviz as az
import pymc as pm
import aesara.tensor as at
import mod17
from numbers import Number
from typing import Callable, Sequence
from pathlib import Path
from matplotlib import pyplot
from mod17 import MOD17, PFT_VALID
from mod17.utils import pft_dominant, restore_bplut, write_bplut
from mod17.calibration import BlackBoxLikelihood, MOD17StochasticSampler, CalibrationAPI

MOD17_DIR = os.path.dirname(mod17.__file__)
# This matplotlib setting prevents labels from overplotting
pyplot.rcParams['figure.constrained_layout.use'] = True


class VNP17StochasticSampler(MOD17StochasticSampler):
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
            self.model, observed, x = drivers, weights = self.weights)
        # With this context manager, "all PyMC3 objects introduced in the indented
        #   code block...are added to the model behind the scenes."
        with pm.Model() as model:
            # (Stochstic) Priors for unknown model parameters
            LUE_max = pm.TruncatedNormal('LUE_max',
                **self.prior['LUE_max'], **self.bounds['LUE_max'])
            # NOTE: All environmental scalars are fixed at their updated
            #   MOD17 values
            tmin0 = self.params['tmin0']
            tmin1 = self.params['tmin1']
            vpd0 = self.params['vpd0']
            vpd1 = self.params['vpd1']
            # Convert model parameters to a tensor vector
            params_list = [LUE_max, tmin0, tmin1, vpd0, vpd1]
            params = at.as_tensor_variable(params_list)
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
            self.model, observed, x = drivers, weights = self.weights)
        # With this context manager, "all PyMC3 objects introduced in the indented
        #   code block...are added to the model behind the scenes."
        with pm.Model() as model:
            # Setting GPP parameters that are known
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
            Q10_froot = pm.TruncatedNormal(
                'Q10_froot', **self.prior['Q10_froot'], **self.bounds['Q10'])
            leaf_mr_base = pm.LogNormal(
                'leaf_mr_base', **self.prior['leaf_mr_base'])
            froot_mr_base = pm.LogNormal(
                'froot_mr_base', **self.prior['froot_mr_base'])
            # For GRS and CRO, livewood mass and respiration are zero
            if list(self.prior['livewood_mr_base'].values()) == [0, 0]:
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
            params = at.as_tensor_variable(params_list)
            # Key step: Define the log-likelihood as an added potential
            pm.Potential('likelihood', log_likelihood(params))
        return model


class VIIRSCalibrationAPI(CalibrationAPI):
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

    def tune_gpp(
            self, pft: int, plot_trace: bool = False, ipdb: bool = False,
            save_fig: bool = False, **kwargs):
        '''
        Run the VNP17 GPP calibration.

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
            `VNP17StochasticSampler.run()`
        '''
        assert pft in PFT_VALID, f'Invalid PFT: {pft}'
        params_dict = restore_bplut(self.config['BPLUT']['GPP'])
        # Load blacklisted sites (if any)
        blacklist = self.config['data']['sites_blacklisted']
        # Filter the parameters to just those for the PFT of interest
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
            dates = hdf['time'][:]
            # For expedience, subset all data to the VIIRS post-launch period
            cutoff = np.argwhere(dates[:,0] == 2012).min()
            weights = hdf['weights'][pft_mask]
            # NOTE: Converting from Kelvin to Celsius
            tday = hdf['MERRA2/T10M_daytime'][:][cutoff:,pft_mask] - 273.15
            qv10m = hdf['MERRA2/QV10M_daytime'][:][cutoff:,pft_mask]
            ps = hdf['MERRA2/PS_daytime'][:][cutoff:,pft_mask]
            drivers = [ # fPAR, Tmin, VPD, PAR
                hdf['VIIRS/VNP15A2HGF_fPAR_interp'][:][cutoff:,pft_mask],
                hdf['MERRA2/Tmin'][:][cutoff:,pft_mask] - 273.15,
                MOD17.vpd(qv10m, ps, tday),
                MOD17.par(hdf['MERRA2/SWGDN'][:][cutoff:,pft_mask]),
            ]
            # Set negative VPD to zero
            drivers[2] = np.where(drivers[2] < 0, 0, drivers[2])
            # Convert fPAR from (%) to [0,1]
            drivers[0] = np.nanmean(drivers[0], axis = -1) / 100
            # If RMSE is used, then we want to pay attention to weighting
            weights = None
            if objective in ('rmsd', 'rmse'):
                weights = hdf['weights'][pft_mask][np.newaxis,:]\
                    .repeat(tday.shape[0], axis = 0)
            for d, each in enumerate(drivers):
                name = ('fPAR', 'Tmin', 'VPD', 'PAR')[d]
                assert not np.isnan(each).any(),\
                    f'Driver dataset "{name}" contains NaNs'
            tower_gpp = hdf['FLUXNET/GPP'][:][cutoff:,pft_mask]
            # Read the validation mask; mask out observations that are
            #   reserved for validation
            print('Masking out validation data...')
            mask = hdf['FLUXNET/validation_mask_VNP17'][pft]
            tower_gpp[mask] = np.nan

        # Clean observations, then mask out driver data where the are no
        #   observations
        tower_gpp = self.clean_observed(
            tower_gpp, drivers, VNP17StochasticSampler.required_drivers['GPP'],
            protocol = 'GPP')
        if weights is not None:
            weights = weights[~np.isnan(tower_gpp)]
        for d, _ in enumerate(drivers):
            drivers[d] = drivers[d][~np.isnan(tower_gpp)]
        tower_gpp = tower_gpp[~np.isnan(tower_gpp)]

        print('Initializing sampler...')
        backend = self.config['optimization']['backend_template'] % ('GPP', pft)
        sampler = VNP17StochasticSampler(
            self.config, MOD17._gpp, params_dict, backend = backend,
            weights = weights)
        if plot_trace or ipdb:
            if ipdb:
                import ipdb
                ipdb.set_trace()
            trace = sampler.get_trace()
            az.plot_trace(trace, var_names = VNP17.required_parameters[0:5])
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
        '''
        Run the VNP17 NPP calibration.

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
            `VNP17StochasticSampler.run()`
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
        # Pass configuration parameters to VNP17StochasticSampler.run()
        for key in ('chains', 'draws', 'tune', 'scaling'):
            if key in self.config['optimization'].keys():
                kwargs[key] = self.config['optimization'][key]

        print('Loading driver datasets...')
        with h5py.File(self.hdf5, 'r') as hdf:
            # NOTE: This is only recorded at the site-level; no need to
            #   determine modal PFT across subgrid
            pft_map = hdf['NPP/PFT'][:]
            # Leave out sites where there is no fPAR (and no LAI) data
            fpar = hdf['NPP/MOD15A2H_fPAR_clim'][:]
            mask = np.logical_and(
                    pft_map == pft, ~np.isnan(np.nanmean(fpar, axis = -1))\
                .all(axis = 0))
            # NOTE: Converting from Kelvin to Celsius
            tday = hdf[f'NPP/{prefix}/T10M_daytime'][:][:,mask] - 273.15
            qv10m = hdf[f'NPP/{prefix}/QV10M_daytime'][:][:,mask]
            ps = hdf[f'NPP/{prefix}/PS_daytime'][:][:,mask]
            drivers = [ # fPAR, Tmin, VPD, PAR, LAI, Tmean, years
                hdf['NPP/VNP15A2H_fPAR_clim'][:][:,mask],
                hdf[f'NPP/{prefix}/Tmin'][:][:,mask]  - 273.15,
                MOD17.vpd(qv10m, ps, tday),
                MOD17.par(hdf[f'NPP/{prefix}/SWGDN'][:][:,mask]),
                hdf['NPP/VNP15A2H_LAI_clim'][:][:,mask],
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
        # Mask out driver data where the are no observations
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

        for k, fold in enumerate(range(1, k_folds + 1)):
            backend = self.config['optimization']['backend_template'] % ('NPP', pft)
            if k_folds > 1:
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
                        'test_indices', (k_folds, size), np.int32, np.stack(out))
                backend = self.config['optimization']['backend_template'] % (f'NPP-k{fold}', pft)
                # Restore the original NPP dataset
                observed_npp = _observed_npp.copy()
                # Set to NaN all the test indices
                idx = idx_test[k]
                observed_npp[idx] = np.nan
                # Same for drivers, after restoring from the original
                drivers = [d.copy()[:,~np.isnan(observed_npp)] for d in _drivers]
                observed_npp = observed_npp[~np.isnan(observed_npp)]

            print('Initializing sampler...')
            sampler = VNP17StochasticSampler(
                self.config, MOD17._npp, params_dict, backend = backend,
                model_name = 'NPP')
            if plot_trace or ipdb:
                if ipdb:
                    import ipdb
                    ipdb.set_trace()
                trace = sampler.get_trace()
                az.plot_trace(trace, var_names = MOD17.required_parameters[5:])
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
            sampler.run(
                observed_npp, drivers, prior = prior, save_fig = save_fig,
                **kwargs)


if __name__ == '__main__':
    import fire
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fire.Fire(VIIRSCalibrationAPI)
