'''
Performs the Sobol' sensitivity analysis for the MOD17 GPP and NPP models.

It's not clear that treating driver data as parameters is appropriate, here,
given that the NSE is calculated based on observed fluxes.
'''

import json
import os
import warnings
import numpy as np
import h5py
import mod17
from tqdm import tqdm
from mod17 import MOD17
from mod17.utils import restore_bplut, pft_dominant
from mod17.science import nash_sutcliffe
from SALib.sample import saltelli
from SALib.analyze import sobol

OUTPUT_TPL = '/home/arthur.endsley/MOD17_sensitivity_%s_analysis.json'


def main(model, config_file, pft = None):
    '''
    Conducts the Sobol' sensitivity analysis for the GPP, NPP models.

    Parameters
    ----------
    model : str
    config_file : str
    pft : int
    '''
    with open(config_file, 'r') as file:
        config = json.load(file)
    if model.lower() == 'gpp':
        saltelli_gpp(config, pft)
    elif model.lower() == 'npp':
        saltelli_npp(config, pft)
    elif model.lower() == 'npp2':
        saltelli_npp_and_gpp(config, pft)
    else:
        raise ValueError('Did not recognize model "%s"' % model)


def saltelli_gpp(config, pft = None):
    '''
    Sensitivity analysis for the GPP model.

    Parameters
    ----------
    pft : int
    '''
    drivers, gpp = load_gpp_data(config, None, pft, validation_mask_only = True)
    params = MOD17.required_parameters[0:5]
    problem = {
        'num_vars': len(params),
        'names': params,
        'bounds': [
            config['optimization']['bounds'][p]
            for p in params
        ]
    }
    # NOTE: Number of samples must be a power of 2
    param_sweep = saltelli.sample(problem, 256)
    Y = np.zeros([param_sweep.shape[0]])
    for i, X in enumerate(tqdm(param_sweep)):
        yhat = MOD17._gpp(X, *drivers[0:4])
        Y[i] = nash_sutcliffe(yhat, gpp, norm = True)
    metrics = sobol.analyze(problem, Y)
    name = 'GPP' if pft is None else 'GPP-PFT%d' % pft
    with open(OUTPUT_TPL % name, 'w') as file:
        json.dump(dict([(k, v.tolist()) for k, v in metrics.items()]), file)


def saltelli_npp(config, pft = None):
    '''
    Sensitivity analysis for the NPP model.

    Parameters
    ----------
    pft : int
    '''
    # Now for NPP; we have to load the BPLUT with the static parameters
    bplut = restore_bplut(config['BPLUT']['NPP'])
    gpp_params = [
        np.nanmean(bplut[p]) for p in MOD17.required_parameters[0:5]
    ]
    drivers, npp = load_npp_data(config, None, pft)
    params = MOD17.required_parameters[5:]
    if pft in (10, 12):
        params = list(set(params).difference(
            ('Q10_livewood', 'livewood_leaf_ratio', 'livewood_mr_base')))
    problem = {
        'num_vars': len(params),
        'names': params,
        'bounds': [
            config['optimization']['bounds'][p] for p in params
        ]
    }
    # NOTE: Number of samples must be a power of 2
    samples = 4096 if pft == 3 else 1024
    samples = 2048 if pft in (10, 12) else samples
    param_sweep = saltelli.sample(problem, samples)
    Y = np.zeros([param_sweep.shape[0]])
    # Index the livewood parameters
    idx = np.array([
        MOD17.required_parameters[5:].index(p)
        for p in ('Q10_livewood', 'livewood_leaf_ratio', 'livewood_mr_base')
    ])
    # There is no generalizable way to index and insert the livewood
    #   parameters because one of them comes at the end of the array and
    #   np.insert() will not work in that case; so, let's assert we have the
    #   right indices and make this part hard-coded
    assert (idx == [1,4,7]).all()
    for i, X in enumerate(tqdm(param_sweep)):
        if pft in (10, 12):
            # Even more confusingly, have to increment each successive index
            #   because of the progressive insertion
            X0 = X.copy()
            for j in idx:
                X = np.insert(X, j, 0)
        yhat = MOD17._npp(np.array([*gpp_params, *X]), *drivers)
        Y[i] = nash_sutcliffe(yhat, npp, norm = True)
    metrics = sobol.analyze(problem, Y)
    name = 'NPP' if pft is None else 'NPP-PFT%d' % pft
    with open(OUTPUT_TPL % name, 'w') as file:
        if pft in (10, 12):
            output = dict()
            for key, value in metrics.items():
                for j in idx:
                    value = np.insert(value, j, 0)
                output[key] = value.tolist()
        else:
            output = dict([(k, v.tolist()) for k, v in metrics.items()])
        json.dump(output, file)


def saltelli_npp_and_gpp(config, pft = None):
    '''
    Sensitivity analysis for the NPP model, including the GPP parameters
    in the analysis.

    Parameters
    ----------
    pft : int
    '''
    # Now for NPP; we have to load the BPLUT with the static parameters
    bplut = restore_bplut(config['BPLUT']['NPP'])
    print('Loading data...')
    drivers, npp = load_npp_data(config, None, pft)
    print('Setting up experiments...')
    params = MOD17.required_parameters
    if pft in (10, 12):
        params = list(set(params).difference(
            ('Q10_livewood', 'livewood_leaf_ratio', 'livewood_mr_base')))
    problem = {
        'num_vars': len(params),
        'names': params,
        'bounds': [
            config['optimization']['bounds'][p] for p in params
        ]
    }
    # NOTE: Number of samples must be a power of 2
    samples = 2048
    if pft is not None:
        samples = 4096 if pft == 3 else 1024
        samples = 2048 if pft in (10, 12) else samples
    param_sweep = saltelli.sample(problem, samples)
    Y = np.zeros([param_sweep.shape[0]])
    # Index the livewood parameters
    idx = np.array([
        MOD17.required_parameters[5:].index(p)
        for p in ('Q10_livewood', 'livewood_leaf_ratio', 'livewood_mr_base')
    ])
    # There is no generalizable way to index and insert the livewood
    #   parameters because one of them comes at the end of the array and
    #   np.insert() will not work in that case; so, let's assert we have the
    #   right indices and make this part hard-coded
    assert (idx == [1,4,7]).all()
    for i, X in enumerate(tqdm(param_sweep)):
        if pft in (10, 12):
            # Even more confusingly, have to increment each successive index
            #   because of the progressive insertion
            X0 = X.copy()
            for j in idx:
                X = np.insert(X, j, 0)
        yhat = MOD17._npp(X, *drivers)
        Y[i] = nash_sutcliffe(yhat, npp, norm = True)
    metrics = sobol.analyze(problem, Y)
    name = 'NPP' if pft is None else 'NPP-PFT%d' % pft
    with open(OUTPUT_TPL % name, 'w') as file:
        if pft in (10, 12):
            output = dict()
            for key, value in metrics.items():
                for j in idx:
                    value = np.insert(value, j, 0)
                output[key] = value.tolist()
        else:
            output = dict([(k, v.tolist()) for k, v in metrics.items()])
        json.dump(output, file)


def load_gpp_data(
        config, filename = None, pft = None, subgrid = False, verbose = True,
        validation_mask_only = False):
    '''
    Loads the data required for running a sensitivity analysis on the
    MOD17 GPP model.

    Parameters
    ----------
    config : dict
    filename : str or None
    pft : int
    subgrid : bool
    verbose : bool
    validation_mask_only : bool

    Returns
    -------
    tuple
        A 3-element tuple: A sequence of driver datasets (first element) and
        the array of valid GPP observations (second element), and the mask
    '''
    if verbose:
        print('Loading GPP datasets...')
    if filename is None:
        filename = config['data']['file']
    with h5py.File(filename, 'r') as hdf:
        sites = hdf['FLUXNET/site_id'][:].tolist()
        if hasattr(sites[0], 'decode'):
            sites = [s.decode('utf-8') for s in sites]
        # NOTE: Converting from Kelvin to Celsius
        tday = hdf['MERRA2/T10M_daytime'][:] - 273.15
        qv10m = hdf['MERRA2/QV10M_daytime'][:]
        ps = hdf['MERRA2/PS_daytime'][:]
        drivers = [ # fPAR, Tmin, VPD, PAR, LAI, Tmean, years
            hdf['MODIS/MOD15A2HGF_fPAR_interp'][:],
            hdf['MERRA2/Tmin'][:]  - 273.15,
            MOD17.vpd(qv10m, ps, tday),
            MOD17.par(hdf['MERRA2/SWGDN'][:]),
        ]
        observed_gpp = hdf['FLUXNET/GPP'][:]
        is_test = hdf['FLUXNET/validation_mask'][:].sum(axis = 0).astype(bool)
        if pft is not None:
            blacklist = config['data']['sites_blacklisted']
            pft_map = pft_dominant(hdf['state/PFT'][:], site_list = sites)
            pft_mask = np.logical_and(pft_map == pft, ~np.in1d(sites, blacklist))
            drivers = [d[:,pft_mask] for d in drivers]
            observed_gpp = observed_gpp[:,pft_mask]
    # Set negative VPD to zero
    drivers[2] = np.where(drivers[2] < 0, 0, drivers[2])
    # Convert fPAR from (%) to [0,1]
    if subgrid:
        drivers[0] = drivers[0] * 0.01
    else:
        # Average over the subgrid
        drivers[0] = np.nanmean(drivers[0], axis = -1) * 0.01
    # Speed things up by focusing only on data points where valid data exist
    mask = ~np.isnan(observed_gpp)
    if validation_mask_only:
        # Stratify the data using the validation mask so that an equal number
        #   of samples from each PFT are used
        mask = np.logical_and(is_test, mask)
    drivers = [d[mask] for d in drivers]
    return (drivers, observed_gpp[mask], mask)


def load_npp_data(
        config, filename = None, pft = None, subgrid = False, verbose = True):
    '''
    Loads the data required for running a sensitivity analysis on the
    MOD17 NPP model.

    Parameters
    ----------
    config : dict
    filename : str
    pft : int
    subgrid : bool
    verbose : bool

    Returns
    -------
    tuple
        A 2-element tuple: A sequence of driver datasets (first element) and
        the array of valid NPP observations (second element)
    '''
    if verbose:
        print('Loading NPP datasets...')
    if filename is None:
        filename = config['data']['file']
    with h5py.File(filename, 'r') as hdf:
        sites = hdf['NPP/site_id'][:].tolist()
        if hasattr(sites[0], 'decode'):
            sites = [s.decode('utf-8') for s in sites]
        # NOTE: Converting from Kelvin to Celsius
        tday = hdf['NPP/surface_met_MERRA2/T10M_daytime'][:] - 273.15
        qv10m = hdf['NPP/surface_met_MERRA2/QV10M_daytime'][:]
        ps = hdf['NPP/surface_met_MERRA2/PS_daytime'][:]
        drivers = [ # fPAR, Tmin, VPD, PAR, LAI, Tmean, years
            hdf['NPP/MOD15A2H_fPAR_clim_filt'][:],
            hdf['NPP/surface_met_MERRA2/Tmin'][:]  - 273.15,
            MOD17.vpd(qv10m, ps, tday),
            MOD17.par(hdf['NPP/surface_met_MERRA2/SWGDN'][:]),
            hdf['NPP/MOD15A2H_LAI_clim_filt'][:],
            hdf['NPP/surface_met_MERRA2/T10M'][:] - 273.15,
            np.full(ps.shape, 1) # i.e., A 365-day climatological year ("Year 1")
        ]
        observed_npp = hdf['NPP/NPP_total'][:]
        if pft is not None:
            blacklist = config['data']['sites_blacklisted']
            pft_map = hdf['NPP/PFT'][:]
            drivers = [d[:,pft_map == pft] for d in drivers]
            observed_npp = observed_npp[pft_map == pft]
    # Set negative VPD to zero
    drivers[2] = np.where(drivers[2] < 0, 0, drivers[2])
    # Convert fPAR from (%) to [0,1] and re-scale LAI; reshape fPAR and LAI
    drivers[0] = np.nanmean(drivers[0], axis = -1) * 0.01
    drivers[4] = np.nanmean(drivers[4], axis = -1) * 0.1
    return (drivers, observed_npp)


if __name__ == '__main__':
    import fire
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fire.Fire(main)
