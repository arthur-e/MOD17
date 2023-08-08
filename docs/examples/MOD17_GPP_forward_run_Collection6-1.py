'''
Runs MOD17 using the Collection 6.1 BPLUT, based on optimization against
FLUXNET GPP data using a stochastic MCMC sampler.

The HDF5 file used in this example can be downloaded from:

    http://doi.org/10.5281/zenodo.7682806
'''

import os
import numpy as np
import h5py
import mod17
from mod17 import MOD17, PFT_VALID
from mod17.utils import restore_bplut, pft_remap

MOD17_DIR = os.path.dirname(mod17.__file__)
BPLUT = os.path.join(MOD17_DIR, 'data/MOD17_BPLUT_C5.1_MERRA_NASA.csv')
MOD17_HDF5 = 'VIIRS_MOD16_MOD17_tower_site_drivers_v9.h5'
OUTPUT_HDF5 = 'MOD17_GPP_C6-1_predictions_at_tower_sites.h5'
N_DAYS = 6575
N_SITES = 356
N_SUBGRID = 9

def main():
    params_dict = restore_bplut(BPLUT)
    with h5py.File(MOD17_HDF5, 'r') as hdf:
        sites = hdf['FLUXNET/site_id'][:].tolist()
        if hasattr(sites[0], 'decode'):
            sites = [s.decode('utf-8') for s in sites]
        # NOTE: Converting from Kelvin to Celsius
        tday = hdf['MERRA2/T10M_daytime'][:] - 273.15
        qv10m = hdf['MERRA2/QV10M_daytime'][:]
        ps = hdf['MERRA2/PS_daytime'][:]
        drivers = [ # fPAR, Tmin, VPD, PAR, LAI, Tmean, years
            hdf['MODIS/MOD15A2HGF_fPAR_interp'][:],
            hdf['MERRA2/Tmin'][:][...,None]  - 273.15,
            MOD17.vpd(qv10m, ps, tday)[...,None],
            # NOTE: Using SWGNT, as is implemented in the original (Collection 6.1) algorithm
            MOD17.par(hdf['MERRA2/SWGNT'][:][...,None]),
        ]
        # Create a map of PFTs, including the site subgrid
        pft_map = pft_remap(
            hdf['state/PFT'][:].astype(np.int16), sites)
    # Set negative VPD to zero
    drivers[2] = np.where(drivers[2] < 0, 0, drivers[2])
    # Convert fPAR from (%) to [0,1]
    drivers[0] = drivers[0] * 0.01
    # Create a vectorized BPLUT; there are 5 GPP parameters
    params_vector = params_dict.copy()
    for p_name in MOD17.required_parameters[0:5]:
        params_vector[p_name] = params_dict[p_name][pft_map]\
            .reshape((1,N_SITES,N_SUBGRID))
    model = MOD17(params_vector)
    # Get model predictions; average over sub-grid
    gpp0 = model.daily_gpp(*drivers)
    gpp = np.nanmean(gpp0, axis = -1)
    with h5py.File(OUTPUT_HDF5, 'a') as hdf:
        if 'MOD17_GPP' in hdf.keys():
            del hdf['MOD17_GPP']
        dataset = hdf.create_dataset('MOD17_GPP', gpp.shape, np.float32, gpp)
        dataset.attrs['units'] = 'g C m-2 d-1'
        dataset.attrs['description'] = 'MOD17 GPP predictions based on the Collection 6.1 BPLUT'


if __name__ == '__main__':
    main()
