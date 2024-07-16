'''
Utilities related to the MOD17 algorithm.
'''

import csv
import os
import numpy as np
import pandas as pd
import mod17
from collections import Counter
from typing import Callable, Sequence, Union, BinaryIO
from pandas._typing import FilePath, ReadCsvBuffer

BPLUT_FIELD_LOOKUP = {
    'LUEmax(KgC/m^2/d/MJ)': 'LUE_max',
    'Tmin_min(C)': 'tmin0',
    'Tmin_max(C)': 'tmin1',
    'VPD_min(Pa)': 'vpd0',
    'VPD_max(Pa)': 'vpd1',
    'SLA(LAI/KgC)': 'SLA',
    'Q10(unitless)': 'Q10',
    'Q10_livewood(unitless)': 'Q10_livewood',
    'Q10_froot(unitless)': 'Q10_froot',
    'froot_leaf_ratio': 'froot_leaf_ratio',
    'livewood_leaf_ratio': 'livewood_leaf_ratio',
    'leaf_mr_base(Kgc/KgC/d)': 'leaf_mr_base',
    'froot_mr_base(Kgc/KgC/d)': 'froot_mr_base',
    'livewood_mr_base(Kgc/KgC/d)': 'livewood_mr_base',
}

def dec2bin_unpack(x: np.ndarray) -> np.ndarray:
    '''
    Unpacks an arbitrary decimal NumPy array into a binary representation
    along a new axis. Assumes decimal digits are on the interval [0, 255],
    i.e., that only 8-bit representations are needed.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    '''
    # Make sure the bit representation is enumerated along a new axis, the
    #   very last axis
    axis = x.ndim
    # unpackbits() returns the bit representation in big-endian order, so we
    #   flip the array (with -8) to get litte-endian order
    return np.unpackbits(x[...,None], axis = axis)[...,-8:]


def haversine(p1: Sequence, p2: Sequence, radius: float = 6371e3) -> float:
    '''
    Haversine formula for great circle distance, in meters. Accurate for
    points separated near and far but for small distances the accuracy is
    improved by providing a different radius of the sphere, say 6356.7523 km
    for polar regions or 6378.1370 km for equatorial regions. Default is the
    mean earth radius.

    NOTE: Distance returned is in the same units as radius.

    Parameters
    ----------
    p1 : tuple or list
        Sequence of two floats, longitude and latitude, respectively
    p2 : tuple or list
        Same as p1 but for the second point
    radius : int or float
        Radius of the sphere to use in distance calculation
        (Default: 6,371,000 meters)

    Returns
    -------
    float
    '''
    x1, y1 = map(np.deg2rad, p1)
    x2, y2 = map(np.deg2rad, p2)
    dphi = np.abs(y2 - y1) # Difference in latitude
    dlambda = np.abs(x2 - x1) # Difference in longitude
    angle = 2 * np.arcsin(np.sqrt(np.add(
        np.power(np.sin(dphi / 2), 2),
        np.cos(y1) * np.cos(y2) * np.power(np.sin(dlambda / 2), 2)
    )))
    return float(angle * radius)


def mod15a2h_qc_fail(x: np.ndarray) -> np.ndarray:
    '''
    Returns pass/fail for QC flags based on the L4C fPAR QC protocol for the
    `FparLai_QC` band: Bad pixels have either `1` in the first bit ("Pixel not
    produced at all") or anything other than `00` ("clear") in bits 3-4.
    Output array is True wherever the array fails QC criteria. Compare to:

        np.vectorize(lambda v: v[0] == 1 or v[3:5] != '00')

    Parameters
    ----------
    x : numpy.ndarray
        Array where the last axis enumerates the unpacked bits
        (ones and zeros)

    Returns
    -------
    numpy.ndarray
        Boolean array with True wherever QC criteria are failed
    '''
    y = dec2bin_unpack(x)
    # Emit 1 = FAIL if these two bits are not == "00"
    c1 = y[...,3:5].sum(axis = -1).astype(np.uint8)
    # Emit 1 = FAIL if 1st bit == 1 ("Pixel not produced at all")
    c2 = y[...,0]
    # Intermediate arrays are 1 = FAIL, 0 = PASS
    return (c1 + c2) > 0


def pft_dominant(
        pft_map: np.ndarray, site_list: list = None,
        valid_pft: list = mod17.PFT_VALID):
    '''
    Returns the dominant PFT type, based on the PFT mode among the sub-grid
    for each site. Note that this is specific to the MOD17 calibration/
    validation (Cal/Val) protocol, i.e., two sites are always classified as
    Deciduous Needleleaf (PFT 3):

        CA-SF2
        CA-SF3

    Three other sites (CN-Do1, CN-Do3, US-WPT) are classified by the PI as
    wetlands, which is not supported, nor are their MCD12Q1 classifications.

    Three other sites have hard-coded PFT classes because they are in urban
    areas (PFT 13):

        IT-Vig: Re-classified to PFT 4 (as reported by PI)
        NL-Hor: Re-classified to PFT 10 (as reported by PI)
        SE-Abi: Re-classified to PFT 4 (as reported by PI)

    At US-ORv, it's clearly a wetland but there is a lot of closed canopy of
    indeterminate composition.

    Parameters
    ----------
    pft_map : numpy.ndarray
        (N x M) array of PFT classes, where N is the number of sites and
        M is the number of sub-grid cells (N PFT classes are returned)
    site_list : list
        (Optional) List of the site names; must be provided to get PFT
        classes that accurately match the Cal/Val protocol
    valid_pft : list
        (Optional) List of valid PFT classes (Default: `mod17.PFT_VALID`)

    Returns
    -------
    numpy.ndarray
        An (N,) array of the dominant PFT classes
    '''
    pft_dom = np.zeros(pft_map.shape[0], dtype = np.float32)
    for i in range(0, pft_map.shape[0]):
        try:
            pft_dom[i] = Counter(
                list(filter(lambda x: x in valid_pft, pft_map[i])))\
                .most_common()[0][0]
        except:
            # Skip those sites that have no valid PFTs
            continue
    if site_list is not None:
        # Fill in the PI-reported PFT for troublesome sites
        pft_dom[197] = 4
        pft_dom[209] = 10
        pft_dom[234] = 4
        # Fill in black-listed sites
        idx = [
            site_list.index(sid)
            for sid in ('CN-Do1', 'CN-Do3', 'US-WPT', 'US-ORv')
        ]
        pft_dom[idx] = np.nan
        # For PFT==3 (DNF) use pre-determined locations
        idx = [site_list.index(sid) for sid in ('CA-SF2', 'CA-SF3')]
        pft_dom[idx] = 3
        # For PFT==6 (CSH) use sites with any amount of CSH pixels
        idx = np.argwhere(
            np.sum(pft_map == 6, axis = 1) > 0).ravel()
        pft_dom[idx] = 6
    return pft_dom


def pft_remap(
        pft_map: np.ndarray, site_list: list = None,
        valid_pft: list = mod17.PFT_VALID):
    '''
    Returns a map of PFTs that is consistent with the model's approved PFT
    classes. Note that this is specific to the MOD17 calibration/
    validation (Cal/Val) protocol, i.e., two sites are always classified as
    Deciduous Needleleaf (PFT 3):

        CA-SF2
        CA-SF3

    Three other sites have hard-coded PFT classes because they are in urban
    areas (PFT 13):

        IT-Vig: Re-classified to PFT 4 (as reported by PI)
        NL-Hor: Re-classified to PFT 10 (as reported by PI)
        SE-Abi: Re-classified to PFT 4 (as reported by PI)

    PFT classes that are not recognized in the MOD17 model are mapped to 0,
    which is not used in the model.

    Parameters
    ----------
    pft_map : numpy.ndarray
        (N x M) array of PFT classes, where N is the number of sites and
        M is the number of sub-grid cells (N PFT classes are returned)
    site_list : list
        (Optional) List of the site names; must be provided to get PFT
        classes that accurately match the Cal/Val protocol
    valid_pft : list
        (Optional) List of valid PFT classes (Default: `mod17.PFT_VALID`)

    Returns
    -------
    numpy.ndarray
        An (N,M) array of the model-consistent PFT classes
    '''
    output_map = pft_map.copy()
    if site_list is not None:
        # Fill in the PI-reported PFT for troublesome sites
        output_map[197] = 4
        output_map[209] = 10
        output_map[234] = 4
        # For PFT==3, DNF, use pre-determined locations
        idx = [site_list.index(sid) for sid in ('CA-SF2', 'CA-SF3')]
        output_map[idx] = 3
    output_map[output_map  > 12] = 0
    output_map[output_map == 11] = 0
    return output_map


def restore_bplut(path_or_buffer: Union[BinaryIO, str]) -> dict:
    '''
    NOTE: I manually exported Maosheng's fixed-width version (fixed-width
    files are a crime) to CSV for easier handling.

    Parameters
    ----------
    path_or_buffer : str or buffer
        File path or buffer representing the BPLUT to be read

    Returns
    -------
    dict
    '''
    # Remaps Maosheng's PFT order to the actual PFT code from MCD12Q1
    #   LC_Type2
    pft_lookup = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    data = pd.read_csv(path_or_buffer)
    # Create a dictionary with an array for every key
    output = dict([
        (k, np.full((13,), np.nan))
        for k in BPLUT_FIELD_LOOKUP.values()
    ])
    # Assumes the first column indexes the parameter/ field names
    field_index = data.columns[0]
    pft_index = list(data.columns)
    pft_index.remove(field_index)
    for k, key in enumerate(data[field_index]):
        values = data.loc[data[field_index] == key, pft_index].values.ravel()
        output[BPLUT_FIELD_LOOKUP[key]][pft_lookup] = values
    return output


def write_bplut(params_dict: dict, output_path: str):
    '''
    Writes a BPLUT parameters dictionary to an output CSV file.

    Parameters
    ----------
    params_dict : dict
    output_path : str
        The output CSV file path
    '''
    template = os.path.join(
        os.path.dirname(mod17.__file__), 'data/MOD17_BPLUT_C5.1_MERRA_NASA.csv')
    with open(template, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if reader.line_num > 1:
                break
            header = line
    with open(output_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for name, key in BPLUT_FIELD_LOOKUP.items():
            values = []
            for pft in mod17.PFT_VALID:
                values.append(params_dict[key][pft])
            writer.writerow((name, *values))


def rmsd(
        params: Sequence, func: Callable = None, observed: Sequence = None,
        drivers: Sequence = None) -> float:
    '''
    The root-mean scquared deviation. This function is intended to be used
    in a multiprocessing context (with functools.partial()).

    Parameters
    ----------
    params : Sequence
        Sequence of expected model parameters
    func : Callable
        The function to call to generate predicted values; function should
        expect to receive positional arguments, the first being a sequence
        of model parameters and every subsequent argument an input array
    observed : Sequence
        The oberved values
    drivers : Sequence
        Sequence of expected model drivers

    Returns
    -------
    float
    '''
    predicted = func(params, *drivers)
    return np.sqrt(np.nanmean((predicted - observed) ** 2))


def sites_by_record_length(
        array: np.ndarray, dates: np.ndarray, pft_map: np.ndarray,
        sites: np.ndarray, n_returned: int = 5, cutoff: float = 0.97,
        pft_passed: Sequence = None) -> tuple:
    '''
    Ranks sites by the total number of site-years with valid data. Returns
    a tuple of (sites, site-years) where sites is the top `n_returned` site
    names with the longest site-year record; site-years is a same-length
    sequence of randomly chosen years, ordered by highest proportion of valid
    data within the year.

    Parameters
    ----------
    array : numpy.ndarray
        The data record, a (T x N) array for N sites
    dates : numpy.ndarray
        (T x 3) array of Year, Month, Day for each time step; years must be
        consecutive
    pft_map : numpy.ndarray
        The map of PFTs, a (N,) array for N sites with a subgrid of M cells
    sites : numpy.ndarray
        (N,) array of site labels
    n_returned : int
        Number of unique sites to return for each PFT
    cutoff : float
        (Optional) A cutoff for the proportion of valid data required in each
        year; i.e., site-years with a proportion below this cutoff are ignored
        when tallying sites by total number of site-years
    pft_passed : Sequence
        (Optional) A sequence of PFT codes for which the `cutoff` will not be
        applied; instead, any site-year proportion above 0 will be considered;
        if None, the `cutoff` is applied to all PFTs (Default: None)

    Returns
    -------
    tuple
        A 2-element tuple of (sites, site-years); each is a (P x Z)
        `numpy.ndarray` where P is the number of unique PFTs and Z is the
        `n_returned`
    '''
    assert array.shape[0] == dates.shape[0],\
        'Data array and dates array should have the same number of time points'
    assert array.shape[1] == pft_map.shape[0],\
        'Data array and PFT map should have the same number of sites'
    assert hasattr(dates, 'size'), '"dates" should be a numpy.ndarray'
    assert hasattr(sites, 'size'), '"sites" should be a numpy.ndarray'
    all_years = np.unique(dates[:,0])
    site_years = np.zeros((len(all_years), pft_map.shape[0]))
    for y, year in enumerate(all_years.ravel()):
        # Count the number of days with valid data in each year; normalize by
        #   the total number of days in the year (366 for leap years)
        dmax = 366 if year % 4 == 0 else 365
        site_years[y,:] = (
            dmax - np.isnan(array[dates[:,0] == year]).sum(axis = 0)) / dmax
    # Ignore site-year proportions below the cutoff
    _site_years = site_years.copy()
    site_years = np.where(site_years < cutoff, 0, site_years)
    # For simplicity, we copy the data from the original site_years array to
    #   the new one for those PFTs that we don't want to apply a cutoff to
    if pft_passed is not None:
        mask = np.repeat(pft_map[None,:], len(all_years), axis = 0)
        for each_pft in pft_passed:
            site_years[mask == each_pft] = _site_years[mask == each_pft]
    # Tally the total "site-years" for each site
    site_years_sum = site_years.sum(axis = 0)
    # Get a list of unique PFTs
    all_pft = np.unique(pft_map[~np.isnan(pft_map)]).tolist()
    all_pft.sort()
    results_sites = np.chararray(
        (len(all_pft), n_returned), itemsize = 6, unicode = True)
    results = np.zeros((len(all_pft), n_returned), dtype = np.int32)
    for p, pft in enumerate(all_pft):
        # Sort the site-year data by number of site-years
        site_max = site_years_sum[pft_map == pft]
        top = sites[pft_map == pft][np.argsort(site_max)][::-1]
        if top.size < n_returned:
            results_sites[p,0:top.size] = top
        else:
            results_sites[p,:] = top[0:n_returned]
        # Indices of those top sites...
        idx = np.argwhere(np.in1d(sites, top[0:n_returned])).ravel()
        # Choose a random year in the top n_returned years, unless it is
        #   PFT 3, in which case we just take the best site-year available
        _cutoff = cutoff
        if pft_passed is not None:
            _cutoff = 0 if pft in pft_passed else cutoff
        choices = [
            all_years[site_years[:,idx][:,i] > _cutoff]
            for i in range(0, len(idx))
        ]
        # Shuffle the years within each site
        for c in choices:
            np.random.shuffle(c)
        results[p,0:len(idx)] = [
            c[0] if c.size > 0 else 0 for c in choices[0:len(idx)]
        ]
    return (results_sites, results)


def report(hdf, by_pft: bool = False):
    '''
    Check that we have everything needed to calibrate MOD17 and print the
    report to the screen

    Parameters
    ----------
    hdf : h5py.File
    by_pft : bool
    '''
    NPP_KEYS = ('MOD15A2H_fPAR_clim', 'MOD15A2H_LAI_clim', 'NPP_total_filled')
    MERRA2_KEYS = (
        'LWGNT', 'LWGNT_daytime', 'LWGNT_nighttime', 'PS', 'PS_daytime',
        'PS_nighttime', 'QV10M', 'QV10M_daytime', 'QV10M_nighttime', 'SWGDN',
        'SWGDN_daytime', 'SWGDN_nighttime', 'SWGNT', 'SWGNT_daytime',
        'SWGNT_nighttime', 'T10M', 'T10M_daytime', 'T10M_nighttime', 'Tmin')

    def find(hdf, prefix, key, pad = 10, mask = None):
        'Find a key, print the report'
        try:
            field = '%s/%s' % (prefix, key)
            pretty = ('"%s"' % key).ljust(pad)
            if mask is None:
                print_stats(hdf[field][:], pad, pretty)
            else:
                shp = hdf[field].shape
                if len(shp) == 1:
                    print_stats(hdf[field][mask], pad, pretty)
                if len(shp) == 2:
                    print_stats(hdf[field][:,mask], pad, pretty)
                elif len(shp) == 3:
                    print_stats(hdf[field][:,mask,:], pad, pretty)
        except KeyError:
            pretty = ('"%s"' % key).ljust(pad)
            print('-- MISSING %s' % pretty)

    def print_stats(data, pad, pretty):
        shp = ' x '.join(map(str, data.shape))
        shp = ('[%s]' % shp).ljust(pad + 7)
        stats = tuple(summarize(data))
        stats_pretty = ''
        if stats[0] is not None:
            stats_pretty = '[%.2f, %.2f] (%.2f)' % (stats[0], stats[2], stats[1])
            if len(key) < 10:
                print('-- Found %s %s %s' % (pretty, shp, stats_pretty))
            else:
                print('-- Found %s' % pretty)
                print('%s%s %s' % (''.rjust(pad + 10), shp, stats_pretty))

    def summarize(data, nodata = -9999):
        'Get summary statistics for a field'
        if str(data.dtype).startswith('int'):
            return (None for i in range(0, 3))
        data[data == -9999] = np.nan
        return (
            getattr(np, f)(data) for f in ('nanmin', 'nanmean', 'nanmax')
        )

    print('\nMOD17: Checking for required driver variables...')
    enum = range(0, 1) if not by_pft else (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12)
    pft_map = None
    for i in enum:
        if by_pft:
            pft_map = np.arange(0, hdf['NPP/PFT'].size)[hdf['NPP/PFT'][:] == i]
            print('\n--------------------')
            print('\n-- PFT == %d' % i)
        for key in NPP_KEYS:
            find(hdf, 'NPP', key, mask = pft_map)
        for key in MERRA2_KEYS:
            find(hdf, 'NPP/surface_met_MERRA2', key, mask = pft_map)
    print('')


def vnp15a2h_qc_fail(x):
    '''
    Returns pass/fail for QC flags based on the L4C fPAR QC protocol for the
    `FparLai_QC` band: Bad pixels have either `11` in the first two bits
    ("Fill Value") or anything other than `0` in the 3rd least-significant
    bits, which combines "Pixel not produced at all". For example, see decimal
    number 80:

        0101|0|000 where "000" is the combined (Fill bit | Retrieval quality)

    Parameters
    ----------
    x : numpy.ndarray
        Unsigned, 8-bit integer array

    Returns
    -------
    numpy.ndarray
        Boolean array
    '''
    y = dec2bin_unpack(x)
    # Emit 1 = FAIL if sum("11") == 2; "BiomeType" == "Filled Value"
    c1 = np.where(y[...,0:2].sum(axis = -1) == 2, 1, 0).astype(np.uint8)
    # Emit 1 = FAIL if 3rd bit == 1; "SCF_QC" == "Pixel not produced at all"
    c2 = y[...,5]
    # Intermediate arrays are 1 = FAIL, 0 = PASS
    return (c1 + c2) > 0


def vnp15a2h_cloud_fail(x):
    '''
    Returns pass/fail for QC flags based on the L4C fPAR QC protocol for the
    `FparExtra_QC` band (cloud QC band): Bad pixels have anything OTHER THAN
    `1` second least-significant bit; `00` and `01` being acceptable cloud QC
    flags ("Confident clear" and "Probably clear", respectively).

    Parameters
    ----------
    x : numpy.ndarray
        Unsigned, 8-bit integer array

    Returns
    -------
    numpy.ndarray
        Boolean array
    '''
    y = dec2bin_unpack(x)
    return y[...,-2] > 0
