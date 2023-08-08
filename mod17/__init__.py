'''
The MOD17 Daily GPP and Annual GPP algorithm.

Note that there are two hidden methods of the MOD17 class:

- `MOD17._gpp()`
- `MOD17._npp()`

These are streamlined implementations of `MOD17.daily_gpp()` and
`MOD17.annual_npp()`, respectively, and were designed for use in calibration,
where repeated function calls can make function overhead a real issue. These
streamlined functions expect a vectorized parameters array as the first
argument and subsequent arguments are driver datasets, e.g.:

    MOD17._gpp(params, fpar, tmin, vpd, par)
    MOD17._npp(params, fpar, tmin, vpd, par, lai, tmean, years)
'''

import warnings
import numpy as np
from typing import Callable, Sequence, Tuple, Union, Iterable
from numbers import Number

PFT_VALID = (1,2,3,4,5,6,7,8,9,10,12)

class MOD17(object):
    '''
    The MODIS MxD17 Gross Primary Productivity and Net Photosynthesis model.
    The required model parameters are:

    - `LUE_max`: Maximum light-use efficiency (kg C MJ-1)
    - `tmin0` and `tmin1`: The lower and upper bounds on the temperature
        response of photosynthesis (deg C); i.e., temperature at which stomata
        are fully closed and fully open, respectively
    - `vpd0` and `vpd0`: The lower and upper bounds on the response to VPD
        of photosynthesis (Pa); i.e., VPD at which stomata are full open and
        fully closed, respectively
    - `SLA`: Specific leaf area, or projected leaf area per kilogram of C
        [LAI kg C-1]
    - `q10`: Exponent shape parameter controlling respiration as a function of
        temperature (in degrees C) (unitless)
    - `froot_leaf_ratio`: The ratio of fine root C to leaf C (unitless).
    - `livewood_leaf_ratio`: Ratio of live wood carbon to annual maximum leaf
        carbon
    - `leaf_mr_base`: Maintenance respiration per unit leaf carbon per day at
        a reference temperature of 20 degrees C [kg C (kg C)-1 day-1]
    - `froot_mr_base`: Maintenance respiration per unit fine root carbon per
        day at a reference temperature of 20 degrees C [kg C (kg C)-1 day-1]
    - `livewood_mr_base`: Maintenance respiration per unit live wood carbon
        per day at a reference temperature of 20 degrees C [kg C (kg C)-1 d-1]

    NOTE: This class includes private class methods `MOD17._gpp()` and
    `MOD17._gpp()`, that avoid the overhead associated with creating a model
    instance; it should be used, e.g., for model calibration because it is faster
    and produces the same results as `MOD17.daily_gpp()`.

    NOTE: For multiple PFTs, vectorized parameters array can be passed; i.e.,
    a dictionary where each value is an (N,) array for N sites.

    Parameters
    ----------
    params : dict
        Dictionary of model parameters
    '''
    required_parameters = [
        'LUE_max', 'tmin0', 'tmin1', 'vpd0', 'vpd1', 'SLA',
        'Q10_livewood', 'Q10_froot', 'froot_leaf_ratio',
        'livewood_leaf_ratio', 'leaf_mr_base', 'froot_mr_base',
        'livewood_mr_base'
    ]

    def __init__(self, params: dict):
        self.params = params
        for key in self.required_parameters:
            setattr(self, key, params[key])

    @staticmethod
    def _gpp(params, fpar, tmin, vpd, par):
        'Daily GPP as static method, avoids overhead of class instantiation'
        # "params" argument should be a Sequence of atomic parameter values
        #   in the order prescribed by "required_parameters"
        tmin_scalar = linear_constraint(params[1], params[2])(tmin)
        vpd_scalar = linear_constraint(
            params[3], params[4], form = 'reversed')(vpd)
        lue = params[0] * tmin_scalar * vpd_scalar
        return 1e3 * lue * fpar * par

    @staticmethod
    def _npp(params, fpar, tmin, vpd, par, lai, tmean, years):
        '''
        Annual NPP as static method, avoids overhead of class instantiation.
        NOTE: It's assumed that the elements of `years` are in chronological
        order on the first axis (time axis).
        '''
        # "params" argument should be a Sequence of atomic parameter values
        #   in the order prescribed by "required_parameters"
        gpp = MOD17._gpp(params, fpar, tmin, vpd, par)
        # Daily respiration
        leaf_mass = lai / params[5] # LAI divided by SLA -> leaf mass [kg m-2]
        froot_mass = leaf_mass * params[8] # Leaf mass times `froot_leaf_ratio`
        # NOTE: Q10 calculated differently depending on the component
        _exp = (tmean - 20) / 10
        q10_leaf = np.power(3.22 - 0.046 * tmean, _exp)
        q10_froot = np.power(params[7], _exp)
        # Convert leaf, froot mass from [kg C m-2] to [g C m-2], then...
        r_leaf = 1e3 * leaf_mass * params[10] * q10_leaf
        r_froot = 1e3 * froot_mass * params[11] * q10_froot
        # Accumulate respiration over each year
        all_years = np.unique(years).tolist()
        # Pre-allocate arrays
        mr_leaf = np.full((len(all_years), *lai.shape[1:],), np.nan)
        mr_froot = np.full((len(all_years), *lai.shape[1:],), np.nan)
        mr_livewood = np.full((len(all_years), *lai.shape[1:],), np.nan)
        diff = np.full((len(all_years), *lai.shape[1:],), np.nan)
        for i, each_year in enumerate(all_years):
            # Sum respiration for each tissue in each year
            mr_leaf[i] = np.nansum(
                np.where(years == each_year, r_leaf, 0), axis = 0)
            mr_froot[i] = np.nansum(
                np.where(years == each_year, r_froot, 0), axis = 0)
            livewood_mass = (np.nanmax(
                np.where(years == each_year, lai, np.nan), axis = 0
                ) / params[5]
            ) * params[9]
            # For consistency with other respiration components, livewood
            #   respiration should be zero, not NaN, when no respiration
            mrl = 1e3 * livewood_mass * params[12] *\
                np.power(params[6], (tmean - 20) / 10).sum(axis = 0)
            mr_livewood[i] = np.where(np.isnan(mrl), 0, mrl)
            # Total plant maintenance respiration
            r_m = mr_leaf[i] + mr_froot[i] + mr_livewood[i]
            # GPP - R_M
            diff[i] =  np.where(years == each_year, gpp, 0).sum(axis = 0) - r_m
        # Annual growth respiration is assumed to be 25% of (GPP - R_M); see
        #   Figure 1 of MOD17 User Guide; the User Guide is TOO CUTE about
        #   derivation; 0.8 == (1/1.25), hence the "25%" figure
        return np.where(diff < 0, 0, 0.8 * diff)

    @staticmethod
    def par(sw_rad: Number, period_hrs: Number = 1) -> Number:
        '''
        Calculates daily total photosynthetically active radiation (PAR) from
        (hourly) incoming short-wave radiation (SW_rad). PAR is assumed to
        be 45% of SW_rad.

        Parameters
        ----------
        swrad : int or float or numpy.ndarray
            Incoming short-wave radiation (W m-2)
        period_hrs : int
            Period over which radiation is measured, in hours (Default: 1)

        Returns
        -------
        int or float or numpy.ndarray
        '''
        # Convert SW_rad from [W m-2] to [MJ m-2], then take 45%;
        #   3600 secs hr-1 times (1 MJ / 1e6 Joules) == 0.0036
        return 0.45 * (0.0036 * (24 / period_hrs) * sw_rad)

    @staticmethod
    def vpd(qv10m: Number, pressure: Number, tmean: Number) -> Number:
        '''
        Computes vapor pressure deficit (VPD) from surface meteorology.

        Parameters
        ----------
        qv10m : int or float or numpy.ndarray
            Water vapor mixing ratio at 10-meter height (Pa)
        pressure : int or float or numpy.ndarray
            Atmospheric pressure (Pa)
        tmean : int or float or numpy.ndarray
            Mean daytime temperature (degrees C)

        Returns
        -------
        int or float or numpy.ndarray
        '''
        # Actual vapor pressure (Gates 1980, Biophysical Ecology, p.311)
        avp = (qv10m * pressure) / (0.622 + (0.379 * qv10m))
        # Saturation vapor pressure (similar to FAO formula)
        svp = 610.7 * np.exp((17.38 * tmean) / (239 + tmean))
        return svp - avp

    def annual_npp(
            self, fpar: Sequence, tmin: Sequence, vpd: Sequence,
            par: Sequence, lai: Sequence, tmean: Sequence, years: Sequence
            ) -> np.ndarray:
        '''
        Annual net primary productivity (NPP).

        Parameters
        ----------
        fpar : Sequence
            Fraction of PAR intercepted by the canopy [0, 1], a (T x ...)
            array for T number of days
        tmin : Sequence
            Daily minimum temperature (degrees C), a (T x ...) array for T
            number of days
        vpd : Sequence
            Daytime vapor pressure deficit (Pa), a (T x ...) array for T
            number of days
        par : Sequence
            Daily photosynthetically active radation (MJ m-2 day-1)
        lai : Sequence
            Leaf area index, daily, a (T x ...) array for T number of days
        tmean : Sequence
            Mean daily temperature (degrees C), a (T x ...) array for T number
            of days
        years : Sequence
            Sequence of integers indicating the year of each daily
            measurement, in order (e.g., [2015, 2015, ..., 2017]); a (T x ...)
            array for T number of days

        Returns
        -------
        numpy.ndarray
            Total annual NPP [g C m-2 year-1]
        '''
        r_leaf, r_froot, r_livewood = self.annual_respiration(
            lai, tmean, years)
        r_m = r_leaf + r_froot + r_livewood
        gpp = self.daily_gpp(fpar, tmin, vpd, par)
        diff = np.empty(r_m.shape)
        all_years = np.unique(years).tolist()
        for i, each_year in enumerate(all_years):
            # GPP - R_M
            diff[i] = np.where(years == each_year, gpp, 0).sum(axis = 0) - r_m[i]
        return np.where(diff < 0, 0, 0.8 * diff)

    def annual_respiration(
            self, lai: Sequence, tmean: Sequence, years: Sequence
            ) -> Iterable[Tuple[Sequence, Sequence, Sequence]]:
        '''
        Annual total maintenance respiration. Input datasets should have daily
        denominations and extend over one or more years.

        Parameters
        ----------
        lai : Sequence
            Leaf area index, daily, a (T x ...) array for T number of days
        tmean : Sequence
            Mean daily temperature (degrees C), a (T x ...) array for T number
            of days
        years : Sequence
            Sequence of integers indicating the year of each daily
            measurement, in order (e.g., [2015, 2015, ..., 2017]); a (T x ...)
            array for T number of days

        Returns
        -------
        tuple
            A 3-tuple of total annual (leaf, fine root, livewood) respiration
            with units of [g C m-2 year-1]
        '''
        assert lai.shape == years.shape,\
            'LAI array should conform with "years" array'
        assert tmean.shape == years.shape,\
            'Mean temperature array should conform with "years" array'
        r_leaf_daily, r_froot_daily = self.daily_respiration(lai, tmean)
        r_leaf, r_froot, r_livewood = [], [], []
        all_years = np.unique(years).tolist()
        all_years.sort()
        for i, each_year in enumerate(all_years):
            r_leaf.append(
                np.nansum(np.where(years == each_year, r_leaf_daily, 0),
                axis = 0))
            r_froot.append(
                np.nansum(np.where(years == each_year, r_froot_daily, 0),
                axis = 0))
            # Annual maximum leaf mass (kg C) converted to livewood mass
            #   by allometric relation (livewood_leaf_ratio)
            livewood_mass = (np.nanmax(
                    np.where(years == each_year, lai, np.nan), axis = 0
                ) / self.SLA) * self.livewood_leaf_ratio
            # Livewood maintenance respiration (g C day-1), converted from
            #   (kg C day-1), as the product of livewood_mass, base
            #   respiration rate (livewood_mr_base), and annual sum of the
            #   maint. respiration term (Q10), see Equation 1.10, User's Guide
            # NOTE: "livewood_mr_base" is denominated in days
            #   (kg C kg C-1 day-1) but that's okay because we took the annual
            #   sum of the Q10 respiration, essentially multipling by ~365
            rl = 1e3 * livewood_mass * self.livewood_mr_base *\
                np.power(self.Q10_livewood, (tmean - 20) / 10).sum(axis = 0)
            # For consistency with other respiration components, livewood
            #   respiration should be zero, not NaN, when no respiration
            r_livewood.append(np.where(np.isnan(rl), 0, rl))
        return (np.stack(r_leaf), np.stack(r_froot), np.vstack(r_livewood))

    def daily_gpp(
            self, fpar: Number, tmin: Number, vpd: Number,
            par: Number) -> Number:
        r'''
        Daily gross primary productivity (GPP).

        $$
        \mathrm{GPP} = \varepsilon\times f(T_{min})\times f(V)\times
            [\mathrm{PAR}]\times [\mathrm{fPAR}]
        $$

        Where $T_{min}$ is the minimum daily temperature, $V$ is the daytime
        vapor pressure deficit (VPD), PAR is daily photosynthetically active
        radiation, fPAR is the fraction of PAR absorbed by the canopy, and
        $\varepsilon$ is the intrinsic (or maximum) light-use efficiency.

        Parameters
        ----------
        fpar : int or float or numpy.ndarray
            Fraction of PAR intercepted by the canopy [0, 1]
        tmin : int or float or numpy.ndarray
            Daily minimum temperature (degrees C)
        vpd : int or float or numpy.ndarray
            Daytime vapor pressure deficit (Pa)
        par : int or float or numpy.ndarray
            Daily photosynthetically active radation (MJ m-2 day-1)

        Returns
        -------
        int or float or numpy.ndarray
            Daily GPP flux in [g C m-2 day-1]
        '''
        return 1e3 * self.lue(tmin, vpd) * fpar * par

    def daily_respiration(
            self, lai: Number, tmean: Number
            ) -> Iterable[Tuple[Number, Number]]:
        r'''
        Daily maintenance respiration for leaves and fine roots.

        Maintenance respiration ($r_m$) for leaves or fine roots is given:

        $$
        $r_m$ = m \times r_0 \times q^{\frac{T - 20}{10}}
        $$

        Where $m$ is either the leaf mass or fine root mass; $r_0$ is the rate
        of maintenance respiration per unit leaf carbon (per day, at 20
        degrees C); and $q$ is the Q10 factor.

        NOTE: For fine roots and live wood, Q10 is a constant value of 2.0.
        For leaves, the temperature-acclimated Q10 equation of Tjoelker et al.
        (2001, Global Change Biology) is used:

        $$
        Q_{10} = 3.22 - 0.046 * T_{avg}
        $$

        The "net photosynthesis" quantity in MOD17, even though it is a bit
        misleading (it does not account for growth respiration and livewood
        $r_m$) can then be calculated as GPP less the maintenance respiration
        of leaves and fine roots:

        $$
        P_{net} = [\mathrm{GPP}] - r_{leaf} - r_{root}
        $$

        Parameters
        ----------
        lai : float or numpy.ndarray
            Leaf area index, daily
        tmean : float or numpy.ndarray
            Mean daily temperature (degrees C)

        Returns
        -------
        tuple
            2-element tuple of (leaf respiration, fine root respiration) in
            units of [g C m-2 day-1]
        '''
        # Leaf mass, fine root mass (Eq 1.4, 1.5 in MOD17 User Guide)
        leaf_mass = lai / self.SLA
        froot_mass = leaf_mass * self.froot_leaf_ratio
        # NOTE: Q10 calculated differently depending on the component
        _exp = (tmean - 20) / 10 # Equations 1.6 and 1.7
        q10_leaf = np.power(3.22 - 0.046 * tmean, _exp) # Equation 1.11
        q10_froot = np.power(self.Q10_froot, _exp)
        # NOTE: Converting from [kg C] to [g C] via *1e3
        r_leaf = 1e3 * leaf_mass * self.leaf_mr_base * q10_leaf
        r_froot = 1e3 * froot_mass * self.froot_mr_base * q10_froot
        return (r_leaf, r_froot)

    def daily_net_photosynthesis(
            self, fpar: Number, tmin: Number, vpd: Number, par: Number,
            lai: Number, tmean: Number) -> Number:
        '''
        Daily net photosynthesis ("PSNet"). See:

        - `MOD17.daily_gpp()`
        - `MOD17.daily_respiration()`

        Parameters
        ----------
        fpar : int or float or numpy.ndarray
            Fraction of PAR intercepted by the canopy [0, 1]
        tmin : int or float or numpy.ndarray
            Daily minimum temperature (degrees C)
        vpd : int or float or numpy.ndarray
            Daytime vapor pressure deficit (Pa)
        par : int or float or numpy.ndarray
            Daily photosynthetically active radation (MJ m-2 day-1)
        lai : float or numpy.ndarray
            Leaf area index, daily
        tmean : float or numpy.ndarray
            Mean daily temperature (degrees C)
        '''
        gpp = self.daily_gpp(fpar, tmin, vpd, par)
        r_leaf, r_froot = self.daily_respiration(lai, tmean)
        # See MOD17 User Gudie, Equation 1.8
        return gpp - r_leaf - r_froot

    def lue(self, tmin: Number, vpd: Number) -> Number:
        '''
        The instantaneous light-use efficiency (LUE), reduced by environmental
        stressors (low minimum temperature, high VPD) from the maximum LUE.

        Parameters
        ----------
        tmin : int or float or numpy.ndarray
        vpd : int or float or numpy.ndarray

        Returns
        -------
        float or numpy.ndarray
        '''
        return self.LUE_max * self.tmin_scalar(tmin) * self.vpd_scalar(vpd)

    def tmin_scalar(self, x: Number) -> Number:
        '''
        Parameters
        ----------
        x : int or float or numpy.ndarray
            Minimum temperature (deg C)

        Returns
        -------
        int or float or numpy.ndarray
        '''
        return linear_constraint(self.tmin0, self.tmin1)(x)

    def vpd_scalar(self, x: Number) -> Number:
        '''
        The environmental scalar for vapor pressure deficit (VPD).

        Parameters
        ----------
        x : int or float or numpy.ndarray
            Vapor pressure deficit (Pa)

        Returns
        -------
        int or float or numpy.ndarray
        '''
        return linear_constraint(self.vpd0, self.vpd1, form = 'reversed')(x)


def linear_constraint(
        xmin: Number, xmax: Number, form: str = None) -> Callable:
    '''
    Returns a linear ramp function, for deriving a value on [0, 1] from
    an input value `x`:

        if x >= xmax:
            return 1
        if x <= xmin:
            return 0
        return (x - xmin) / (xmax - xmin)

    Parameters
    ----------
    xmin : int or float
        Lower bound of the linear ramp function
    xmax : int or float
        Upper bound of the linear ramp function
    form : str
        Type of ramp function: "reversed" decreases as x increases;
        "binary" returns xmax when x == 1; default (None) is increasing
        as x increases.

    Returns
    -------
    function
    '''
    assert form is None or form in ('reversed', 'binary'),\
        'Argument "form" must be None or one of: "reversed", "binary"'
    assert form == 'binary' or np.any(xmax >= xmin),\
        'xmax must be greater than/ equal to xmin'
    if form == 'reversed':
        return lambda x: np.where(x >= xmax, 0,
            np.where(x < xmin, 1, 1 - np.divide(
                np.subtract(x, xmin), xmax - xmin)))
    if form == 'binary':
        return lambda x: np.where(x == 1, xmax, xmin)
    return lambda x: np.where(x >= xmax, 1,
        np.where(x < xmin, 0,
            np.divide(np.subtract(x, xmin), xmax - xmin)))


def suppress_warnings(func):
    'Decorator to suppress NumPy warnings'
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return func(*args, **kwargs)
    return inner
