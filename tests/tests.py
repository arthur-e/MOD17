'''
Unit tests for the mod17 Python utilities library.
'''

import os
import unittest
import numpy as np
import mod17
from mod17 import MOD17
from mod17.utils import restore_bplut
from mod17.srs import modis_from_wgs84, modis_to_wgs84, modis_tile_from_wgs84, modis_row_col_from_wgs84, modis_row_col_to_wgs84

MOD17_BPLUT = os.path.join(
    os.path.dirname(mod17.__file__), 'data/MOD17_BPLUT_CX.X_MERRA_NASA.csv')

class GPP(unittest.TestCase):
    '''
    Suite of GPP test cases based on the Collection 5.1 BPLUT.
    '''
    @classmethod
    def setUp(cls):
        cls.pft = np.arange(0, 13)
        cls.params = dict([
            (k, v[cls.pft]) for k, v in restore_bplut(MOD17_BPLUT).items()
        ])
        cls.drivers_annual = [ # Sequence of daily driver data over 1 year
            np.repeat(np.array((0.5,)), 365)[:,np.newaxis]\
                .repeat(cls.pft.size, axis = -1), # fPAR
            np.repeat(np.array((15,)), 365)[:,np.newaxis]\
                .repeat(cls.pft.size, axis = -1), # Tmin
            np.repeat(np.array((1000,)), 365)[:,np.newaxis]\
                .repeat(cls.pft.size, axis = -1), # VPD
            np.repeat(np.array((10,)), 365)[:,np.newaxis]\
                .repeat(cls.pft.size, axis = -1), # PAR
            np.repeat(np.array((1,)), 365)[:,np.newaxis]\
                .repeat(cls.pft.size, axis = -1), # LAI
            np.repeat(np.array((25,)), 365)[:,np.newaxis]\
                .repeat(cls.pft.size, axis = -1), # Tmean
            np.repeat(np.array((2010,)), 365)[:,np.newaxis]\
                .repeat(cls.pft.size, axis = -1) # years
        ]

    def test_gpp(self):
        'Should model GPP as expected'
        model = MOD17(self.params)
        par = model.par(100)
        answer = np.array([
            np.nan, 2., 2.73, 2.04, 2.51, 2.01, 2.63,
            1.81, 2.37, 2.31, 1.91, np.nan, 2.07
        ])
        pred = model.daily_gpp(fpar = 0.5, tmin = 10, vpd = 1000, par = par)\
            .round(2)
        self.assertTrue(
            np.logical_or(np.isnan(pred), np.equal(pred, answer)).all())

    def test_gpp_static_method(self):
        'Should model GPP as expected, from class method'
        model = MOD17(self.params)
        params = [
            self.params[p][2] # i.e., PFT = 2
            for p in ('LUE_max', 'tmin0', 'tmin1', 'vpd0', 'vpd1')
        ]
        par = model.par(100)
        predicted = MOD17._gpp(params, 0.5, 10, 1000, par).round(2)
        self.assertTrue(np.equal(predicted, 2.73).all())

    def test_gpp_temperature_ramp(self):
        'Should accurately model GPP over a range of temperatures'
        tmin_sweep = np.arange(-10, 35, 5)
        answer = np.array([
            np.full((9,), np.nan), # i.e., PFT == 0
            [0., 0.37, 0.98, 1.6,  2.,   2.,   2.,   2.,   2.  ],
            [0., 0.48, 1.28, 2.08, 2.73, 2.73, 2.73, 2.73, 2.73],
            [0., 0.34, 0.91, 1.48, 2.04, 2.09, 2.09, 2.09, 2.09],
            [0., 0.16, 0.94, 1.73, 2.51, 2.51, 2.51, 2.51, 2.51],
            [0., 0.24, 0.85, 1.46, 2.01, 2.01, 2.01, 2.01, 2.01],
            [0., 0.47, 1.27, 2.06, 2.63, 2.63, 2.63, 2.63, 2.63],
            [0., 0.32, 0.86, 1.4,  1.81, 1.81, 1.81, 1.81, 1.81],
            [0., 0.4,  1.05, 1.71, 2.37, 2.55, 2.55, 2.55, 2.55],
            [0., 0.39, 1.03, 1.67, 2.31, 2.49, 2.49, 2.49, 2.49],
            [0., 0.32, 0.85, 1.38, 1.91, 2.13, 2.13, 2.13, 2.13],
            np.full((9,), np.nan), # i.e, PFT == 11
            [0., 0.34, 0.92, 1.49, 2.07, 2.3,  2.3,  2.3,  2.3 ]])
        model = MOD17(self.params)
        for i, tmin in enumerate(tmin_sweep.tolist()):
            par = model.par(100)
            pred = model.daily_gpp(fpar = 0.5, tmin = tmin, vpd = 1000, par = par).round(2)
            self.assertTrue(np.logical_or(
                np.isnan(pred),
                np.equal(answer[:,i], pred)).all())

    def test_gpp_vpd_ramp(self):
        'Should accurately model GPP over a range of VPD'
        vpd_sweep = np.arange(0, 6000, 1000)
        answer = np.array([
            np.full((6,), np.nan), # i.e., PFT == 0
            [2.35, 2.  , 1.  , 0.  , 0.  , 0.],
            [2.73, 2.73, 1.82, 0.91, 0.  , 0.],
            [2.39, 2.09, 1.26, 0.42, 0.  , 0.],
            [2.97, 2.51, 1.19, 0.  , 0.  , 0.],
            [2.38, 2.01, 0.95, 0.  , 0.  , 0.],
            [2.91, 2.63, 1.83, 1.04, 0.24, 0.],
            [2.  , 1.81, 1.28, 0.75, 0.21, 0.],
            [2.91, 2.55, 1.53, 0.51, 0.  , 0.],
            [2.83, 2.49, 1.53, 0.57, 0.  , 0.],
            [2.36, 2.13, 1.46, 0.8 , 0.13, 0.],
            np.full((6,), np.nan), # i.e., PFT == 11
            [2.53, 2.3 , 1.64, 0.98, 0.33, 0.]])
        model = MOD17(self.params)
        for i, vpd in enumerate(vpd_sweep.tolist()):
            par = model.par(100)
            pred = model.daily_gpp(
                fpar = 0.5, tmin = 20, vpd = vpd, par = par).round(2)
            self.assertTrue(np.logical_or(
                np.isnan(pred),
                np.equal(answer[:,i], pred)).all())

    def test_daily_respiration(self):
        'Should accurately calculate daily respiration'
        lai = 1
        tmean = 25
        model = MOD17(self.params)
        pred_leaf, pred_froot = model.daily_respiration(lai, tmean)
        self.assertTrue(np.logical_or(
            np.isnan(pred_leaf),
            np.equal(pred_leaf.round(3), np.array([
                np.nan, 0.579, 0.323, 0.694, 0.453, 0.495, 1.33, 0.622, 0.434,
                0.433, 0.371, np.nan, 0.371
            ]))
        ).all())
        self.assertTrue(np.logical_or(
            np.isnan(pred_froot),
            np.equal(pred_froot.round(3), np.array([
                np.nan, 0.587, 0.3, 0.738, 0.327, 0.357, 0.781, 0.795, 0.459,
                0.457, 0.792, np.nan, 0.61
            ]))
        ).all())

    def test_daily_respiration_lai_ramp(self):
        'Should accurately calculate daily respiration over range of LAI'
        lai_sweep = np.array((0.1, 1, 2, 3, 4.5))
        tmean = 25
        model = MOD17(self.params)
        answer1 = np.array([
            [np.nan, 0.06, 0.03, 0.07, 0.05, 0.05, 0.13, 0.06, 0.04, 0.04, 0.04, np.nan, 0.04],
            [np.nan, 0.58, 0.32, 0.69, 0.45, 0.50, 1.33, 0.62, 0.43, 0.43, 0.37, np.nan, 0.37],
            [np.nan, 1.16, 0.65, 1.39, 0.91, 0.99, 2.66, 1.24, 0.87, 0.87, 0.74, np.nan, 0.74],
            [np.nan, 1.74, 0.97, 2.08, 1.36, 1.49, 3.99, 1.87, 1.3 , 1.30, 1.11, np.nan, 1.11],
            [np.nan, 2.61, 1.45, 3.12, 2.04, 2.23, 5.99, 2.80, 1.95, 1.95, 1.67, np.nan, 1.67],
        ])
        answer2 = np.array([
            [np.nan, 0.06, 0.03, 0.07, 0.03, 0.04, 0.08, 0.08, 0.05, 0.05, 0.08, np.nan, 0.06],
            [np.nan, 0.59, 0.30, 0.74, 0.33, 0.36, 0.78, 0.80, 0.46, 0.46, 0.79, np.nan, 0.61],
            [np.nan, 1.17, 0.60, 1.48, 0.65, 0.71, 1.56, 1.59, 0.92, 0.91, 1.58, np.nan, 1.22],
            [np.nan, 1.76, 0.90, 2.21, 0.98, 1.07, 2.34, 2.39, 1.38, 1.37, 2.38, np.nan, 1.83],
            [np.nan, 2.64, 1.35, 3.32, 1.47, 1.61, 3.51, 3.58, 2.06, 2.06, 3.57, np.nan, 2.74],
        ])
        for i, lai in enumerate(lai_sweep.tolist()):
            pred1, pred2 = model.daily_respiration(lai, tmean)
            pred1 = pred1.round(2)
            pred2 = pred2.round(2)
            self.assertTrue(np.logical_or(
                np.isnan(answer1[i]), np.equal(answer1[i], pred1)).all())
            self.assertTrue(np.logical_or(
                np.isnan(answer2[i]), np.equal(answer2[i], pred2)).all())

    def test_daily_respiration_temperature_ramp(self):
        'Should accurately calculate daily respiration over temperature range'
        tmean_sweep = np.arange(-10, 35, 5)
        lai = 1
        model = MOD17(self.params)
        answer1 = np.array([
            [np.nan, 0.01, 0.  , 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, np.nan, 0.01],
            [np.nan, 0.02, 0.01, 0.02, 0.01, 0.02, 0.04, 0.02, 0.01, 0.01, 0.01, np.nan, 0.01],
            [np.nan, 0.04, 0.02, 0.05, 0.03, 0.03, 0.09, 0.04, 0.03, 0.03, 0.02, np.nan, 0.02],
            [np.nan, 0.08, 0.04, 0.09, 0.06, 0.07, 0.18, 0.08, 0.06, 0.06, 0.05, np.nan, 0.05],
            [np.nan, 0.15, 0.08, 0.17, 0.11, 0.12, 0.33, 0.16, 0.11, 0.11, 0.09, np.nan, 0.09],
            [np.nan, 0.25, 0.14, 0.3 , 0.2 , 0.22, 0.58, 0.27, 0.19, 0.19, 0.16, np.nan, 0.16],
            [np.nan, 0.4 , 0.22, 0.48, 0.31, 0.34, 0.92, 0.43, 0.3 , 0.3 , 0.26, np.nan, 0.26],
            [np.nan, 0.58, 0.32, 0.69, 0.45, 0.5 , 1.33, 0.62, 0.43, 0.43, 0.37, np.nan, 0.37],
            [np.nan, 0.74, 0.41, 0.89, 0.58, 0.63, 1.7 , 0.8 , 0.56, 0.55, 0.47, np.nan, 0.47],
        ])
        answer2 = np.array([
            [np.nan, 0.05, 0.03, 0.07, 0.03, 0.03, 0.07, 0.07, 0.04, 0.04, 0.07, np.nan, 0.05],
            [np.nan, 0.07, 0.04, 0.09, 0.04, 0.04, 0.1 , 0.1 , 0.06, 0.06, 0.1 , np.nan, 0.08],
            [np.nan, 0.1 , 0.05, 0.13, 0.06, 0.06, 0.14, 0.14, 0.08, 0.08, 0.14, np.nan, 0.11],
            [np.nan, 0.15, 0.08, 0.18, 0.08, 0.09, 0.2 , 0.2 , 0.11, 0.11, 0.2 , np.nan, 0.15],
            [np.nan, 0.21, 0.11, 0.26, 0.12, 0.13, 0.28, 0.28, 0.16, 0.16, 0.28, np.nan, 0.22],
            [np.nan, 0.29, 0.15, 0.37, 0.16, 0.18, 0.39, 0.4 , 0.23, 0.23, 0.4 , np.nan, 0.3 ],
            [np.nan, 0.42, 0.21, 0.52, 0.23, 0.25, 0.55, 0.56, 0.32, 0.32, 0.56, np.nan, 0.43],
            [np.nan, 0.59, 0.3 , 0.74, 0.33, 0.36, 0.78, 0.8 , 0.46, 0.46, 0.79, np.nan, 0.61],
            [np.nan, 0.83, 0.42, 1.04, 0.46, 0.51, 1.1 , 1.12, 0.65, 0.65, 1.12, np.nan, 0.86],
        ])
        for i, tmean in enumerate(tmean_sweep.tolist()):
            pred1, pred2 = model.daily_respiration(lai, tmean)
            pred1 = pred1.round(2)
            pred2 = pred2.round(2)
            self.assertTrue(np.logical_or(
                np.isnan(answer1[i]), np.equal(answer1[i], pred1)).all())
            self.assertTrue(np.logical_or(
                np.isnan(answer2[i]), np.equal(answer2[i], pred2)).all())

    def test_annual_respiration(self):
        'Should correctly calculate annual respiration'
        lai = np.repeat(np.array((1,)), 365)[:,np.newaxis]\
            .repeat(self.pft.size, axis = -1)
        tmean = np.repeat(np.array((25,)), 365)[:,np.newaxis]\
            .repeat(self.pft.size, axis = -1)
        years = np.repeat(np.array((2010,)), 365)[:,np.newaxis]\
            .repeat(self.pft.size, axis = -1)
        model = MOD17(self.params)
        r_leaf, r_froot, r_livewood = model.annual_respiration(
            lai, tmean, years)
        self.assertTrue(np.equal(r_leaf.round(1), np.array([[
            0., 211.5, 117.9, 253.2, 165.4, 180.8, 485.5, 227.1, 158.5,
            157.9, 135.4, 0., 135.4
        ]])).all())
        self.assertTrue(np.equal(r_froot.round(1), np.array([[
            0., 214.3, 109.6, 269.5, 119.3, 130.4, 285., 290.2, 167.4,
            166.9, 289.3, 0., 222.5
        ]])).all())
        self.assertTrue(np.equal(r_livewood.round(1), np.array([[
            0., 24.9, 12.3, 20., 15.7, 17.2, 18.9, 3.8, 5.1, 0.9, 0., 0., 0.
        ]])).all())

    def test_annual_npp(self):
        'Should correctly calculate annual NPP'
        fpar, tmin, vpd, par, lai, tmean, years = self.drivers_annual
        model = MOD17(self.params)
        npp = model.annual_npp(fpar, tmin, vpd, par, lai, tmean, years)
        npp[np.isnan(npp)] = 0
        self.assertTrue(np.equal(npp.round(0), np.array([[
            0., 1144., 1859., 1137., 1641., 1249., 1342., 943., 1654.,
            1610., 1259., 0., 1439.
        ]])).all())

    def test_annual_npp_static_method(self):
        'Should calculate annual NPP same when using low-level API'
        pft = 1
        params = [
            self.params[p][pft] for p in MOD17.required_parameters
        ]
        # Get the driver data for the specific PFT
        drivers = map(lambda x: x[:,pft], self.drivers_annual)
        npp = MOD17._npp(params, *drivers)
        self.assertEqual(npp.round(2), 1144.22)


class CoordinateTransformations(unittest.TestCase):
    '''
    Suite of tests related to coordinate transformations involving the MODIS
    Sinusoidal projection.
    '''
    wgs84_coords = [ # (Longitude, Latitude)
        ( 30.5, -25.5),
        (-50.5, -30.1),
        (-10.1,  45.6),
        (125.1,  65.5),
    ]
    sinusoidal_coords = [
        ( 3061072.0, -2835473.8),
        (-4858128.1, -3346971.1),
        ( -785770.9,  5070494.4),
        ( 5768590.8,  7283275.9),
    ]
    tiles = [ # (h, v)
        (20, 11), (13, 12), (17, 4), (23, 2),
    ]
    row_col_500m = [
        (1319, 1806), (23, 1513), (1055, 703), (1079, 450)
    ]
    row_col_1000m = [
        (659, 902), (11, 756), (527, 351), (539, 224)
    ]

    def test_modis_from_wgs84(self):
        'Should correctly determine Sinusoidal coordinates from WGS84 coords.'
        for i, pair in enumerate(self.wgs84_coords):
            x, y = modis_from_wgs84(pair)
            answer = self.sinusoidal_coords[i]
            self.assertTrue(
                x.round(1) == answer[0] and y.round(1) == answer[1])
        # Test vectorized version
        self.assertTrue(np.equal(
            np.stack(self.sinusoidal_coords, axis = -1),
            modis_from_wgs84(np.stack(self.wgs84_coords, axis = -1)).round(1)
        ).all())

    def test_modis_to_wgs84(self):
        'Should correctly determine WGS84 coordinates from Sinusoidal coords.'
        for i, pair in enumerate(self.sinusoidal_coords):
            x, y = modis_to_wgs84(pair)
            answer = self.wgs84_coords[i]
            self.assertTrue(
                x.round(1) == answer[0] and y.round(1) == answer[1])
        # Test vectorized version
        self.assertTrue(np.equal(
            np.stack(self.wgs84_coords, axis = -1),
            modis_to_wgs84(np.stack(self.sinusoidal_coords, axis = -1)).round(1)
        ).all())

    def test_modis_row_col_from_wgs84_500m(self):
        'Should correctly determine MODIS row, column from WGS84 coordinates'
        for i, pair in enumerate(self.wgs84_coords):
            r, c = modis_row_col_from_wgs84(pair, nominal = 500)
            answer = self.row_col_500m[i]
            self.assertTrue(r == answer[0] and c == answer[1])
        # Test vectorized version
        self.assertTrue(np.equal(
            np.stack(self.row_col_500m, axis = -1),
            modis_row_col_from_wgs84(np.stack(self.wgs84_coords, axis = -1))
        ).all())

    def test_modis_row_col_from_wgs84_1000m(self):
        'Should correctly determine MODIS row, column from WGS84 coordinates'
        for i, pair in enumerate(self.wgs84_coords):
            r, c = modis_row_col_from_wgs84(pair, nominal = 1000)
            answer = self.row_col_1000m[i]
            self.assertTrue(r == answer[0] and c == answer[1])

    def test_modis_row_col_to_wgs84_500m(self):
        'Should correctly determine WGS84 coordinates from MODIS row, column'
        for i, pair in enumerate(self.row_col_500m):
            x, y = modis_row_col_to_wgs84(
                pair, h = self.tiles[i][0], v = self.tiles[i][1], nominal = 500)
            x, y = list(map(lambda x: round(x, 1), (x, y)))
            answer = self.wgs84_coords[i]
            self.assertTrue(x == answer[0] and y == answer[1])
        # Test vectorized version
        self.assertTrue(np.equal(
            np.stack(self.wgs84_coords, axis = -1),
            modis_row_col_to_wgs84(
                np.stack(self.row_col_500m, axis = -1),
                *np.stack(self.tiles, axis = -1)).round(1)
        ).all())

    def test_modis_row_col_to_wgs84_1000m(self):
        'Should correctly determine WGS84 coordinates from MODIS row, column'
        for i, pair in enumerate(self.row_col_1000m):
            x, y = modis_row_col_to_wgs84(
                pair, h = self.tiles[i][0], v = self.tiles[i][1], nominal = 1000)
            x, y = list(map(lambda x: round(x, 1), (x, y)))
            answer = self.wgs84_coords[i]
            self.assertTrue(x == answer[0] and y == answer[1])

    def test_modis_tile_from_wgs84(self):
        'Should correctly determine the MODIS tile from WGS84 coordinates'
        for i, pair in enumerate(self.wgs84_coords):
            h, v = modis_tile_from_wgs84(pair)
            self.assertTrue(h == self.tiles[i][0] and v == self.tiles[i][1])
        # Test vectorized version
        self.assertTrue(np.equal(
            np.stack(self.tiles, axis = -1),
            modis_tile_from_wgs84(np.stack(self.wgs84_coords, axis = 1))
        ).all())


if __name__ == '__main__':
    unittest.main()
