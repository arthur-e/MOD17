'''
MODIS sinusoidal projection forward and backward coordinate transformations,
courtesy of Giglio et al. (2018), Collection 6 MODIS Burned Area Product
User's Guide, Version 1, Appendix B:

- https://modis-fire.umd.edu/files/MODIS_C6_BA_User_Guide_1.2.pdf
'''

import h5py
import re
import numpy as np
from numbers import Number
from typing import Sequence, Tuple, Iterable

SPHERE_RADIUS = 6371007.181 # Radius of ideal sphere, meters
TILE_LINES = {250: 5000, 500: 2400, 1000: 1200} # Num. lines by nominal res.
TILE_SIZE = 1111950 # Width and height of MODIS tile in projection plane
XMIN = -20015109 # Western limit of projection plane, meters
YMAX = 10007555 # Nothern limit of projection plane, meters
VIIRS_METADATA = re.compile(
    r'.*XDim=(?P<xdim>\d+)'
    r'.*YDim=(?P<ydim>\d+)'
    r'.*UpperLeftPointMtrs=\((?P<ul>[0-9,\-\.]+)\)'
    r'.*LowerRightMtrs=\((?P<lr>[0-9,\-\.]+)\).*', re.DOTALL)
# Taken from a MOD15A2H EOS-HDF file; this seems to fit best
MODIS_SINUSOIDAL_PROJ_WKT = '''
PROJCS["unnamed",GEOGCS["Unknown datum based upon the custom spheroid",
  DATUM["Not specified (based on custom spheroid)",
  SPHEROID["Custom spheroid",6371007.181,0]],PRIMEM["Greenwich",0],
  UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],
  PROJECTION["Sinusoidal"],
  PARAMETER["longitude_of_center",0],
  PARAMETER["false_easting",0],
  PARAMETER["false_northing",0],
  UNIT["Meter",1],AXIS["Easting",EAST],
  AXIS["Northing",NORTH]]
'''.replace('\n', '')


def geotransform(
        hdf: h5py.File,
        ps: float = 463.31271653,
        nrow: int = 2400,
        ncol: int = 2400,
        metadata = VIIRS_METADATA
    ) -> Iterable[Tuple[Number, Number, Number, Number, Number, Number]]:
    '''
    Prescribe a geotransform tuple for the output GeoTIFF. For MODIS/VIIRS
    sinsuoidal projections, the lower right corner coordinates are "the only
    metadata that accurately reflect the extreme corners of the gridded image"
    (Myneni et al. 2018, VIIRS LAI/fPAR User Guide). So, instead of using the
    reported upper-left (UL) corner coordinates, it is more accurate to use
    the lower-right (LR) corner coordinates and calculate the position of the
    UL corner based on the width and height of the image and the pixel size.
    NOTE that a rather odd pixel size is required to get the correct results
    verified against the HDF-EOS-to-GeoTIFF (HEG) Conversion Tool; see also
    Giglio et al. (2018), "Collection 6 MODIS Burned Area Product User's
    Guide, Version 1" Table 1.

        https://modis-land.gsfc.nasa.gov/pdf/MODIS_C6_BA_User_Guide_1.2.pdf

    Parameters
    ----------
    hdf : h5py.File
    ps : int or float
        The pixel size; in units matching the linear units of the SRS
        (Default: 463.3127 meters)
    nrow : int
        Number of rows in the output image (Default: 2400 for MODIS/VIIRS)
    ncol : int
        Number of columns in the output image (Default: 2400 for MODIS/VIIRS)
    metadata : re.Pattern
        Compiled regex that captures important metadata fields

    Returns
    -------
    tuple
        (x_min, pixel_width, 0, y_max, 0, -pixel_height)
    '''
    meta = hdf['HDFEOS INFORMATION/StructMetadata.0'][()].decode()
    lr = VIIRS_METADATA.match(meta).groupdict()['lr'].split(',')
    return ( # Subtract distance (meters) from LR corner to obtain UR corner
        float(lr[0]) - (ncol * ps), ps, 0, float(lr[1]) + (nrow * ps), 0, -ps)


def modis_from_wgs84(coords: Sequence) -> Sequence:
    '''
    Given longitude-latitude coordinates, return the coordinates on the
    sinusoidal projection plane.

    This function is vectorized such that `coords` (a 2-element sequence),
    can contain one number each (a single coordinate pair) or multiple
    numbers; i.e., the first element is a vector of (multiple) longitudes and
    the second element is a vector of latitudes of the same length.

    Parameters
    ----------
    coords : tuple or list or numpy.ndarray
        (Longitude, Latitude) coordinate pair; for a numpy.ndarray, assumes
        that this is described by the first axis, which is length 2

    Returns
    -------
    numpy.ndarray
        (X, Y) coordinate pair in MODIS sinusoidal projection; a (2 x ...) array
    '''
    x, y = np.deg2rad(coords)
    return np.stack((SPHERE_RADIUS * x * np.cos(y), SPHERE_RADIUS * y))


def modis_to_wgs84(coords: Sequence) -> Sequence:
    '''
    Convert coordinates on the MODIS sinusoidal plane to longitude-latitude
    coordinates (WGS84).

    Parameters
    ----------
    coords : tuple or list or numpy.ndarray
        (X, Y) coordinate pair in MODIS sinusoidal projection; for a
        numpy.ndarray, assumes that this is described by the first axis,
        which is length 2

    Returns
    -------
    numpy.ndarray
        (Longitude, Latitude) coordinate pair; a (2 x ...) array
    '''
    x, y = coords
    lat = y / SPHERE_RADIUS # i.e., return value in radians
    lng = x / (SPHERE_RADIUS * np.cos(lat))
    return np.stack((np.rad2deg(lng), np.rad2deg(lat)))


def modis_tile_from_wgs84(coords: Sequence) -> Sequence:
    '''
    Given longitude-latitude coordinates, return the MODIS tile (H,V) that
    contains them.

    Parameters
    ----------
    coords : tuple or list or numpy.ndarray
        (Longitude, Latitude) coordinate pair; for a numpy.ndarray, assumes
        that this is described by the first axis, which is length 2

    Returns
    -------
    numpy.ndarray
        (H,V) tile identifier; a (2 x ...) array
    '''
    x, y = modis_from_wgs84(coords) # Get coordinates in the projection plane
    return np.stack((
        np.floor((x - XMIN) / TILE_SIZE),
        np.floor((YMAX - y) / TILE_SIZE)))


def modis_row_col_from_wgs84(
        coords: Sequence, nominal: int = 500) -> Sequence:
    '''
    Given longitude-latitude coordinates, return the corresponding row-column
    coordinates within a MODIS tile. NOTE: You'll need to determine which
    MODIS tile contains this row-column index with `modis_tile_from_wgs84()`.

    This function is vectorized such that `coords` (a 2-element sequence),
    can contain one number each (a single coordinate pair) or multiple
    numbers; i.e., the first element is a vector of (multiple) longitudes and
    the second element is a vector of latitudes of the same length.

    Parameters
    ----------
    coords : tuple or list or numpy.ndarray
        (Longitude, Latitude) coordinate pair in WGS84 projection
    nominal : int
        Nominal resolution of MODIS raster: 250 (meters), 500, or 1000

    Returns
    -------
    numpy.ndarray
        (Row, Column) coordinates; a (2 x ...) array
    '''
    x, y = modis_from_wgs84(coords) # Get coordinates in the projection plane
    # Get actual size of, e.g., "500-m" MODIS sinusoidal cell
    res = TILE_SIZE / float(TILE_LINES[nominal])
    out = np.stack((
        np.floor((((YMAX - y) % TILE_SIZE) / res) - 0.5),
        np.floor((((x - XMIN) % TILE_SIZE) / res) - 0.5),
    ))
    # Fix some edge cases where subtracting 0.5, taking the floor leads to -1
    return np.where(out == -1, 0, out)


def modis_row_col_to_wgs84(
        coords: Sequence, h: Number, v: Number, nominal: int = 500
    ) -> Sequence:
    '''
    Convert pixel coordinates in a specific MODIS tile to longitude-latitude
    coordinates. The "h" and "v" arguments if passed as arrays, must be
    conformable to the "coords" array.

    Parameters
    ----------
    coords : tuple or list or numpy.ndarray
        (X, Y) coordinate pair in MODIS sinusoidal projection
    h : int or numpy.ndarray
        MODIS tile "h" index
    v : int or numpy.ndarray
        MODIS tile "v" index
    nominal : int
        Nominal resolution of MODIS raster: 250 (meters), 500, or 1000

    Returns
    -------
    numpy.ndarray
        (Longitude, Latitude) coordinates; a (2 x ...) array
    '''
    r, c = coords
    lines = TILE_LINES[nominal]
    assert np.logical_and(
        np.logical_and(0 <= r, r <= (lines - 1)),
        np.logical_and(0 <= c, c <= (lines - 1))).all(),\
        'Row and column indices must be in the range [0, %d]' % (lines - 1)
    # Get actual size of, e.g., "500-m" MODIS sinusoidal cell
    res = TILE_SIZE / float(TILE_LINES[nominal])
    x = ((c + 0.5) * res) + (h * TILE_SIZE) + XMIN
    y = YMAX - ((r + 0.5) * res) - (v * TILE_SIZE)
    return modis_to_wgs84((x, y))
