[![DOI](https://zenodo.org/badge/607343238.svg)](https://zenodo.org/badge/latestdoi/607343238)

MODIS MOD17 Terrestrial Productivity Algorithm
==============================================

The MODIS MOD17 algorithm provided the first global, continuous, weekly estimates of ecosystem gross primary productivity (GPP) and annual estimates of net primary productivity (NPP). This source code can be used for comprehensive calibration, validation, sensitivity, and uncertainty analysis of the MOD17 algorithm. It was used by Endsley et al. (In Review) for the final recalibration of MODIS MOD17 and the development of a new, VIIRS-based VNP17 global productivity algorithm.

MOD17 consists of three potentially independent sub-models:

- 8-day gross primary productivity (GPP)
- 8-day net photosynthesis
- Annual net primary productivity (NPP)

8-day composite products are given the designation MOD17A2H, for Terra MODIS, or MYD17A2H, for Aqua MODIS. Annual products, including annual GPP (the sum of one year's 8-day GPP composites), are carried by MOD17A3H (or MYD17A3H). The new VIIRS products would be designated VNP17A2H and VNP17A3H. GPP is calculated using a classic light-use efficiency (LUE) approach (Running et al. 2004, Yuan et al. 2014, Madani et al. 2017), where the carbon (C) uptake by plants is assumed to be proportional to canopy absorbed photosynthetically active radiation (APAR) under prevailing daytime environmental conditions for diel or longer time scales. Low temperatures or high vapor pressure deficit (VPD) reduce the efficiency of photosynthetic C uptake, thus, MOD17 GPP is described as a product of APAR, the light-use efficiency under optimal conditions ($\varepsilon_{\mathrm{max}}$), and environmental scalars.


Installation
------------

Within the `MOD17` repository's root directory:

```sh
pip install .
```

Tests can be run with:

```sh
python tests/tests.py
```


Example Use
-----------

**To make model predictions, e.g., for daily GPP:**

```py
from mod17 import MOD17, PFT_VALID
from mod17.utils import restore_bplut

N_SITES = 10 # Number of sites

# Read-in a BPLUT file, which contains model parameters
params_dict = restore_bplut(BPLUT_FILE)

# Create a vectorized BPLUT; there are 5 GPP parameters
params_vector = params_dict.copy()
for p_name in MOD17.required_parameters[0:5]:
    # "pft_map" refers to a 1D array of numeric PFT codes
    params_vector[p_name] = params_dict[p_name][pft_map].reshape((1, N_SITES))

# Create model, get predictions; "drivers" refers to a list of the expected
#   arguments to the daily_gpp() function, each is an array representing
#   a driver dataset (e.g., minimum temperature)
model = MOD17(params_vector)
gpp = model.daily_gpp(*drivers)
```

**For a more complete example of GPP simulation, see:**

```
docs/examples/MOD17_GPP_forward_run_Collection6-1.py
```

And download the following driver dataset: [http://doi.org/10.5281/zenodo.7682806](http://doi.org/10.5281/zenodo.7682806)

**To calibrate the GPP model from the command line**

```sh
# Optimize the parameters for PFT 1 (Evergreen Needleleaf); calibration
#   data and other options are described by the configuration file, e.g.:
#       mod17/data/MOD17_calibration.config
python calibration.py tune-gpp --pft=1

# Export the thinned, mean a posteriori estimates from the command line;
#   in this example a burn-in of 1000 samples, taking every 10th sample
python calibration.py export-bplut output.csv --burn=1000 --thin=10
```


Citation
--------

**If using this software, please refer to the DOI:**

```
10.5281/zenodo.8045097
```

**And cite the following paper:**

Endsley, K.A., M. Zhao, J.S. Kimball, S. Devadiga. In Review. Continuity of global MODIS terrestrial primary productivity estimates in the VIIRS era using model-data fusion. *Journal of Geophysical Research: Biogeosciences.*



References
----------

Madani, N., J. S. Kimball, and S. W. Running. 2017. Improving global gross primary productivity estimates by computing optimum light use efficiencies using flux tower data. *Journal of Geophysical Research: Biogeosciences* 122 (11):2939–2951.

Running, S. W., R. R. Nemani, F. A. Heinsch, M. Zhao, M. Reeves, and H. Hashimoto. 2004. A continuous satellite-derived measure of global terrestrial primary production. *BioScience* 54 (6):547.

Yuan, W., W. Cai, J. Xia, J. Chen, S. Liu, W. Dong, et al. 2014. Global comparison of light use efficiency models for simulating terrestrial vegetation gross primary production based on the LaThuile database. *Agricultural and Forest Meteorology* 192–193:108–120.
