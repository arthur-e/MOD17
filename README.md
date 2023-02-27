MODIS MOD17 Terrestrial Productivity Algorithm
==============================================

The MODIS MOD17 algorithm provided the first global, continuous, weekly estimates of ecosystem gross primary productivity (GPP) and annual estimates of net primary productivity (NPP). This source code can be used for comprehensive calibration, validation, sensitivity, and uncertainty analysis of the MOD17 algorithm. It was used by Endsley et al. (In Review) for the final recalibration of MODIS MOD17 and the development of a new, VIIRS-based VNP17 global productivity algorithm.


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


Citation
--------

**If using this software, please cite the following paper:**

```
Endsley, K.A., M. Zhao, J.S. Kimball, S. Devadiga. In Review. Continuity of global MODIS terrestrial primary productivity estimates in the VIIRS era using model-data fusion. Journal of Geophysical Research: Biogeosciences.
```
