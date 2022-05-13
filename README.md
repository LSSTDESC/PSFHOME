# Impact of PSF Higher Order Moments Error (HOME) on Weak Lensing (WL)

## Introduction

This repository includes software to quantitatively test the relation between the weak lensing shear bias and the modeling error of PSF higher order moments (beyond second).

It is based on the following paper:
 - Impact of Point Spread Function Higher Moments Error on Weak Gravitational Lensing; Zhang and Mandelbaum for LSST (2021) https://arxiv.org/abs/2107.05644
 - Impact of Point Spread Function Higher Moments Error on Weak Gravitational Lensing II: A Comprehensive Study; Zhang et al. (2022) *in prep.* to be released in May 2022.

 This code has the following functionalities:
 - change PSF second and higher moments through shapelet decomposition
 - conduct shear measurement for galaxy 90-deg rotated pairs, for the true and model PSF
 - measure the higher moments of the PSF
 - measure the two point correlation functions of the PSF moments residuals
 - fisher forecast for the unbiased and biased data vector (see https://github.com/hsnee/PZ_project)

## Installment

Standardized installment is not available at the moment, please do
```
git clone https://github.com/LSSTDESC/PSFHOME.git
```
and install the dependency in a conda environment with python 3. 

## Guideboard

Class objects and help function can be found in the folder "./psfhome"
- ``HOMExShapeletPair.py`` is the class to do single galaxy simulations to compute the shear response to any PSF higher moments. 
- ``great3pipe.py`` is a class that do single galaxy simulation for the GREAT3 galaxies. https://arxiv.org/abs/1404.1593
- ``homesm.py`` is the class to do single galaxy simulations to compute the shear response to radial kurtosis. 
- ``metasm.py`` is the class to do metacalibration, but is not presented in the paper. 
- ``moments.py``is a class that allows the user to change the PSF higher moments at their wish, using shapelet decomposition. This class also contains the function for measuring higher moments, called `get_all_moments()`.


Data are analyzed in the notebooks in the folder "./notebooks"
- ``Additive_bpd_tomographic.ipynb`` is a notebook that compute the additive bias on the DC2 cosmic shear 2PCF. 
- ``CorrGRF`` is a notebook for generating Gaussian random fields for the DC2 galaxies. 
- ``HSC_higher_moments_analysis`` is a notebook for analysis of the HSC PSF higher moments and their residuals
- ``HSC_moment_measure_pdr1`` is a notebook for measuring the PSF higher moments of the HSC stars and PSFs.
- ``Single-simulation`` is a notebook for conducting single galaxy simulation
- ``bpd_simulation``is a notebook to compute the additive shear response for a grid of bulge-disk galaxy, which is then used to compute the shear response for the CosmoDC2 galaxies. 
- ``fisher_forecast`` is a notebook that conduct fisher forecast to predict the cosmology parameters bias induced by the PSF higher moments. 


Figure can be found in the "./plots"
- ``All_plots.ipynb`` is a notebook that reproduce all the plots in the paper. Users are welcomed to play with it. (many data needed are stored in ./plots/pickle)


## Contact us

Please contact Tianqing Zhang (tianqinz "at" andrew.cmu.edu), if you need help for using the code. 

Please use the [issues](https://github.com/LSSTDESC/PSFHOME/issues) on this repository to suggest changes, request support, or otherwise contact the developer.


## License

The code has been publicly released; it is available under the terms of our [LICENSE](LICENSE.txt).

If you make use of the software, please cite papers listed at the top of this README, and provide a link to this code repository.




