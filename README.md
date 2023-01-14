# Impact of PSF Higher Order Moments Error (HOME) on Weak Lensing (WL)

## Introduction

This repository includes software to quantitatively test the relation between the weak lensing shear bias and the modeling error of PSF higher order moments (beyond second).

It is based on the following papers:
 - Impact of Point Spread Function Higher Moments Error on Weak Gravitational Lensing; Zhang and Mandelbaum for LSST (2021), MNRAS - [ADS entry](https://ui.adsabs.harvard.edu/abs/2022MNRAS.510.1978Z/abstract)
 - Impact of Point Spread Function Higher Moments Error on Weak Gravitational Lensing II: A Comprehensive Study; Zhang et al. (2022), MNRAS [ADS entry](https://ui.adsabs.harvard.edu/abs/2022MNRAS.tmp.3141Z/abstract)
 - A General Framework for Removing Point Spread Function Additive Systematics in Cosmological Weak Lensing Analysis; Zhang et al. (2022), MNRAS [ADS entry](https://ui.adsabs.harvard.edu/abs/2022arXiv221203257Z/abstract)

 This code has functionality to do the following tasks:
 - change PSF second and higher moments through shapelet decomposition
 - conduct shear measurement for galaxy 90-deg rotated pairs, for the true and model PSF
 - measure the higher moments of the PSF
 - measure the two point correlation functions of the PSF moments residuals
 - carry out a Fisher forecast for the unbiased and biased data vector (see https://github.com/hsnee/PZ_project)
 - Utilize a PSF higher moments catalog and galaxy shape catalog to estimate the PSF systematics model parameters

## Installation

Standardized installation is not available at the moment. Please do
```
git clone https://github.com/LSSTDESC/PSFHOME.git
```
In a python 3 environment, you can then install the package by 
```
pip install ./PSFHOME
```

Notice that this process will not install metacalibration, to use metacalibration, you need to install version `1.3.8` of `ngmix`.


## Guide to repository contents

Class objects and helper functions can be found in the folder "./psfhome"
- ``HOMExShapeletPair.py`` defines a class to do single galaxy simulations to compute the shear response to any PSF higher moments. 
- ``great3pipe.py`` defines a class that carries out single galaxy simulation for the [GREAT3](https://arxiv.org/abs/1404.1593) galaxy sample. 
- ``homesm.py`` defines a class to carry out single galaxy simulations to compute the shear response to radial kurtosis. 
- ``metasm.py`` defines a class to measure shear for simulated images using metacalibration with ngmix, but it is not presented in the paper. 
- ``moments.py``defines a class that allows the user to change the PSF higher moments as they wish, using shapelet decomposition. This class also contains the function for measuring higher moments, called `get_all_moments()`.

Update: 
- ``psfmod.py``defines a class that helps fitting the galaxy-PSF correlation function with PSF-PSF correlation functions, by running an MCMC. It save the posterior of the PSF parameters, which are provided to the cosmic shear analysis


Data are analyzed in the notebooks in the folder "./notebooks"
- ``Additive_bpd_tomographic.ipynb`` is a notebook that computes the additive bias on the DC2 cosmic shear 2PCF. 
- ``CorrGRF`` is a notebook for generating Gaussian random fields for the DC2 galaxies. 
- ``HSC_higher_moments_analysis`` is a notebook for analysis of the HSC PSF higher moments and their residuals
- ``HSC_moment_measure_pdr1`` is a notebook for measuring the PSF higher moments of the HSC stars and PSFs.
- ``Single-simulation`` is a notebook for conducting single galaxy simulations
- ``bpd_simulation``is a notebook to compute the additive shear response for a grid of bulge-disk galaxy parameters, which is then used to compute the shear response for the CosmoDC2 galaxies. 
- ``fisher_forecast`` is a notebook that conducts a Fisher forecast to predict the cosmological parameters biases induced by the PSF higher moments if their impact on shear is not modeled. 

Update:
- ``Fit_PSF_systematics`` shows how to psfmod.py to compute the galaxy-PSF correlation function, PSF-PSF correlation function, and use  MCMC to infer the posterior of the PSF parameters. Users can use the argument in the class to turn on/off PSF higher moments parameters, constant shear, PSF-PSF correlation noise, and whether to consider higher moment g-p in the chi2. 





Figure can be found in the "./plots"
- ``All_plots.ipynb`` is a notebook that reproduces all plots in the second paper. Users are welcomed to play with it. (The necessary data are stored in ./plots/pickle)
Update: 
- ``read_psfmod`` is a notebook for producing the PSF parameter posterior and other related plots in the paper III.


## Contact us

Please contact Tianqing Zhang (tianqinz "at" andrew.cmu.edu), if you need help using the code. 

Please use the [issues](https://github.com/LSSTDESC/PSFHOME/issues) on this repository to suggest changes, request support, or otherwise contact the developer.


## License

The code has been publicly released; it is available under the terms of our [LICENSE](LICENSE.txt).

If you make use of the software, please cite papers listed at the top of this README, and provide a link to this code repository.




