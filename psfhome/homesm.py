#Stands for higher order moment error shape measurement

#In this class, user can detect the bias caused by higher order moment error of the PSF modeling. This class will perform the convolution of
#galaxy and PSF, then measure the shape of the galaxy by the model PSF. The current available shape measurement methods are GalSim.HSM and 
#metacalibration class implemented by the assist of ngmix.

# galaxy and PSF can be specified in two different ways: 

#1. GalSim simulation: appoint galaxy size and shape; PSF size and shape, the class will generate a PSF model that has the 
#same second order mements, but with different higher order moments. Then, get_measurement() will simulate the shape measurement
#bias caused by the mis-modeling of the higher order moment for the PSF model. Specifically, the program perform shape measurement
#twice. The first time, measure the shape using the truth PSF, and the second time, using the flawed PSF model. The difference 
#between the two shape meansurement will be the systematic bias cause by the PSF modeling process. 

#2. Real data measurement: In this case, the galaxy or the PSF is specified by image instead of the galsim.gsobject class. In this case,
#user have to feed the image of the PSF and the PSF model into the instance. During initilization,  the code will interpolate for both
#truth PSF and model PSF. The model PSF then will be transformed by GSObject.shear() and GSObject.expand() to have identical second order
#moment as the truth. Rest of the procedure is the same as the first kind of shape measurement

import galsim
import numpy as np
import scipy
#from .metasm import *

def find_kurtosis_2(image):
    return galsim.hsm.FindAdaptiveMom(image).moments_rho4



class PSFSameSecondTest:
    
    def __init__(self, gal_type, gal_sigma,e1,e2,psf_type,psf_sigma,psf_e1,psf_e2,
                gal_flux=1.e2,pixel_scale = 0.1,sersicn = -1,subtract_intersection = True,
                is_self_defined_PSF = False , self_defined_PSF=None, self_define_PSF_model = None, 
                psf_sersicn = -1, metacal_method = None, mod_kol_radius_ratio = -1):
        
        #Define basic variables
        self.pixel_scale = pixel_scale        
        self.subtract_intersection = subtract_intersection        
        self.is_self_defined_PSF = is_self_defined_PSF
        
        #Define galaxy
        self.gal_type = gal_type
        self.gal_sigma = gal_sigma
        self.gal_flux=gal_flux
        self.e1 = e1
        self.e2 = e2
        self.e = np.sqrt(e1**2+e2**2)
        self.sersicn=sersicn
        self.e_truth = np.sqrt(e1**2+e2**2)
        self.metacal_method = metacal_method

#         if gal_type == "gaussian":
#             self.gal_light = galsim.Gaussian(flux = self.gal_flux, sigma = Gaussian_sigma(self.gal_sigma,self.pixel_scale))
#         elif gal_type == "sersic":
#             self.gal_light = self.findAdaptiveSersic(gal_sigma,sersicn)
        if gal_type == 'gaussian':
            gaussian_profile = galsim.Gaussian(sigma = 1.0)
            gaussian_profile = self.toSize(gaussian_profile, self.gal_sigma)
            self.gal_light = gaussian_profile.withFlux(self.gal_flux)
        elif gal_type == 'sersic':
            sersic_profile = galsim.Sersic(sersicn, half_light_radius = self.gal_sigma)
            #sersic_profile = self.toSize(sersic_profile, self.gal_sigma)
            self.gal_light = sersic_profile.withFlux(self.gal_flux)



        
        self.gal_light = self.gal_light.shear(e1=e1, e2=e2)
        
        if not is_self_defined_PSF:
            self.psf_type = psf_type
            self.psf_sigma = psf_sigma
            self.psf_model_sigma = psf_sigma
            self.psf_e1 = psf_e1
            self.psf_e2 = psf_e2
            self.psf_model_e1 = psf_e1
            self.psf_model_e2 = psf_e2

          
            if psf_type == 'gaussian':
                self.psf_light = galsim.Gaussian(flux = 1.0, sigma = self.psf_sigma)
                self.psf_light = self.toSize(self.psf_light, self.psf_sigma)
            elif psf_type == 'kolmogorov':
                self.psf_light  = galsim.Kolmogorov(flux = 1.0, half_light_radius = 1.0)
                self.psf_light = self.toSize(self.psf_light, self.psf_sigma)
                
                psf_model_light = 0.5*self.psf_light + 0.5*self.psf_light.expand(mod_kol_radius_ratio)
                
                new_sigma = galsim.hsm.FindAdaptiveMom(psf_model_light.drawImage(scale = self.pixel_scale, method = 'no_pixel')).moments_sigma*self.pixel_scale
                ratio = self.psf_sigma/new_sigma
                self.psf_model_light = psf_model_light.expand(ratio).withFlux(1.0)
                
                
            elif psf_type == 'opticalPSF':
                self.psf_light = galsim.OpticalPSF(1.0,flux = 1.0)
                self.psf_light = self.toSize(self.psf_light, self.psf_sigma)
            elif psf_type == 'sersic':
                self.psf_light = galsim.Sersic(psf_sersicn, half_light_radius = 1.0)
                self.psf_light = self.toSize(self.psf_light, self.psf_sigma)
                self.psf_model_light = galsim.Gaussian(flux = 1.0, sigma = self.psf_model_sigma)

            self.psf_light = self.psf_light.shear(e1=psf_e1, e2=psf_e2)
            self.psf_model_light = self.psf_model_light.shear(e1=self.psf_model_e1,e2=self.psf_model_e2)       
        else:
            self.psf_type = "self_define"
            truth_image = self_defined_PSF
            truth_psf = galsim.InterpolatedImage(truth_image,scale = pixel_scale)
            truth_measure = galsim.hsm.FindAdaptiveMom(truth_image)
            truth_sigma = truth_measure.moments_sigma

            model_image = self_define_PSF_model
            model_measure = galsim.hsm.FindAdaptiveMom(model_image)
            model_sigma = model_measure.moments_sigma
            model_psf = galsim.InterpolatedImage(model_image,scale = pixel_scale)

            delta_g1 = truth_measure.observed_shape.g1 - model_measure.observed_shape.g1
            delta_g2 = truth_measure.observed_shape.g2 - model_measure.observed_shape.g2

            this_ratio = truth_sigma/model_sigma


            model_psf_after = model_psf.expand(this_ratio)
            model_psf_after = model_psf_after.shear(g1 = delta_g1, g2 = delta_g2)


            self.psf_sigma = truth_sigma
            self.psf_model_sigma = self.psf_sigma
            self.psf_e1 = truth_measure.observed_shape.e1
            self.psf_model_e1 = self.psf_e1
            self.psf_e2 = truth_measure.observed_shape.e2
            self.psf_model_e2 = self.psf_e2
            self.psf_light = truth_psf
            self.psf_model_light = model_psf_after
            
            
    def get_intersection(self,metacal):
        image_epsf = self.psf_light.drawImage(scale=self.pixel_scale)
        final = galsim.Convolve([self.gal_light,self.psf_light])
        image = final.drawImage(scale = self.pixel_scale)
        if metacal == False:
            results = galsim.hsm.EstimateShear(image,image_epsf)
            intersection = np.sqrt(results.corrected_e1**2+results.corrected_e2**2)-self.e_truth
            return intersection
        else:
            results = self.perform_metacal(image,image_epsf)
            intersection = np.sqrt(results['g_cal'].e1**2+results['g_cal'].e2**2) - self.e_truth
            return intersection
        
        
    def get_prediction(self):
        return self.e_truth*(self.psf_model_sigma**2-self.psf_sigma**2)/self.gal_sigma**2 + (self.psf_sigma/self.gal_sigma)**2*(self.psf_e2-self.psf_model_e2)
    
    def get_measurement(self,metacal = False):
        image_epsf = self.psf_model_light.drawImage(scale = self.pixel_scale)
        final = galsim.Convolve([self.gal_light,self.psf_light])
        image = final.drawImage(scale = self.pixel_scale)
        if metacal==False:
            results = galsim.hsm.EstimateShear(image,image_epsf)
            bias = np.sqrt(results.corrected_e1**2+results.corrected_e2**2)-self.e_truth
        else:
            results = self.perform_metacal(image,image_epsf)
            bias = np.sqrt(results['g_cal'].e1**2+results['g_cal'].e2**2) - self.e_truth
        if self.subtract_intersection==True:
            return bias-self.get_intersection(metacal)
        else:
            return bias
        
    def perform_metacal(self,image,image_epsf):
        metacal = metasm.metacal_shear_measure(image,image_epsf)
        metacal.measure_shear(self.metacal_method)
        results = metacal.get_results()
        return results
    
    def findAdaptiveSersic(self,sigma,n):
        good_half_light_re = bisect(Sersic_sigma,sigma/3,sigma*5,args=(n,self.pixel_scale,sigma))
        return galsim.Sersic(n=n,half_light_radius=good_half_light_re)
    
    def findAdaptiveKolmogorov(self,sigma):
        good_half_light_re = bisect(Kolmogorov_sigma,max(self.psf_sigma/5,self.pixel_scale),self.psf_sigma*5,args = (self.pixel_scale,sigma))
        return galsim.Kolmogorov(half_light_radius = good_half_light_re)
    
    def findAdaptiveOpticalPSF(self,sigma):
        good_fwhm = bisect(OpticalPSF_sigma,max(self.psf_sigma/3,self.pixel_scale),self.psf_sigma*5,args = (self.pixel_scale,sigma))
        return galsim.OpticalPSF(good_fwhm)
    
    def toSize(self, profile, sigma):
        true_sigma = galsim.hsm.FindAdaptiveMom(profile.drawImage(scale =self.pixel_scale,method = 'no_pixel')).moments_sigma*self.pixel_scale
        ratio = sigma/true_sigma
        new_profile = profile.expand(ratio)
        return new_profile
        
    
    def real_gal_sigma(self):
        image = self.gal_light.drawImage(scale = self.pixel_scale,method = 'no_pixel')
        return galsim.hsm.FindAdaptiveMom(image).moments_sigma*self.pixel_scale
    
    def get_results(self,metacal = False):
        results = dict()
        results["prediction"] = self.get_prediction()
        results["measurement"] = self.get_measurement(metacal = metacal)

        results["gal_type"] = self.gal_type
        results["psf_type"] = self.psf_type
        results["gal_sigma"] = self.gal_sigma
        results["psf_sigma"] = self.psf_sigma
        results["e1"] = self.e1
        results["e2"] = self.e2
        results["e"] = self.e
        results["sersicn"] = self.sersicn
        results["psf_e1"] = self.psf_e1
        results["psf_e2"] = self.psf_e2
        
        results["psf_model_sigma"] = self.psf_model_sigma
        results["psf_model_e1"] = self.psf_model_e1
        results["psf_model_e2"] = self.psf_model_e2


        results["gal_hlr"] = self.gal_light.calculateHLR()
        
        results["prediction-error"] = results["prediction"]-results["measurement"]
        results["prediction-percenterror"] = results["prediction-error"]/results["e"]
        
        results["psf_kurtosis"] = find_kurtosis_2(self.psf_light.drawImage(scale = self.pixel_scale))
        results["psf_model_kurtosis"] = find_kurtosis_2(self.psf_model_light.drawImage(scale = self.pixel_scale))
        results["kurtosis_perc_error"] = 100*(results['psf_model_kurtosis']-results['psf_kurtosis'])/results['psf_model_kurtosis']
        
        return results
    
    
    
    
    
