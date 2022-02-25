import galsim
import numpy as np
import scipy
import ngmix

import sys
sys.path.append('/Users/zhangtianqing/Documents/Research/PSF_Rachel/HOMESim')
import homesm
import metasm


class GREAT3Pipe:
    
    def __init__(self, psf_type,psf_sigma,psf_e1,psf_e2,great3_ind,cosmic_shear,
                 pixel_scale = 0.2,subtract_intersection = True,
                is_self_defined_PSF = False , self_defined_PSF=None, self_define_PSF_model = None, 
                psf_sersicn = -1, metacal_method = None, mod_kol_radius_ratio = -1,
                  great3_cat = None, shift = False, beta0 = -1, delta_beta = -1):
        
        #Define basic variables
        self.pixel_scale = pixel_scale        
        self.subtract_intersection = subtract_intersection        
        self.is_self_defined_PSF = is_self_defined_PSF
        self.metacal_method = metacal_method
        self.great3_ind = great3_ind
        self.cosmic_shear = cosmic_shear
        self.results = {}
        self.shift = shift


        self.gal_light = great3_cat.makeGalaxy(gal_type='parametric',index=great3_ind)
        self.gal_rotate_light = self.gal_light.rotate(90 * galsim.degrees)
        
        sersicn = great3_cat.getParametricRecord(great3_ind)['sersicfit'][2]

        self.shift1_ori = np.random.uniform(low = -1.0, high = 1.0)
        self.shift2_ori = np.random.uniform(low = -1.0, high = 1.0)

        self.shift1_rot = np.random.uniform(low = -1.0, high = 1.0)
        self.shift2_rot = np.random.uniform(low = -1.0, high = 1.0)


        if not is_self_defined_PSF:
            self.psf_type = psf_type
            self.psf_sigma = psf_sigma
            self.psf_model_sigma = psf_sigma
            self.psf_e1 = psf_e1
            self.psf_e2 = psf_e2
            self.psf_model_e1 = psf_e1
            self.psf_model_e2 = psf_e2

            self.results["psf_type"] = psf_type
            self.results["psf_sigma"] = psf_sigma
          
            if psf_type == 'gaussian':
                self.psf_light = galsim.Gaussian(flux = 1.0, sigma = self.psf_sigma)
                self.psf_light = self.toSize(self.psf_light, self.psf_sigma)
            elif psf_type == 'kolmogorov':
                self.psf_light  = galsim.Kolmogorov(flux = 1.0, half_light_radius = self.psf_sigma)
                #self.psf_light = self.toSize(self.psf_light, self.psf_sigma)
                
                psf_model_light = 0.5*self.psf_light + 0.5*self.psf_light.expand(mod_kol_radius_ratio)

                old_sigma = galsim.hsm.FindAdaptiveMom(self.psf_light.drawImage(scale = self.pixel_scale)).moments_sigma*self.pixel_scale
                #new_sigma = galsim.hsm.FindAdaptiveMom(psf_model_light.drawImage(scale = self.pixel_scale)).moments_sigma*self.pixel_scale
                #ratio = old_sigma/new_sigma
                #print(ratio)
                self.psf_model_light = self.toSize(psf_model_light, old_sigma).withFlux(1.0)
            elif psf_type == 'moffat':
                self.psf_light = galsim.Moffat(beta0, half_light_radius = self.psf_sigma)
                self.psf_light = self.toSize(self.psf_light, self.psf_sigma, weighted = True)
                self.psf_model_light = galsim.Moffat(beta0+delta_beta, half_light_radius = self.psf_sigma)
                self.psf_model_light = self.toSize(self.psf_model_light, self.psf_sigma, weighted = True)


                #print(true_sigma, model_sigma)
                
                
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
    
    def find_kurtosis_2(self, image):
        return galsim.hsm.FindAdaptiveMom(image).moments_rho4

            
            
    def measure(self,metacal=True,rot = False, base = False):
        if base:
            image_epsf = self.psf_light.drawImage(scale=self.pixel_scale)
        else:
            image_epsf = self.psf_model_light.drawImage(scale=self.pixel_scale)
            
        if rot:
            galaxy = self.gal_rotate_light
            galaxy = galaxy.shear(self.cosmic_shear)
        else:
            galaxy = self.gal_light
            galaxy = galaxy.shear(self.cosmic_shear)
        
        final = galsim.Convolve([galaxy,self.psf_light])

        if self.shift:
            if rot:
                image = final.drawImage(scale = self.pixel_scale, offset = (self.shift1_rot,self.shift2_rot) )
            else:
                image = final.drawImage(scale = self.pixel_scale, offset = (self.shift1_ori, self.shift2_ori))
        else:
            image = final.drawImage(scale = self.pixel_scale)

        if metacal == False:
            results = galsim.hsm.EstimateShear(image,image_epsf)
            shape = galsim.Shear(e1 = results.corrected_e1, e2 = results.corrected_e2)
            return np.array([[1.0,0,0,1.0]]),np.array([shape.g1,shape.g2])
        else:
            results = self.perform_metacal(image,image_epsf)
            return results["R"].reshape((-1)), results["noshear"]
        
        
        
    def perform_metacal(self,image,image_epsf):
        metacal = metasm.metacal_shear_measure(image,image_epsf,great3 = True)
        metacal.measure_shear(self.metacal_method)
        results = metacal.get_results()
        return results
    
    def toSize(self, profile, sigma , weighted = True, tol = 1e-4):


        if weighted:
            apply_pixel =  max(self.pixel_scale, sigma/5)
            #apply_pixel =  self.pixel_scale
            true_sigma = galsim.hsm.FindAdaptiveMom(profile.drawImage(scale =apply_pixel)).moments_sigma*apply_pixel
        else:
            #true_sigma = profile.calculateMomentRadius(scale = self.pixel_scale, rtype='trace')

            
            image = profile.drawImage(scale = self.pixel_scale)
            true_sigma = image.calculateMomentRadius()

        ratio = sigma/true_sigma
        new_profile = profile.expand(ratio)

        while abs(true_sigma - sigma)>tol:
            ratio = sigma/true_sigma
            new_profile = new_profile.expand(ratio)

            if weighted:
                
                apply_pixel =  max(self.pixel_scale, sigma/5)
                #apply_pixel =  self.pixel_scale
                true_sigma = galsim.hsm.FindAdaptiveMom(new_profile.drawImage(scale =apply_pixel),hsmparams=galsim.hsm.HSMParams(max_mom2_iter = 2000)).moments_sigma*apply_pixel
            else:
                #true_sigma = profile.calculateMomentRadius(scale = self.pixel_scale, rtype='trace')
                image = new_profile.drawImage(scale = self.pixel_scale)
                true_sigma = image.calculateMomentRadius()
        return new_profile
        

    def real_gal_sigma(self):
        image = self.gal_light.drawImage(scale = self.pixel_scale,method = 'no_pixel')
        return galsim.hsm.FindAdaptiveMom(image).moments_sigma*self.pixel_scale
     
def df_block(params,index, metacal = True):
    
    test = great3pipe.GREAT3Pipe(*params[:-1],**params[-1])
    
    try:
        base_ori_r, base_ori_e = test.measure(metacal = metacal,rot = False, base = True)
        mod_ori_r, mod_ori_e = test.measure(metacal = metacal,rot = False, base = False)
        base_rot_r, base_rot_e = test.measure(metacal = metacal,rot = True, base = True)
        mod_rot_r, mod_rot_e = test.measure(metacal = metacal,rot = True, base = False)

        truth_kurtosis = test.find_kurtosis_2(test.psf_light.drawImage(scale = test.pixel_scale))
        model_kurtosis = test.find_kurtosis_2(test.psf_model_light.drawImage(scale = test.pixel_scale))
        
        df = pd.DataFrame([[index,params[5],self.psf_sigma,truth_kurtosis,model_kurtosis,base_ori_r,base_ori_e,mod_ori_r,mod_ori_e], [index,params[5],self.psf_sigma,truth_kurtosis,model_kurtosis,base_rot_r,base_rot_e,mod_rot_r,mod_rot_e]],
                          columns=["index","true_shear","psf_hlr", "truth_kurtosis","model_kurtosis","R_base","e_base","R_mod","e_mod"])
        return df
    except:
        print ("error occured")
        return pd.DataFrame()
        
        
    
    
