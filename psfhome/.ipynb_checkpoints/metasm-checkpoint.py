#Stands for metacalibration shape measurement

#To perform metacalibration with this class, you will need to
#1. Provide the image and the psf image during initiation
#2. Provide the elementary method you want to use when calling measure_shear(). Currently, the options are "estimate_shear", 
# "ngmix" (stands for N-Gaussian mixture) and admombootstrap (a bootstrap in the ngmix class)

#To run this class, you will have to import Galsim and NGMix

#The return results is a library of galsim.Shear(), the 'g_cal' correspond to the shape measured by the metacalibration
#The shape for the 'g' key is the shape measured by the basic method that the user have chosen.



import galsim
import numpy as np
import scipy
import ngmix


# Metacalibration implemented in this code is based on ngmix=1.3.8. The code breaks if you install the >2.0.0 versions.

# from ngmix.observation import Observation, ObsList, MultiBandObsList
# from ngmix.fitting import LMSimple


class metacal_shear_measure:
    
    def __init__(self,final_image,psf_image):
        self.final_image = final_image
        self.final_image_array = final_image.array
        self.psf_image = psf_image
        self.psf_image_array = psf_image.array
        return None
    
    def measure_shear(self,method):
        self.results={}
        if method == 'estimateShear':
            shear = self.measure_shear_estimateShear()
        elif method == 'ngmix':
            shear = self.measure_shear_ngmix()
        elif method == 'admomBootstrap':
            shear = self.measure_shear_admombootstrap()
        self.results['g_cal'] = shear
        return 0
        
    def measure_shear_estimateShear(self):
        obs_results = galsim.hsm.EstimateShear(self.final_image,self.psf_image)
                
        psf_obs=Observation(self.psf_image_array)
        obs = Observation(self.final_image_array,psf=psf_obs)
        
        obdic = ngmix.metacal.get_all_metacal(obs,fixnoise = False)
        
        g_obs = galsim.Shear(e1 = obs_results.corrected_e1, e2 = obs_results.corrected_e2)
        
        self.results['g'] = g_obs
        
        
        mcal_results={}

        for key in obdic:

            mobs = obdic[key]
            mpsf_array = mobs.get_psf().image
            mimage_array = mobs.image

            this_image = galsim.Image(mimage_array)
            this_image_epsf = galsim.Image(mpsf_array)

            res = galsim.hsm.EstimateShear(this_image,this_image_epsf)

            res_shear = galsim.Shear(e1 = res.corrected_e1, e2 = res.corrected_e2)
            this_res = {'g1':res_shear.g1,'g2':res_shear.g2}
            #print key,this_res
            mcal_results[key] = this_res


        # calculate response R11. The shear by default
        # is 0.01, so dgamma=0.02
        
        g = np.array([mcal_results['noshear']['g1'], mcal_results['noshear']['g2']])
        

        R11 = (mcal_results['1p']['g1'] - mcal_results['1m']['g1'])/(0.02)
        R22 = (mcal_results['2p']['g2'] - mcal_results['2m']['g2'])/(0.02)
        R12 = (mcal_results['1p']['g2'] - mcal_results['1m']['g2'])/(0.02)
        R21 = (mcal_results['2p']['g1'] - mcal_results['2m']['g1'])/(0.02)
        
        R = np.array([[R11,R12],[R21,R22]])
        Rinv = np.linalg.inv(R)
        
        #print R11,R22
        
        g_truth = np.matmul(Rinv,g)
        return galsim.Shear(g1 = g_truth[0], g2 = g_truth[1])
    
    def measure_shear_ngmix(self):
        psf_obs=Observation(self.psf_image_array)
        pfitter=LMSimple(psf_obs,'gauss')
        pfit_guess = self.make_guess(self.psf_image_array)
        
        pfitter.go(pfit_guess)

        psf_gmix_fit=pfitter.get_gmix()
        
        psf_obs.set_gmix(psf_gmix_fit)
        
        weight=np.ones(self.final_image_array.shape)

        obs = Observation(self.final_image_array,weight = weight, psf=psf_obs)

        fitter=LMSimple(obs,'gauss')

        final_guess = self.make_guess(self.final_image_array)
        #final_guess = [0.0, 0.0, 0.01, 0.0, 50.0, 1.e5]
        
        fitter.go(final_guess)

        obs_res=fitter.get_result()
        
        self.results['g'] = obs_res['g']

        ngmix.print_pars(obs_res['pars'],front="meas: ")
        
        obdic = ngmix.metacal.get_all_metacal(obs)
        
        mcal_results={}

        for key in obdic:
            mobs = obdic[key]
            this_psf = mobs.get_psf()
            this_image = mobs.image


            #psf fitting
            this_pfitter=LMSimple(this_psf,'gauss')
            this_pguess = self.make_guess(this_psf.image)
            this_pfitter.go(this_pguess)
            this_psf_gmix_fit=this_pfitter.get_gmix()

            # set the gmix; needed for galaxy fitting later
            this_psf.set_gmix(this_psf_gmix_fit)

            #image fitting
            weight=np.ones(this_image.shape)

            # When constructing the Observation we include a weight map and a psf
            # observation

            this_obs = Observation(this_image, weight=weight, psf=this_psf)

            this_fitter=LMSimple(this_obs,'gauss')

            this_guess = self.make_guess(this_image)
            this_fitter.go(this_guess)

            this_res=this_fitter.get_result()
            #print key,res
            mcal_results[key] = this_res
            
            #print key,this_res['g']

        R11 = (mcal_results['1p']['g'][0] - mcal_results['1m']['g'][0])/(0.02)
        R22 = (mcal_results['2p']['g'][1] - mcal_results['2m']['g'][1])/(0.02)
        R12 = (mcal_results['1p']['g'][1] - mcal_results['1m']['g'][1])/(0.02)
        R21 = (mcal_results['2p']['g'][0] - mcal_results['2m']['g'][0])/(0.02)
        R = np.array([[R11,R12],[R21,R22]])
        Rinv = np.linalg.inv(R)


        #print R11,R22
        
        g_truth = np.matmul(Rinv,obs_res['g'])
        return galsim.Shear(g1 = g_truth[0], g2 = g_truth[1])
    
    
    
    def measure_shear_admombootstrap(self):
        psf_obs=Observation(self.psf_image_array)
        obs = Observation(self.final_image_array, psf=psf_obs)
        meta_boot = ngmix.bootstrap.AdmomMetacalBootstrapper(obs)
        meta_boot.fit_metacal(Tguess = 100, psf_Tguess=60)
        res = meta_boot.get_metacal_result()
        R11 = (res['1p']['g'][0] - res['1m']['g'][0])/(0.02)
        R22 = (res['2p']['g'][1] - res['2m']['g'][1])/(0.02)
        R12 = (res['1p']['g'][1] - res['1m']['g'][1])/(0.02)
        R21 = (res['2p']['g'][0] - res['2m']['g'][0])/(0.02)
        R = np.array([[R11,R12],[R21,R22]])
        Rinv = np.linalg.inv(R)
        e_est = res['noshear']['e']
        self.results['g'] = e_est
        
        g_truth = np.matmul(Rinv,e_est)
        return galsim.Shear(g1 = g_truth[0], g2 = g_truth[1])

        
    def make_guess(self,array):
        
        eps = 0.01
        #shape = galsim.hsm.FindAdaptiveMom(galsim.Image(array))
        pars = np.zeros(6)
        pars[0] = 0
        pars[1] = 0
        pars[2] = 0
        pars[3] = 0
        pars[4] = 100
        pars[5] = 1
        #print "guess",pars

        return pars
    
    def get_results(self):
        
        return self.results
        
        
        




        