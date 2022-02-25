from scipy.special import jn, jn_zeros,jv
from scipy.interpolate import interp1d,interp2d,RectBivariateSpline
from scipy.optimize import fsolve
from wigner_functions import *
import numpy as np
import itertools

class wigner_transform():
    def __init__(self,theta=[],l=[],s1_s2=[(0,0)],logger=None,ncpu=None,**kwargs):
        self.name='Wigner'
        self.logger=logger
        self.l=l
        self.grad_l=np.gradient(l)
        self.norm=(2*l+1.)/(4.*np.pi)
        self.theta=theta
        self.grad_theta=np.gradient(theta)
        self.inv_norm=np.sin(self.theta)*2*np.pi
        self.inv_wig_norm=self.inv_norm*self.grad_theta

        self.wig_d={}
        # self.wig_3j={}
        self.s1_s2s=s1_s2
        self.theta={}
        # self.theta=theta
        for (m1,m2) in s1_s2:
            self.wig_d[(m1,m2)]=wigner_d_parallel(m1,m2,theta,self.l,ncpu=ncpu)
#             self.wig_d[(m1,m2)]=wigner_d_recur(m1,m2,theta,self.l)
            self.theta[(m1,m2)]=theta #FIXME: Ugly

    def reset_theta_l(self,theta=None,l=None):
        """
        In case theta ell values are changed. This can happen when we implement the binning scheme.
        """
        if theta is None:
            theta=self.theta
        if l is None:
            l=self.l
        self.__init__(theta=theta,l=l,s1_s2=self.s1_s2s,logger=self.logger)

    def cl_grid(self,l_cl=[],cl=[],wig_l=None,taper=False,**kwargs):
        """
        Interpolate a given C_ell onto the grid of ells for which WT is intialized. 
        This is to generalize in case user doesnot want to compute C_ell at every ell.
        Also apply tapering if needed.
        """
        if taper:
            self.taper_f=self.taper(l=l_cl,**kwargs)
            taper_f=self.taper_f['taper_f']
            cl=cl*taper_f
            print('taper:',taper_f)
        if np.all(wig_l==l_cl):
            return cl
        cl_int=interp1d(l_cl,cl,bounds_error=False,fill_value=0,
                        kind='linear')
        if wig_l is None:
            wig_l=self.l
        cl2=cl_int(wig_l)
        return cl2
    
    def cl_cov_grid(self,l_cl=[],cl_cov=[],taper=False,**kwargs):
        """
        Interpolate a given C_ell covariance onto the grid of ells for which WT is intialized. 
        This is to generalize in case user doesnot want to compute C_ell at every ell.
        Also apply tapering if needed.
        """
        if taper:#FIXME there is no check on change in taper_kwargs
            if self.taper_f2 is None or not np.all(np.isclose(self.taper_f['l'],cl)):
                self.taper_f=self.taper(l=l,**kwargs)
                taper_f2=np.outer(self.taper_f['taper_f'],self.taper_f['taper_f'])
                self.taper_f2={'l':l,'taper_f2':taper_f2}
            cl=cl*self.taper_f2['taper_f2']
        if l_cl_cl==[]:#In this case pass a function that takes k with kwargs and outputs cl
            cl2=cl_cov(l=self.l,**kwargs)
        else:
            cl_int=RectBivariateSpline(l_cl,l_cl,cl_cov,)#bounds_error=False,fill_value=0,
                            #kind='linear')
                    #interp2d is slow. Make sure l_cl is on regular grid.
            cl2=cl_int(self.l,self.l)
        return cl2

    def projected_correlation(self,l_cl=[],cl=[],s1_s2=[],taper=False,wig_d=None,**kwargs):
        """
        Get the projected correlation function from given c_ell.
        """
        if wig_d is None: #when using default wigner matrices, interpolate to ensure grids match.
            cl2=self.cl_grid(l_cl=l_cl,cl=cl,taper=taper,**kwargs)
            w=np.dot(self.wig_d[s1_s2]*self.grad_l*self.norm,cl2)
        else:
            w=np.dot(wig_d,cl)
        return self.theta[s1_s2]*1.,w
    
    def inv_projected_correlation(self,theta_xi=[],xi=[],s1_s2=[],taper=False,**kwargs):
        """
        Get the projected power spectra (c_ell) from given xi.
        """
#         if wig_d is None: #when using default wigner matrices, interpolate to ensure grids match.
        wig_d=self.wig_d[s1_s2].T
        wig_theta=self.theta[s1_s2]
        wig_norm=self.inv_wig_norm

        xi2=self.cl_grid(l_cl=theta_xi,cl=xi,taper=taper,wig_l=wig_theta,**kwargs)
        cl=np.dot(wig_d*wig_norm,xi2)
        return self.l,cl

    def projected_covariance(self,l_cl=[],cl_cov=[],s1_s2=[],s1_s2_cross=None,
                            taper=False,**kwargs):
        if s1_s2_cross is None:
            s1_s2_cross=s1_s2
        #when cl_cov can be written as vector, eg. gaussian covariance
        cl2=self.cl_grid(l_cl=l_cl,cl=cl_cov,taper=taper,**kwargs)
        cov=np.einsum('rk,k,sk->rs',self.wig_d[s1_s2]*np.sqrt(self.norm),cl2*self.grad_l,
                    self.wig_d[s1_s2_cross]*np.sqrt(self.norm),optimize=True)
        #FIXME: Check normalization
        #FIXME: need to allow user to input wigner matrices.
        return self.theta[s1_s2]*1.,cov

    def projected_covariance2(self,l_cl=[],cl_cov=[],s1_s2=[],s1_s2_cross=None,
                                taper=False,**kwargs):
        #when cl_cov is a 2-d matrix
        if s1_s2_cross is None:
            s1_s2_cross=s1_s2
        cl_cov2=cl_cov  #self.cl_cov_grid(l_cl=l_cl,cl_cov=cl_cov,s1_s2=s1_s2,taper=taper,**kwargs)

        cov=np.einsum('rk,kk,sk->rs',self.wig_d[s1_s2]*np.sqrt(self.norm)*self.grad_l,cl_cov2,
                    self.wig_d[s1_s2_cross]*np.sqrt(self.norm),optimize=True)
#         cov=np.dot(self.wig_d[s1_s2]*self.grad_l*np.sqrt(self.norm),np.dot(self.wig_d[s1_s2_cross]*np.sqrt(self.norm),cl_cov2).T)
        # cov*=self.norm
        #FIXME: Check normalization
        return self.theta[s1_s2],cov

    def taper(self,l=[],large_k_lower=10,large_k_upper=100,low_k_lower=0,low_k_upper=1.e-5):
        #FIXME there is no check on change in taper_kwargs
        if self.taper_f is None or not np.all(np.isclose(self.taper_f['k'],k)):
            taper_f=np.zeros_like(k)
            x=k>large_k_lower
            taper_f[x]=np.cos((k[x]-large_k_lower)/(large_k_upper-large_k_lower)*np.pi/2.)
            x=k<large_k_lower and k>low_k_upper
            taper_f[x]=1
            x=k<low_k_upper
            taper_f[x]=np.cos((k[x]-low_k_upper)/(low_k_upper-low_k_lower)*np.pi/2.)
            self.taper_f={'taper_f':taper_f,'k':k}
        return self.taper_f

    def diagonal_err(self,cov=[]):
        return np.sqrt(np.diagonal(cov))

    def skewness(self,l_cl=[],cl1=[],cl2=[],cl3=[],s1_s2=[],taper=False,**kwargs):
        """
        Because we can do 6 point functions as well :). 
        """
        cl1=self.cl_grid(l_cl=l_cl,cl=cl1,s1_s2=s1_s2,taper=taper,**kwargs)
        cl2=self.cl_grid(l_cl=l_cl,cl=cl2,s1_s2=s1_s2,taper=taper,**kwargs)
        cl3=self.cl_grid(l_cl=l_cl,cl=cl3,s1_s2=s1_s2,taper=taper,**kwargs)
        skew=np.einsum('ji,ki,li',self.wig_d[s1_s2],self.wig_d[s1_s2],
                        self.wig_d[s1_s2]*cl1*cl2*cl3)
        skew*=self.norm
        #FIXME: Check normalization
        return self.theta[s1_s2],skew