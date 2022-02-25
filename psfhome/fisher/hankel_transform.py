from scipy.special import jn, jn_zeros,jv
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import numpy as np
import itertools

class hankel_transform():
    def __init__(self,rmin=0.1,rmax=100,kmax=10,kmin=1.e-4,n_zeros=1000,n_zeros_step=1000,
                 j_nu=[0,2],prune_r=0,prune_log_space=True):
        self.rmin=rmin
        self.rmax=rmax
        self.kmax=kmax
        self.kmin=kmin
        self.n_zeros=n_zeros
        self.n_zeros_step=n_zeros_step
        self.k={}
        self.r={}
        self.J={}
        self.J_nu1={}
        self.zeros={}
        for i in j_nu:
            self.k[i],self.r[i],self.J[i],self.J_nu1[i],self.zeros[i]=self.get_k_r_j(j_nu=i,
                                                   n_zeros=n_zeros,rmin=rmin,rmax=rmax,
                                                   kmax=kmax,kmin=kmin,n_zeros_step=n_zeros_step,
                                                   prune_r=prune_r,
                                                   prune_log_space=prune_log_space)

    def get_k_r_j(self,j_nu=0,n_zeros=1000,rmin=0.1,rmax=100,kmax=10,kmin=1.e-4,
                  n_zeros_step=1000,prune_r=0,prune_log_space=True):
        while True:
            if isinstance(j_nu,int):
                zeros=jn_zeros(j_nu,n_zeros)
            else:
                def jv2(x):
                    return jv(j_nu,x)
                zeros_t=jn_zeros(j_nu-0.5,n_zeros)+0.7852
                zeros=np.zeros_like(zeros_t)
                zeros[:5500]= fsolve(jv2,zeros_t[:5500])
                zi=interp1d(zeros_t[:5500],zeros[:5500]-zeros_t[:5500],
                                bounds_error=False,fill_value='extrapolate',kind=0)
                zeros=zi(zeros_t)+zeros_t
                #this is bad, but can't find zeros of spherical right now. .7852 does make it
                #better by ensuring most values are <1.e-3
            k=zeros/zeros[-1]*kmax
            r=zeros/kmax
            if min(r)>rmin:
                kmax=min(zeros)/rmin
                print ('changed kmax to',kmax,' to cover rmin')
                continue
            elif max(r)<rmax:
                n_zeros+=n_zeros_step
                print ('j-nu=',j_nu,' not enough zeros to cover rmax, increasing by ',n_zeros_step,' to',n_zeros)
            elif min(k)>kmin:
                n_zeros+=n_zeros_step
                print ('j-nu=',j_nu,' not enough zeros to cover kmin, increasing by ',n_zeros_step,' to',n_zeros)
            else:
                break
        rmin2=r[r<=rmin][-1]
        rmax2=r[r>=rmax][0]
        x=r<=rmax2
        x*=r>=rmin2
        r=r[x]
        if prune_r!=0:
            print ('pruning r, log_space,n_f:',prune_log_space,prune_r)
            N=len(r)
            if prune_log_space:
                idx=np.unique(np.int64(np.logspace(0,np.log10(N-1),N/prune_r)))#pruning can be worse than prune_r factor due to repeated numbers when logspace number are convereted to int.
                idx=np.append([0],idx)
            else:
                idx=np.arange(0,N-1,step=prune_r)
            idx=np.append(idx,[N-1])
            r=r[idx]
            print ('pruned r:',len(r))
        r=np.unique(r)
        print ('nr:',len(r))
        J=jn(j_nu,np.outer(r,k))
        J_nu1=jn(j_nu+1,zeros)
        return k,r,J,J_nu1,zeros

    def pk_grid(self,k_pk=[],pk=[],j_nu=[],taper=False,**kwargs):
        if taper:
            pk=self.taper(k=k_pk,pk=pk,**kwargs)
        if k_pk==[]:#In this case pass a function that takes k with kwargs and outputs pk
            pk2=pk(k=self.k[j_nu],**kwargs)
        else:
            pk_int=interp1d(k_pk,pk,bounds_error=False,fill_value=0,
                            kind='linear')
            pk2=pk_int(self.k[j_nu])
        return pk2

    def projected_correlation(self,k_pk=[],pk=[],j_nu=[],taper=False,**kwargs):
        pk2=self.pk_grid(k_pk=k_pk,pk=pk,j_nu=j_nu,taper=taper,**kwargs)
        w=np.dot(self.J[j_nu],pk2/self.J_nu1[j_nu]**2)
        #w*=(2.*self.kmax**2/self.zeros[j_nu][-1]**2)/(2*np.pi)
        w*=4*np.pi / self.zeros[j_nu][-1]**2
        #print((2.*self.kmax**2/self.zeros[j_nu][-1]**2)/(2*np.pi))
        #print(self.J_nu1[j_nu])
        return self.r[j_nu],w

    def spherical_correlation(self,k_pk=[],pk=[],j_nu=[],taper=False,**kwargs):
    #we will use relation spherical_jn(z)=j{n+0.5}(z)*sqrt(pi/2z)
    #pk will be written as k*pk
        pk2=self.pk_grid(k_pk=k_pk,pk=pk,j_nu=j_nu,taper=taper,**kwargs)
        j_f=np.sqrt(np.pi/2./np.outer(self.r[j_nu],self.k[j_nu]))
        w=np.dot(self.J[j_nu],pk2*self.k[j_nu]/self.J_nu1[j_nu]**2)
        w*=(self.kmax**2/self.zeros[j_nu][-1]**2)/(2*np.pi)
        return self.r[j_nu],w

    def projected_covariance(self,k_pk=[],pk1=[],pk2=[],j_nu=[],taper=False,**kwargs):
        pk1=self.pk_grid(k_pk=k_pk,pk=pk1,j_nu=j_nu,taper=taper,**kwargs)
        pk2=self.pk_grid(k_pk=k_pk,pk=pk2,j_nu=j_nu,taper=taper,**kwargs)
        cov=np.dot(self.J[j_nu],(self.J[j_nu]*pk1*pk2/self.J_nu1[j_nu]**2).T)
        cov*=(2.*self.kmax**2/self.zeros[j_nu][-1]**2)/(2*np.pi)
        return self.r[j_nu],cov

    def taper(self,k=[],pk=[],large_k_lower=10,large_k_upper=100,low_k_lower=0,low_k_upper=1.e-5):
        pk_out=np.copy(pk)
        x=k>large_k_lower
        pk_out[x]*=np.cos((k[x]-large_k_lower)/(large_k_upper-large_k_lower)*np.pi/2.)
        x=k>large_k_upper
        pk_out[x]=0
        x=k<low_k_upper
        pk_out[x]*=np.cos((k[x]-low_k_upper)/(low_k_upper-low_k_lower)*np.pi/2.)
        x=k<low_k_lower
        pk_out[x]=0
        return pk_out

    def corr_matrix(self,cov=[]):
        corr=np.zeros_like(cov)
        n_bins=len(cov)
        for i in np.arange(n_bins):
            for j in np.arange(n_bins):
                corr[i][j]=cov[i][j]/np.sqrt(cov[i][i]*cov[j][j])
        corr=np.nan_to_num(corr)
        return corr

    def diagonal_err(self,cov=[]):
        return np.sqrt(np.diagonal(cov))

    def bin_cov(self,r=[],cov=[],r_bins=[]):
        bin_center=np.sqrt(r_bins[1:]*r_bins[:-1])
        n_bins=len(bin_center)
        cov_int=np.zeros((n_bins,n_bins),dtype='float64')
        bin_idx=np.digitize(r,r_bins)-1
        r2=np.sort(np.unique(np.append(r,r_bins))) #this takes care of problems around bin edges
        dr=np.gradient(r2)
        r2_idx=[i for i in np.arange(len(r2)) if r2[i] in r]
        dr=dr[r2_idx]
        r_dr=r*dr
        cov_r_dr=cov*np.outer(r_dr,r_dr)
        for i in np.arange(min(bin_idx),n_bins):
            xi=bin_idx==i
            for j in np.arange(min(bin_idx),n_bins):
                xj=bin_idx==j
                norm_ij=np.sum(r_dr[xi])*np.sum(r_dr[xj])
                if norm_ij==0:
                    continue
                cov_int[i][j]=np.sum(cov_r_dr[xi,:][:,xj])/norm_ij
        #cov_int=np.nan_to_num(cov_int)
        return bin_center,cov_int

    def skewness(self,k_pk=[],pk1=[],pk2=[],pk3=[],j_nu=[],taper=False,**kwargs):
        pk1=self.pk_grid(k_pk=k_pk,pk=pk1,j_nu=j_nu,taper=taper,**kwargs)
        pk2=self.pk_grid(k_pk=k_pk,pk=pk2,j_nu=j_nu,taper=taper,**kwargs)
        pk3=self.pk_grid(k_pk=k_pk,pk=pk3,j_nu=j_nu,taper=taper,**kwargs)
        skew=np.einsum('ji,ki,li',self.J[j_nu],self.J[j_nu],
                        self.J[j_nu]*pk1*pk2*pk3/self.J_nu1[j_nu]**2)
        skew*=(2.*self.kmax**2/self.zeros[j_nu][-1]**2)/(2*np.pi)
        return self.r[j_nu],skew

    def bin_mat(self,r=[],mat=[],r_bins=[]):#works for cov and skewness
        bin_center=np.sqrt(r_bins[1:]*r_bins[:-1])
        n_bins=len(bin_center)
        ndim=len(mat.shape)
        mat_int=np.zeros([n_bins]*ndim,dtype='float64')
        norm_int=np.zeros([n_bins]*ndim,dtype='float64')
        bin_idx=np.digitize(r,r_bins)-1
        r2=np.sort(np.unique(np.append(r,r_bins))) #this takes care of problems around bin edges
        dr=np.gradient(r2)
        r2_idx=[i for i in np.arange(len(r2)) if r2[i] in r]
        dr=dr[r2_idx]
        r_dr=r*dr

        ls=['i','j','k','l']
        s1=ls[0]
        s2=ls[0]
        r_dr_m=r_dr
        for i in np.arange(ndim-1):
            s1=s2+','+ls[i+1]
            s2+=ls[i+1]
            r_dr_m=np.einsum(s1+'->'+s2,r_dr_m,r_dr)#works ok for 2-d case

        mat_r_dr=mat*r_dr_m
        for indxs in itertools.product(np.arange(min(bin_idx),n_bins),repeat=ndim):
            x={}#np.zeros_like(mat_r_dr,dtype='bool')
            norm_ijk=1
            mat_t=[]
            for nd in np.arange(ndim):
                slc = [slice(None)] * (ndim)
                #x[nd]=bin_idx==indxs[nd]
                slc[nd]=bin_idx==indxs[nd]
                if nd==0:
                    mat_t=mat_r_dr[slc]
                else:
                    mat_t=mat_t[slc]
                norm_ijk*=np.sum(r_dr[slc[nd]])
            if norm_ijk==0:
                continue
            mat_int[indxs]=np.sum(mat_t)/norm_ijk
            norm_int[indxs]=norm_ijk
        return bin_center,mat_int





        