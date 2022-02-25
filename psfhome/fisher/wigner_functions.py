"""
This file contains functions to compute wigner matrices used in wigner transform (power spectra to correlation functions) 
and wigner_3j matrices used in window function calculations. 
"""

import numpy as np
from scipy.special import binom,jn,loggamma
from scipy.special import eval_jacobi as jacobi
from multiprocessing import Pool,cpu_count
from functools import partial
import sparse
from sympy import Integer
from sympy import sqrt as sy_sqrt
from sympy import exp as sy_exp
from sympy import log as sy_log

from mpmath import exp as mp_exp
from mpmath import log as mp_log
from sympy.physics.wigner import wigner_3j


def wigner_d(m1,m2,theta,l,l_use_bessel=1.e4):
    """
    Function to compute wigner matrices used in wigner transforms.
    """
    l0=np.copy(l)
    if l_use_bessel is not None:
    #FIXME: This is not great. Due to a issues with the scipy hypergeometric function,
    #jacobi can output nan for large ell, l>1.e4
    # As a temporary fix, for ell>1.e4, we are replacing the wigner function with the
    # bessel function. Fingers and toes crossed!!!
    # mpmath is slower and also has convergence issues at large ell.
    #https://github.com/scipy/scipy/issues/4446
    
        l=np.atleast_1d(l)
        x=l<l_use_bessel
        l=np.atleast_1d(l[x])
    k=np.amin([l-m1,l-m2,l+m1,l+m2],axis=0)
    a=np.absolute(m1-m2)
    lamb=0 #lambda
    if m2>m1:
        lamb=m2-m1
    b=2*l-2*k-a
    d_mat=(-1)**lamb
    d_mat*=np.sqrt(binom(2*l-k,k+a)) #this gives array of shape l with elements choose(2l[i]-k[i], k[i]+a)
    d_mat/=np.sqrt(binom(k+b,b))
    d_mat=np.atleast_1d(d_mat)
    x=k<0
    d_mat[x]=0

    d_mat=d_mat.reshape(1,len(d_mat))
    theta=theta.reshape(len(theta),1)
    d_mat=d_mat*((np.sin(theta/2.0)**a)*(np.cos(theta/2.0)**b))
    x=d_mat==0
    d_mat*=jacobi(k,a,b,np.cos(theta)) #l
    d_mat[x]=0
    
    if l_use_bessel is not None:
        l=np.atleast_1d(l0)
        x=l>=l_use_bessel
        l=np.atleast_1d(l[x])
#         d_mat[:,x]=jn(m1-m2,l[x]*theta)
        d_mat=np.append(d_mat,jn(m1-m2,l*theta),axis=1)
    return d_mat

def wigner_d_parallel(m1,m2,theta,l,ncpu=None,l_use_bessel=1.e4):
    """
    Compute wigner matrix in parallel.
    """
    if ncpu is None:
        ncpu=cpu_count()
    p=Pool(ncpu)
    d_mat=np.array(p.map(partial(wigner_d,m1,m2,theta,l_use_bessel=l_use_bessel),l))
    p.close()
    p.join()
    return d_mat[:,:,0].T