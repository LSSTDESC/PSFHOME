import galsim
import numpy as np
from numpy import mgrid, sum
import scipy.linalg as alg
import scipy.stats as stats 


class shapeletXmoment:

    """
    A class to manipulate PSF using radial shapelets, and change higher moments by gradient descent.

    """

    def __init__(self, psf, n, bmax=10, pixel_scale=1.0):

        """
        Initialize class.

        Parameters:

        psf: PSF galsim object
        n: highest order of higher moments we want to measure
        bmax: highest shapelet moment we use to expand PSF
        pixel_scale: pixel scale.

        """
        self.n = n
        self.bmax = bmax
        self.pixel_scale = pixel_scale
        self.base_psf = psf
        self.base_psf_image = psf.drawImage(scale=pixel_scale)
        self.base_psf_result = galsim.hsm.FindAdaptiveMom(self.base_psf_image)
        self.base_shapelet = galsim.Shapelet.fit(
            self.base_psf_result.moments_sigma,
            bmax,
            self.base_psf_image,
            normalization="sb",
        )
        self.base_bvec = self.base_shapelet.bvec

    def moment_measure(self, image, p, q):
        """
        Measure higher moment M_{pq} of the image. If we are measuring 2nd moments, return e2,sigma,e1.

        Parameters:
        image: galsim image we want to measrue
        p: x index
        q: y index
        """
        n = p + q

        if n < 2:
            print("Does not support moment measure less than second order.")
            return 0
        elif n == 2:
            return self.get_second_moment(image, p, q)
        else:
            return self.higher_weighted_moment(image, p, q)

    def get_second_moment(self, image, p, q):
        """
        Return higher moments
        p=2, q=0: e1
        p=1, q=1: sigma
        o=0, q=2: e2
        """
        image_results = galsim.hsm.FindAdaptiveMom(image)
        if p == 2:
            return image_results.observed_shape.e1
        elif q == 2:
            return image_results.observed_shape.e2
        else:
            return image_results.moments_sigma

    def higher_weighted_moment(self, gsimage, p, q):

        """
        Calculate the PSF higher moments.

        Input:
        gsimage: galsim image object of the PSF
        p: x index
        q: y index
        """

        image = gsimage.array

        y, x = mgrid[: image.shape[0], : image.shape[1]] + 1

        psfresults = galsim.hsm.FindAdaptiveMom(
            galsim.Image(image, scale=self.pixel_scale)
        )
        M = np.zeros((2, 2))
        e1 = psfresults.observed_shape.e1
        e2 = psfresults.observed_shape.e2
        sigma4 = psfresults.moments_sigma**4
        c = (1 + e1) / (1 - e1)
        M[1][1] = np.sqrt(sigma4 / (c - 0.25 * e2**2 * (1 + c) ** 2))
        M[0][0] = c * M[1][1]
        M[0][1] = 0.5 * e2 * (M[1][1] + M[0][0])
        M[1][0] = M[0][1]

        pos = np.array(
            [x - psfresults.moments_centroid.x, y - psfresults.moments_centroid.y]
        )
        pos = np.swapaxes(pos, 0, 1)
        pos = np.swapaxes(pos, 1, 2)

        inv_M = np.linalg.inv(M)
        sqrt_inv_M = alg.sqrtm(inv_M)
        std_pos = np.zeros(pos.shape)
        weight = np.zeros(pos.shape[0:2])
        for i in range(pos.shape[0]):
            for j in range(pos.shape[1]):
                this_pos = pos[i][j]
                this_standard_pos = np.matmul(sqrt_inv_M, this_pos)
                std_pos[i][j] = this_standard_pos
                weight[i][j] = np.exp(-0.5 * this_standard_pos.dot(this_standard_pos))

        std_x, std_y = std_pos[:, :, 0], std_pos[:, :, 1]

        return sum(std_x**p * std_y**q * weight * image) / sum(image * weight)

    def modify_pq(self, m, c, delta=0.0001):
        
        """
        Modify the higher moments of the image by a multiplicative factor of m, 
        or additive factor of c. m and c are arrays in the order of the default
        order of the PSF higher moments.
        """

        n = self.n
        mu = self.get_mu(n)
        pq_list = self.get_pq_full(n)
        shapelet_list = self.pq2shapelet(pq_list)

        ori_moments = self.get_all_moments(self.base_psf_image, pq_list)

        A = np.zeros(shape=(mu, mu))

        # i is the mode index
        # j is the moment index
        # measure d_moment_j / d_mode_i

        for i in range(mu):

            mode_index = shapelet_list[i]

            pert_bvec = self.base_bvec.copy()
            pert_bvec[mode_index] += delta
            ith_pert = galsim.Shapelet(
                self.base_psf_result.moments_sigma, self.bmax, pert_bvec
            )
            pert_moment = self.get_all_moments(
                ith_pert.drawImage(scale=self.pixel_scale, method="no_pixel"), pq_list
            )
            for j in range(mu):
                A[i][j] = (pert_moment[j] - ori_moments[j]) / delta
        self.A = A

        dm = np.zeros(mu)
        dm += m * ori_moments + c
        ds = np.linalg.solve(A.T, dm)
        # print ds
        # print dm

        true_mod_bvec = self.base_bvec.copy()
        for i in range(mu):
            true_mod_bvec[shapelet_list[i]] += ds[i]

        self.true_mod = galsim.Shapelet(
            self.base_psf_result.moments_sigma, self.bmax, true_mod_bvec
        )
        return self.true_mod

    def step_modify_pq(
        self,
        current_moments,
        current_dm,
        current_mod_bvec,
        current_psf,
        mu,
        shapelet_list,
        delta,
        pq_list,
    ):
        """
        helper function of 'modify_pq'
        """

        A = np.zeros(shape=(mu, mu))

        for i in range(mu):
            mode_index = shapelet_list[i]

            pert_bvec = current_mod_bvec.copy()
            pert_bvec[mode_index] += delta
            ith_pert = galsim.Shapelet(
                self.base_psf_result.moments_sigma, self.bmax, pert_bvec
            )
            pert_moment = self.get_all_moments(
                ith_pert.drawImage(scale=self.pixel_scale), pq_list
            )
            for j in range(mu):
                A[i][j] = (pert_moment[j] - current_moments[j]) / delta

        ds = np.linalg.solve(A.T, current_dm)

        for i in range(mu):
            current_mod_bvec[shapelet_list[i]] += ds[i]
        return current_mod_bvec

    def iterative_modify_pq(self, m, c, delta=0.0001, threshold=1e-6):
        """
        helper function of 'modify_pq'
        """
        iterative_n = 10

        n = self.n
        mu = self.get_mu(n)
        pq_list = self.get_pq_full(n)
        shapelet_list = self.pq2shapelet(pq_list)
        base_shapelet_image = self.base_shapelet.drawImage(scale=self.pixel_scale)
        original_moment = self.get_all_moments(base_shapelet_image, pq_list)
        current_moment = self.get_all_moments(base_shapelet_image, pq_list)

        current_dm = np.zeros(mu)
        current_dm += m * current_moment + c

        destiny_moment = current_moment + current_dm

        current_mod_bvec = self.base_bvec.copy()
        current_psf = galsim.Shapelet(
            self.base_psf_result.moments_sigma, self.bmax, current_mod_bvec
        )

        while np.max(np.abs(current_dm)) > threshold:

            current_mod_bvec = self.step_modify_pq(
                current_moment,
                current_dm,
                current_mod_bvec,
                current_psf,
                mu,
                shapelet_list,
                delta,
                pq_list,
            )
            current_psf = galsim.Shapelet(
                self.base_psf_result.moments_sigma, self.bmax, current_mod_bvec
            )
            current_moment = self.get_all_moments(
                current_psf.drawImage(scale=self.pixel_scale), pq_list
            )

            current_dm = destiny_moment - current_moment
            # print current_dm
            # print current_moment - original_moment
        return current_psf

    def get_all_moments(self, image, pq_list):
        """
        get the moments of `image` in the same order of (p,q)
        in `pq_list`
        """
        results_list = []
        for tup in pq_list:
            results_list.append(self.moment_measure(image, tup[0], tup[1]))

        return np.array(results_list)

    def pq2mode(self, p, q):
        
        if p <= q:
            return (p + q) * (p + q + 1) // 2 + 2 * min(p, q)
        else:
            return (p + q) * (p + q + 1) // 2 + 2 * min(p, q) + 1

    def pq2shapelet(self, pq_list):
        
        """
        convert (p,q) to the shapelet index
        """
        
        shapelet_index = []
        for tup in pq_list:
            shapelet_index.append(self.pq2mode(tup[0], tup[1]))
        return shapelet_index

    def get_mu(self, n):

        mu = 0
        for i in range(2, n + 1):
            mu += i + 1
        return mu

    def get_pq_full(self, nmax):
        
        """
        get a list of (p,q) index from n=2 to n=`nmax`
        """

        pq_list = []

        for n in range(2, nmax + 1):
            p = 0
            q = n

            pq_list.append((p, q))

            while p < n:
                p += 1
                q -= 1
                pq_list.append((p, q))
        return pq_list

    def get_pq_except(self, nmax, p, q):

        pq_full = self.get_pq_full(nmax)
        pq_except = []
        for tup in pq_full:
            if tup != (p, q):
                pq_except.append(tup)

        return pq_except


def get_all_moments_fast(image, pqlist):
    
    """
    A faster function for computing the PSF higher moments for an image. This only does the standardization
    transformation to the image once for all the higher moments.
    """

    results_list = []
    
    image_array = image.array
    
    y, x = mgrid[:image_array.shape[0],:image_array.shape[1]]+1
    psfresults = galsim.hsm.FindAdaptiveMom(image)
    M = np.zeros((2,2))
    e1 = psfresults.observed_shape.e1
    e2 = psfresults.observed_shape.e2
    sigma4 = psfresults.moments_sigma**4
    c = (1+e1)/(1-e1)
    M[1][1] = np.sqrt(sigma4/(c-0.25*e2**2*(1+c)**2))
    M[0][0] = c*M[1][1]
    M[0][1] = 0.5*e2*(M[1][1]+M[0][0])
    M[1][0] = M[0][1]

    pos = np.array([x-psfresults.moments_centroid.x, y-psfresults.moments_centroid.y])
    inv_M = np.linalg.inv(M)
    sqrt_inv_M = alg.sqrtm(inv_M)
    
    std_pos = np.einsum('ij,jqp->iqp',sqrt_inv_M,pos)
    weight = np.exp(-0.5* np.einsum('ijk,ijk->jk',std_pos,std_pos ))

    std_x, std_y = std_pos[0],std_pos[1]
    
    normalization = sum(image_array*weight)
    image_weight = weight*image_array
    for tup in pqlist:
        p = tup[0]
        q = tup[1]
        
        if q+p==2:
            if p==2:
                this_moment = e1
            elif q==2:
                this_moment = e2
            else:
                this_moment = psfresults.moments_sigma
        else:
            this_moment = sum(std_x**p*std_y**q*image_weight)/normalization
        results_list.append(this_moment)
        
    t3 = time.time()        
    return np.array(results_list)


def get_all_moments_raw(image,pqlist):
    
    results_list = []
    
    image_array = image.array
    
    y, x = mgrid[:image_array.shape[0],:image_array.shape[1]]+1
    psfresults = galsim.hsm.FindAdaptiveMom(image)
    M = np.zeros((2,2))
    e1 = psfresults.observed_shape.e1
    e2 = psfresults.observed_shape.e2
    sigma4 = psfresults.moments_sigma**4
    c = (1+e1)/(1-e1)
    M[1][1] = np.sqrt(sigma4/(c-0.25*e2**2*(1+c)**2))
    M[0][0] = c*M[1][1]
    M[0][1] = 0.5*e2*(M[1][1]+M[0][0])
    M[1][0] = M[0][1]
    
    
    pos = np.array([x-psfresults.moments_centroid.x, y-psfresults.moments_centroid.y])
    inv_M = np.linalg.inv(M)
    sqrt_inv_M = alg.sqrtm(inv_M)
    
    std_pos = np.einsum('ij,jqp->iqp',sqrt_inv_M,pos)
    weight = np.exp(-0.5* np.einsum('ijk,ijk->jk',std_pos,std_pos ))

    pos_x, pos_y = pos[1],pos[0]
    normalization = sum(image_array*weight)
    image_weight = weight*image_array
    for tup in pqlist:
        p = tup[0]
        q = tup[1]
        
        if q+p==2:
            if p==2:
                this_moment = e1
            elif q==2:
                this_moment = e2
            else:
                this_moment = psfresults.moments_sigma
        else:
            this_moment = sum(pos_x**p*pos_y**q*image_weight)/normalization
        results_list.append(this_moment)
        
    return np.array(results_list)
        

def standard_from_raw(e1,e2,sigma,m40,m31,m22,m13,m04):
    
    """
    Convert raw moments to standardized moments
    m40,m31,m22,m13,m04 are raw fourth moments defined in the cartesian coordinate 
    """
    
    sigma4 = sigma**4
    c = (1 + e1) / (1 - e1)
    myy = np.sqrt(sigma4 / (c - 0.25 * e2**2 * (1 + c) ** 2))
    mxx = c * myy
    mxy = 0.5 * e2 * (myy + mxx)
    
    d = mxx*myy-mxy**2
    zeta = d*(mxx+myy+2*np.sqrt(d))
    
    a = (myy + np.sqrt(d))/np.sqrt(zeta)
    b = -mxy/np.sqrt(zeta)
    c = (mxx+np.sqrt(d))/np.sqrt(zeta)
        
    stan_m40 = a**4*m40 + 4*a**3*b*m31 + 6*a**2*b**2*m22 + 4*a*b**3*m13 + b**4*m04
    stan_m31 = (a**3*b)*m40 + (a**3*c+3*a**2*b**2)*m31 + (3*a**2*b*c+3*a*b**3)*m22 + (3*a*b**2*c+b**4)*m13 + b**3*c*m04
    stan_m22 = a**2*b**2*m40 + (2*a**2*b*c+2*a*b**3)*m31 + (a**2*c**2+4*a*b**2*c+b**4)*m22 + (2*a*b*c**2+2*b**3*c)*m13+ b**2*c**2*m04
    stan_m13 = a*b**3*m40 + (3*a*b**2*c+b**4)*m31 + (3*a*b*c**2+3*b**3*c)*m22 + (a*c**3+3*b**2*c**2)*m13 + b*c**3*m04
    stan_m04 = b**4*m40 + 4*b**3*c*m31 + 6*b**2*c**2*m22 + 4*b*c**3*m13 + c**4*m04

    return stan_m40, stan_m31, stan_m22, stan_m13, stan_m04




        