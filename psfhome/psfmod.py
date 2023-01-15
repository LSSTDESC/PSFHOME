import emcee
import numpy as np
from astropy.table import Table, vstack
import treecorr
import scipy
import pickle
from scipy import stats


class psfmod:

    """
    perform PSF modeling fitting to the galaxy-PSF cross correlation
    This class can measure the galaxy-PSF cross correlation, and PSF-PSF auto correlation
    You can specify a type of model by setting "mode"
    You can include a constant term in the e_sys model by switching on "constant mode"
    This class also performs MCMC to find the bestfit paramters and will auto save the chain
    The postprocessing will produce delta_xip, and its decomposition
    """

    def __init__(
        self,
        mode,
        cov_file,
        chain_len=5000,
        correlation_ready=False,
        correlation_file=None,
        theta_min=1.0,
        theta_max=200,
        constant_mode=True,
        n_walker=50,
        save_chain=True,
        chain_root="./chains/",
        prefix="s19a",
        gal_shape_file="data/egal.fits",
        fitsecond=False,
        nonpsf=False,
        ximmode=False,
        seeing_weight=False,
        include_pp_cov=True,
        shear_convention=False,
    ):
        """
        Initialization
        mode: "second" for second moments only, "four" for second + fourth moments
              "fourth_only" for fourth moments only.
        cov_file: the filename for the full covariance matrix of the data vector
        chain_len = MCMC chain length
        correlation_ready: if True, read the correlation functions from correlation_file
                            else, measure the correlation functions
        correlation_file: if correlation_ready==True, read correlation from this file
        n_walker: number of mcmc walker
        save_chain: if True, save the chain file
        chain_root: if save_chain==True, save the chain at this root directory
        prefix: prefix for the chain file
        """

        self.mode = mode
        self.cov_file = cov_file
        self.chain_len = chain_len
        self.correlation_ready = correlation_ready
        self.correlation_file = correlation_file
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.constant_mode = constant_mode
        self.n_walker = n_walker
        self.save_chain = save_chain
        self.chain_root = chain_root
        self.prefix = prefix
        self.gal_shape_file = gal_shape_file
        self.fitsecond = fitsecond
        self.nonpsf = nonpsf
        self.ximmode = ximmode
        self.seeing_weight = seeing_weight
        self.num_of_corr = 4
        self.include_pp_cov = include_pp_cov
        self.shear_convention = shear_convention

    def go(self):

        """
        Run everything in this class
        """

        self.measure_correlation()

        self.read_cov()

        self.generate_ang_slice()

        self.run_mcmc()

        self.generate_delta_xip(self.mean_params)

    def get_corr(self, cat1, cat2, min_sep=1, max_sep=200, nbins=20):

        """
        Get the correlation functions
        cat1: catalog 1
        cat2: catalog 2
        min_sep: minimum angular separation in arcmin
        max_sep: maximum angular separation in arcmin
        nbins: number of bins
        """

        gg_ij = treecorr.GGCorrelation(
            min_sep=min_sep,
            max_sep=max_sep,
            nbins=nbins,
            sep_units="arcmin",
            var_method="jackknife",
        )

        gg_ij.process(cat1, cat2)

        if self.ximmode == True:
            return gg_ij.meanlogr, gg_ij.xim, gg_ij.cov[20:, 20:], gg_ij
        else:

            return gg_ij.meanlogr, gg_ij.xip, gg_ij.cov[:20, :20], gg_ij

    def measure_correlation(self):

        """
        This function measures the correlation function between the galaxy-PSF and PSF-PSF
        """

        # read some variables for later use in this function
        mode = self.mode
        constant_mode = self.constant_mode
        num_of_corr = self.num_of_corr

        # if correlation is ready, just process to get the cross and auto
        if self.correlation_ready:

            with open(self.correlation_file, "rb") as f:
                (
                    r,
                    gp_corr,
                    pp_corr,
                    psf_const1,
                    psf_const2,
                    egal_mean,
                    pp_corr_cov,
                    pp_joint_cov,
                ) = pickle.load(f)

            self.r = r
            self.gp_corr = gp_corr
            self.pp_corr = pp_corr
            self.egal_mean = egal_mean

            self.psf_const1 = psf_const1
            self.psf_const2 = psf_const2

            # self.gp_corr_cov = gp_corr_cov
            self.pp_corr_cov = pp_corr_cov
            self.pp_joint_cov = pp_joint_cov

            self.pp_joint_cov_inv = np.linalg.inv(pp_joint_cov)

        # if the correlation is not ready, measure them
        else:

            # load galaxy shape and PSF moments tables
            cat_egal = treecorr.Catalog(
                self.gal_shape_file,
                w_col="weight",
                ra_col="ra",
                dec_col="dec",
                ra_units="deg",
                dec_units="deg",
                g1_col="shear1",
                g2_col="shear2",
                npatch=20,
                kmeans_init="kmeans++",
            )

            if self.nonpsf:

                cat_epsf = treecorr.Catalog(
                    "data/epsf_nonpsf.fits",
                    ra_col="ra",
                    dec_col="dec",
                    ra_units="deg",
                    dec_units="deg",
                    g1_col="shear1",
                    g2_col="shear2",
                    patch_centers=cat_egal.patch_centers,
                )
                cat_depsf = treecorr.Catalog(
                    "data/depsf_nonpsf.fits",
                    ra_col="ra",
                    dec_col="dec",
                    ra_units="deg",
                    dec_units="deg",
                    g1_col="shear1",
                    g2_col="shear2",
                    patch_centers=cat_egal.patch_centers,
                )
                cat_dMpsf = treecorr.Catalog(
                    "data/dMpsf_nonpsf.fits",
                    ra_col="ra",
                    dec_col="dec",
                    ra_units="deg",
                    dec_units="deg",
                    g1_col="shear1",
                    g2_col="shear2",
                    patch_centers=cat_egal.patch_centers,
                )
                cat_Mpsf = treecorr.Catalog(
                    "data/Mpsf_nonpsf.fits",
                    ra_col="ra",
                    dec_col="dec",
                    ra_units="deg",
                    dec_units="deg",
                    g1_col="shear1",
                    g2_col="shear2",
                    patch_centers=cat_egal.patch_centers,
                )
            else:

                cat_epsf = treecorr.Catalog(
                    "data/epsf_psf.fits",
                    ra_col="ra",
                    dec_col="dec",
                    ra_units="deg",
                    dec_units="deg",
                    g1_col="shear1",
                    g2_col="shear2",
                    patch_centers=cat_egal.patch_centers,
                )
                cat_depsf = treecorr.Catalog(
                    "data/depsf_psf.fits",
                    ra_col="ra",
                    dec_col="dec",
                    ra_units="deg",
                    dec_units="deg",
                    g1_col="shear1",
                    g2_col="shear2",
                    patch_centers=cat_egal.patch_centers,
                )
                cat_dMpsf = treecorr.Catalog(
                    "data/dMpsf_psf.fits",
                    ra_col="ra",
                    dec_col="dec",
                    ra_units="deg",
                    dec_units="deg",
                    g1_col="shear1",
                    g2_col="shear2",
                    patch_centers=cat_egal.patch_centers,
                )
                cat_Mpsf = treecorr.Catalog(
                    "data/Mpsf_psf.fits",
                    ra_col="ra",
                    dec_col="dec",
                    ra_units="deg",
                    dec_units="deg",
                    g1_col="shear1",
                    g2_col="shear2",
                    patch_centers=cat_egal.patch_centers,
                )

            gal_cat = cat_egal

            # measure the mean galxy shape, that's the last two data points
            e_gal_1_mean = np.mean(np.array(gal_cat.g1))
            e_gal_2_mean = np.mean(np.array(gal_cat.g2))

            self.egal_mean = np.array([e_gal_1_mean, e_gal_2_mean])

            # define the PSF moments list, [second leakage, second modeling, fourth leakage, fourth modeling]
            psf_cat_list = [cat_epsf, cat_depsf, cat_Mpsf, cat_dMpsf]

            # measure the psf moments average, e1 and e2
            self.psf_const1 = np.array(
                [np.mean(psf_cat_list[i].g1) for i in range(num_of_corr)]
            )
            self.psf_const2 = np.array(
                [np.mean(psf_cat_list[i].g2) for i in range(num_of_corr)]
            )

            # measure the galaxy-PSF cross correlations
            gp_corr_xip = np.zeros(shape=(num_of_corr, 20))
            # gp_corr_cov = np.zeros(shape = (num_of_corr,20,20))

            for i in range(num_of_corr):
                logr, this_xip, this_cov, _ = self.get_corr(gal_cat, psf_cat_list[i])
                # gp_corr_cov[i] = this_cov
                gp_corr_xip[i] = this_xip

            # measure the PSF-PSF correlations
            pp_corr_xip = np.zeros(shape=(num_of_corr, num_of_corr, 20))
            pp_corr_cov = np.zeros(shape=(num_of_corr, num_of_corr, 20, 20))

            pp_corr_list = []

            for i in range(num_of_corr):
                for j in range(num_of_corr):
                    logr, this_xip, this_cov, pp_corr_item = self.get_corr(
                        psf_cat_list[i], psf_cat_list[j]
                    )
                    pp_corr_cov[i][j] = this_cov
                    pp_corr_xip[i][j] = this_xip
                    pp_corr_list.append(pp_corr_item)

            slice_ = []
            for i in range(num_of_corr**2):
                slice_.append(np.arange(0, 20) + i * 40)
            slice_ = np.array(slice_).reshape(-1)

            self.pp_joint_cov = treecorr.estimate_multi_cov(pp_corr_list, "jackknife")[
                slice_, :
            ][:, slice_]
            self.pp_joint_cov_inv = np.linalg.inv(self.pp_joint_cov)
            #             for i in range(num_of_corr):
            #                 for j in range(0,i):
            #                     pp_corr_xip[i][j] = pp_corr_xip[j][i]
            #                     pp_corr_cov[i][j] = pp_corr_cov[j][i]

            self.r = np.exp(logr)

            self.pp_corr = pp_corr_xip
            self.pp_corr_cov = pp_corr_cov

            self.gp_corr = gp_corr_xip
            # self.gp_corr_cov = gp_corr_cov

    def read_cov(self):

        cov = np.loadtxt(self.cov_file)

        # if self.shear_convention==False:
        #     cov[0:80,0:40] *= 2
        #     cov[0:40,0:80] *= 2
        self.cov = cov
        self.full_cov_inv = np.linalg.inv(cov)

        # self.cov = np.diag(np.diag(self.cov))

    def generate_ang_slice(self):

        slice_ = []
        r = self.r

        slice_low = np.where(r > self.theta_min)[0][0]
        slice_high = np.where(r < self.theta_max)[0][-1] + 1
        if self.fitsecond == False:
            for i in range(self.num_of_corr):
                slice_.append(np.arange(slice_low, slice_high) + i * 20)
            slice_ = np.array(slice_).reshape(-1)

            slice_ = np.concatenate([slice_, np.array([80, 81], dtype="int")])
        else:
            for i in [0, 1]:
                slice_.append(np.arange(slice_low, slice_high) + i * 20)

            slice_ = np.array(slice_).reshape(-1)

        # slice_ = np.arange(20,40)

        self.slice_ = slice_

        self.new_cov = self.cov[slice_][:, slice_]
        self.new_cov_inv = np.linalg.inv(self.new_cov)

    def forward_model(self, param, pg, cov_inv, pp, slice_, rand_pp, n_b=20):

        # t1 = time.time()

        num_of_corr = self.num_of_corr

        gp_theory = np.zeros(shape=(num_of_corr, n_b))

        prior = self.get_prior(param)

        # print(pp.reshape(-1).shape, self.pp_joint_cov.shape)

        param_matrix = self.construct_param_matrix(param)

        if self.include_pp_cov:
            pp_cov = np.zeros(shape=(82, 82))
            pp_cov[0:80, 0:80] = (param_matrix @ self.pp_joint_cov) @ param_matrix.T

            new_cov = (self.cov + pp_cov)[self.slice_, :][:, self.slice_]
            new_cov_inv = np.linalg.inv(new_cov)

            self.norm_factor = compute_logdet(new_cov, 1e12)

        else:
            new_cov_inv = cov_inv
            self.norm_factor = 0

        # print(pp.reshape(-1).shape,param_matrix.shape )
        gp_theory = param_matrix @ pp.reshape(-1)

        ec_mpsf = np.array(
            [param[-2] * self.psf_const1[i].repeat(20) for i in range(4)]
        ).reshape(-1) + np.array(
            [param[-1] * self.psf_const2[i].repeat(20) for i in range(4)]
        ).reshape(
            -1
        )

        gp_theory += ec_mpsf

        e1gal_theory = (
            np.sum([param[i] * self.psf_const1[i] for i in range(num_of_corr)])
            + param[-2]
        )
        e2gal_theory = (
            np.sum([param[i] * self.psf_const2[i] for i in range(num_of_corr)])
            + param[-1]
        )

        gp_theory_db = np.concatenate(
            [gp_theory.reshape(-1), np.array([e1gal_theory, e2gal_theory])]
        )
        # print(gp_theory_db)
        res = gp_theory_db - pg

        new_res = res[slice_]

        chi2 = new_res.dot(new_cov_inv).dot(new_res)

        # t2 = time.time()
        # print(t2 - t1)
        # print(chi2)
        return chi2 + prior

    def construct_param_matrix(self, param):

        res = np.zeros(shape=(80, 320))

        basic_block = np.zeros(shape=(20, 80))
        for i in range(4):
            basic_block[0:20, i * 20 : (i + 1) * 20] = np.diag([param[i]] * 20)

        # print(basic_block)
        for i in range(4):
            res[i * 20 : (i + 1) * 20, i * 80 : (i + 1) * 80] = basic_block

        return res

    def compute_cov_inv_pp(self, param_matrix, pp_cov_inv, gp_cov_inv):

        res = np.zeros(shape=(82, 82))

        C0_alpha = gp_cov_inv @ param_matrix
        denominator = (param_matrix.T @ gp_cov_inv) @ param_matrix + pp_cov_inv

        pp_cov_inv = (C0_alpha @ np.linalg.inv(denominator)) @ C0_alpha.T

        res[0:80, 0:80] = pp_cov_inv

        return res

    def get_prior(self, param):

        if self.constant_mode == False:
            boo = abs(param[-1]) > 1e-8 or abs(param[-2]) > 1e-8
            if boo:
                return np.inf

        if self.mode == "second":
            boo = abs(param[2]) > 1e-8 or abs(param[3]) > 1e-8
            if boo:
                return np.inf

        elif self.mode == "fourth_only":
            boo = abs(param[0]) > 1e-8 or abs(param[1]) > 1e-8
            if boo:
                return np.inf

        return 0

    def log_prob(
        self,
        param,
        pg,
        cov_inv,
        pp,
        slice_,
        rand_pp,
        num_of_corr=4,
    ):
        loss = self.forward_model(param, pg, cov_inv, pp, slice_, rand_pp)
        return -0.5 * (loss + self.norm_factor)

    def run_mcmc(self):

        constant_mode = self.constant_mode
        num_of_corr = self.num_of_corr

        num_of_param = num_of_corr + 2

        observe_dv = np.concatenate([self.gp_corr.reshape(-1), self.egal_mean])
        self.observe_dv = observe_dv

        bounds = [
            (-np.inf, np.inf),
            (-np.inf, np.inf),
            (-np.inf, np.inf),
            (-np.inf, np.inf),
            (-np.inf, np.inf),
            (-np.inf, np.inf),
        ]

        if self.constant_mode == False:
            bounds[-1] = (0.0, 0.0)
            bounds[-2] = (0.0, 0.0)

        if self.mode == "second":
            bounds[2] = (0.0, 0.0)
            bounds[3] = (0.0, 0.0)

        elif self.mode == "fourth_only":
            bounds[0] = (0.0, 0.0)
            bounds[1] = (0.0, 0.0)

        best_fit = scipy.optimize.minimize(
            self.forward_model,
            x0=np.zeros(num_of_param),
            bounds=bounds,
            args=(observe_dv, self.new_cov_inv, self.pp_corr, self.slice_, False),
        )

        print(best_fit.x)

        nwalkers = self.n_walker
        pos = best_fit.x + np.random.randn(nwalkers, num_of_param) * 0.1
        # print(pos.shape)
        if self.constant_mode == False:
            pos[:, -1] = np.random.randn(nwalkers) * 1e-10
            pos[:, -2] = np.random.randn(nwalkers) * 1e-10

        if self.mode == "second":
            pos[:, 2] = np.random.randn(nwalkers) * 1e-10
            pos[:, 3] = np.random.randn(nwalkers) * 1e-10

        elif self.mode == "fourth_only":
            pos[:, 0] = np.random.randn(nwalkers) * 1e-10
            pos[:, 1] = np.random.randn(nwalkers) * 1e-10

        # self.pp_pool = np.random.multivariate_normal(self.pp_corr.reshape(-1), self.pp_joint_cov, size=100000)

        print("start running mcmc")

        sampler = emcee.EnsembleSampler(
            nwalkers,
            num_of_param,
            self.log_prob,
            args=(observe_dv, self.new_cov_inv, self.pp_corr, self.slice_, True),
        )

        # print(pos)
        state = sampler.run_mcmc(pos, self.chain_len)
        flatchain = sampler.get_chain(discard=200, flat=True)

        mean_params = np.mean(flatchain, axis=0)

        # print(mean_params)

        self.mean_params = mean_params
        self.best_fit = best_fit.x
        self.flatchain = flatchain

        if self.save_chain == True:
            filename = "{}_chain_{}_const={}.txt".format(
                self.prefix, self.mode, constant_mode
            )

            np.savetxt(self.chain_root + filename, flatchain)

        df = len(self.slice_) - 1

        if self.constant_mode == True:
            df -= 2

        if self.mode == "four":
            df -= 4
        elif self.mode == "second" or self.mode == "fourth_only":
            df -= 2

        self.df = df
        self.chi2 = self.forward_model(
            mean_params, observe_dv, self.new_cov_inv, self.pp_corr, self.slice_, False
        )
        self.p_value = 1 - stats.chi2.cdf(self.chi2, df)

        bestfit_list = []

        for i in range(num_of_corr):
            this_gp = np.zeros(20)
            for j in range(num_of_corr):
                this_gp += mean_params[j] * self.pp_corr[i][j]

            if self.constant_mode:
                this_gp += mean_params[-2] * self.psf_const1[i]
                this_gp += mean_params[-1] * self.psf_const2[i]
            bestfit_list.append(this_gp)

        bestfit_e1 = (
            np.sum([mean_params[i] * self.psf_const1[i] for i in range(num_of_corr)])
            + mean_params[-2]
        )
        bestfit_e2 = (
            np.sum([mean_params[i] * self.psf_const2[i] for i in range(num_of_corr)])
            + mean_params[-1]
        )

        self.bestfit_dv = np.array(bestfit_list)
        self.bestfit_ec = [bestfit_e1, bestfit_e2]

    def generate_delta_xip(self, param, mean_param=True):

        num_of_corr = self.num_of_corr
        mean_params = param

        pp_corr = self.pp_corr

        delta_xip = np.zeros(20)
        delta_xip_components = []

        for i in range(num_of_corr):
            for j in range(num_of_corr):

                this_component = mean_params[i] * mean_params[j] * pp_corr[i][j]

                delta_xip += this_component
                delta_xip_components.append(this_component)

        for i in range(num_of_corr):

            this_component = (
                2
                * mean_params[i]
                * (
                    self.psf_const1[i] * mean_params[-2]
                    + self.psf_const2[i] * mean_params[-1]
                )
            )
            delta_xip += this_component
            delta_xip_components.append(this_component)

        const_const = mean_params[-2] ** 2 + mean_params[-1] ** 2
        delta_xip += const_const
        delta_xip_components.append(const_const)

        if mean_param == True:

            self.delta_xip = delta_xip

            self.delta_xip_components = delta_xip_components

            self.corr_corr_dxip = np.sum(np.array(delta_xip_components[:16]), axis=0)
            self.corr_const_dxip = np.sum(np.array(delta_xip_components[16:20]))
            self.const_const_dxip = const_const

        return delta_xip

    def sample_param(self, n):

        flatchain = self.flatchain

        params_mean = np.mean(flatchain, axis=0)

        params_cov = np.cov(flatchain.T)

        return np.random.multivariate_normal(params_mean, params_cov, size=n)

    def delta_xip_error(self, n=100):

        sample_params = self.sample_param(n)

        delta_xip_list = []

        for i in range(len(sample_params)):
            this_param = sample_params[i]

            this_delta_xip = self.generate_delta_xip(param=this_param, mean_param=False)

            delta_xip_list.append(this_delta_xip)

        return np.cov(np.array(delta_xip_list).T)
