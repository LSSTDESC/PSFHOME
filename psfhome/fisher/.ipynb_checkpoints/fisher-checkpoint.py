import pandas as pd
import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import fsum
from functools import lru_cache
from numpy.linalg import inv
import sys, scipy, emcee
import numdifftools as nd
from scipy.stats import uniform, norm
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from copy import deepcopy
from numbers import Number
from itertools import chain
import seaborn as sns
from sklearn.neighbors import KernelDensity
import pickle


def FoM(matrix):
    return np.sqrt(np.linalg.det(matrix))


def plot_contours(matrix, sigmas, fid):
    prefactor = {1: 1.52, 2: 2.48}
    prefactor = prefactor[sigmas]
    matrix = np.linalg.inv(matrix)
    s00, s01, s11 = matrix[0][0], matrix[0][1], matrix[1][1]
    a = np.sqrt(0.5 * (s00 + s11) + np.sqrt(s01**2 + 0.25 * (s00 - s11) ** 2))
    b = np.sqrt(0.5 * (s00 + s11) - np.sqrt(s01**2 + 0.25 * (s00 - s11) ** 2))
    b *= prefactor
    a *= prefactor
    theta = np.arctan(2 * s01 / (s00 - s11)) / 2
    eig = np.linalg.eig(matrix)
    maxeig = eig[1][np.argmax(eig[0])]
    theta = np.arctan2(maxeig[1], maxeig[0])
    el = matplotlib.patches.Ellipse(
        fid, 2 * a, 2 * b, angle=-np.degrees(theta), alpha=0.3
    )
    return el, ((fid[0] - 2 * a, fid[0] + 2 * a), (fid[1] - 2 * a, fid[1] + 2 * a))


def marginalize(fisher_matrix, i, j):
    return np.linalg.inv(np.linalg.inv(fisher_matrix)[np.ix_([i, j], [i, j])])


def get_cov_matrix(l, data_order, cl_vals, orderings, fsky):
    prefac = 1 / (2 * l + 1) / fsky
    out_cov = np.zeros((len(data_order), len(data_order)))
    for z in range(out_cov.shape[0]):
        for y in range(out_cov.shape[1]):
            i, j = (data_order[z][_] for _ in [0, 1])
            k, l = (data_order[y][_] for _ in [0, 1])

            try:
                cl_ik = cl_vals[orderings.index([i, k])]
            except ValueError:
                # this happens if only autocorrelation is queried
                cl_ik = 0

            try:
                cl_jl = cl_vals[orderings.index([j, l])]
            except ValueError:
                cl_jl = 0

            try:
                cl_il = cl_vals[orderings.index([i, l])]
            except ValueError:
                cl_il = 0

            try:
                cl_jk = cl_vals[orderings.index([j, k])]
            except ValueError:
                cl_jk = 0

            out_cov[z, y] = prefac * (cl_ik * cl_jl + cl_il * cl_jk)
    return out_cov


def multi_bin_cov(fsky, Clbins, Cl_ordering, num_dens):
    # first column --> l values

    combination_cl = Clbins[:, 0]
    orderings = []
    for i in range(1, Clbins.shape[1]):
        # Cl_orderings has no l entry
        index_comb = Cl_ordering[i - 1, :].tolist()
        if index_comb[0] != index_comb[1]:
            # add the entry plus the
            # reversed entry --> would NOT be valid for cross correlations since different
            # bins!!!
            combination_cl = np.column_stack((combination_cl, Clbins[:, i]))
            combination_cl = np.column_stack((combination_cl, Clbins[:, i]))
            orderings.append(index_comb)
            orderings.append([index_comb[1], index_comb[0]])
        else:
            # autocorrelations
            combination_cl = np.column_stack((combination_cl, Clbins[:, i]))
            orderings.append(index_comb)

    # remove the first column from combination_cl --> makes it easier since now
    # it corresponds to orderings vector

    combination_cl = combination_cl[:, 1:]

    assert len(orderings) == combination_cl.shape[1]

    # add the shotnoise to each of the cl combinations:
    # TODO: make this an option for clustering vs lensing
    shotnoise = []
    for el in orderings:
        shotnoise.append(
            1.0 / num_dens[int(el[0] - 1)]
        )  # because ordering starts with 1
    shotnoise = np.array(shotnoise)
    assert len(shotnoise) == combination_cl.shape[1]

    for i in range(combination_cl.shape[1]):
        # only the autocorrelation is affected by shot noise
        if orderings[i][0] == orderings[i][1]:
            combination_cl[:, i] += shotnoise[i]

    # now calculate the covariance matrice for each of the Cl_orderings
    out_mat = []
    for i in range(Clbins.shape[0]):
        curr_l = Clbins[i, 0]
        matrix_out = get_cov_matrix(
            curr_l, Cl_ordering, combination_cl[i, :], orderings, fsky
        )
        out_mat.append(matrix_out)
    return out_mat


def one_bin_cov(fsky, Clbins, num_dens):
    added_shotnoise = (Clbins[:, 1] + 1.0 / num_dens) ** 2
    prefactor = 2.0 / ((2.0 * Clbins[:, 0] + 1.0) * fsky)
    covariance = prefactor * added_shotnoise
    cov_matrix = np.diag(covariance)
    return cov_matrix


class PhotoZ_core(object):
    """Defines p(zp | z) of the core distribution (i.e. no outliers)"""

    def __init__(self, zp_support, zbias, sigma_z):
        self.zp_support = zp_support
        self.zbias = zbias
        self.sigma_z = sigma_z

    def get_pdf_given(self, z: Number):
        rv = norm()
        scale = self.sigma_z * (1 + z)
        loc = z - self.zbias
        return rv.pdf((np.array(self.zp_support) - loc) / scale) / scale


class Core(object):
    """Defines the core and outlier population"""

    def __init__(self, zp_support, zbias, sigma_z):
        self.zp_support = zp_support
        self.core = PhotoZ_core(zp_support, zbias, sigma_z)

    def get_pdf_given(self, z: Number):
        core_pdf = self.core.get_pdf_given(z)
        return core_pdf / np.trapz(core_pdf, self.zp_support)


class SmailZ(object):
    """Define the True photometric redshift distribution
    that follows a Smail Type distribution
    """

    def __init__(self, z_support, pdf_values_evaluated_at_zsupport):
        self.z_support = z_support
        self.pdf = pdf_values_evaluated_at_zsupport / np.trapz(
            pdf_values_evaluated_at_zsupport, z_support
        )

    def get_pdf(self):
        return self.pdf

    def get_pdf_convoled(self, filter_list):
        output_tomo_list = np.array([el * self.pdf for el in filter_list]).T
        output_tomo_list = np.column_stack((self.z_support, output_tomo_list))
        return output_tomo_list


class PhotozModel(object):
    """Convolve the joint distribution p(z_s, z_p) with
    a set of filter functions (e.g. gaussian or tophat)

    The class function get_pdf produces an array of tomographic
    bins

    """

    def __init__(self, pdf_z, pdf_zphot_given_z, filters):
        self.pdf_zphot_given_z = pdf_zphot_given_z
        self.pdf_z = pdf_z
        self.filter_list = filters

    def get_pdf(self):
        return np.array([self.get_pdf_tomo(el) for el in self.filter_list])

    def get_pdf_tomo(self, filter):
        z_support = self.pdf_z.z_support
        zp_support = self.pdf_zphot_given_z.zp_support
        z_pdf = self.pdf_z.get_pdf()
        pdf_joint = np.zeros((len(zp_support), len(z_support)))
        for i in range(len(z_support)):
            pdf_joint[:, i] = (
                self.pdf_zphot_given_z.get_pdf_given(z_support[i])
                * z_pdf[i]
                * filter[i]
            )

        pdf_zp = np.zeros((len(zp_support),))
        for i in range(len(zp_support)):
            pdf_zp[i] = np.trapz(pdf_joint[i, :], z_support)

        return pdf_zp


def centroid_shift(unbiased, biased):
    cl_unbiased = np.array(unbiased.ccl_cls["C_ell"]).reshape(15, 15).T
    cl_biased = np.array(biased.ccl_cls["C_ell"]).reshape(15, 15).T

    diff_cl = np.column_stack(cl_biased - cl_unbiased)
    bias_vec = []
    for i, param in enumerate(unbiased.param_order):
        bias_vec.append(
            sum(
                diff_cl[idx].dot(
                    np.array(unbiased.invcov_list[idx]).dot(
                        unbiased.derivs_sig[param][idx]
                    )
                )
                for idx in range(len(unbiased.ell))
            )
        )
    bias_vec = np.array(bias_vec)
    para_bias = np.linalg.inv(fisher).dot(bias_vec)
    para_bias = {param_order[i]: para_bias[i] for i in range(7)}
    return para_bias


def FullPlot(params, *args, labels="LongString"):
    colors = sns.color_palette(palette="colorblind", n_colors=len(args))
    es = []
    for i, arg in enumerate(args):
        if not arg.has_run:
            raise ValueError("This class has not been processed yet.")
        matrix = arg.fisher
        e, (xlim, ylim) = plot_contours(
            marginalize(
                matrix,
                arg.param_order.index(params[0]),
                arg.param_order.index(params[1]),
            ),
            sigmas=2,
            fid=(arg.vals[params[0]], arg.vals[params[1]]),
        )
        e.set_facecolor(colors[i])
        e.set_label(labels[i])
        e.set_alpha(1 / (1 + len(args)))
        es.append(e)

    fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
    for i, e in enumerate(es):
        ax.add_artist(e)

    _ = plt.xlim((xlim))
    _ = plt.ylim((ylim))
    # _ = plt.legend(handles=es, loc=(0.7, 0.7))
    _ = plt.xlabel(arg.param_labels[arg.param_order.index(params[0])])
    _ = plt.ylabel(arg.param_labels[arg.param_order.index(params[1])])
    _ = plt.figure()


class Fisher:
    def __init__(
        self,
        cosmo,
        zvariance=[1, 1, 1, 1, 1],
        zbias=[0, 0, 0, 0, 0],
        outliers=[1, 1, 1, 1, 1],
        calcCov=False,
        plot_label=None,
    ):
        self.zvariance = zvariance
        self.zbias = zbias
        self.outliers = outliers
        self.has_run = False
        self.A0 = 5
        self.beta = 0
        self.etal = 0
        self.etah = 0
        self.calcCov = calcCov
        self.plot_label = plot_label
        self.cosmo = cosmo
        self.funcs = {
            "sigma_8": self.getC_ellOfSigma8,
            "omega_b": self.getC_ellOfOmegab,
            "h": self.getC_ellOfh,
            "n_s": self.getC_ellOfn_s,
            "omega_m": self.getC_ellOfOmegam,
            "w_0": self.getC_ellOfw0,
            "w_a": self.getC_ellOfwa,
            "A0": self.getC_ellOfIA_amp,
            "beta": self.getC_ellOfIA_beta,
            "etal": self.getC_ellOfIA_lowz,
            "etah": self.getC_ellOfIA_highz,
            "zbias1": self.getC_ellOfzbias1,
            "zbias2": self.getC_ellOfzbias2,
            "zbias3": self.getC_ellOfzbias3,
            "zbias4": self.getC_ellOfzbias4,
            "zbias5": self.getC_ellOfzbias5,
            "zvariance1": self.getC_ellOfzvariance1,
            "zvariance2": self.getC_ellOfzvariance2,
            "zvariance3": self.getC_ellOfzvariance3,
            "zvariance4": self.getC_ellOfzvariance4,
            "zvariance5": self.getC_ellOfzvariance5,
            "zoutlier1": self.getC_ellOfzoutlier1,
            "zoutlier2": self.getC_ellOfzoutlier2,
            "zoutlier3": self.getC_ellOfzoutlier3,
            "zoutlier4": self.getC_ellOfzoutlier4,
            "zoutlier5": self.getC_ellOfzoutlier5,
        }
        self.vals = {
            "sigma_8": 0.831,
            "omega_b": 0.049,
            "h": 0.6727,
            "n_s": 0.9645,
            "omega_m": 0.3156,
            "w_0": -1,
            "w_a": 0,
            "A0": 5.0,
            "etal": 0,
            "etah": 0,
            "beta": 2,
        }
        for i in range(5):
            self.vals["zbias" + str(i + 1)] = 0
            self.vals["zvariance" + str(i + 1)] = 1
            self.vals["zoutlier" + str(i + 1)] = 1
        self.priors = {
            "sigma_8": 1 / 0.14**2,
            "omega_b": 1 / 0.006**2,
            "h": 1 / 0.063**2,
            "n_s": 1 / 0.08**2,
            "omega_m": 1 / 0.2**2,
            "w_0": 1 / 0.8**2,
            "w_a": 1 / 2**2,
            "A0": 1 / 2**2,
            "beta": 1 / 2**2,
            "etal": 1 / 2**2,
            "etah": 1 / 2**2,
            "zbias1": 1 / 2**2,
            "zbias2": 1 / 2**2,
            "zbias3": 1 / 2**2,
            "zbias4": 1 / 2**2,
            "zbias5": 1 / 2**2,
            "zvariance1": 1 / 2**2,
            "zvariance2": 1 / 2**2,
            "zvariance3": 1 / 2**2,
            "zvariance4": 1 / 2**2,
            "zvariance5": 1 / 2**2,
            "zoutlier1": 1 / 2**2,
            "zoutlier2": 1 / 2**2,
            "zoutlier3": 1 / 2**2,
            "zoutlier4": 1 / 2**2,
            "zoutlier5": 1 / 2**2,
        }
        self.param_order = (
            [
                "omega_m",
                "sigma_8",
                "n_s",
                "w_0",
                "w_a",
                "omega_b",
                "h",
                "A0",
                "beta",
                "etal",
                "etah",
            ]
            + ["zbias" + str(i) for i in range(1, 6)]
            + ["zvariance" + str(i) for i in range(1, 6)]
            + ["zoutlier" + str(i) for i in range(1, 6)]
        )
        self.param_labels = (
            [
                r"$\Omega_m$",
                r"$\sigma_8$",
                r"$n_s$",
                r"$w_0$",
                r"$w_a$",
                r"$\Omega_b$",
                r"$h$",
                r"$A_0$",
                r"$\beta$",
                r"$\eta_l$",
                r"$\eta_h$",
            ]
            + [r"$z_{bias}$" + str(i) for i in range(1, 6)]
            + [r"$\std{z}$" + str(i) for i in range(1, 6)]
            + [r"$out$" + str(i) for i in range(1, 6)]
        )

    def __repr__(self):
        return f"Run status: {self.has_run} \
                with cosmology: {self.cosmo} \
                with Photo-z error model:  \
                - bias:  {self.zbias} \
                - variance {[v*0.05 for v in self.zvariance]} \
                - outliers: {[o*0.03 for o in self.outliers]}"

    def _makeLensPZ(self):
        raise NotImplemented("Lens distributions not implemented yet.")

    def _NormalizePZ(self, qs, dNdz_dict_source, m=1):
        for q, k in zip(qs, dNdz_dict_source.keys()):
            dNdz_dict_source[k] = dNdz_dict_source[k] * sum(qs) / q
            f = CubicSpline(self.zmid, dNdz_dict_source[k])
            d = quad(f, 0, 4)[0]
            for k in dNdz_dict_source.keys():
                dNdz_dict_source[k] /= d * m
            return dNdz_dict_source

    def _makeSourcePZ(
        self, file="fisher_data/nzdist.txt", implement_outliers=True, y1=False
    ):
        print("Making Source Photo-z")
        if y1:
            file = "fisher_data/nzdist_y1.txt"
        df = pd.read_csv(file, sep=" ")
        self.zmid = list(df["zmid"])
        self.dneff = df["dneff"]
        self.z = [0] + self.zmid
        n = len(self.zmid)
        datapts = [
            list(np.ones(int(self.dneff[i] / min(self.dneff))) * self.zmid[i])
            for i in range(n)
        ]
        datapts = list(chain.from_iterable(datapts))  # flatten
        bins = datapts[0 :: int(len(datapts) / 5)]
        self.bins = bins
        bin_centers = [
            0.5 * fsum([bins[i] + bins[i + 1]]) for i in range(len(bins[:-1]))
        ]
        self.bin_centers = bin_centers
        self.pdf_z = SmailZ(self.zmid, np.array(self.dneff))
        dNdz_dict_source = {}
        qs = []
        for index, (x, x2) in enumerate(zip(bins[:-1], bins[1:])):
            bias = self.zbias[index]
            variance = self.zvariance[index]
            core = Core(self.zmid, zbias=0.1 * bias, sigma_z=variance * 0.05)
            tomofilter = uniform.pdf(self.zmid, loc=x, scale=x2 - x)
            photoz_model = PhotozModel(self.pdf_z, core, [tomofilter])
            dNdz_dict_source[bin_centers[index]] = photoz_model.get_pdf()[0]
            f = CubicSpline(self.zmid, dNdz_dict_source[bin_centers[index]])
            qs.append(quad(f, 0, 4)[0])
        norm = 1

        if implement_outliers:
            # inbin = [0.009780926739896716, 0.0022554556214481247, 0.020947481173020983, 0.04831089597165704, 0.07238322930340861]
            inbin = [0.03] * 5
            inbin = [inbin[i] * self.outliers[i] for i in range(len(inbin))]
            norm = 1 - sum(inbin)

        dNdz_dict_source = self._NormalizePZ(qs, dNdz_dict_source, 5 / norm)
        self.dNdz_dict_source = dNdz_dict_source

        if implement_outliers:
            self.scores = []
            self.KDEs = pickle.load(open("fisher_data/KDEs.p", "rb"))
            for i, b in enumerate(list(sorted(self.dNdz_dict_source.keys()))):
                kde = self.KDEs[i]
                self.scores.append(
                    np.exp(kde.score_samples(np.array(self.zmid).reshape(-1, 1)))
                )
                self.dNdz_dict_source[b] += self.scores[i] * inbin[i]

    def getElls(self, file="fisher_data/ell-values.txt"):
        print("Getting Ells")
        ell = pd.read_csv(file, names=["ell"])
        self.ell = list(ell.to_dict()["ell"].values())[:15]

    def A_h(self, z, etah):
        if z > 2:
            return (1 + z) ** etah
        return 1

    def A_l(self, z, etal):
        return (1 + z) ** etal

    def makeCells(self, cosmo=None):
        print("making C_ells")
        if not cosmo:
            cosmo = self.cosmo
        ccl_cls = pd.DataFrame()
        zbin = 0
        j = 0
        ia0 = (
            self.A0
            * np.array([self.A_l(zi, self.etal) for zi in self.zmid])
            * np.array([self.A_h(zi, self.etah) for zi in self.zmid])
        )
        lst = list(self.dNdz_dict_source.keys())
        for i, key in enumerate(lst):
            ia = (
                self.getAi(
                    self.beta,
                    cosmo,
                    dndz=(tuple(self.zmid), tuple(self.dNdz_dict_source[key])),
                )
                * ia0
            )
            lens1 = ccl.WeakLensingTracer(
                cosmo,
                dndz=(self.zmid, self.dNdz_dict_source[key]),
                ia_bias=(self.zmid, ia),
            )
            for keyj in lst[i:]:
                ia = (
                    self.getAi(
                        self.beta,
                        cosmo,
                        dndz=(tuple(self.zmid), tuple(self.dNdz_dict_source[keyj])),
                    )
                    * ia0
                )
                lens2 = ccl.WeakLensingTracer(
                    cosmo,
                    dndz=(self.zmid, self.dNdz_dict_source[keyj]),
                    ia_bias=(self.zmid, ia),
                )
                cls = ccl.angular_cl(cosmo, lens1, lens2, self.ell)
                newdf = pd.DataFrame(
                    {
                        "zbin": [int(k) for k in j * np.ones(len(cls))],
                        "ell": self.ell,
                        "C_ell": cls,
                    }
                )
                ccl_cls = pd.concat((ccl_cls, newdf))
                j += 1

        self.ccl_cls = ccl_cls.reset_index()

        C_ells = []
        for i in set(self.ccl_cls["zbin"]):
            C_ells.append(list(self.ccl_cls[self.ccl_cls["zbin"] == i]["C_ell"]))
        return C_ells

    def _makeShearShearCells(self):
        pass

    def _makePosPosCells(self):
        pass

    def buildCovMatrix(self, y1=False):
        print("Getting covariance matrix")
        if self.calcCov:
            numdens = [2 * 5.95e7 / 0.28**2] * 5
            fsky = 0.4
            C_ells = []
            for i in set(self.ccl_cls["zbin"]):
                C_ells.append(list(self.ccl_cls[self.ccl_cls["zbin"] == i]["C_ell"]))
            self.C_ells = C_ells
            l = []
            for j in range(5):
                l.extend([[j, j + i] for i in range(5 - j)])
            ordering = np.array(l)
            cl_bins = np.vstack((self.ell, self.C_ells)).T
            self.cov_matrix_list = multi_bin_cov(fsky, cl_bins, ordering, numdens)
            self.invcov_list = []
            for cov_l in self.cov_matrix_list:
                self.invcov_list.append(np.linalg.inv(cov_l))
        else:
            if y1 == True:
                invcov_SRD = pd.read_csv(
                    "fisher_data/Y1_shear_shear_inv.txt",
                    names=["a", "b"],
                    delimiter=" ",
                )
            else:
                invcov_SRD = pd.read_csv(
                    "fisher_data/Y10_shear_shear_inv.txt",
                    names=["a", "b"],
                    delimiter=" ",
                )

            matrix = np.array(invcov_SRD["b"]).reshape(300, 300)
            total_ells = 20
            self.invcov_list = []
            for l in range(total_ells):
                m = []
                for k in range(15):
                    m.append([matrix[k * 20 + l][i * 20 + l] for i in range(15)])
                self.invcov_list.append(m)

    def getDerivs(self, cosmo=True):
        print("Getting derivatives")
        if cosmo:
            # end = 7
            end = 11

        else:
            end = len(self.param_order)
        dNdz_dict_source = self.dNdz_dict_source
        self.derivs_sig = {}
        for var in self.param_order[:end]:
            print(var)
            zbin = 0
            j = 0
            derivs = []
            lst = list(self.dNdz_dict_source.keys())
            for i, key in enumerate(lst):
                for keyj in lst[i:]:
                    self.key = key
                    self.keyj = keyj
                    f = nd.Derivative(self.funcs[var], full_output=True, step=0.01)
                    val, info = f(self.vals[var])
                    # print(val)
                    derivs.append(val)
            self.derivs_sig[var] = np.array(derivs).T

    def getFisher(self, cosmo=True):
        print("Building fisher matrix")
        if cosmo:
            # end = 7
            end = 11

        else:
            end = len(self.param_order)
        self.fisher = np.zeros((end, end))
        for i, var1 in enumerate(self.param_order[:end]):
            for j, var2 in enumerate(self.param_order[:end]):
                res = [
                    self.derivs_sig[var1][l].T
                    @ self.invcov_list[l]
                    @ self.derivs_sig[var2][l]
                    for l in range(len(self.derivs_sig[var1]))
                ]
                self.fisher[i][j] = sum(res)  # *400
        for i in range(end):
            self.fisher[i][i] += self.priors[self.param_order[i]]

    def process(self, cosmo=False, y1=False):
        self._makeSourcePZ(y1=y1)
        self.getElls()
        _ = self.makeCells()
        self.buildCovMatrix(y1=y1)
        self.getDerivs(cosmo)
        self.getFisher(cosmo)
        self.has_run = True
        print("Done")

    def emcee(self, mcparams):
        data = self.C_ells

        def lnprob(theta):
            cosmo = ccl.Cosmology(
                Omega_c=theta[1],
                Omega_b=0.049,
                h=0.9727,
                sigma8=theta[0],
                n_s=0.9645,
                transfer_function="eisenstein_hu",
            )
            theory = self.makeCells(cosmo)
            diff = np.array(data) - np.array(theory)
            # return -sum(np.dot(diff[l], np.dot(self.invcov_list[l], diff[l]))/2
            #           for l in range(len(self.invcov_list)))
            # return -np.dot(diff[0], np.dot(self.invcov_list[0], diff[0]))/2
            return -sum(diff**2)

        ndim = 1
        nwalkers = 8
        nsteps = 600

        # p00 = [np.array([0.2666, 0.831]) * np.ones(2) \
        #    + 0.05 * numpy.random.uniform(-1, 2, 2)
        #       for i in range(nwalkers)]
        p00 = [np.array([0.831]) + 0.05 * numpy.random.uniform(-1, 2) for i in range(8)]

        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
        self.pos, self.prob, self.state = self.sampler.run_mcmc(p00, 50)
        self.sampler.reset()
        self.sampler.run_mcmc(self.pos, nsteps, rstate0=self.state)

    # @lru_cache
    def getAi(
        self,
        beta,
        cosmo,
        dndz,
        Mr_s=-20.70,
        Q=1.23,
        alpha_lum=-1.23,
        phi_0=0.0094,
        P=-0.3,
        mlim=25.3,
        Lp=1.0,
    ):
        """Get the amplitude of the 2-halo part of w_{l+}
        A0 is the amplitude, beta is the power law exponent (see Krause et al. 2016)
        cosmo is a CCL cosmology object
        Lp is a pivot luminosity (default = 1)
        dndz is (z, dNdz) - vector of z and dNdz for the galaxies
        (does not need to be normalised)
        """
        z_input, dNdz = dndz
        z_input = list(z_input)
        dNdz = list(dNdz)
        (z_k, kcorr, x, x, x) = np.loadtxt("fisher_data/kcorr.dat", unpack=True)
        (z_e, ecorr, x, x, x) = np.loadtxt("fisher_data/ecorr.dat", unpack=True)
        zmaxke = min(max(z_k), max(z_e))
        zminke = max(min(z_k), min(z_e))
        if zminke > min(z_input) and zmaxke < max(z_input):
            z = np.linspace(zminke, zmaxke, 1000)
            interp_dNdz = scipy.interpolate.interp1d(z_input, dNdz)
            dNdz = interp_dNdz(z)
        elif zmaxke < max(z_input):
            z = np.linspace(min(z_input), zmaxke, 1000)
            interp_dNdz = scipy.interpolate.interp1d(z_input, dNdz)
            dNdz = interp_dNdz(z)
        elif zminke > min(z_input):
            z = np.linspace(zminke, max(z_input), 1000)
            interp_dNdz = scipy.interpolate.interp1d(z_input, dNdz)
            dNdz = interp_dNdz(z)
        else:
            z = z_input

        # Get the luminosity function
        (L, phi_normed) = self.get_phi(
            z, self.cosmo, Mr_s, Q, alpha_lum, phi_0, P, mlim
        )
        # Pivot luminosity:
        Lp = 1.0

        # Get Ai as a function of lens redshift.
        Ai_ofzl = np.zeros(len(z))
        for zi in range(len(z)):
            Ai_ofzl[zi] = scipy.integrate.simps(
                np.asarray(phi_normed[zi]) * (np.asarray(L[zi]) / Lp) ** (beta),
                np.asarray(L[zi]),
            )

        # Integrate over dNdz
        Ai = scipy.integrate.simps(Ai_ofzl * dNdz, z) / scipy.integrate.simps(dNdz, z)

        return Ai

    def get_phi(self, z, cosmo, Mr_s, Q, alpha_lum, phi_0, P, mlim, Mp=-22.0):

        """This function outputs the Schechter luminosity function with parameters fit in Loveday 2012, following the same procedure as Krause et al. 2015, as a function of z and L
        The output is L[z][l], list of vectors of luminosity values in z, different at each z due to the different lower luminosity limit, and phi[z][l], a list of luminosity functions at these luminosity vectors, at each z
        cosmo is a CCL cosmology object
        mlim is the magnitude limit of the survey
        Mp is the pivot absolute magnitude.
        other parameteres are the parameters of the luminosity function that are different for different samples, e.g. red vs all. lumparams = [Mr_s, Q, alpha_lum, phi_0, P]
        Note that the luminosity function is output normalized (appropriate for getting Ai)."""

        # Get the amplitude of the Schechter luminosity function as a function of redshift.
        phi_s = phi_0 * 10.0 ** (0.4 * P * z)

        # Get M_* (magnitude), then convert to L_*
        Ms = Mr_s - Q * (z - 0.1)
        Ls = 10 ** (-0.4 * (Ms - Mp))

        # Import the kcorr and ecorr correction from Poggianti (assumes elliptical galaxies)
        # No data for sources beyon z = 3, so we keep the same value at higher z as z=3
        (z_k, kcorr, x, x, x) = np.loadtxt("fisher_data/kcorr.dat", unpack=True)
        (z_e, ecorr, x, x, x) = np.loadtxt("fisher_data/ecorr.dat", unpack=True)
        kcorr_interp = scipy.interpolate.interp1d(z_k, kcorr)
        ecorr_interp = scipy.interpolate.interp1d(z_e, ecorr)
        kcorr = kcorr_interp(z)
        ecorr = ecorr_interp(z)

        # Get the absolute magnitude and luminosity corresponding to limiting apparent magntiude (as a function of z)
        dl = ccl.luminosity_distance(cosmo, 1.0 / (1.0 + z))
        Mlim = mlim - (5.0 * np.log10(dl) + 25.0 + kcorr + ecorr)
        Llim = 10.0 ** (-0.4 * (Mlim - Mp))

        L = [0] * len(z)
        for zi in range(0, len(z)):
            L[zi] = scipy.logspace(np.log10(Llim[zi]), 2.0, 1000)

        # Now get phi(L,z), where this exists for each z because the lenghts of the L vectors are different.
        phi_func = [0] * len(z)
        for zi in range(0, len(z)):
            phi_func[zi] = np.zeros(len(L[zi]))
            for li in range(0, len(L[zi])):
                phi_func[zi][li] = (
                    phi_s[zi]
                    * (L[zi][li] / Ls[zi]) ** (alpha_lum)
                    * np.exp(-L[zi][li] / Ls[zi])
                )

        norm = np.zeros(len(z))
        phi_func_normed = [0] * len(z)
        for zi in range(len(z)):
            norm[zi] = scipy.integrate.simps(phi_func[zi], L[zi])
            phi_func_normed[zi] = phi_func[zi] / norm[zi]

        return (L, phi_func_normed)

    def _outlier_helper(self, idx, zoutlier):
        dNdz_dict_source = {}
        qs = []
        for index, (x, x2) in enumerate(zip(bins[:-1], bins[1:])):
            core = Core(
                self.zmid,
                zbias=0.1 * self.zbias[index],
                sigma_z=self.zvariance[index] * 0.05,
            )
            tomofilter = uniform.pdf(self.zmid, loc=x, scale=x2 - x)
            photoz_model = PhotozModel(self.pdf_z, core, [tomofilter])
            dNdz_dict_source[bin_centers[index]] = photoz_model.get_pdf()[0]
            f = CubicSpline(self.zmid, dNdz_dict_source[bin_centers[index]])
            qs.append(quad(f, 0, 4)[0])

        inbin = [0.03] * 5
        for index in range(len(self.bins) - 1):
            if index == idx:
                inbin[index] = inbin[index] * self.outliers[index] * zoutlier
            else:
                inbin[index] = inbin[index] * self.outliers[index]
        norm = 1 - sum(inbin)

        dNdz_dict_source = self._NormalizePZ(qs, dNdz_dict_source, 5 / norm)

        for i, b in enumerate(list(sorted(dNdz_dict_source.keys()))):
            dNdz_dict_source[b] += self.scores[i] * inbin[i]
        return dNdz_dict_source

    def getC_ellOfzoutlier1(self, zoutlier):
        index = 0
        dNdz_dict_source = self._outlier_helper(index, zoutlier)
        return self._helper(self.cosmo, dNdz_dict_source)

    def getC_ellOfzoutlier2(self, zoutlier):
        index = 1
        dNdz_dict_source = self._outlier_helper(index, zoutlier)
        return self._helper(self.cosmo, dNdz_dict_source)

    def getC_ellOfzoutlier3(self, zoutlier):
        index = 2
        dNdz_dict_source = self._outlier_helper(index, zoutlier)
        return self._helper(self.cosmo, dNdz_dict_source)

    def getC_ellOfzoutlier4(self, zoutlier):
        index = 3
        dNdz_dict_source = self._outlier_helper(index, zoutlier)
        return self._helper(self.cosmo, dNdz_dict_source)

    def getC_ellOfzoutlier5(self, zoutlier):
        index = 4
        dNdz_dict_source = self._outlier_helper(index, zoutlier)
        return self._helper(self.cosmo, dNdz_dict_source)

    def _bias_helper(self, idx, zbias):
        dNdz_dict_source = {}
        qs = []
        for index, (x, x2) in enumerate(zip(self.bins[:-1], self.bins[1:])):
            if index == idx:
                core = Core(
                    self.zmid,
                    zbias=self.zbias[index] + zbias,
                    sigma_z=0.05 * self.zvariance[index],
                )
            else:
                core = Core(
                    self.zmid,
                    zbias=self.zbias[index],
                    sigma_z=0.05 * self.zvariance[index],
                )
            tomofilter = uniform.pdf(self.zmid, loc=x, scale=x2 - x)
            photoz_model = PhotozModel(self.pdf_z, core, [tomofilter])
            dNdz_dict_source[self.bin_centers[index]] = photoz_model.get_pdf()[0]
            f = CubicSpline(self.zmid, dNdz_dict_source[self.bin_centers[index]])
            qs.append(quad(f, 0, 4)[0])

        inbin = [0.03] * 5
        for index in range(len(self.bins) - 1):
            inbin[index] = inbin[index] * self.outliers[index]
        norm = 1 - sum(inbin)

        dNdz_dict_source = self._NormalizePZ(qs, dNdz_dict_source, 5 / norm)

        for i, b in enumerate(list(sorted(self.dNdz_dict_source.keys()))):
            dNdz_dict_source[b] += self.scores[i] * inbin[i]
        return dNdz_dict_source

    def getC_ellOfIA_amp(self, A0):
        return self._helper(self.cosmo, self.dNdz_dict_source, A0=A0)

    def getC_ellOfIA_highz(self, etah):
        return self._helper(self.cosmo, self.dNdz_dict_source, etah=etah)

    def getC_ellOfIA_lowz(self, etal):
        return self._helper(self.cosmo, self.dNdz_dict_source, etal=etal)

    def getC_ellOfIA_beta(self, beta):
        return self._helper(self.cosmo, self.dNdz_dict_source, beta=beta)

    def getC_ellOfzbias1(self, zbias):
        index = 0
        dNdz_dict_source = self._bias_helper(index, zbias)
        return self._helper(self.cosmo, dNdz_dict_source)

    def getC_ellOfzbias2(self, zbias):
        index = 1
        dNdz_dict_source = self._bias_helper(index, zbias)
        return self._helper(self.cosmo, dNdz_dict_source)

    def getC_ellOfzbias3(self, zbias):
        index = 2
        dNdz_dict_source = self._bias_helper(index, zbias)
        return self._helper(self.cosmo, dNdz_dict_source)

    def getC_ellOfzbias4(self, zbias):
        index = 3
        dNdz_dict_source = self._bias_helper(index, zbias)
        return self._helper(self.cosmo, dNdz_dict_source)

    def getC_ellOfzbias5(self, zbias):
        index = 4
        dNdz_dict_source = self._bias_helper(index, zbias)
        return self._helper(self.cosmo, dNdz_dict_source)

    def _variance_helper(self, idx, zvar):
        dNdz_dict_source = {}
        qs = []
        for index, (x, x2) in enumerate(zip(self.bins[:-1], self.bins[1:])):
            if index == idx:
                core = Core(
                    self.zmid,
                    zbias=self.zbias[index],
                    sigma_z=0.05 * self.zvariance[index] * zvar,
                )
            else:
                core = Core(
                    self.zmid,
                    zbias=self.zbias[index],
                    sigma_z=0.05 * self.zvariance[index],
                )
            tomofilter = uniform.pdf(self.zmid, loc=x, scale=x2 - x)
            photoz_model = PhotozModel(self.pdf_z, core, [tomofilter])
            dNdz_dict_source[self.bin_centers[index]] = photoz_model.get_pdf()[0]
            f = CubicSpline(self.zmid, dNdz_dict_source[self.bin_centers[index]])
            qs.append(quad(f, 0, 4)[0])

        inbin = [0.03] * 5
        for index in range(len(self.bins) - 1):
            inbin[index] = inbin[index] * self.outliers[index]
        norm = 1 - sum(inbin)

        dNdz_dict_source = self._NormalizePZ(qs, dNdz_dict_source, 5 / norm)

        for i, b in enumerate(list(sorted(dNdz_dict_source.keys()))):
            dNdz_dict_source[b] += self.scores[i] * inbin[i]
        return dNdz_dict_source

    def getC_ellOfzvariance1(self, zvariance):
        index = 0
        dNdz_dict_source = self._variance_helper(index, zvariance)
        return self._helper(self.cosmo, dNdz_dict_source)

    def getC_ellOfzvariance2(self, zvariance):
        index = 1
        dNdz_dict_source = self._variance_helper(index, zvariance)
        return self._helper(self.cosmo, dNdz_dict_source)

    def getC_ellOfzvariance3(self, zvariance):
        index = 2
        dNdz_dict_source = self._variance_helper(index, zvariance)
        return self._helper(self.cosmo, dNdz_dict_source)

    def getC_ellOfzvariance4(self, zvariance):
        index = 3
        dNdz_dict_source = self._variance_helper(index, zvariance)
        return self._helper(self.cosmo, dNdz_dict_source)

    def getC_ellOfzvariance5(self, zvariance):
        index = 4
        dNdz_dict_source = self._variance_helper(index, zvariance)
        return self._helper(self.cosmo, dNdz_dict_source)

    def _helper(
        self, cosmo, dNdz_dict_source, A0=None, beta=None, etal=None, etah=None
    ):
        if not beta:
            beta = self.beta
        if not etal:
            etal = self.etal
        if not etah:
            etah = self.etah
        if not A0:
            A0 = self.A0
        ia0 = (
            A0
            * np.array([self.A_l(zi, etal) for zi in self.zmid])
            * np.array([self.A_h(zi, etah) for zi in self.zmid])
        )
        # print(ia0)
        ia = (
            self.getAi(
                beta, cosmo, dndz=(tuple(self.zmid), tuple(dNdz_dict_source[self.key]))
            )
            * ia0
        )
        # print(ia)
        lens1 = ccl.WeakLensingTracer(
            cosmo, dndz=(self.zmid, dNdz_dict_source[self.key]), ia_bias=(self.zmid, ia)
        )
        ia = (
            self.getAi(
                beta, cosmo, dndz=(tuple(self.zmid), tuple(dNdz_dict_source[self.keyj]))
            )
            * ia0
        )
        lens2 = ccl.WeakLensingTracer(
            cosmo,
            dndz=(self.zmid, dNdz_dict_source[self.keyj]),
            ia_bias=(self.zmid, ia),
        )
        return ccl.angular_cl(cosmo, lens1, lens2, self.ell)

    def getC_ellOfSigma8(self, sigma8):
        cosmo = ccl.Cosmology(
            Omega_c=self.vals["omega_m"] - self.vals["omega_b"],
            Omega_b=self.vals["omega_b"],
            h=self.vals["h"],
            sigma8=sigma8,
            n_s=self.vals["n_s"],
            transfer_function="eisenstein_hu",
            matter_power_spectrum=self.cosmo._config_init_kwargs[
                "matter_power_spectrum"
            ],
        )
        return self._helper(cosmo, dNdz_dict_source=self.dNdz_dict_source)

    def getC_ellOfOmegab(self, Omega_b):
        cosmo = ccl.Cosmology(
            Omega_c=self.vals["omega_m"] - self.vals["omega_b"],
            Omega_b=Omega_b,
            h=self.vals["h"],
            sigma8=self.vals["sigma_8"],
            n_s=self.vals["n_s"],
            transfer_function="eisenstein_hu",
            matter_power_spectrum=self.cosmo._config_init_kwargs[
                "matter_power_spectrum"
            ],
        )
        return self._helper(cosmo, dNdz_dict_source=self.dNdz_dict_source)

    def getC_ellOfh(self, h):
        cosmo = ccl.Cosmology(
            Omega_c=self.vals["omega_m"] - self.vals["omega_b"],
            Omega_b=self.vals["omega_b"],
            h=h,
            sigma8=self.vals["sigma_8"],
            n_s=self.vals["n_s"],
            transfer_function="eisenstein_hu",
            matter_power_spectrum=self.cosmo._config_init_kwargs[
                "matter_power_spectrum"
            ],
        )
        return self._helper(cosmo, dNdz_dict_source=self.dNdz_dict_source)

    def getC_ellOfn_s(self, n_s):
        cosmo = ccl.Cosmology(
            Omega_c=self.vals["omega_m"] - self.vals["omega_b"],
            Omega_b=self.vals["omega_b"],
            h=self.vals["h"],
            sigma8=self.vals["sigma_8"],
            n_s=n_s,
            transfer_function="eisenstein_hu",
            matter_power_spectrum=self.cosmo._config_init_kwargs[
                "matter_power_spectrum"
            ],
        )
        return self._helper(cosmo, dNdz_dict_source=self.dNdz_dict_source)

    def getC_ellOfOmegam(self, Omega_m):
        cosmo = ccl.Cosmology(
            Omega_c=Omega_m - self.vals["omega_b"],
            Omega_b=self.vals["omega_b"],
            h=self.vals["h"],
            sigma8=self.vals["sigma_8"],
            n_s=self.vals["n_s"],
            transfer_function="eisenstein_hu",
            matter_power_spectrum=self.cosmo._config_init_kwargs[
                "matter_power_spectrum"
            ],
        )
        return self._helper(cosmo, dNdz_dict_source=self.dNdz_dict_source)

    def getC_ellOfw0(self, w_0):
        cosmo = ccl.Cosmology(
            Omega_c=self.vals["omega_m"] - self.vals["omega_b"],
            Omega_b=self.vals["omega_b"],
            h=self.vals["h"],
            sigma8=self.vals["sigma_8"],
            n_s=self.vals["n_s"],
            w0=w_0,
            transfer_function="eisenstein_hu",
            matter_power_spectrum=self.cosmo._config_init_kwargs[
                "matter_power_spectrum"
            ],
        )
        return self._helper(cosmo, dNdz_dict_source=self.dNdz_dict_source)

    def getC_ellOfwa(self, w_a):
        cosmo = ccl.Cosmology(
            Omega_c=self.vals["omega_m"] - self.vals["omega_b"],
            Omega_b=self.vals["omega_b"],
            h=self.vals["h"],
            sigma8=self.vals["sigma_8"],
            n_s=self.vals["n_s"],
            wa=w_a,
            transfer_function="eisenstein_hu",
            matter_power_spectrum=self.cosmo._config_init_kwargs[
                "matter_power_spectrum"
            ],
        )
        return self._helper(cosmo, dNdz_dict_source=self.dNdz_dict_source)
