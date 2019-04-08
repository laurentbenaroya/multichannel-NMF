# -*- coding: utf-8 -*-
"""
Initialisation of the Multi-channel NMF, for the mixing matrix in the instantaneous case.
ELB implementation of the conference paper algorithm:
"Identifying Single Source Data for Mixing Matrix Estimation
in Instantaneous Blind Source Separation"
Conference: Artificial Neural Networks - ICANN 2008
Slower than Pau Bofill's implementation, but easier to read!
"""

# Author : E.L. Benaroya - laurent.benaroya@gmail.com
# Date : 04/2019
# License : GNU GPL v3

import numpy as np
import warnings
import scipy.signal


class par:
    # parameters
    def __init__(self):
        # parS, ssSelect
        self.R = 0.001
        self.theta = 0.05 * np.pi / 180
        # parE, ssEstimate
        self.Lambda = 4
        self.P = 120


def cart2pol(x, y):
    """
    cartesian to polar coordinates
    Parameters
    ----------
    x
    y

    Returns
    -------
    rho
    phi
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(theta, rho):
    """

    Parameters
    ----------
    theta
    rho

    Returns
    -------

    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def cgmaxs(vec, nsrc):
    """
    Finds the largest local maxima of a vector
    Parameters
    ----------
    vec : numpy vector (nv, )
        input vector
    nsrc : int > 0
        number of peaks to be found

    Returns
    -------
    ks: numpy vector (nv, )
        vector with True located at largest peaks.
        Plateaux and other extrema are ignored.
    """

    nv = len(vec)
    # get slope sign differences
    vec = np.hstack((vec[-2], vec[-1], vec, vec[0], vec[1]))
    dvec = np.sign(np.diff(vec))
    dvec = np.hstack((dvec[0], dvec))
    ddvec = np.hstack((np.diff(dvec), 0.))
    # single point peaks
    epks = ddvec == -2
    val = np.sort(vec[epks])
    if len(val) < nsrc:
        warnings.warn('Found less sources than specified.')
        print('Found less sources than specified.')
        nsrc = len(val)
        print(nsrc)
    aux = vec >= val[len(val)-nsrc]  # ou bien val[len(val)-nsrc]
    pks = epks & aux
    pks = pks[2:nv+2]
    return pks


def cosine_kernel(Lambda, angle_range, theta):
    """
    KERNEL Cosine kernel function, as described in the paper.
    Parameters
    ----------
    Lambda : scale parameter
    angle_range : range of the angle
    theta : numpy array (a, b)
        input angles

    Returns
    -------
    phi_kernel : numpy array (a, b)
        result of the kernel applied to theta
    """

    theta = Lambda*theta
    theta = np.abs(theta)
    ind_theta = theta < (angle_range / 2)
    phi_kernel = np.zeros_like(theta)
    phi_kernel[ind_theta] = 1/2+1/2*np.cos(np.pi * 2 * theta[ind_theta] / angle_range)
    return phi_kernel


def compute_mask(X, param):
    """
    compute mask corresponding to single source in the TF plane
    Parameters
    ----------
    X : numpy array (F, N, 2)
        input spectrogram
    param : class par
        parameters
    Returns
    -------
    mask : boolean numpy array (F, N)
        TF mask
    """
    F, N, Ii = X.shape
    amplitude, theta = cart2pol(X[:, :, 0], X[:, :, 1])
    # compute the mask corresponding to a single source in the TF plane

    abs_theta = np.abs(theta)
    d_theta = np.diff(abs_theta, axis=1)
    d_theta2 = np.hstack((d_theta, 1e10*np.ones((F, 1))))
    mask = (np.abs(d_theta2) < param.theta) & (abs_theta > param.R*np.max(abs_theta))

    return mask


def compute_hist(X, nsrc, param):
    """

    Parameters
    ----------
    X
    nsrc
    param

    Returns
    -------

    """
    if X.ndim == 2:
        amplitude, theta = cart2pol(X[:, 0], X[:, 1])
    elif X.ndim == 3:
        amplitude, theta = cart2pol(X[:, :, 0], X[:, :, 1])
    # abs_theta = np.abs(theta)

    theta_min = np.min(theta)
    theta_max = np.max(theta)

    Delta = (theta_max-theta_min)/param.P
    grid = np.linspace(theta_min+Delta/2, theta_max-Delta/2, param.P)

    # Return the indices of the bins to which each value in input array belongs.
    ind_bin = np.digitize(theta, grid)

    Hp = np.zeros(grid.shape)  # sum of the amplitudes in each histogram bin
    # Each observation vector is assigned by angle to each corresponding bin,
    # with a weight corresponding to it’s length,
    # thus putting more emphasis on the most significant data.
    for pp in range(param.P):
        ind_p = ind_bin == pp  # indices of bin pp
        Hp[pp] = np.sum(amplitude[ind_p])  # amplitude rho in this bin and sum

    range_theta = theta_max-theta_min

    Fp = np.zeros_like(Hp)  # smoothed histogram
    for pp in range(param.P):
        ind_p = ind_bin == pp
        if len(ind_p) == 0:
            continue
        theta_p = theta[ind_p]
        for th_p in theta_p:
            for qq in range(param.P):
                ind_q = ind_bin == qq
                theta_q = theta[ind_q]
                Fp[pp] += np.sum(cosine_kernel(param.Lambda,
                                               range_theta, th_p - theta_q))*Hp[qq]

    # max_pos = cgmaxs(Fp, nsrc)  # Take the nsrc largest peaks

    # Take the nsrc largest peaks using scipy.signal
    max_val = scipy.signal.find_peaks_cwt(Fp, np.arange(1, 10))
    # get nsrc largest peak indices
    max_pos_tmp = np.argsort(max_val, axis=0)
    max_pos = max_pos_tmp[-nsrc:]
    max_angles = grid[max_pos]
    # print(max_angles)
    unit_amplitude = np.ones(max_angles.shape)

    A = np.zeros((2, nsrc))
    ax, ay = pol2cart(max_angles, unit_amplitude)
    A[0, :] = 1.
    A[1, :] = ay/(ax+np.spacing(1))

    return A


def estmix_inst(X, nsrc):
    """

    Parameters
    ----------
    X
    nsrc

    Returns
    -------

    """
    n_bin, n_fram, n_chan = X.shape

    # # Errors # #
    if n_bin == 2*(n_bin//2):
        raise Exception('The number of frequency bins must be odd.')
    if n_chan != 2:
        raise Exception('The number of channels must be equal to 2.')

    param = par()
    abs_X = np.abs(X)

    mask = compute_mask(abs_X, param)
    # Estimates the mixing matrix
    A = compute_hist(np.abs(abs_X[mask]), nsrc, param)

    return A
