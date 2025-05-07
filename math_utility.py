""" various utility files """
import numpy as np
from cmath import rect, phase
from math import radians, degrees
from scipy.stats import pearsonr
from scipy.stats import chi2
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


# source: https://rosettacode.org/wiki/Averages/Mean_angle#Python
def mean_angle(deg):
    return degrees(phase(sum(rect(1, radians(d)) for d in deg)/len(deg)))


def circ_corrcc(alpha, x):
    """Correlation coefficient between one circular and one linear random
    variable.

    Args:
        alpha: vector
            Sample of angles in radians

        x: vector
            Sample of linear random variable

    Returns:
        rho: float
            Correlation coefficient

        pval: float
            p-value

    Code taken from the Circular Statistics Toolbox for Matlab
    By Philipp Berens, 2009
    Python adaptation by Etienne Combrisson
    """
    if len(alpha) is not len(x):
        raise ValueError('The length of alpha and x must be the same')
    n = len(alpha)

    # Compute correlation coefficent for sin and cos independently
    rxs = pearsonr(x, np.sin(alpha))[0]
    rxc = pearsonr(x, np.cos(alpha))[0]
    rcs = pearsonr(np.sin(alpha), np.cos(alpha))[0]

    # Compute angular-linear correlation (equ. 27.47)
    rho = np.sqrt((rxc ** 2 + rxs ** 2 - 2 * rxc * rxs * rcs) / (1 - rcs ** 2));

    # Compute pvalue
    pval = 1 - chi2.cdf(n * rho ** 2, 2);

    return rho, pval


def circ_r(alpha, w=None, d=0, axis=0):
    """Computes mean resultant vector length for circular data.

    Args:
        alpha: array
            Sample of angles in radians

    Kargs:
        w: array, optional, [def: None]
            Number of incidences in case of binned angle data

        d: radians, optional, [def: 0]
            Spacing of bin centers for binned data, if supplied
            correction factor is used to correct for bias in
            estimation of r

        axis: int, optional, [def: 0]
            Compute along this dimension

    Return:
        r: mean resultant length

    Code taken from the Circular Statistics Toolbox for Matlab
    By Philipp Berens, 2009
    Python adaptation by Etienne Combrisson
    """
    #     alpha = np.array(alpha)
    #     if alpha.ndim == 1:
    #         alpha = np.matrix(alpha)
    #         if alpha.shape[0] is not 1:
    #             alpha = alpha

    if w is None:
        w = np.ones(alpha.shape)
    elif (alpha.size is not w.size):
        raise ValueError("Input dimensions do not match")

    # Compute weighted sum of cos and sin of angles:
    r = np.multiply(w, np.exp(1j * alpha)).sum(axis=axis)

    # Obtain length:
    r = np.abs(r) / w.sum(axis=axis)

    # For data with known spacing, apply correction factor to
    # correct for bias in the estimation of r
    if d is not 0:
        c = d / 2 / np.sin(d / 2)
        r = c * r

    return np.array(r)


def circ_rtest(alpha, w=None, d=0):
    """Computes Rayleigh test for non-uniformity of circular data.
    H0: the population is uniformly distributed around the circle
    HA: the populatoin is not distributed uniformly around the circle
    Assumption: the distribution has maximally one mode and the data is
    sampled from a von Mises distribution!

    Args:
        alpha: array
            Sample of angles in radians

    Kargs:
        w: array, optional, [def: None]
            Number of incidences in case of binned angle data

        d: radians, optional, [def: 0]
            Spacing of bin centers for binned data, if supplied
            correction factor is used to correct for bias in
            estimation of r

    Code taken from the Circular Statistics Toolbox for Matlab
    By Philipp Berens, 2009
    Python adaptation by Etienne Combrisson
    """
    alpha = np.array(alpha)
    if alpha.ndim == 1:
        alpha = np.matrix(alpha)
    if alpha.shape[1] > alpha.shape[0]:
        alpha = alpha.T

    if w is None:
        r = circ_r(alpha)
        n = len(alpha)
    else:
        if len(alpha) is not len(w):
            raise ValueError("Input dimensions do not match")
        r = circ_r(alpha, w, d)
        n = w.sum()

    # Compute Rayleigh's
    R = n * r

    # Compute Rayleigh's
    z = (R ** 2) / n

    # Compute p value using approxation in Zar, p. 617
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))

    return np.squeeze(pval), np.squeeze(z)


def smooth_signal(x, window_len=10, window='flat'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.
    The code taken from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                'flat' window will produce a moving average smoothing.

    output:
        the smoothed signal
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]

    if window == 'flat':  # Moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y[(int(window_len / 2) - 1):-int(window_len / 2)]


def WhittakerSmooth(x, w, lambda_, differences=1):
    '''
    Penalized least squares algorithm for background fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,
                 the smoother the resulting background
        differences: integer indicating the order of the difference of penalties

    output
        the fitted background vector
    '''
    X = np.matrix(x)
    m = X.size
    i = np.arange(0, m)
    E = eye(m, format='csc')
    D = E[1:] - E[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * D.T * D))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)


def airPLS(x, lambda_=100, porder=1, itermax=15):
    """
    Adaptive iteratively reweighted penalized least squares for baseline fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,
                 the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting

    output
        the fitted background vector
    """

    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
            if (i == itermax): print('WARNING max iteration reached!')
            break
        w[d >= 0] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    return z
