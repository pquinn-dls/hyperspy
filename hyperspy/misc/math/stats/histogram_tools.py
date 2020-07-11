# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import logging
import math
import functools
from skimage.transform.pyramids import pyramid_gaussian
from hyperspy.external.astroML.histtools import (scotts_bin_width,
                                                 knuth_bin_width,
                                                 freedman_bin_width)
from hyperspy.external.astroML.bayesian_blocks import bayesian_blocks
from scipy.ndimage import shift
from scipy import stats
from scipy.optimize import minimize
_logger = logging.getLogger(__name__)


def histogram2d(arr1, arr2, bins='scotts', **kwargs):
    """Enhanced histogram2d.

    This is a 2d histogram function that enables the use of more sophisticated
    algorithms for determining bins.  Aside from the `bins` argument allowing
    a string specified how bins are computed, the parameters are the same
    as numpy.histogram().

    Parameters
    ----------
    arr1 : array_like
        array of data to be histogrammed
    arr2 : array_like
        array of data to be histogrammed
    bins : int or list or str (optional)
        If bins is a string, then it must be one of:
        'blocks' : use bayesian blocks for dynamic bin widths
        'knuth' : use Knuth's rule to determine bins
        'scotts' : use Scott's rule to determine bins
        'freedman' : use the Freedman-diaconis rule to determine bins

    other keyword arguments are described in numpy.hist().

    Returns
    -------
    hist : array
        The values of the histogram. See `normed` and `weights` for a
        description of the possible semantics.
    bin_edges : array of dtype float
        Return the bin edges ``(length(hist)+1)``.

    See Also
    --------
    numpy.histogram2d
    """

    if bins == 'auto':
        if len(arr1) < 200:
            bins = 'sturges'
        else:
            bins = 'scotts'
    if bins == 'sturges':
        dh, bins = sturges_bin_width(arr1, True)
        Nbins = len(bins)
    elif bins == 'scotts':
        dh, bins = scotts_bin_width(arr1, True)
        Nbins = len(bins)
    elif bins == 'freedman':
        Nbins = freedman_diaconis(arr1, returnas="bins")
    elif isinstance(bins, str):
        raise ValueError("unrecognized bin code: '%s'" % bins)

    return np.histogram2d(arr1, arr2, bins=Nbins)

def freedman_diaconis(data, returnas="bins"):
    """
    Use Freedman Diaconis rule to compute optimal histogram bin width. 
    ``returnas`` can be one of "width" or "bins", indicating whether
    the bin width or number of bins should be returned respectively. 


    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.

    returnas: {"width", "bins"}
        If "width", return the estimated width for each histogram bin. 
        If "bins", return the number of bins suggested by rule.
    """
    data = np.asarray(data, dtype=np.float_)
    IQR  = stats.iqr(data, rng=(25, 75), scale="raw", nan_policy="omit")
    N    = data.size
    bw   = (2 * IQR) / np.power(N, 1/3)

    if returnas=="width":
        result = bw
    else:
        datmin, datmax = np.nanmin(data), np.nanmax(data)
        datrng = datmax - datmin
        if bw==0.0:
            result = 2
        else:
            result = int((datrng / bw) + 1)
    return result

    
def sturges_bin_width(data, return_bins=False):
    
    """Return the optimal histogram bin width using sturges's rule
    
    1 + log2(N) where N is the number of samples. 
    
    This is generally considered good for low N (e.g. N<200)

    Parameters
    ----------
    data : array-like, ndim=1
        observed (one-dimensional) data
    Returns
    -------
    width : float
        optimal bin width using Sturges rule

    """
    data = np.asarray(data)
    bad_indices = np.isnan(data)
    good_indices = ~bad_indices
    data = data[good_indices]

    if data.ndim != 1:
        raise ValueError("data should be one-dimensional")

    n = data.size
    if n < 4:
        raise ValueError("data should have more than three entries")

    Nbins = np.ceil(1. + np.log(n))
    dx = (np.nanmax(data) - np.nanmin(data)) / Nbins
    if return_bins:
        bins = np.nanmin(data) + dx * np.arange(Nbins + 1)
        return dx,bins
    return dx


def scotts_bin_width(data, return_bins=False):
    r"""Return the optimal histogram bin width using Scott's rule:

    Parameters
    ----------
    data : array-like, ndim=1
        observed (one-dimensional) data
    return_bins : bool (optional)
        if True, then return the bin edges

    Returns
    -------
    width : float
        optimal bin width using Scott's rule
    bins : ndarray
        bin edges: returned if `return_bins` is True

    Notes
    -----
    The optimal bin width is

    .. math::
        \Delta_b = \frac{3.5\sigma}{n^{1/3}}

    where :math:`\sigma` is the standard deviation of the data, and
    :math:`n` is the number of data points.

    See Also
    --------
    knuth_bin_width
    freedman_bin_width
    astroML.plotting.hist
    """

    data = np.asarray(data)
    bad_indices = np.isnan(data)
    good_indices = ~bad_indices
    data = data[good_indices]
    if data.ndim != 1:
        raise ValueError("data should be one-dimensional")

    n = data.size
    sigma = np.std(data)

    dx = 3.5 * sigma * 1. / (n ** (1. / 3))

    if return_bins:
        Nbins = np.ceil((np.nanmax(data) - np.nanmin(data)) * 1. / dx)
        Nbins = max(1, Nbins)
        bins = np.nanmin(data) + dx * np.arange(Nbins + 1)
        return dx, bins
    else:
        return dx


def freedman_bin_width(data, return_bins=False):
    r"""Return the optimal histogram bin width using the Freedman-Diaconis rule

    Parameters
    ----------
    data : array-like, ndim=1
        observed (one-dimensional) data
    return_bins : bool (optional)
        if True, then return the bin edges

    Returns
    -------
    width : float
        optimal bin width using Scott's rule
    bins : ndarray
        bin edges: returned if `return_bins` is True

    Notes
    -----
    The optimal bin width is

    .. math::
        \Delta_b = \frac{2(q_{75} - q_{25})}{n^{1/3}}

    where :math:`q_{N}` is the :math:`N` percent quartile of the data, and
    :math:`n` is the number of data points.

    See Also
    --------
    knuth_bin_width
    scotts_bin_width
    astroML.plotting.hist
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("data should be one-dimensional")

    n = data.size
    if n < 4:
        raise ValueError("data should have more than three entries")

    bad_indices = np.isnan(data)
    good_indices = ~bad_indices
    dsorted = np.sort(data[good_indices])
    v25 = dsorted[n // 4 - 1]
    v75 = dsorted[(3 * n) // 4 - 1]

    dx = 2 * (v75 - v25) * 1. / (n ** (1. / 3))

    if return_bins:
        Nbins = np.ceil((dsorted[-1] - dsorted[0]) * 1. / dx)
        Nbins = max(1, Nbins)
        bins = dsorted[0] + dx * np.arange(Nbins + 1)
        return dx, bins
    else:
        return dx

