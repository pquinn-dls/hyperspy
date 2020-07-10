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


def mutual_information(arr1, arr2, norm=False, bin_rule="scotts"):
    """Calculate the mutual information between two images.

    Computes mutual information between two images variate from a
    joint histogram.

    Parameters
    ----------
    arr1 : 1D array
    arr2 : 1D array

    bins:  number of bins to use, default = 'auto'.

    Returns
    -------
     mi: float  the computed similariy measure
    """
    # Convert bins counts to probability values
    hgram, edges_x, edges_y = histogram2d(arr1, arr2, bins=bin_rule)
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum

    if norm:
        nxzx = px > 0
        nxzy = py > 0
        h_x = -np.sum(px[nxzx] * np.log(px[nxzx]))
        h_y = -np.sum(py[nxzy] * np.log(py[nxzy]))
        norm = 1.0 / (max(np.amax(h_x), np.amax(h_y)))
    else:
        norm = 1.0

    i_xy = norm*(np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs])))

    return i_xy


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
    if bins == 'scotts':
        dh, bins = scotts_bin_width(arr1, True)
        Nbins = len(bins)
    elif bins == 'freedman':
        Nbins = freedman_diaconis(arr1, returnas="bins")
    elif isinstance(bins, str):
        raise ValueError("unrecognized bin code: '%s'" % bins)
    return np.histogram2d(arr1, arr2, bins=Nbins+1, **kwargs)


def _shift_cost_function(reference_image, moving_image, shift,
                         bin_rule='scott', cval=0.0, order=1):
    transformed = shift_image(moving_image, shift, order=order,
                              cval=cval)
    nmi = -mutual_information(reference_image.ravel(),
                               transformed.ravel(),
                               bin_rule=bin_rule)
    return nmi


def shift_image(im, offset=0, order=1, cval=np.nan):
    if np.any(offset):
        fractional, integral = np.modf(offset)
        if fractional.any():
            order = order
        else:
            # Disable interpolation
            order = 0
    return shift(im, offset, cval=cval, order=order)


def estimate_image_shift_mi(reference_image, moving_image,
                            initial_shift=(0.0, 0.0),
                            bin_rule='scotts',
                            method="Powell",
                            use_pyramid_levels=True,
                            pyramid_scale=2,
                            pyramid_minimum_size=16,
                            cval=0.0,
                            order=1,
                            **kwargs):
    """Register two images by translation using a metric.

    Registration by optimization of the mutual information metric

    Parameters
    ----------
    reference_image : ndarray
        A reference image to compare against the target.
    moving_image : ndarray
        Our target for registration. Transforming this image using the
        returned matrix aligns it with the reference.
    cost : function, optional
        A cost function which takes two images and returns a score which is
        at a minimum when images are aligned. Uses the normalized mutual
        information by default.
    pyramid_scale : float, optional
        Scaling factor to generate the image pyramid. The affine transformation
        is estimated first for a downscaled version of the image, then
        progressively refined with higher resolutions. This parameter controls
        the increase in resolution at each level.
    pyramid_minimum_size : integer, optional
        The smallest size for an image along any dimension. This value
        determines the size of the image pyramid used. Choosing a smaller value
        here can cause registration errors, but a larger value could speed up
        registration when the alignment is easy.
    method : string or callable
        Method of minimization.  See ``scipy.optimize.minimize`` for available
        options.
    **kwargs : keyword arguments
        Keyword arguments passed through to ``scipy.optimize.minimize``

    Returns
    -------
    matrix : array, or object coercible to array
        A transformation matrix used to obtain a new image.
        ``ndi.affine_transform(moving, matrix)`` will align the moving image to
        the reference.

    Example
    -------
    >>> from skimage.data import astronaut
    >>> reference_image = astronaut()[..., 1]
    >>> r = -0.12  # radians
    >>> c, s = np.cos(r), np.sin(r)
    >>> matrix_transform = np.array([[c, -s, 0], [s, c, 50], [0, 0, 1]])
    >>> moving_image = ndi.affine_transform(reference_image, matrix_transform)
    >>> matrix = affine(reference_image, moving_image)
    >>> registered_moving = ndi.affine_transform(moving_image, matrix)
    """
    ndim = reference_image.ndim
    if ndim == 0:
        raise ValueError(
            'Input images must have at least 1 spatial dimension.'
        )
    min_dim = min(reference_image.shape[:ndim])
    # number of pyramid levels depends on scaling used
    nlevels = int(np.floor(math.log(min_dim, pyramid_scale)
                           - math.log(pyramid_minimum_size, pyramid_scale)))
    if use_pyramid_levels:
        pyramid_ref = pyramid_gaussian(reference_image,
                                       downscale=pyramid_scale,
                                       max_layer=nlevels)
        pyramid_mvg = pyramid_gaussian(moving_image,
                                       downscale=pyramid_scale,
                                       max_layer=nlevels)
        image_pairs = reversed(list(zip(pyramid_ref, pyramid_mvg)))
    else:
        image_pairs = zip(reference_image[:, :, np.newaxis],
                          moving_image[:, :, np.newaxis])
        pyramid_scale = 1.0
        nlevels = 1

    if initial_shift is None:
        parameters = np.zeros(2)
    else:
        parameters = np.array(initial_shift)

    for i in range(len(parameters)):
        if abs(parameters[i]) > 0.0:
            parameters[i] /= (pyramid_scale ** nlevels)

    for ref, mvg in image_pairs:
        parameters *= pyramid_scale
        _cost = functools.partial(_shift_cost_function, ref, mvg,
                                  bin_rule=bin_rule,
                                  order=order, cval=cval)
        result = minimize(_cost, x0=parameters,
                          method=method, **kwargs)
        parameters = result.x

    return parameters, -result.fun


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
        datmin, datmax = data.min(), data.max()
        print(datmin,datmax,bw)
        datrng = datmax - datmin
        if bw==0.0:
            result = 2
        else:
            result = int((datrng / bw) + 1)
    return result
