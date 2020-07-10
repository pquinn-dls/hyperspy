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
from hyperspy.misc.image.similarity import mutual_information
from scipy import stats
from scipy.optimize import minimize
from hyperspy.misc.math_tools import optimal_fft_size
import dask.array as da
import scipy as sp
import matplotlib.pyplot as plt
try:
    # For scikit-image >= 0.17.0
    from skimage.registration._phase_cross_correlation import _upsampled_dft
except ModuleNotFoundError:
    from skimage.feature.register_translation import _upsampled_dft

_logger = logging.getLogger(__name__)



def shift_image(im, shift=0, interpolation_order=1, fill_value=0.0):
    if np.any(shift):
        fractional, integral = np.modf(shift)
        if fractional.any():
            order = interpolation_order
        else:
            # Disable interpolation
            order = 0
        return sp.ndimage.shift(im, shift, cval=fill_value, order=order)
    else:
        return im


def triu_indices_minus_diag(n):
    """Returns the indices for the upper-triangle of an (n, n) array
    excluding its diagonal

    Parameters
    ----------
    n : int
        The length of the square array

    """
    ti = np.triu_indices(n)
    isnotdiag = ti[0] != ti[1]
    return ti[0][isnotdiag], ti[1][isnotdiag]


def hanning2d(M, N):
    """
    A 2D hanning window created by outer product.
    """
    return np.outer(np.hanning(M), np.hanning(N))


def sobel_filter(im):
    sx = sp.ndimage.sobel(im, axis=0, mode='constant')
    sy = sp.ndimage.sobel(im, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    return sob


def fft_correlation(in1, in2, normalize=False, real_only=False):
    """Correlation of two N-dimensional arrays using FFT.

    Adapted from scipy's fftconvolve.

    Parameters
    ----------
    in1, in2 : array
        Input arrays to convolve.
    normalize: bool, default False
        If True performs phase correlation.
    real_only : bool, default False
        If True, and in1 and in2 are real-valued inputs, uses
        rfft instead of fft for approx. 2x speed-up.

    """
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    size = s1 + s2 - 1

    # Calculate optimal FFT size
    complex_result = (in1.dtype.kind == 'c' or in2.dtype.kind == 'c')
    fsize = [optimal_fft_size(a, not complex_result) for a in size]

    # For real-valued inputs, rfftn is ~2x faster than fftn
    if not complex_result and real_only:
        fft_f, ifft_f = np.fft.rfftn, np.fft.irfftn
    else:
        fft_f, ifft_f = np.fft.fftn, np.fft.ifftn

    fprod = fft_f(in1, fsize)
    fprod *= fft_f(in2, fsize).conjugate()

    if normalize is True:
        fprod = np.nan_to_num(fprod / np.absolute(fprod))

    ret = ifft_f(fprod).real.copy()

    return ret, fprod


def estimate_image_shift(ref, image, roi=None, sobel=True,
                         medfilter=True, hanning=True, plot=False,
                         dtype='float', normalize_corr=False,
                         sub_pixel_factor=1,
                         return_maxval=True):
    """Estimate the shift in a image using phase correlation

    This method can only estimate the shift by comparing
    bidimensional features that should not change the position
    in the given axis. To decrease the memory usage, the time of
    computation and the accuracy of the results it is convenient
    to select a region of interest by setting the roi keyword.

    Parameters
    ----------
    ref : 2D numpy.ndarray
        Reference image
    image : 2D numpy.ndarray
        Image to register
    roi : tuple of ints (top, bottom, left, right)
         Define the region of interest
    sobel : bool
        apply a sobel filter for edge enhancement
    medfilter :  bool
        apply a median filter for noise reduction
    hanning : bool
        Apply a 2d hanning filter
    plot : bool or matplotlib.Figure
        If True, plots the images after applying the filters and the phase
        correlation. If a figure instance, the images will be plotted to the
        given figure.
    reference : 'current' or 'cascade'
        If 'current' (default) the image at the current
        coordinates is taken as reference. If 'cascade' each image
        is aligned with the previous one.
    dtype : str or dtype
        Typecode or data-type in which the calculations must be
        performed.
    normalize_corr : bool
        If True use phase correlation instead of standard correlation
    sub_pixel_factor : float
        Estimate shifts with a sub-pixel accuracy of 1/sub_pixel_factor parts
        of a pixel. Default is 1, i.e. no sub-pixel accuracy.

    Returns
    -------
    shifts: np.array
        containing the estimate shifts
    max_value : float
        The maximum value of the correlation

    Notes
    -----
    The statistical analysis approach to the translation estimation
    when using reference='stat' roughly follows [*]_ . If you use
    it please cite their article.

    References
    ----------
    .. [*] Bernhard Schaffer, Werner Grogger and Gerald Kothleitner. 
       “Automated Spatial Drift Correction for EFTEM Image Series.” 
       Ultramicroscopy 102, no. 1 (December 2004): 27–36.

    """

    ref, image = da.compute(ref, image)
    # Make a copy of the images to avoid modifying them
    ref = ref.copy().astype(dtype)
    image = image.copy().astype(dtype)
    if roi is not None:
        top, bottom, left, right = roi
    else:
        top, bottom, left, right = [None, ] * 4

    # Select region of interest
    ref = ref[top:bottom, left:right]
    image = image[top:bottom, left:right]

    # Apply filters
    for im in (ref, image):
        if hanning is True:
            im *= hanning2d(*im.shape)
        if medfilter is True:
            # This is faster than sp.signal.med_filt,
            # which was the previous implementation.
            # The size is fixed at 3 to be consistent
            # with the previous implementation.
            im[:] = sp.ndimage.median_filter(im, size=3)
        if sobel is True:
            im[:] = sobel_filter(im)

    # If sub-pixel alignment not being done, use faster real-valued fft
    real_only = (sub_pixel_factor == 1)

    phase_correlation, image_product = fft_correlation(
        ref, image, normalize=normalize_corr, real_only=real_only)

    # Estimate the shift by getting the coordinates of the maximum
    argmax = np.unravel_index(np.argmax(phase_correlation),
                              phase_correlation.shape)
    threshold = (phase_correlation.shape[0] / 2 - 1,
                 phase_correlation.shape[1] / 2 - 1)
    shift0 = argmax[0] if argmax[0] < threshold[0] else  \
        argmax[0] - phase_correlation.shape[0]
    shift1 = argmax[1] if argmax[1] < threshold[1] else \
        argmax[1] - phase_correlation.shape[1]
    max_val = phase_correlation.real.max()
    shifts = np.array((shift0, shift1))

    # The following code is more or less copied from
    # skimage.feature.register_feature, to gain access to the maximum value:
    if sub_pixel_factor != 1:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * sub_pixel_factor) / sub_pixel_factor
        upsampled_region_size = np.ceil(sub_pixel_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        sub_pixel_factor = np.array(sub_pixel_factor, dtype=np.float64)
        normalization = (image_product.size * sub_pixel_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * sub_pixel_factor
        correlation = _upsampled_dft(image_product.conj(),
                                     upsampled_region_size,
                                     sub_pixel_factor,
                                     sample_region_offset).conj()
        correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
            np.argmax(np.abs(correlation)),
            correlation.shape),
            dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + maxima / sub_pixel_factor
        max_val = correlation.real.max()

    # Plot on demand
    if plot is True or isinstance(plot, plt.Figure):
        if isinstance(plot, plt.Figure):
            fig = plot
            axarr = plot.axes
            if len(axarr) < 3:
                for i in range(3):
                    fig.add_subplot(1, 3, i + 1)
                axarr = fig.axes
        else:
            fig, axarr = plt.subplots(1, 3)
        full_plot = len(axarr[0].images) == 0
        if full_plot:
            axarr[0].set_title('Reference')
            axarr[1].set_title('Image')
            axarr[2].set_title('Phase correlation')
            axarr[0].imshow(ref)
            axarr[1].imshow(image)
            d = (np.array(phase_correlation.shape) - 1) // 2
            extent = [-d[1], d[1], -d[0], d[0]]
            axarr[2].imshow(np.fft.fftshift(phase_correlation),
                            extent=extent)
            plt.show()
        else:
            axarr[0].images[0].set_data(ref)
            axarr[1].images[0].set_data(image)
            axarr[2].images[0].set_data(np.fft.fftshift(phase_correlation))
            # TODO: Renormalize images
            fig.canvas.draw_idle()
    # Liberate the memory. It is specially necessary if it is a
    # memory map
    del ref
    del image
    if return_maxval:
        return -shifts, max_val
    else:
        return -shifts

def _shift_cost_function(reference_image, moving_image, shift,
                         bin_rule='scott', cval=0.0, order=1):
    transformed = shift_image(moving_image, shift, interpolation_order=order,
                              fill_value=cval)
    nmi = -mutual_information(reference_image.ravel(),
                               transformed.ravel(),
                               bin_rule=bin_rule)
    return nmi



def estimate_image_shift_mi(reference_image, moving_image,
                            roi=None, sobel=False,
                            medfilter=False, hanning=False,
                            initial_shift=(0.0, 0.0),
                            bin_rule='scotts',
                            method="Powell",
                            use_pyramid_levels=True,
                            pyramid_scale=2,
                            pyramid_minimum_size=16,
                            cval=0.0,
                            order=1,
                            dtype='float',
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
    reference_image, moving_image = da.compute(reference_image, moving_image)
    # Make a copy of the images to avoid modifying them
    reference_image = reference_image.copy().astype(dtype)
    moving_image = moving_image.copy().astype(dtype)
    if roi is not None:
        top, bottom, left, right = roi
    else:
        top, bottom, left, right = [None, ] * 4

    # Select region of interest
    reference_image = reference_image[top:bottom, left:right]
    moving_image = moving_image[top:bottom, left:right]

    # Apply filters
    for im in (reference_image, moving_image):
        if hanning is True:
            im *= hanning2d(*im.shape)
        if medfilter is True:
            # This is faster than sp.signal.med_filt,
            # which was the previous implementation.
            # The size is fixed at 3 to be consistent
            # with the previous implementation.
            im[:] = sp.ndimage.median_filter(im, size=3)
        if sobel is True:
            im[:] = sobel_filter(im)

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

    return -parameters, -result.fun
