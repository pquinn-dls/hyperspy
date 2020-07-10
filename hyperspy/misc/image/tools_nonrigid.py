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
from scipy.optimize import minimize,basinhopping
from scipy.ndimage import affine_transform
from skimage.transform.pyramids import pyramid_gaussian
from hyperspy.external.astroML.histtools import (scotts_bin_width,
                                                 knuth_bin_width,
                                                 freedman_bin_width)
from hyperspy.external.astroML.bayesian_blocks import bayesian_blocks
#from hyperspy.misc.math.stats.mutual_information import similarity_measure
from scipy.ndimage import gaussian_filter,shift


import numpy as np
from scipy import stats
_logger = logging.getLogger(__name__)


def _mi_cost_function(reference_image, moving_image, parameters,
                   translation_only=False, starting_parameters=None,center=(0,0), bin_rule='scott',
                   cval=0.0, order=1):
#    print("oo",parameters)
 #   affine_order      = [2,4,0,5,3,1]
    affine_order      = [4,2,0,3,5,1]
#    affine_order      = [2,4,0,3,5,1]
    if not translation_only:
        pp = parameters[affine_order]
    else:
        pp = parameters
    st = starting_parameters[affine_order]
    affine_matrix = _parameters_to_affine(pp,translation_only,
                                          st,
          center=center)
    transformed = affine_transform(moving_image, affine_matrix, order=order,
                                   cval=cval)
    
#    nmi= -mutual_information(reference_image.ravel(), transformed.ravel(),
#                               bin_rule=bin_rule)

    nmi = -similarity_measure(reference_image,transformed,bin_rule=bin_rule,measure="PMI")
#    print(parameters, nmi)
    return nmi



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
    if bins == 'blocks':
        bins = bayesian_blocks(arr1)
        Nbins = len(bins)
    elif bins == 'knuth':
        dh, bins = knuth_bin_width(arr1, True)
        Nbins = len(bins)
    elif bins == 'scotts':
        dh, bins = scotts_bin_width(arr1, True)
        Nbins = len(bins)
    elif bins == 'freedman':
        Nbins =freedman_diaconis(arr1, returnas="bins")
        #Nbins = freedman_draconis(arr1, True)
    elif isinstance(bins, str):
        raise ValueError("unrecognized bin code: '%s'" % bins)
    return np.histogram2d(arr1, arr2, bins=Nbins+1, **kwargs)


def _get_affine_transform(scale=1, shear=0, rotation=0,
                          translation=(0, 0), center=(0, 0)):
    """Return affine matrix from a set of input parameters.

    Convert scale, shear, rotation, translation, center to an
    affine matrix

    Parameters
    ----------
    rotation : TYPE, optional
        DESCRIPTION. The default is 0.
    scale : TYPE, optional
        DESCRIPTION. The default is 1.
    shear : TYPE, optional
        DESCRIPTION. The default is 0.
    translation : TYPE, optional
        DESCRIPTION. The default is (0, 0).
    center : TYPE, optional
        DESCRIPTION. The default is (0, 0).

    Returns
    -------
    affine_matrix : TYPE
        DESCRIPTION.

    """
    scale = (scale, scale) if isinstance(scale, (int, float)) else scale
    shear = (shear, 0) if isinstance(shear, (int, float)) else shear
    translation = np.array(translation)
    center = np.array(center).reshape(2, 1)
    scale_x, scale_y = scale
    shear_x, shear_y = shear
    t_x, t_y = translation

    affine_matrix = np.array([
        [scale_x*np.cos(rotation+shear_x),
         -scale_y*np.sin(rotation+shear_y), t_x],
        [scale_x*np.sin(rotation+shear_x),
         scale_y*np.cos(rotation+shear_y), t_y],
        [0, 0, 1]
    ])
    center_translation = affine_matrix[:2, :2].dot(center)
    affine_matrix[:2, 2:3] += center - center_translation
    return affine_matrix


def apply_affine_transform(moving_image, rotation=0, scale=1, shear=0,
                         translation=(0, 0), center=(0, 0), **kwargs):
    """
    Parameters
    ----------
    rotation : TYPE, optional
        DESCRIPTION. The default is 0.
    scale : TYPE, optional
        DESCRIPTION. The default is 1.
    shear : TYPE, optional
        DESCRIPTION. The default is 0.
    translation : TYPE, optional
        DESCRIPTION. The default is (0, 0).
    center : TYPE, optional
        DESCRIPTION. The default is (0, 0).

    Returns
    -------
    None.

    """
    affine_matrix = _get_affine_transform(scale=scale, shear=shear,
                                               rotation=rotation,
                                               translation=translation,
                                               center=center)
    transformed = affine_transform(moving_image,
                                   affine_matrix, **kwargs)
    return transformed


def _affine_params_to_matrix(params, center=(0, 0)):
    '''
    Converts a list of affine transformation parameters to the corresponding 3x3 matrix.
    '''
    assert len(params) == 6
    [scale_x, scale_y, shear, rotation, offset_x, offset_y] = params
    center = np.array(center).reshape(2, 1)
    print(scale_y,rotation,shear)
    a0 = scale_x * math.cos(rotation)
    a1 = -scale_y * math.sin(rotation + shear)
    a2 = offset_x
    b0 = scale_x * math.sin(rotation)
    b1 = scale_y * math.cos(rotation + shear)
    b2 = offset_y
    affine_matrix = np.array([[a0, a1, a2], [b0, b1, b2], [0, 0, 1]])
    print("affine",affine_matrix)
    center_translation = affine_matrix[:2, :2].dot(center)
    affine_matrix[:2, 2:3] += center - center_translation
    print("affine",affine_matrix)
    return affine_matrix


def _affine_matrix_to_params(matrix):
    """Converts a 3x3 affine transformation matrix to a set of a
    six affine transformation parameters.

    """
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == 3
    assert matrix.shape[1] == 3
    a0 = matrix[0][0]
    a1 = matrix[0][1]
    a2 = matrix[0][2]
    b0 = matrix[1][0]
    b1 = matrix[1][1]
    b2 = matrix[1][2]
    scale_x = math.sqrt(a0**2 + b0**2)
    scale_y = math.sqrt(a1**2 + b1**2)
    rotation = math.atan2(b0, a0)
    shear = math.atan2(-a1, b1) - rotation
    offset_x = a2
    offset_y = b2
    return [scale_x, scale_y, shear, rotation, offset_x, offset_y]

def _reduce_parameters(scale=1, shear=0, rotation=0,
                        translation=(0, 0), center=(0, 0)):
    scale = (scale, scale) if isinstance(scale, (int, float)) else scale
    translation = np.array(translation)

    parameters = np.array([scale[0], scale[1],
                          shear, rotation, translation[0], translation[1]])
    parameters = _affine_params_to_matrix(parameters,center=center).ravel()[:6]
    return parameters

def _parameters_to_affine(params_in, translation_only,
                          starting_parameters,
                          center=(0, 0)):

    if starting_parameters is None:
        params_out = np.array([1, 0, 0, 0, 1, 0], dtype=float)
    else:
        params_out = starting_parameters

    if translation_only:
        params_out[[2, 5]] = params_in
    else:
        params_out = params_in

    top_matrix = np.reshape(params_out, (2, 3))
    bottom_row = np.array([[0] * 2 + [1]])
    return np.concatenate((top_matrix, bottom_row), axis=0)
#    return _affine_params_to_matrix(params_out)


def _shift_cost_function(reference_image, moving_image, shift,
                           bin_rule='scott', cval=0.0, order=1):
    transformed = shift_image(moving_image, shift, order=order,
                                   cval=cval)
    print("shift",shift,reference_image.shape,moving_image.shape)
#    nmi= -mutual_information(reference_image.ravel(), transformed.ravel(),
#                               bin_rule=bin_rule)
    nmi = -similarity_measure(reference_image,transformed,bin_rule=bin_rule,measure="DPMI")

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
                             shift=(0.0, 0.0),
                             bin_rule='scotts',
                             method="Powell",
                             use_pyramid_levels=True,
                             pyramid_scale=2,
                             pyramid_minimum_size=16,
                             cval=0.0,
                             order=1,
                             **kwargs):
    """Register two imaages by translation using the mutual information metric.

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
    inverse : bool, optional
        Whether to return the inverse transform, which converts coordinates
        in the reference space to coordinates in the target space. For
        technical reasons, this is the transform expected by
        ``scipy.ndimage.affine_transform`` to map the target image to the
        reference space. Defaults to True.
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

    if shift is None:
        parameters = np.zeros(2)
    else:
        parameters = np.array(shift)

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

def estimate_image_transform(reference_image, moving_image,
                             initial_rotation=0.0,
                             initial_scale=1.0,
                             initial_shear=0.0,
                             initial_translation=(0.0, 0.0),
                             translation_only=False,
                             center=(0, 0),
                             bin_rule='freedman',
                             method="Powell",
                             pyramid_levels=True,
                             pyramid_scale=2,
                             pyramid_minimum_size=16,
                             cval=0.0,
                             order=1,
                             **kwargs):
    """Find a transformation matrix to register a moving image to a reference.

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
    inverse : bool, optional
        Whether to return the inverse transform, which converts coordinates
        in the reference space to coordinates in the target space. For
        technical reasons, this is the transform expected by
        ``scipy.ndimage.affine_transform`` to map the target image to the
        reference space. Defaults to True.
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
    if pyramid_levels:
        pyramid_ref = pyramid_gaussian(reference_image,
                                       downscale=pyramid_scale,
                                       sigma=2,
                                       max_layer=nlevels)
        pyramid_mvg = pyramid_gaussian(moving_image,
                                       downscale=pyramid_scale,
                                       sigma=2,
                                       max_layer=nlevels)
        image_pairs = reversed(list(zip(pyramid_ref, pyramid_mvg)))
    else:
        image_pairs = zip(reference_image[:, :, np.newaxis],
                          moving_image[:, :, np.newaxis])
        pyramid_scale = 1.0
        nlevels = 1

    starting_parameters = \
        _reduce_parameters(scale=initial_scale,
                           shear=initial_shear,
                           rotation=initial_rotation,
                           translation=initial_translation)

    translation_indices = [0, 1]
    translation_order = [2,5,1,3,0,4]
    affine_order      = [4,2,0,3,5,1]

    if translation_only:
        starting_parameters = starting_parameters[translation_order]
        parameters = starting_parameters[translation_indices]
    else:
        starting_parameters = starting_parameters[translation_order]
        parameters = starting_parameters
    for ind in translation_indices:
        if abs(parameters[ind]) > 0.0:
            parameters[translation_indices] /= (pyramid_scale ** nlevels)
    for ref, mvg in image_pairs:
        parameters[translation_indices] *= pyramid_scale
#        print(parameters,nlevels)
 #       exit(0)
        _cost = functools.partial(_mi_cost_function, ref, mvg,
                                  translation_only=translation_only,
                                  starting_parameters=starting_parameters,
                                  center=center, bin_rule=bin_rule,
                                  order=order, cval=cval)
#        result = basinhopping(_cost, x0=parameters,stepsize=0.05,minimizer_kwargs={"method":"powell"})
        result = minimize(_cost, x0=parameters,
                          method=method, **kwargs)
        parameters = result.x

        print("best", parameters)
    if not translation_only:
        pp = parameters[affine_order]
    else:
        pp = parameters
    affine_matrix = _parameters_to_affine(pp,
                                                     translation_only,
                                          starting_parameters[affine_order],
                                          center=center)
    c=affine_transform(moving_image,affine_matrix)
    return _affine_matrix_to_params(affine_matrix), -result.fun,c


def make_capital_A(shape):
    """Generates an image in the shape of a capital A (for testing).
    """
    (height, width) = shape
    slope = 0.4375 * width / height
    arr_i = np.arange(height).reshape(height, 1)
    arr_j = np.arange(width).reshape(1, width)
    # Exclude top and bottom
    cond_1 = abs(arr_i - 0.5 * height) < 0.4 * height
    # Exclude the outside of each 'leg'
    cond_2 = abs(arr_j - 0.5 * width) < 0.00625 * width + slope * arr_i
    # Make middle bar
    cond_3 = abs(arr_i - 0.6 * height) < 0.05 * height
    # Cut out holes
    cond_4 = abs(arr_j - 0.5 * width) > slope * arr_i - 0.09375 * width
    # Combine conditions
    cond = cond_1 & cond_2 & (cond_3 | cond_4)
    return cond.astype(np.float64)


def _parameter_vector_to_matrix(parameter_vector):
    """Convert m optimization parameters to a (n+1, n+1) transformation matrix.
    By default (the case of this function), the parameter vector is taken to
    be the first n rows of the affine transformation matrix in homogeneous
    coordinate space.
    Parameters
    ----------
    parameter_vector : (ndim*(ndim+1)) array
        A vector of M = N * (N+1) parameters.
    Returns
    -------
    matrix : (ndim+1, ndim+1) array
        A transformation matrix used to affine-map coordinates in an
        ``ndim``-dimensional space.
    """
    m = parameter_vector.shape[0]
    ndim = int((np.sqrt(4*m + 1) - 1) / 2)
    top_matrix = np.reshape(parameter_vector, (ndim, ndim+1))
    bottom_row = np.array([[0] * ndim + [1]])
    return np.concatenate((top_matrix, bottom_row), axis=0)


def cost_nmi(image0, image1, *, bins=100):
    """Negative of the normalized mutual information.
    See :func:`skimage.metrics.normalized_mutual_information` for more info.
    Parameters
    ----------
    image0, image1 : array
        The images to be compared. They should have the same shape.
    bins : int or sequence of int, optional
        The granularity of the histogram with which to compare the images.
        If it's a sequence, each number is the number of bins for that image.
    Returns
    -------
    cnmi : float
        The negative of the normalized mutual information between ``image0``
        and ``image1``.
    """
    return -mutual_information(image0, image1, bin_rule="scotts")


def _param_cost(reference_image, moving_image, parameter_vector, *,
                vector_to_matrix, cost, multichannel):
    transformation = vector_to_matrix(parameter_vector)
    if not multichannel:
        transformed = affine_transform(moving_image, transformation,
                                           order=1)
    else:
        transformed = np.zeros_like(moving_image)
        for ch in range(moving_image.shape[-1]):
            affine_transform(moving_image[..., ch], transformation,
                                 order=1, output=transformed[..., ch])
    return cost(reference_image.ravel(), transformed.ravel())


def affine(reference_image, moving_image,
           *,
           cost=cost_nmi,
           initial_parameters=None,
           vector_to_matrix=None,
           translation_indices=None,
           inverse=True,
           pyramid_scale=2,
           pyramid_minimum_size=16,
           multichannel=False,
           level_callback=None,
           method='Powell',
           **kwargs):
    """Find a transformation matrix to register a moving image to a reference.
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
    initial_parameters : array of float, optional
        The initial vector to optimize. This vector should have the same
        dimensionality as the transform being optimized. For example, a 2D
        affine transform has 6 parameters. A 2D rigid transform, on the other
        hand, only has 3 parameters.
    vector_to_matrix : callable, array (M,) -> array-like (N+1, N+1), optional
        A function to convert a linear vector of parameters, as used by
        `scipy.optimize.minimize`, to an affine transformation matrix in
        homogeneous coordinates.
    translation_indices : array of int, optional
        The location of the translation parameters in the parameter vector. If
        None, the positions of the translation parameters in the raveled
        affine transformation matrix, in homogeneous coordinates, are used. For
        example, in a 2D transform, the translation parameters are in the
        top two positions of the third column of the 3 x 3 matrix, which
        corresponds to the linear indices [2, 5].
        The translation parameters are special in this class of transforms
        because they are the only ones not scale-invariant. This means that
        they need to be adjusted for each level of the image pyramid.
    inverse : bool, optional
        Whether to return the inverse transform, which converts coordinates
        in the reference space to coordinates in the target space. For
        technical reasons, this is the transform expected by
        ``scipy.ndimage.affine_transform`` to map the target image to the
        reference space. Defaults to True.
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
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension. By default, this is False.
    level_callback : callable, optional
        If given, this function is called once per pyramid level with a tuple
        containing the current downsampled image, transformation matrix, and
        cost as the argument. This is useful for debugging or for plotting
        intermediate results during the iterative process.
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

    # ignore the channels if present
    ndim = reference_image.ndim if not multichannel else reference_image.ndim - 1
    if ndim == 0:
        raise ValueError(
            'Input images must have at least 1 spatial dimension.'
        )

    min_dim = min(reference_image.shape[:ndim])
    nlevels = int(np.floor(np.log2(min_dim) - np.log2(pyramid_minimum_size)))

    pyramid_ref = pyramid_gaussian(reference_image, downscale=pyramid_scale,
                                   max_layer=nlevels,
                                   multichannel=multichannel)
    pyramid_mvg = pyramid_gaussian(moving_image, downscale=pyramid_scale,
                                   max_layer=nlevels,
                                   multichannel=multichannel)
    image_pairs = reversed(list(zip(pyramid_ref, pyramid_mvg)))

    if initial_parameters is None:
        initial_parameters = np.eye(ndim, ndim + 1).ravel()
    parameter_vector = initial_parameters

    if vector_to_matrix is None:
        vector_to_matrix = _parameter_vector_to_matrix

    if translation_indices is None:
        translation_indices = slice(ndim, ndim**2 - 1, ndim)

    for ref, mvg in image_pairs:
        parameter_vector[translation_indices] *= pyramid_scale
        _cost = functools.partial(_param_cost, ref, mvg,
                                  vector_to_matrix=vector_to_matrix,
                                  cost=cost, multichannel=multichannel)
        result = minimize(_cost, x0=parameter_vector, method=method, **kwargs)
        parameter_vector = result.x
        if level_callback is not None:
            level_callback(
                (mvg,
                 vector_to_matrix(parameter_vector),
                 result.fun)
            )

    matrix = vector_to_matrix(parameter_vector)

    if not inverse:
        # estimated is already inverse, so we invert for forward transform
        matrix = np.linalg.inv(matrix)

    return matrix


#from fastkde import fastKDE


TINY = float(np.finfo(np.double).tiny)
SIGMA_FACTOR = 0.05
# A lambda function to force positive values
nonzero = lambda x: np.maximum(x, TINY)




def dist2loss(q, qI=None, qJ=None):
    """
    Convert a joint distribution model q(i,j) into a pointwise loss:

    L(i,j) = - log q(i,j)/(q(i)q(j))

    where q(i) = sum_j q(i,j) and q(j) = sum_i q(i,j)

    See: Roche, medical image registration through statistical
    inference, 2001.
    """
    qT = q.T
    if qI is None:
        qI = q.sum(0)
    if qJ is None:
        qJ = q.sum(1)
    q /= nonzero(qI)
    qT /= nonzero(qJ)
    return -np.log(nonzero(q))


class SimilarityMeasure(object):
    """
    Template class
    """
    def __init__(self, renormalize=False):
        self.renormalize = renormalize

    def loss(self, H):
        return np.zeros(H.shape)

    def npoints(self, H):
        return H.sum()

    def __call__(self, H):
        total_loss = np.sum(H * self.loss(H))
        if not self.renormalize:
            total_loss /= nonzero(self.npoints(H))
        return -total_loss

class MutualInformation(SimilarityMeasure):
    """
    Use the normalized joint histogram as a distribution model
    """
    def loss(self, H):
        return dist2loss(H / nonzero(self.npoints(H)))


class ParzenMutualInformation(SimilarityMeasure):
    """
    Use Parzen windowing to estimate the distribution model
    """
    def loss(self, H):
        if not hasattr(self, 'sigma'):
            self.sigma = SIGMA_FACTOR * np.array(H.shape)
        npts = nonzero(self.npoints(H))
        Hs = H / npts
        gaussian_filter(Hs, sigma=self.sigma, mode='constant', output=Hs)
        return dist2loss(Hs)


class DiscreteParzenMutualInformation(SimilarityMeasure):
    """
    Use Parzen windowing in the discrete case to estimate the
    distribution model
    """
    def loss(self, H):
        if not hasattr(self, 'sigma'):
            self.sigma = SIGMA_FACTOR * np.array(H.shape)
        Hs = gaussian_filter(H, sigma=self.sigma, mode='constant')
        Hs /= nonzero(Hs.sum())
        return dist2loss(Hs)


class NormalizedMutualInformation(SimilarityMeasure):
    """
    NMI = 2*(1 - H(I,J)/[H(I)+H(J)])
        = 2*MI/[H(I)+H(J)])
    """
    def __call__(self, H):
        H = H / nonzero(self.npoints(H))
        hI = H.sum(0)
        hJ = H.sum(1)
        entIJ = -np.sum(H * np.log(nonzero(H)))
        entI = -np.sum(hI * np.log(nonzero(hI)))
        entJ = -np.sum(hJ * np.log(nonzero(hJ)))
        return 2 * (1 - entIJ / nonzero(entI + entJ))




def similarity_measure(image1,image2,norm=True,bin_rule="scotts",measure="PMI"):
    """
    
    Computes mutual information between two images variate from a
    joint histogram.
    
    Parameters
    ----------
    
    arr1 : 1D array
    arr2 : 1D array
    
    bins:  number of bins to use 
           Default = None.  If None specificed then 
           the inital estimate is set to be int(sqrt(size/5.))
           where size is the number of points in arr1  
           
    Returns
    -------
     mi: float  the computed similariy measure
     
     
    """
    arr1 = image1.ravel()
    arr2 = image2.ravel()
    if bin_rule ==  None or bin_rule == "sturges":
        dx,Nbins = sturges_bin_width(arr1)
    elif bin_rule == "scotts":
        dx,Nbins = scotts_bin_width(arr1,True)
    elif bin_rule == "freedman":
        dx,Nbins = freedman_bin_width(arr1,True)
    elif bin_rule == 'auto':
        if len(arr1)<400:
            dx,Nbins = sturges_bin_width(arr1,True)
        else:
            dx,Nbins = scotts_bin_width(arr1,True)
    else:
        raise ValueError("Unrecognised bin width rule: please use auto, scott, sturges or freedman")

    # Convert bins counts to probability values
    hgram, x_edges, y_edges = np.histogram2d(arr1,arr2,Nbins)
    if measure == "FKDE":
        hgram,edges = fastKDE.pdf(arr1,arr2,Nbins)
        measure     =  "NMI"                
    if measure == "MI":
        pxy = MutualInformation(renormalize=norm)
    elif measure == "NMI":
        pxy = NormalizedMutualInformation(renormalize=norm)
    elif measure == "PMI":
        pxy = ParzenMutualInformation(renormalize=norm)
    elif measure == "DPMI":
        pxy = DiscreteParzenMutualInformation(renormalize=norm)
    else:
        pxy = NormalizedMutualInformation(renormalize=norm)
    return pxy(hgram)





