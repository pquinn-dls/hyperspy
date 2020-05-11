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

import matplotlib.pyplot as plt
import numpy as np
import dask.array as da
import scipy as sp
import logging
from scipy.fftpack import fftn, ifftn
import skimage
from skimage.feature.register_translation import _upsampled_dft
#import pybobyqa
from hyperspy.misc.math.stats.mutual_information import similarity_measure


_logger = logging.getLogger(__name__)

def shift_image(im, shift=0, interpolation_order=1, fill_value=np.nan):
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


def fft_correlation(in1, in2, normalize=False):
    """Correlation of two N-dimensional arrays using FFT.

    Adapted from scipy's fftconvolve.

    Parameters
    ----------
    in1, in2 : array
    normalize: bool
        If True performs phase correlation

    """
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    size = s1 + s2 - 1
    # Use 2**n-sized FFT
    fsize = (2 ** np.ceil(np.log2(size))).astype("int")
    fprod = fftn(in1, fsize)
    fprod *= fftn(in2, fsize).conjugate()
    if normalize is True:
        fprod = np.nan_to_num(fprod / np.absolute(fprod))
    ret = ifftn(fprod).real.copy()
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
    plot : bool | matplotlib.Figure
        If True, plots the images after applying the filters and the phase
        correlation. If a figure instance, the images will be plotted to the
        given figure.
    reference : 'current' | 'cascade'
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
    when using `reference`='stat' roughly follows [1]_ . If you use
    it please cite their article.

    References
    ----------

    .. [1] Bernhard Schaffer, Werner Grogger and Gerald
        Kothleitner. “Automated Spatial Drift Correction for EFTEM
        Image Series.”
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
            im[:] = sp.signal.medfilt(im)
        if sobel is True:
            im[:] = sobel_filter(im)
    phase_correlation, image_product = fft_correlation(
        ref, image, normalize=normalize_corr)

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



def Similarity_ErrFunc(param,ref_images,moving_images,interpolation_order,fill_value,bin_rule='auto',measure="PMI"):
    
    if isinstance(ref_images, np.ndarray):
        ref_images    = [ref_images]
        moving_images = [moving_images]
    
    mi_sum=0.0
    for ref,move in zip(ref_images,moving_images):
        # Perform a translational shift
        _img = shift_image(move,param,interpolation_order, 0.0)#fill_value)
        # crop both images....
#        n_y, n_x = _img.shape
#        y, x = -int(np.trunc(param[0]+0.5)), -int(np.trunc(param[1]+0.5))
#        y, x = -int(param[0]//2), -int(param[1]//2)

#        top, bot = max(0, y), min(n_y, n_y+y)
#        lef, rig = max(0, x), min(n_x, n_x+x),
#        _img_crop = _img[ top-y:bot-y, lef-x:rig-x]
#        ref_crop  = ref[ top-y:bot-y, lef-x:rig-x]
        # compare the images
        #mi_sum+=similarity_measure(ref_crop,_img_crop,bin_rule=bin_rule,measure=measure)
        mi_sum+=similarity_measure(ref,_img,bin_rule=bin_rule,measure=measure)

    # measure the mutual information between the reference and moving image
    mi_sum = max(mi_sum,1.0e-10)
    return -mi_sum

def gradient_measure(im1,im2):
    gx,gy=np.gradient(im1)
    g_stack = np.dstack([gx,gy])
    gnorm = np.linalg.norm(g_stack)
    gx=gx/gnorm
    gy=gy/gnorm
    
    hx,hy=np.gradient(im2)
    h_stack = np.dstack([hx,hy])
    hnorm = np.linalg.norm(h_stack)
    hx=hx/hnorm
    hy=hy/hnorm
    h = np.dstack([hx,hy])
    g = np.dstack([gx,gy])
    cross_product = np.cross(g,h) 
    result = np.linalg.norm(cross_product)

def jacobian_image(x,dx=0.2,**args):
    n=len(x)
    func=f(x,**args)
    jac=np.zeros((n,n))
    for j in range(n): #through columns
        Dxj = dx 
        x_plus=[(xi if k!=j else xi+Dxj) for k,xi in enumerate(x)]
        jac[:,j]=(f(x_plus,**args)-func)/Dxj
    return jac
    

def Gradient_ErrFunc(param,ref_images,moving_images,interpolation_order,fill_value,bin_rule='auto',measure="MI"):
    
    if isinstance(ref_images, np.ndarray):
        ref_images    = [ref_images]
        moving_images = [moving_images]
    
    mi_sum=0.0
    for ref,move in zip(ref_images,moving_images):
        # Perform a translational shift
        _img = shift_image(move,param,interpolation_order, fill_value)
        # crop both images....
        n_y, n_x = _img.shape
        y, x = -int(np.round(param[0])), -int(np.round(param[1]))
        top, bot = max(0, y), min(n_y, n_y+y)
        lef, rig = max(0, x), min(n_x, n_x+x),
        _img_crop = _img[ top-y:bot-y, lef-x:rig-x]
        ref_crop  = ref[ top-y:bot-y, lef-x:rig-x]
        # compare the images
        mi_sum+=similarity_measure(ref_crop,_img_crop,bin_rule=bin_rule,measure=measure)
    # measure the mutual information between the reference and moving image
    return 1.0/mi_sum

def estimate_similarity_image_shift(ref,image,guess=None,interpolation_order=1,fill_value=-1.0,\
                                    roi=None,
                         medfilter=True,  plot=False,
                         dtype='float',bin_rule='auto',measure="MI"):
    # Rough search...
    # rough estimate of number of bins based on the starting image size...
    #print("img0",img[0].shape)
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
        if medfilter is True:
#            im[:] = sp.signal.medfilt(im)
            im[:] = sp.ndimage.filters.gaussian_filter(im, sigma=1.5)

    xsize,ysize = image.shape
    xspan = min(int(0.2*xsize),20)
    yspan = min(int(0.2*ysize),20)
    

    x_range = np.array(range(-xspan,xspan),np.float64)
    y_range = np.array(range(-yspan,yspan),np.float64)
    if guess is not None:
        bestx = guess[0]
        besty = guess[1]
    else:
        bestx = 0.
        besty = 0.
    besterr = 1.e10
    for xpos in x_range:   
        for ypos in y_range:
            err = Similarity_ErrFunc([xpos,ypos],ref,image,interpolation_order,\
                                     fill_value,bin_rule=bin_rule,measure=measure)
            if err < besterr:
                besterr=err
                bestx = xpos
                besty = ypos
    besterr = 1.e10
    x_range = np.arange(bestx-1.,bestx+1.,0.2)
    y_range = np.arange(besty-1.,besty+1.,0.2)

    for xpos in x_range:   
        for ypos in y_range:   
            err = Similarity_ErrFunc([xpos,ypos],ref,image,interpolation_order,\
                                     fill_value,bin_rule=bin_rule,measure=measure)
            if err < besterr:
                besterr=err
                bestx = xpos
                besty = ypos

    param = [bestx,besty]
    param = np.array(param,np.float64)
#    lower = np.array([-xspan, -yspan])
#    upper = np.array([xspan, yspan])

    # Call Py-BOBYQA (with bounds)
    #soln = pybobyqa.solve(Similarity_ErrFunc,param,\
    #                      args=(ref,image,interpolation_order,fill_value,bin_rule),
    #                      bounds=(lower,upper))                
    default_options = {'maxcor': 10, 'ftol': 1e-7, 'gtol': 1e-5,'eps': 1e-8, 'maxiter': 1000,'epsilon':0.2}
    bounds=((bestx-10,bestx+10),(besty-10.,bestx+10.))
#    optim = sp.optimize.fmin_l_bfgs_b(Similarity_ErrFunc,param, 
#        args=(ref,image,interpolation_order,fill_value,"auto",measure),
#                               approx_grad=True,bounds=bounds,epsilon=0.2,disp=1)
#    optim = sp.optimize.minimize(Similarity_ErrFunc,param, method='COBYLA',\
#        args=(ref,image,interpolation_order,fill_value,bin_rule,measure),
#                              constraints=(), \
#                              tol=None, \
#                              callback=None,\
#                               options={'rhobeg': 2.0,
##                                      'maxiter': 200, 
 #                                     'disp': True, 
 #                                     'catol': 0.0002})
    #optim = sp.optimize.minimize(Similarity_ErrFunc,param, method='POWELL',\
    #    args=(ref,image,interpolation_order,fill_value,bin_rule,measure))

  #  param = optimize.cobyla(MIErrFunc,param,approx_grad=True,epsilon=0.05,
    #                               args=(img,ximg,interpolation_order,fill_value,bins), bounds=[(-14,14),(-14,14)])        
    #return soln.x,soln.f
    #param=optim.x
    #print(optim)
#    print(optim[2])
 #   print(optim)
    #besterr = optim.fun
    return -param,besterr

#
#def estimate_gradient_image_shift(ref,image,interpolation_order=1,fill_value=-1.0,\
#                                    roi=None,
#                         medfilter=True,  plot=False,
#                         dtype='float',bin_rule='auto',measure="MI"):
#    # Rough search...
#    # rough estimate of number of bins based on the starting image size...
#    #print("img0",img[0].shape)
#    ref, image = da.compute(ref, image)
#    # Make a copy of the images to avoid modifying them
#    ref = ref.copy().astype(dtype)
#    image = image.copy().astype(dtype)
#    if roi is not None:
#        top, bottom, left, right = roi
#    else:
#        top, bottom, left, right = [None, ] * 4
#
#    # Select region of interest
#    ref = ref[top:bottom, left:right]
#    image = image[top:bottom, left:right]
#    # Apply filters
#    for im in (ref, image):
#        if medfilter is True:
#            im[:] = sp.signal.medfilt(im)
#
#    xsize,ysize = image.shape
#    xspan = int(0.3*xsize)
#    yspan = int(0.3*ysize)
#    
#
#    x_range = np.array(range(-xspan,xspan),np.float64)
#    y_range = np.array(range(-yspan,yspan),np.float64)
#    bestx = 0.
#    besty = 0.
#    besterr = 1.e10
#    for xpos in x_range:   
#        for ypos in y_range:
#            err = Gradient_ErrFunc([xpos,ypos],ref,image,interpolation_order,\
#                                     fill_value)
#           # print(xpos,ypos,err)
#            if err < besterr:
#                besterr=err//
#                bestx = xpos
#                besty = ypos
#    param = [bestx,besty]
#    param = np.array(param,np.float64)
#    lower = np.array([-xspan, +xspan])
#    upper = np.array([-yspan, +yspan])
#    lower = np.array([-xspan, -yspan])
#    upper = np.array([xspan, yspan])
#
##    bounds =(lower,upper)
#    # Call Py-BOBYQA (with bounds)
#    soln = pybobyqa.solve(Gradient_ErrFunc,param,\
#                          args=(ref,image,interpolation_order,fill_value),\
#                          bounds=(lower,upper))
#    #param = sp.optimize.fmin_l_bfgs_b(Gradient_ErrFunc,param,approx_grad=True,epsilon=0.1,
#    #                              args=(ref,image,interpolation_order,fill_value), bounds=bounds)        
#    return soln.x,soln.f
#    #return param#,besterr
#

import functools
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize

from skimage.transform.pyramids import pyramid_gaussian
from skimage.metrics import normalized_mutual_information

__all__ = ['affine']


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
    return -normalized_mutual_information(image0, image1, bins=bins)


def _param_cost(reference_image, moving_image, parameter_vector, *,
                vector_to_matrix, cost, multichannel):
    transformation = vector_to_matrix(parameter_vector)
    if not multichannel:
        transformed = ndi.affine_transform(moving_image, transformation,
                                           order=1)
    else:
        transformed = np.zeros_like(moving_image)
        for ch in range(moving_image.shape[-1]):
            ndi.affine_transform(moving_image[..., ch], transformation,
                                 order=1, output=transformed[..., ch])
    return cost(reference_image, transformed)



def affine(reference_image, moving_image,
           *,
           cost=cost_nmi,
           initial_parameters=None,
           vector_to_matrix=None,
           translation_indices=None,
           inverse=True,
           pyramid_scale=2,
           pyramid_minimum_size=32,
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




def affine_matrix_to_params(matrix: np.array):

    '''

    Converts a 3x3 affine transformation matrix to a list of six affine transformation parameters.

    '''

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

def transform_using_values(arr_in: np.array, values: list, cval=-1):
    '''Applies a translation or affine transformation to input_image
    using the parameter values in `values
  
    Parameters
    ----------
    input_image : ndarray
    values : ndarray or length 2 or 6
        array of length 2 is a translation containing offset_x,offset_y
        array of length 6 is an affine transformation containing offset_x,offset_y,scale_x,scale_y,shear,rotation
    cval : int or float , default : -1 
        Fill value for border regions after transformation
        
    Returns
    -------
    ndarray
        Transformed image    
    
    '''
    scale_x = 0.0
    scale_y = 0.0
    shear_radians = 0.0
    rotate_radians = 0.0

    if len(values==2):
        offset_x = values[0]
        offset_y = values[1]
    elif len(values) == 6:
        offset_x = values[0]
        offset_y = values[1]
        scale_x = values[2]
        scale_y = values[3]
        shear_radians = values[4]
        rotate_radians = values[5]

    # Image must be shifted by minus half each dimension, then transformed, then shifted back.
    # This way, rotations and shears will be about the centre of the image rather than the top-left corner.
    shift_x = -0.5 * arr_in.shape[1]
    shift_y = -0.5 * arr_in.shape[0]
    a0 = scale_x * np.cos(rotate_radians)
    a1 = -scale_y * np.sin(rotate_radians + shear_radians)
    a2 = a0 * shift_x + a1 * shift_y + offset_x - shift_x
    b0 = scale_x * np.sin(rotate_radians)
    b1 = scale_y * np.cos(rotate_radians + shear_radians)
    b2 = b0 * shift_x + b1 * shift_y + offset_y - shift_y
    tform = skimage.transform.AffineTransform(matrix=np.array([[a0, a1, a2], [b0, b1, b2], [0, 0, 1]]))
    arr_out = skimage.transform.warp(arr_in.astype(float), tform.inverse, cval=cval)
    return arr_out


def scale(input_image: np.array, scale_factor_x, scale_factor_y):
    '''Scales the image represented by `arr_in` by scale factors `scale_factor_x` 
    and `scale_factor_y`.

    Parameters
    ----------
    input_image : ndarray
    scale_x : float
    scale_y : float

    Returns
    ndarray
        Scaled version of input_image

    '''

    return transform_using_values(input_image, [scale_factor_x, scale_factor_y, 0.0, 0.0, 0.0, 0.0])





def rotate(arr_in: np.array, rotate_radians):

    '''

    Rotates the image represented by `arr_in` by `rotate_radians` radians in the clockwise direction.

    '''

    return transform_using_values(arr_in, [1.0, 1.0, 0.0, rotate_radians, 0.0, 0.0])

        



def transform_using_matrix(arr_in: np.array, matrix):

    '''

    Applies an affine transformation specified by `matrix` to `arr_in`.

    '''

    tform = skimage.transform.AffineTransform(matrix=matrix)

    return skimage.transform.warp(arr_in/arr_in.max(), tform.inverse, cval=arr_in.mean()/arr_in.max())*arr_in.max()



def optimise_affine(arr_moving: np.array, arr_ref: np.array, scale_x=1.0, scale_y=1.0, shear_radians=0.0, rotate_radians=0.0, offset_x=0.0, offset_y=0.0, method='Powell', bounds=None, isotropic_scaling=False, lock_scale=False, lock_shear=False, lock_rotation=False, lock_translation=False, debug=False):

    '''

    Uses a local optimisation algorithm to obtain a set of affine transform parameters that maximises the mutual information between `arr_transformed` and `arr_ref`, where `arr_transformed` is the transformed version of `arr_moving`.

    

    `arr_moving`: The moving image.

    `arr_ref`: The reference image. The function will attempt to align `arr_moving` with `arr_ref`.

    `scale_x`, `scale_y`, `shear_radians`, `rotate_radians`, `offset_x`, `offset_y`: Initial guesses of parameter values, to be fed into the optimisation algorithm.

    `method`: The name of the local optimisation algorithm to be used.

    `bounds`: If not None, this is a Numpy array of six 2-tuples. Each tuple contains the minimum and maximum permitted values (in that order) of the corresponding affine transformation parameter. If `bounds` is None, default minimum and maximum values will be used.

    `isotropic_scaling`: If True, the horizontal and vertical scale parameters will be treated as a single scale parameter, such that any change in scale will be by the same scale factor in the horizontal and vertical directions.

    `lock_scale`, `lock_shear`, `lock_rotation`, `lock_translation`: If True, the corresponding affine translation parameter(s) will be set to a default value and no attempt will be made to optimise them.

    `debug`: If True, debugging messages will be printed during execution.

    

    In theory, `optimise_affine_v2` does everything this function does and optionally more on top of that. However, `optimise_affine` seems to be faster, even when the parameters of `optimise_affine_v2` are set in such a way that it might be expected to do almost exactly the same thing.

    '''

    params = np.array([scale_x, scale_y, shear_radians, rotate_radians, offset_x, offset_y])

    (height, width) = arr_moving.shape

    if bounds is None:

        bounds = np.array([(0.5, 2), (0.5, 2), (-math.pi/6, math.pi/6), (-math.pi/6, math.pi/6), (-height*0.2, height*0.2), (-width*0.2, width*0.2)])

    

    def inverse_mutual_information_after_transform(free_params):

        def _outside_limits(x: np.array):

            [xmin, xmax] = np.array(bounds).T

            return (np.any(x < xmin) or np.any(x > xmax))

        def _fit_params_to_bounds(params):

            assert len(params) == 6

            params_out = np.array(params)

            for i in range(6):

                params_out[i] = max(params[i], bounds[i][0])

                params_out[i] = min(params_out[i], bounds[i][1])

            return params_out

        transform_params = fill_missing_parameters(free_params, isotropic_scaling, lock_scale, lock_shear, lock_rotation, lock_translation)

        arr_transformed = transform_using_values(arr_moving, transform_params)

        mi = im.similarity_measure(arr_transformed, arr_ref)

        mi_scaled = mi/(arr_ref.size)

        outside = _outside_limits(transform_params)

        metric = 1/(mi_scaled + 1)

        if outside:

            fitted_params = _fit_params_to_bounds(transform_params)

            arr_transformed_fitted = transform_using_values(arr_moving, fitted_params)

            mi_fitted = im.similarity_measure(arr_transformed_fitted, arr_ref)/(arr_ref.size)

            metric = 1/mi_fitted

        if debug:

            print((1/metric, free_params))

        assert (outside and 1/metric <= 1) or ((not outside) and 1/metric >= 1)

        return metric

    

    initial_guess = remove_locked_parameters(params, isotropic_scaling, lock_scale, lock_shear, lock_rotation, lock_translation)

    

    optimisation_result = scipy.optimize.minimize(inverse_mutual_information_after_transform, initial_guess, method=method)

    if optimisation_result.success:

        optimised_parameters = optimisation_result.x

        # If there is only one optimised parameter, optimisation_parameters will be of the form np.array(param), which has zero length.

        # Otherwise, it will be of the form np.array([param1, param2, ...]).

        # np.array(param) should therefore be converted to the form np.array([param]), which has length 1.

        if len(optimised_parameters.shape) == 0:

            optimised_parameters = np.array([optimised_parameters])

        return fill_missing_parameters(optimised_parameters, isotropic_scaling, lock_scale, lock_shear, lock_rotation, lock_translation)

    else:

        raise ValueError(optimisation_result.message)

    

    

def optimise_affine_v2(

    arr_moving: np.array,

    arr_ref: np.array,

    arr_ref_ns: np.array=None,

    bounds: dict={},

    lock_strings: list=[],

    initial_guess: dict={},

    debug=False,

    local_optimisation_method='Powell',

    similarity_measure: str='',

    ns_max_groups=6,

    basinhopping=False,

    basinhopping_kwargs={}):

    '''

    Uses a local optimisation algorithm to obtain a set of affine transform parameters that maximises the mutual information between `arr_transformed` and `arr_ref`, where `arr_transformed` is the transformed version of `arr_moving`.

    

    `arr_moving`: The moving image.

    `arr_ref`: The reference image. The function will attempt to align `arr_moving` with `arr_ref`.

    `arr_ref_ns`: If not None, this should be the output of `get_neighbour_similarity(arr_ref)`. Only used if `similarity_measure` is 'neighbour_similarity'.

    `bounds`: A dictionary of up to six 2-tuples. The key corresponding to each tuple is the name of an affine transformation parameter. Each tuple contains the minimum and maximum permitted values (in that order) of the corresponding affine transformation parameter. If `bounds` is missing any parameters, default minimum and maximum values will be used for those parameters.

    `lock_strings`: a list of some of the following strings (or an empty list):

        - 'isotropic_scaling'

        - 'lock_scale'

        - 'lock_scale_x'

        - 'lock_scale_y'

        - 'lock_shear'

        - 'lock_rotation'

        - 'lock_translation'

        - 'lock_translation_x'

        - 'lock_translation_y'

    If 'isotropic_scaling' is present, the horizontal and vertical scale parameters will be treated as a single scale parameter, such that any change in scale will be by the same scale factor in the horizontal and vertical directions.

    If one or more of the other strings is present, the corresponding affine translation parameter(s) will be set to a default value and no attempt will be made to optimise them.

    `initial_guess`: A dictionary of up to six float values. The key corresponding to each float is the name of an affine transformation parameter. Each float represents the initial guess for the corresponding affine transformation parameter, which will be fed into the optimisation algorithm. If `initial_guess` is missing any parameters, default values will be used for those parameters.

    `debug`: If True, debugging messages will be printed during execution.

    `local_optimisation_method`: The name of the local optimisation algorithm to be used.

    `similarity_measure`: A string representing the similarity measure to be used. Currently the following strings are recognised:

        - 'neighbour_similarity' (corresponds to `similarity_measure_using_neighbour_similarity`)

        - 'overlap' (corresponds to `similarity_measure_area_of_overlap`)

    If `similarity_measure` does not match any of these, the similarity measure used will default to `similarity_measure_after_transform`.

    `ns_max_groups`: The value of the `max_groups` parameter passed into `similarity_measure_using_neighbour_similarity`. Only used if `similarity_measure` is 'neighbour_similarity'.

    `basinhopping`: If True, a basin-hopping algorithm is used rather than just the local optimisation method. The function will take longer to execute if this is the case, but the results may be better.

    `basinhopping_kwargs`: Any keyword arguments to pass into the basin-hopping algorithm. Only used if `basinhopping` is True.

    '''

        

    params = params_dict_to_array(initial_guess)

    bounds_list = bounds_dict_to_array(bounds)

    (height, width) = arr_moving.shape

    if arr_ref_ns is None and similarity_measure == 'neighbour_similarity':

        arr_ref_ns = get_neighbour_similarity(arr_ref)

    

    if basinhopping:

        if 'T' not in basinhopping_kwargs:

            basinhopping_kwargs['T'] = 0.5

        if 'take_step' not in basinhopping_kwargs:

            basinhopping_kwargs['take_step'] = RandomDisplacementBounds(xmin=0, xmax=1, stepsize=0.5)

        if 'minimizer_kwargs' in basinhopping_kwargs:

            if 'method' not in basinhopping_kwargs['minimizer_kwargs']:

                basinhopping_kwargs['minimizer_kwargs']['method'] = local_optimisation_method

        else:

            basinhopping_kwargs['minimizer_kwargs'] = {'method': local_optimisation_method}

    

    def _chosen_similarity_measure(params: np.array):

        if similarity_measure == 'neighbour_similarity':

            return similarity_measure_using_neighbour_similarity(arr_moving, arr_ref, arr_ref_ns, params, debug=debug, max_groups=ns_max_groups)

        elif similarity_measure == 'overlap':

            return similarity_measure_area_of_overlap(arr_ref, arr_moving, params)

        else:

            return similarity_measure_after_transform(arr_ref, arr_moving, params)

            

    def _error_metric_after_transform(params_scaled):

        transform_params = recover_parameters_from_scaled_guess(params_scaled, bounds_list, lock_strings)

        sm = _chosen_similarity_measure(transform_params)

        outside = outside_bounds(transform_params, bounds_list)

        error_metric = 1/(sm + 1)

        if outside:

            fitted_params = fit_to_bounds(transform_params, bounds_list)

            sm_fitted = _chosen_similarity_measure(fitted_params)

            error_metric = np.float_(np.finfo(np.float_).max)

            if sm_fitted > 0:

                error_metric = 1/sm_fitted

        if debug:

            print((1/error_metric, list_free_parameters(transform_params, lock_strings)))

        if outside:

            assert 1/error_metric <= 1

        else:

            assert 1/error_metric >= 1

        return error_metric

    

    guess = list_free_parameters_scaled_to_bounds(params, bounds_list, lock_strings)

    

    if basinhopping:

        optimisation_result = scipy.optimize.basinhopping(_error_metric_after_transform, guess, **basinhopping_kwargs)

    else:

        optimisation_result = scipy.optimize.minimize(_error_metric_after_transform, guess, method=local_optimisation_method)

    if debug:

        print(optimisation_result)

    optimised_parameters = optimisation_result.x.reshape(optimisation_result.x.size)

    return recover_parameters_from_scaled_guess(optimised_parameters, bounds_list, lock_strings)



        