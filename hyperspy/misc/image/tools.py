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

