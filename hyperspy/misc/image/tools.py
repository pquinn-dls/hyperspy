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
