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
from hyperspy.misc.math.stats.histogram_tools import histogram2d
from scipy import stats
_logger = logging.getLogger(__name__)


def mutual_information(arr1, arr2, norm=False,
                       bin_rule="scotts"):
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
