# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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
import numpy.ma as ma
import dask.array as da
import scipy as sp
import logging
import warnings
try:
    # For scikit-image >= 0.17.0
    from skimage.registration._phase_cross_correlation import _upsampled_dft
except ModuleNotFoundError:
    from skimage.feature.register_translation import _upsampled_dft

from hyperspy.defaults_parser import preferences
from hyperspy.external.progressbar import progressbar
from hyperspy.misc.math_tools import symmetrize, antisymmetrize, optimal_fft_size
from hyperspy.signal import BaseSignal
from hyperspy._signals.lazy import LazySignal
from hyperspy._signals.common_signal2d import CommonSignal2D
from hyperspy.docstrings.plot import (
    BASE_PLOT_DOCSTRING, PLOT2D_DOCSTRING, KWARGS_DOCSTRING)
from hyperspy.docstrings.signal import SHOW_PROGRESSBAR_ARG, PARALLEL_ARG
from hyperspy.misc.image.tools_shift import (estimate_image_shift_mi,
                                             estimate_image_shift,shift_image)


_logger = logging.getLogger(__name__)


class Signal2D(BaseSignal, CommonSignal2D):

    """
    """
    _signal_dimension = 2
    _lazy = False

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if self.axes_manager.signal_dimension != 2:
            self.axes_manager.set_signal_dimension(2)

    def plot(self,
             colorbar=True,
             scalebar=True,
             scalebar_color="white",
             axes_ticks=None,
             axes_off=False,
             saturated_pixels=None,
             vmin=None,
             vmax=None,
             gamma=1.0,
             no_nans=False,
             centre_colormap="auto",
             min_aspect=0.1,
             **kwargs
             ):
        """%s
        %s
        %s

        """
        super(Signal2D, self).plot(
            colorbar=colorbar,
            scalebar=scalebar,
            scalebar_color=scalebar_color,
            axes_ticks=axes_ticks,
            axes_off=axes_off,
            saturated_pixels=saturated_pixels,
            vmin=vmin,
            vmax=vmax,
            gamma=gamma,
            no_nans=no_nans,
            centre_colormap=centre_colormap,
            min_aspect=min_aspect,
            **kwargs
        )
    plot.__doc__ %= (BASE_PLOT_DOCSTRING, PLOT2D_DOCSTRING, KWARGS_DOCSTRING)

    def create_model(self, dictionary=None):
        """Create a model for the current signal

        Parameters
        ----------
        dictionary : {None, dict}, optional
            A dictionary to be used to recreate a model. Usually generated
            using :meth:`hyperspy.model.as_dictionary`

        Returns
        -------
        A Model class

        """
        from hyperspy.models.model2d import Model2D
        return Model2D(self, dictionary=dictionary)

    def estimate_shift2D(self,
                         reference='current',
                         correlation_threshold=None,
                         chunk_size=30,
                         roi=None,
                         normalize_corr=False,
                         sobel=True,
                         medfilter=True,
                         hanning=True,
                         plot=False,
                         dtype='float',
                         show_progressbar=None,
                         sub_pixel_factor=1):
        """Estimate the shifts in an image using phase correlation.

        This method can only estimate the shift by comparing
        bi-dimensional features that should not change position
        between frames. To decrease the memory usage, the time of
        computation and the accuracy of the results it is convenient
        to select a region of interest by setting the ``roi`` argument.

        Parameters
        ----------
        reference : {'current', 'cascade' ,'stat'}
            If 'current' (default) the image at the current
            coordinates is taken as reference. If 'cascade' each image
            is aligned with the previous one. If 'stat' the translation
            of every image with all the rest is estimated and by
            performing statistical analysis on the result the
            translation is estimated.
        correlation_threshold : {None, 'auto', float}
            This parameter is only relevant when reference='stat'.
            If float, the shift estimations with a maximum correlation
            value lower than the given value are not used to compute
            the estimated shifts. If 'auto' the threshold is calculated
            automatically as the minimum maximum correlation value
            of the automatically selected reference image.
        chunk_size : {None, int}
            If int and reference='stat' the number of images used
            as reference are limited to the given value.
        roi : tuple of ints or floats (left, right, top, bottom)
            Define the region of interest. If int(float) the position
            is given axis index(value). Note that ROIs can be used
            in place of a tuple.
        normalize_corr : bool, default False
            If True, use phase correlation to align the images, otherwise
            use cross correlation.
        sobel : bool, default True
            Apply a Sobel filter for edge enhancement
        medfilter : bool, default True
            Apply a median filter for noise reduction
        hanning : bool, default True
            Apply a 2D hanning filter
        plot : bool or 'reuse'
            If True plots the images after applying the filters and
            the phase correlation. If 'reuse', it will also plot the images,
            but it will only use one figure, and continuously update the images
            in that figure as it progresses through the stack.
        dtype : str or dtype
            Typecode or data-type in which the calculations must be
            performed.
        %s
        sub_pixel_factor : float
            Estimate shifts with a sub-pixel accuracy of 1/sub_pixel_factor
            parts of a pixel. Default is 1, i.e. no sub-pixel accuracy.

        Returns
        -------
        shifts : list of array
            List of estimated shifts

        Notes
        -----
        The statistical analysis approach to the translation estimation
        when using ``reference='stat'`` roughly follows [Schaffer2004]_.
        If you use it please cite their article.

        References
        ----------
        .. [Schaffer2004] Schaffer, Bernhard, Werner Grogger, and Gerald Kothleitner.
           “Automated Spatial Drift Correction for EFTEM Image Series.”
           Ultramicroscopy 102, no. 1 (December 2004): 27–36.

        See Also
        --------
        * :py:meth:`~._signals.signal2d.Signal2D.align2D`

        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_two()
        if roi is not None:
            # Get the indices of the roi
            yaxis = self.axes_manager.signal_axes[1]
            xaxis = self.axes_manager.signal_axes[0]
            roi = tuple([xaxis._get_index(i) for i in roi[2:]] +
                        [yaxis._get_index(i) for i in roi[:2]])

        ref = None if reference == 'cascade' else \
            self.__call__().copy()
        shifts = []
        nrows = None
        images_number = self.axes_manager._max_index + 1
        if plot == 'reuse':
            # Reuse figure for plots
            plot = plt.figure()
        if reference == 'stat':
            nrows = images_number if chunk_size is None else \
                min(images_number, chunk_size)
            pcarray = ma.zeros((nrows, self.axes_manager._max_index + 1,
                                ),
                               dtype=np.dtype([('max_value', np.float),
                                               ('shift', np.int32,
                                                (2,))]))
            nshift, max_value = estimate_image_shift(
                self(),
                self(),
                roi=roi,
                sobel=sobel,
                medfilter=medfilter,
                hanning=hanning,
                normalize_corr=normalize_corr,
                plot=plot,
                dtype=dtype,
                sub_pixel_factor=sub_pixel_factor)
            np.fill_diagonal(pcarray['max_value'], max_value)
            pbar_max = nrows * images_number
        else:
            pbar_max = images_number

        # Main iteration loop. Fills the rows of pcarray when reference
        # is stat
        with progressbar(total=pbar_max,
                         disable=not show_progressbar,
                         leave=True) as pbar:
            for i1, im in enumerate(self._iterate_signal()):
                if reference in ['current', 'cascade']:
                    if ref is None:
                        ref = im.copy()
                        shift = np.array([0, 0])
                    nshift, max_val = estimate_image_shift(
                        ref, im, roi=roi, sobel=sobel, medfilter=medfilter,
                        hanning=hanning, plot=plot,
                        normalize_corr=normalize_corr, dtype=dtype,
                        sub_pixel_factor=sub_pixel_factor)
                    if reference == 'cascade':
                        shift += nshift
                        ref = im.copy()
                    else:
                        shift = nshift
                    shifts.append(shift.copy())
                    pbar.update(1)
                elif reference == 'stat':
                    if i1 == nrows:
                        break
                    # Iterate to fill the columns of pcarray
                    for i2, im2 in enumerate(
                            self._iterate_signal()):
                        if i2 > i1:
                            nshift, max_value = estimate_image_shift(
                                im,
                                im2,
                                roi=roi,
                                sobel=sobel,
                                medfilter=medfilter,
                                hanning=hanning,
                                normalize_corr=normalize_corr,
                                plot=plot,
                                dtype=dtype,
                                sub_pixel_factor=sub_pixel_factor)
                            pcarray[i1, i2] = max_value, nshift
                        del im2
                        pbar.update(1)
                    del im
        if reference == 'stat':
            # Select the reference image as the one that has the
            # higher max_value in the row
            sqpcarr = pcarray[:, :nrows]
            sqpcarr['max_value'][:] = symmetrize(sqpcarr['max_value'])
            sqpcarr['shift'][:] = antisymmetrize(sqpcarr['shift'])
            ref_index = np.argmax(pcarray['max_value'].min(1))
            self.ref_index = ref_index
            shifts = (pcarray['shift'] +
                      pcarray['shift'][ref_index, :nrows][:, np.newaxis])
            if correlation_threshold is not None:
                if correlation_threshold == 'auto':
                    correlation_threshold = \
                        (pcarray['max_value'].min(0)).max()
                    _logger.info("Correlation threshold = %1.2f",
                                 correlation_threshold)
                shifts[pcarray['max_value'] <
                       correlation_threshold] = ma.masked
                shifts.mask[ref_index, :] = False

            shifts = shifts.mean(0)
        else:
            shifts = np.array(shifts)
            del ref
        return shifts

    estimate_shift2D.__doc__ %= SHOW_PROGRESSBAR_ARG


    def estimate_shift2D_mutual(self,
                       reference='current',
                       chunk_size=30,
                       roi=None,
                       medfilter=False,
                       plot=False,
                       dtype='float',
                       bin_rule='scotts',
                       method="Powell",
                       use_pyramid_levels=True,
                       pyramid_scale=2,
                       pyramid_minimum_size=16,
                       fill_value=0.0,
                       interpolation_order=1,
                       show_progressbar=None):
        """Estimate the shifts in an image using phase correlation.

        This method can only estimate the shift by comparing
        bi-dimensional features that should not change position
        between frames. To decrease the memory usage, the time of
        computation and the accuracy of the results it is convenient
        to select a region of interest by setting the ``roi`` argument.

        Parameters
        ----------
        reference : {'current', 'cascade' ,'stat'}
            If 'current' (default) the image at the current
            coordinates is taken as reference. If 'cascade' each image
            is aligned with the previous one. If 'stat' the translation
            of every image with all the rest is estimated and by
            performing statistical analysis on the result the
            translation is estimated.
        chunk_size : {None, int}
            If int and reference='stat' the number of images used
            as reference are limited to the given value.
        roi : tuple of ints or floats (left, right, top, bottom)
            Define the region of interest. If int(float) the position
            is given axis index(value). Note that ROIs can be used
            in place of a tuple.
        medfilter : bool, default True
            Apply a median filter for noise reduction
        hanning : bool, default True
            Apply a 2D hanning filter
        plot : bool or 'reuse'
            If True plots the images after applying the filters and
            the phase correlation. If 'reuse', it will also plot the images,
            but it will only use one figure, and continuously update the images
            in that figure as it progresses through the stack.
        dtype : str or dtype
            Typecode or data-type in which the calculations must be
            performed.

        Returns
        -------
        shifts : list of array
            List of estimated shifts

        Notes
        -----
        The statistical analysis approach to the translation estimation
        when using ``reference='stat'`` roughly follows [Schaffer2004]_.
        If you use it please cite their article.

        References
        ----------
        .. [Schaffer2004] Schaffer, Bernhard, Werner Grogger, and Gerald Kothleitner.
           “Automated Spatial Drift Correction for EFTEM Image Series.”
           Ultramicroscopy 102, no. 1 (December 2004): 27–36.

        See Also
        --------
        * :py:meth:`~._signals.signal2d.Signal2D.align2D`

        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        self._check_signal_dimension_equals_two()
        if roi is not None:
            # Get the indices of the roi
            yaxis = self.axes_manager.signal_axes[1]
            xaxis = self.axes_manager.signal_axes[0]
            roi = tuple([xaxis._get_index(i) for i in roi[2:]] +
                        [yaxis._get_index(i) for i in roi[:2]])

        ref = None if reference == 'cascade' else \
            self.__call__().copy()
        shifts = []
        nrows = None
        images_number = self.axes_manager._max_index + 1
        if plot == 'reuse':
            # Reuse figure for plots
            plot = plt.figure()
        if reference == 'stat':
            nrows = images_number if chunk_size is None else \
                min(images_number, chunk_size)
            pcarray = ma.zeros((nrows, self.axes_manager._max_index + 1,
                                ),
                               dtype=np.dtype([('max_value', np.float),
                                               ('shift', np.int32,
                                                (2,))]))
            nshift, max_value = estimate_image_shift_mi(
                self(),
                self(),
                roi=roi,
                medfilter=medfilter,
                dtype=dtype,
                bin_rule=bin_rule,
                use_pyramid_levels=use_pyramid_levels,
                pyramid_scale=pyramid_scale,
                pyramid_minimum_size=pyramid_minimum_size,
                cval=fill_value,
                order=interpolation_order)
            np.fill_diagonal(pcarray['max_value'], max_value)
            pbar_max = nrows * images_number
        else:
            pbar_max = images_number

        # Main iteration loop. Fills the rows of pcarray when reference
        # is stat
        with progressbar(total=pbar_max,
                         disable=not show_progressbar,
                         leave=True) as pbar:
            for i1, im in enumerate(self._iterate_signal()):
                if reference in ['current', 'cascade']:
                    if ref is None:
                        ref = im.copy()
                        shift = np.array([0, 0])
                    nshift, max_val = estimate_image_shift_mi(
                        ref, im, roi=roi, medfilter=medfilter,
                        dtype=dtype,
                        bin_rule=bin_rule,
                        use_pyramid_levels=use_pyramid_levels,
                        pyramid_scale=pyramid_scale,
                        pyramid_minimum_size=pyramid_minimum_size,
                        cval=fill_value,
                        order=interpolation_order)
                    if reference == 'cascade':
                        shift += nshift
                        ref = im.copy()
                    else:
                        shift = nshift
                    shifts.append(shift.copy())
                    pbar.update(1)
                elif reference == 'stat':
                    if i1 == nrows:
                        break
                    # Iterate to fill the columns of pcarray
                    for i2, im2 in enumerate(
                            self._iterate_signal()):
                        if i2 > i1:
                            nshift, max_value = estimate_image_shift_mi(
                                im,
                                im2,
                                roi=roi,
                                medfilter=medfilter,
                                dtype=dtype,
                                use_pyramid_levels=use_pyramid_levels,
                                pyramid_scale=pyramid_scale,
                                pyramid_minimum_size=pyramid_minimum_size,
                                cval=fill_value,
                                order=interpolation_order)
                            pcarray[i1, i2] = max_value, nshift
                        del im2
                        pbar.update(1)
                    del im
        if reference == 'stat':
            # Select the reference image as the one that has the
            # higher max_value in the row
            sqpcarr = pcarray[:, :nrows]
            sqpcarr['max_value'][:] = symmetrize(sqpcarr['max_value'])
            sqpcarr['shift'][:] = antisymmetrize(sqpcarr['shift'])
            ref_index = np.argmax(pcarray['max_value'].min(1))
            self.ref_index = ref_index
            shifts = (pcarray['shift'] +
                      pcarray['shift'][ref_index, :nrows][:, np.newaxis])

            shifts = shifts.mean(0)
        else:
            shifts = np.array(shifts)
            del ref
        return shifts


    def align2D(
        self,
        crop=True,
        fill_value=np.nan,
        shifts=None,
        expand=False,
        interpolation_order=1,
        show_progressbar=None,
        parallel=None,
        max_workers=None,
        metric = 'phase',
        **kwargs,
    ):
        """Align the images in-place using :py:func:`scipy.ndimage.shift`.

        The images can be aligned using either user-provided shifts or
        by first estimating the shifts.

        See :py:meth:`~._signals.signal2d.Signal2D.estimate_shift2D`
        for more details on estimating image shifts.

        Parameters
        ----------
        crop : bool
            If True, the data will be cropped not to include regions
            with missing data
        fill_value : int, float, nan
            The areas with missing data are filled with the given value.
            Default is nan.
        shifts : None or list of tuples
            If None the shifts are estimated using
            :py:meth:`~._signals.signal2D.estimate_shift2D`.
        expand : bool
            If True, the data will be expanded to fit all data after alignment.
            Overrides `crop`.
        interpolation_order: int, default 1.
            The order of the spline interpolation. Default is 1, linear
            interpolation.
        %s
        %s
        %s
        **kwargs :
            Keyword arguments passed to :py:meth:`~._signals.signal2d.Signal2D.estimate_shift2D`

        Returns
        -------
        shifts : np.array
            The estimated shifts are returned only if ``shifts`` is None

        See Also
        --------
        * :py:meth:`~._signals.signal2d.Signal2D.estimate_shift2D`

        """
        self._check_signal_dimension_equals_two()

        return_shifts = False

        if shifts is None:
            if metric == "phase":
                shifts = self.estimate_shift2D(**kwargs)
                return_shifts = True
            elif metric == "mutual":
                shifts = self.estimate_shift2D_mutual(**kwargs)
                return_shifts = True
            if not np.any(shifts):
                warnings.warn(
                    "The estimated shifts are all zero, suggesting "
                    "the images are already aligned",
                    UserWarning,
                )
                return shifts

        elif not np.any(shifts):
            warnings.warn(
                "The provided shifts are all zero, no alignment done",
                UserWarning,
            )
            return None

        if expand:
            # Expand to fit all valid data
            left, right = (
                int(np.floor(shifts[:, 1].min())) if shifts[:, 1].min() < 0 else 0,
                int(np.ceil(shifts[:, 1].max())) if shifts[:, 1].max() > 0 else 0,
            )
            top, bottom = (
                int(np.floor(shifts[:, 0].min())) if shifts[:, 0].min() < 0 else 0,
                int(np.ceil(shifts[:, 0].max())) if shifts[:, 0].max() > 0 else 0,
            )
            xaxis = self.axes_manager.signal_axes[0]
            yaxis = self.axes_manager.signal_axes[1]
            padding = []

            for i in range(self.data.ndim):
                if i == xaxis.index_in_array:
                    padding.append((right, -left))
                elif i == yaxis.index_in_array:
                    padding.append((bottom, -top))
                else:
                    padding.append((0, 0))

            self.data = np.pad(
                self.data, padding, mode="constant", constant_values=(fill_value,)
            )

            if left < 0:
                xaxis.offset += left * xaxis.scale
            if np.any((left < 0, right > 0)):
                xaxis.size += right - left
            if top < 0:
                yaxis.offset += top * yaxis.scale
            if np.any((top < 0, bottom > 0)):
                yaxis.size += bottom - top

        # Translate, with sub-pixel precision if necesary,
        # note that we operate in-place here
        self._map_iterate(
            shift_image,
            iterating_kwargs=(("shift", -shifts),),
            show_progressbar=show_progressbar,
            parallel=parallel,
            max_workers=max_workers,
            ragged=False,
            inplace=True,
            fill_value=fill_value,
            interpolation_order=interpolation_order,
        )

        if crop and not expand:
            max_shift = np.max(shifts, axis=0) - np.min(shifts, axis=0)

            if np.any(max_shift >= np.array(self.axes_manager.signal_shape)):
                raise ValueError("Shift outside range of signal axes. Cannot crop signal.")

            # Crop the image to the valid size
            shifts = -shifts
            bottom, top = (
                int(np.floor(shifts[:, 0].min())) if shifts[:, 0].min() < 0 else None,
                int(np.ceil(shifts[:, 0].max())) if shifts[:, 0].max() > 0 else 0,
            )
            right, left = (
                int(np.floor(shifts[:, 1].min())) if shifts[:, 1].min() < 0 else None,
                int(np.ceil(shifts[:, 1].max())) if shifts[:, 1].max() > 0 else 0,
            )
            self.crop_image(top, bottom, left, right)
            shifts = -shifts

        self.events.data_changed.trigger(obj=self)

        if return_shifts:
            return shifts

    def crop_image(self, top=None, bottom=None,
                   left=None, right=None, convert_units=False):
        """Crops an image in place.

        Parameters
        ----------
        top, bottom, left, right : {int | float}
            If int the values are taken as indices. If float the values are
            converted to indices.
        convert_units : bool
            Default is False
            If True, convert the signal units using the 'convert_to_units'
            method of the 'axes_manager'. If False, does nothing.

        See also
        --------
        crop

        """
        self._check_signal_dimension_equals_two()
        self.crop(self.axes_manager.signal_axes[1].index_in_axes_manager,
                  top,
                  bottom)
        self.crop(self.axes_manager.signal_axes[0].index_in_axes_manager,
                  left,
                  right)
        if convert_units:
            self.axes_manager.convert_units('signal')

    def add_ramp(self, ramp_x, ramp_y, offset=0):
        """Add a linear ramp to the signal.

        Parameters
        ----------
        ramp_x: float
            Slope of the ramp in x-direction.
        ramp_y: float
            Slope of the ramp in y-direction.
        offset: float, optional
            Offset of the ramp at the signal fulcrum.

        Notes
        -----
            The fulcrum of the linear ramp is at the origin and the slopes are
            given in units of the axis with the according scale taken into
            account. Both are available via the `axes_manager` of the signal.

        """
        yy, xx = np.indices(self.axes_manager._signal_shape_in_array)
        if self._lazy:
            import dask.array as da
            ramp = offset * da.ones(self.data.shape, dtype=self.data.dtype,
                                    chunks=self.data.chunks)
        else:
            ramp = offset * np.ones(self.data.shape, dtype=self.data.dtype)
        ramp += ramp_x * xx
        ramp += ramp_y * yy
        self.data += ramp


    def crop_image(self, top=None, bottom=None,
                   left=None, right=None, convert_units=False):
        """Crops an image in place.

        Parameters
        ----------
        top, bottom, left, right : {int | float}
            If int the values are taken as indices. If float the values are
            converted to indices.
        convert_units : bool
            Default is False
            If True, convert the signal units using the 'convert_to_units'
            method of the 'axes_manager'. If False, does nothing.

        See also
        --------
        crop

        """
        self._check_signal_dimension_equals_two()
        self.crop(self.axes_manager.signal_axes[1].index_in_axes_manager,
                  top,
                  bottom)
        self.crop(self.axes_manager.signal_axes[0].index_in_axes_manager,
                  left,
                  right)
        if convert_units:
            self.axes_manager.convert_units('signal')

    def add_ramp(self, ramp_x, ramp_y, offset=0):
        """Add a linear ramp to the signal.

        Parameters
        ----------
        ramp_x: float
            Slope of the ramp in x-direction.
        ramp_y: float
            Slope of the ramp in y-direction.
        offset: float, optional
            Offset of the ramp at the signal fulcrum.

        Notes
        -----
            The fulcrum of the linear ramp is at the origin and the slopes are
            given in units of the axis with the according scale taken into
            account. Both are available via the `axes_manager` of the signal.

        """
        yy, xx = np.indices(self.axes_manager._signal_shape_in_array)
        if self._lazy:
            import dask.array as da
            ramp = offset * da.ones(self.data.shape, dtype=self.data.dtype,
                                    chunks=self.data.chunks)
        else:
            ramp = offset * np.ones(self.data.shape, dtype=self.data.dtype)
        ramp += ramp_x * xx
        ramp += ramp_y * yy
        self.data += ramp



class LazySignal2D(LazySignal, Signal2D):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
