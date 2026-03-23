import math
from math import cos, sin
import json
from functools import wraps
from astropy.convolution.convolve import convolve_fft
from astropy.convolution.kernels import Gaussian2DKernel

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.utils import isiterable
from fast_histogram import histogram2d

from scanning.coordinates import TelescopePattern

class Simulation():

    # other attributes (for development)
    # _sky_hist, _sky_hist_rem
    # _det_hist, _det_hist_rem
    # _time_hist, _time_hist_rem

    # _param_unit
    # _sky_pattern
    # _field_rotation (degrees)
    # _module
    # _mem_limit

    # _pxan
    # _param: ...
    # _prep: max_pixel, x_range_pxan, y_range_pxan, num_pxan
    # _det_mask
    # _validity_mask

    # what unit each parameter is stored in
    _param_units = {'pixel_size': u.deg, 'max_acc': u.deg/u.s/u.s, 'min_speed': u.deg/u.s, 'det_radius': u.deg, 'det_lim': u.deg, 'det_list': u.deg, 'pxan_list': u.deg, 'pxan_lim': u.deg}

    # INITIALIZATION

    def __init__(self, telescope_pattern, instrument, module_identifier, sim_param=None, **kwargs) -> None:
        """
        Creating a `Simulation` object works by passing a `TelescopePattern` object, which stores the AZ/EL coordinates of 
        the telescope boresight. By passing an `Instrument` object and the name of one of its modules, we use the detector 
        element positions of that camera module and use the coordinates of the scan pattern that that camera module would make. 
        
        Use **kwargs to specify conditions. Note that if multiple detector filters are used (`pols_on`, `det_lim`, etc.), 
        only detector elements that satisfy all will be used. 

        Parameters
        ----------------------------
        telescope_pattern : TelescopePattern
            A `TelescopePattern` object representing the boresight. 
        instrument : Instrument
            `Instrument` object to get `module` from. 
        module_identifier : str
            Gets the `Module` and AZ/ALT coordinates from a string indicating a module name in the instrument e.g. 'SFH'. 
        sim_param : str
            File path to a json file containing parameters for simulation (see **kwargs).
            

        Keyword Args
        --------------------------
        mem_limit : int; default 640 MB
            Approximate memory limit in megabytes when simulating the scan. 
        pixel_size : float/Quantity/str; default 10 arcsec, default unit arcsec
            Length of a square pixel in x-y offsets. 
        max_acc : float/Quantity/str; default None, default unit deg/s^2
            Maximum absolute accleration in x-y offsets. If `None`, all points are considered valid.
        min_acc : float/Quantity/str; default None, default unit deg/s
            Minimum speed in x-y offsets. If `None`, all points are considered valid.
        
        pols_on : int, int sequence or None; default None FIXME allow other units
            Which polarization geometries (in degrees) of the detector elements to keep on. If `None`, include all. 
        rhombi_on : int, int sequence or None; default None
            Which rhombi to keep on. If `None`, include all rhombi. 
        wafers_on : int, int sequence or None; default None
            Which wafers to keep on. If `None`, include all wafers. 
        det_radius : two-length tuple or None; default None; default unit arcsec
            Take the (min_radius, max_radius) of detector elements to keep. Use `None` in the tuple for an unspecified limit. If `None`, include all. 
        det_lim : two-length sequence of two-length tuples or None; default None; default unit arcsec
            Take the [(x_min, x_max), (y_min, y_max)] range of detector elements to keep. Use `None` in the tuple for an unspecified limit. If `None`, include all.
            FIXME is this with mod_rot? instr_rot? field rotation? 
        det_list : sequence of int, sequence of two-length tuples, or None; default None 
            If a sequence of `int`, a list of detector numbers to keep on exact detector elements.
            If a sequence of two-length tuples, a list of (x, y) positions to keep on the detector element closest to each position; default unit arcsec
            If `None, include all. 
            FIXME is this with mod_rot? instr_rot? field rotation?
        
        pxan_lim : two-length sequence of two-length tuples or None; default None; default unit arcsec
            A recutangular range of x-y pixels within [(xmin, xmax), (ymin, ymax)] to perform analysis.
            Cannot be used with `pxan_list`. If both are `None`, no pixel analysis is performed.
        pxan_list : sequence of two-length tuples, or None; default None; default unit arcsec
            A list of pixels such as [(x1, y1), (x2, y2)] to analyze. Each point (x, y) corresponds to a singular pixel that contains this point. 
            Cannot be used with `pxan_lim`. If both are `None`, no pixel analysis is performed.
        max_pixel : int;
            In order to externally give the number of pixel of the simulation (length will be max_pixel*2)
            If None, calculated from the range of the data.
        high_precision : boolean;
            If True, float64 is used for the coordinates. If False, float32 is used
            for a map with max_pixel > 256, otherwise float16 is used. Default is True
        ra_c, dec_c : float;
            Reference coordinates in degrees where the pixel (0, 0) corresponds
            If set to None, the first data point is used
        """

        # get Module object from instrument
        self._module = instrument.get_module(module_identifier, with_instr_rot=True, with_mod_rot=True)

        # get the telescope pattern
        module_loc = instrument.get_module_location(module_identifier, from_boresight=True)
        telescope_pattern = telescope_pattern.view_module(module_loc, includes_instr_offset=True)
        
        ra_c = kwargs.pop('ra_c', None)
        dec_c = kwargs.pop('dec_c', None)

        self._sky_pattern = telescope_pattern.get_sky_pattern(ra_c, dec_c)
        self._field_rotation = telescope_pattern.rot_angle.value

        # clean parameters
        if not sim_param is None:
            if kwargs:
                raise TypeError('passed "sim_param" but also passed keywords')
            with open(sim_param, 'r') as f:
                kwargs = json.load(f)

        self._param = self._clean_param(**kwargs)
        self._prep = self._preprocessing()

        # filer detector elements
        self._det_mask = self._filter_detectors()
        self._validity_mask = self._filter_ts()

        # initialize histograms
        self._sky_hist = self._sky_hist_rem = self._det_hist = self._det_hist_rem = self._time_hist = self._time_hist_rem = self._pol_hist = self._pol_hist_rem = None

    def _clean_param(self, **kwargs):
        new_kwargs = dict()

        new_kwargs['mem_limit'] = kwargs.pop('mem_limit', 640)

        new_kwargs['pixel_size'] = u.Quantity(kwargs.pop('pixel_size', 10), u.arcsec).to(self._param_units['pixel_size']).value

        max_acc = kwargs.pop('max_acc', None)
        if not max_acc is None:
            new_kwargs['max_acc'] = u.Quantity(max_acc, self._param_units['max_acc']).value

        min_speed = kwargs.pop('min_speed', None)
        if not min_speed is None:
            new_kwargs['min_speed'] = u.Quantity(min_speed, self._param_units['min_speed']).value

        # parameters for keeping on certain parts of the module

        pols_on = kwargs.pop('pols_on', None)
        if not pols_on is None:
            all_pols = np.unique(self._module.pol.value)
            if not isiterable(pols_on):
                pols_on = [pols_on]
            if not set(pols_on).issubset(all_pols):
                raise ValueError(f'pols_on = {pols_on} is not a subset of available polarizations {all_pols}')
            new_kwargs['pols_on'] = pols_on
  
        rhombi_on = kwargs.pop('rhombi_on', None)
        if not rhombi_on is None:
            all_rhombi = np.unique(self._module.rhombus)
            if not isiterable(rhombi_on):
                rhombi_on = [rhombi_on]
            if not set(rhombi_on).issubset(all_rhombi):
                raise ValueError(f'rhombi_on = {rhombi_on} is not a subset of available rhombuses {all_rhombi}')
            new_kwargs['rhombi_on'] = rhombi_on

        wafers_on = kwargs.pop('wafers_on', None)
        if not wafers_on is None:
            all_wafers = np.unique(self._module.wafer)
            if not isiterable(wafers_on):
                wafers_on = [wafers_on]
            if not set(wafers_on).issubset(all_wafers):
                raise ValueError(f'wafers_on = {wafers_on} is not a subset of available wafers {all_wafers}')
            new_kwargs['wafers_on'] = wafers_on
        
        det_radius = kwargs.pop('det_radius', None)
        if not det_radius is None:
            try:
                min_radius = det_radius[0]
                min_radius = -math.inf if min_radius is None else u.Quantity(min_radius, u.arcsec).to(self._param_units['det_radius']).value
                max_radius = det_radius[1]
                max_radius = math.inf if max_radius is None else u.Quantity(max_radius, u.arcsec).to(self._param_units['det_radius']).value
            except (IndexError, TypeError):
                raise TypeError('"det_radius" is of incorrect shape')
            new_kwargs['det_radius'] = (min_radius, max_radius)
        
        det_lim = kwargs.pop('det_lim', None)
        if not det_lim is None:
            try:
                min_x = det_lim[0][0]
                min_x = -math.inf if min_x is None else u.Quantity(min_x, u.arcsec).to(self._param_units['det_lim']).value
                max_x = det_lim[0][1]
                max_x = math.inf if max_x is None else u.Quantity(max_x, u.arcsec).to(self._param_units['det_lim']).value
                min_y = det_lim[1][0]
                min_y = -math.inf if min_y is None else u.Quantity(min_y, u.arcsec).to(self._param_units['det_lim']).value
                max_y = det_lim[1][1]
                max_y = math.inf if max_y is None else u.Quantity(max_y, u.arcsec).to(self._param_units['det_lim']).value
            except (IndexError, TypeError):
                raise TypeError('"det_lim" is of incorrect shape or type')
            new_kwargs['det_lim'] = ((min_x, max_x), (min_y, max_y))
        
        det_list = kwargs.pop('det_list', None)
        if not det_list is None:
            try:
                new_det_list = []
                for point in det_list:
                    new_det_list.append( (u.Quantity(point[0], u.arcsec).to(self._param_units['det_list']).value, u.Quantity(point[1], u.arcsec).to(self._param_units['det_list']).value) )
                new_kwargs['det_list'] = np.array(new_det_list)
            except TypeError:
                if not (len(np.shape(det_list)) == 1):
                    raise TypeError(f'"det_list" is of incorrect shape or type')
                new_kwargs['det_list'] = np.array(det_list, dtype=int)

        # for pixel analysis

        self._pxan = False
        pxan_list = kwargs.pop('pxan_list', None)
        pxan_lim = kwargs.pop('pxan_lim', None)

        if not pxan_list is None and not pxan_lim is None:
            raise TypeError('pxan_list and pxan_lim cannot be used at the same time')

        elif not pxan_lim is None:
            self._pxan = True
            try:
                x_min = u.Quantity(pxan_lim[0][0], u.arcsec).to(u.deg).value
                x_max = u.Quantity(pxan_lim[0][1], u.arcsec).to(u.deg).value
                y_min = u.Quantity(pxan_lim[1][0], u.arcsec).to(u.deg).value
                y_max = u.Quantity(pxan_lim[1][1], u.arcsec).to(u.deg).value
            except (IndexError, TypeError):
                raise TypeError('"pxan_lim" is of incorrect shape or type')
            new_kwargs['pxan_lim']= [(x_min, x_max), (y_min, y_max)]

        elif not pxan_list is None:
            self._pxan = True
            new_pxan_list = []
            try:
                for px in pxan_list:
                    x = u.Quantity(px[0], u.arcsec).to(u.deg).value
                    y = u.Quantity(px[1], u.arcsec).to(u.deg).value
                    new_pxan_list.append( (x, y) )
            except (IndexError, TypeError):
                raise TypeError('"pxan_lim" is of incorrect shape or type')
            new_kwargs['pxan_list'] = new_pxan_list

        # max_pixel
        max_pixel = kwargs.pop('max_pixel', None)
        new_kwargs['max_pixel'] = max_pixel

        # precision
        high_precision = kwargs.pop('high_precision', True)
        new_kwargs['high_precision'] = high_precision

        # return
        if kwargs:
            raise TypeError(f'uncessary keywords: {kwargs.keys()}')

        return new_kwargs

    def _preprocessing(self):
        pixel_size = self._param['pixel_size']

        if self._param['max_pixel'] is None:
            farthest_det_elem = math.sqrt(max( (self._module.x.value)**2 + (self._module.y.value)**2 ))
            #farthest_ts = math.sqrt(max(self._sky_pattern.distance.value))
            farthest_ts = max(self._sky_pattern.distance.value)
            max_pixel = math.ceil((farthest_det_elem + farthest_ts)/pixel_size)
        else:
            max_pixel = self._param['max_pixel']

        if self._pxan:

            pxan_lim = self._param.get('pxan_lim')
            pxan_list = self._param.get('pxan_list')

            if not pxan_lim is None:
                x_min = math.floor(pxan_lim[0][0]/pixel_size)
                x_max = math.ceil(pxan_lim[0][1]/pixel_size)
                y_min = math.floor(pxan_lim[1][0]/pixel_size)
                y_max = math.ceil(pxan_lim[1][1]/pixel_size)
                x_range_pxan = [ (x_min, x_max) ]
                y_range_pxan = [ (y_min, y_max) ]
                num_pxan = (x_max-x_min)*(y_max-y_min)
            else:
                x_range_pxan = []
                y_range_pxan = []
                for px in pxan_list:
                    x_min = math.floor(px[0]/pixel_size)
                    x_range_pxan.append( (x_min, x_min + 1) )

                    y_min = math.floor(px[1]/pixel_size)
                    y_range_pxan.append( (y_min, y_min + 1) )
                
                num_pxan = len(x_range_pxan)
        else:
            x_range_pxan = y_range_pxan = num_pxan = None
        

        return {'max_pixel': max_pixel, 'x_range_pxan': x_range_pxan, 'y_range_pxan': y_range_pxan, 'num_pxan': num_pxan}

    def _filter_detectors(self):
        
        module = self._module
        det_mask = np.full(module.x.value.size, True)
        param = self._param

        if not param.get('pols_on') is None:
            det_mask = np.in1d(module.pol.value, param['pols_on'])
        if not param.get('rhombi_on') is None:
            det_mask = det_mask & np.in1d(module.rhombus, param['rhombi_on'])
        if not param.get('wafers_on') is None:
            det_mask = det_mask & np.in1d(module.wafer, param['wafers_on'])
        
        if not param.get('det_radius') is None:
            radius = np.sqrt(module.x.value**2 + module.y.value**2)
            det_mask = det_mask & (radius >= param['det_radius'][0]) & (radius <= param['det_radius'][1])
        if not param.get('det_lim') is None:
            det_lim = param['det_lim']
            det_mask = det_mask & (module.x.value >= det_lim[0][0]) & (module.x.value <= det_lim[0][1]) & (module.y.value >= det_lim[1][0]) & (module.y.value <= det_lim[1][1])
        if not param.get('det_list') is None:
            det_list = param['det_list']
            if len(np.shape(det_list)) == 1:
                det_mask = det_mask & np.in1d(module.pixel_num, det_list)
            else:
                det_list_mask = np.full(module.x.value.size, False)
                for point in det_list:
                    distance_away = np.sqrt((module.x.value - point[0])**2 + (module.y.value - point[1])**2)
                    point_i = distance_away.argmin()
                    det_list_mask[point_i] = True
                det_mask = det_mask & det_list_mask

        return det_mask

    def _filter_ts(self):
        param = self._param
        
        # filter ts (rows)
        if not param.get('max_acc') is None:
            ts_mask_acc = abs(self._sky_pattern.acc.value) <= param['max_acc']
        else:
            ts_mask_acc = np.full(self._sky_pattern.x_coord.value.size, True)
        if not param.get('min_speed') is None:
            ts_mask_speed = abs(self._sky_pattern.vel.value) >= param['min_speed']
        else:
            ts_mask_speed = np.full(self._sky_pattern.x_coord.value.size, True)

        ts_mask = (ts_mask_acc & ts_mask_speed)

        return ts_mask
    
    # SIMULATING SCAN

    def _set_histograms(self, kept):

        # select (in)valid points
        if kept:
            validity_mask = self._validity_mask
            print('Generating histograms for kept hits...')
        else:
            validity_mask = ~self._validity_mask
            print('Generating histograms for removed hits...')

        # prep x_coord, y_coord, and rot_angle for simulating
        pixel_size = self._param['pixel_size']
        x_coord_px = self._sky_pattern.x_coord.value[validity_mask]/pixel_size
        y_coord_px = self._sky_pattern.y_coord.value[validity_mask]/pixel_size
        rot_angle_rad = np.radians(self._field_rotation[validity_mask])

        sky_hist, det_hist, time_hist = self._simulate_scan(x_coord_px, y_coord_px, rot_angle_rad)

        # clean up histograms
        hitmap_range = np.arange(-self._prep['max_pixel'], self._prep['max_pixel'])
        sky_hist = pd.DataFrame(sky_hist, index=hitmap_range, columns=hitmap_range)

        if self._pxan:
            det_num = self._module.pixel_num[self._det_mask]
            det_hist = pd.Series(det_hist, index=det_num).reindex(self._module.pixel_num)

            sky_pattern_index = np.arange(self._sky_pattern.x_coord.value.size)
            time_num = sky_pattern_index[validity_mask]
            time_hist = pd.Series(time_hist, index=time_num).reindex(sky_pattern_index)
            time_hist.index = self._sky_pattern.time_offset.value

        # store histograms
        if kept:
            self._sky_hist = sky_hist
            self._det_hist = det_hist
            self._time_hist = time_hist
        else:
            self._sky_hist_rem = sky_hist
            self._det_hist_rem = det_hist
            self._time_hist_rem = time_hist

    def _simulate_scan(self, x_coord_px, y_coord_px, rot_angle_rad):

        max_pixel = self._prep['max_pixel']
        num_bins = 2*max_pixel

        x_range_pxan = self._prep['x_range_pxan']
        y_range_pxan = self._prep['y_range_pxan']

        # get detector elements
        pixel_size = self._param['pixel_size']
        x_mod = self._module.x.value[self._det_mask]/pixel_size
        y_mod = self._module.y.value[self._det_mask]/pixel_size
        num_det_elem = len(x_mod)

        # number of timestamps
        num_ts = len(x_coord_px)

        # initialize histograms to be returned
        sky_hist = np.zeros((num_bins, num_bins))
        if self._pxan:
            det_hist = np.zeros(num_det_elem)
            time_hist = np.zeros(num_ts)
        else:
            det_hist = None
            time_hist = None

        # this section if for removed hits (if there are none)
        if num_ts == 0 or num_det_elem == 0:
            return sky_hist, det_hist, time_hist

        # Divide process into chunks to abide by memory limits 
        # We store using np.float16, which takes 2 bytes for each additional value
        # -> updated, use float64 is high_precision is True.
        # Otherwise adjust the precision depending on max_pixel
        # max_pixel <= 256: float16 (interval 1/8)
        # the rest should be sufficient with float32

        if self._param['high_precision']:
            dtype = np.float64
            byte_per_point = 8
        else:
            if max_pixel <= 256:
                dtype = np.float32
                byte_per_point = 4
            else:
                dtype = np.float16
                byte_per_point = 2

        max_number_points = (self._param['mem_limit']*10**6)/byte_per_point

        chunk_ts = math.floor(max_number_points/num_det_elem)
        for chunk in range(math.ceil(num_ts/chunk_ts)):

            # initialize empty arrays (rows of ts and cols of det elements) to store hits 
            if (chunk+1)*chunk_ts <= num_ts:
                num_rows = chunk_ts
            else:
                num_rows = num_ts - chunk*chunk_ts

            all_x_coords = np.empty((num_rows, num_det_elem), dtype=dtype)
            all_y_coords = np.empty((num_rows, num_det_elem), dtype=dtype)

            # range of ts to loop over
            start = chunk*chunk_ts
            end = start + num_rows
            print(f'...{start}/{num_ts} completed...')

            # add all hits from the detector elements at each ts 
            for i, x_coord1, y_coord1, rot1 in zip(range(num_rows), x_coord_px[start:end], y_coord_px[start:end], rot_angle_rad[start:end]):
                all_x_coords[i] = x_coord1 + x_mod*cos(rot1) - y_mod*sin(rot1)
                all_y_coords[i] = y_coord1 + x_mod*sin(rot1) + y_mod*cos(rot1)

            sky_hist += histogram2d(all_x_coords, all_y_coords, range=[[-max_pixel, max_pixel], [-max_pixel, max_pixel]], bins=[num_bins, num_bins])

            # apply pixel(s) analysis
            if self._pxan:
                mask = False
                for x_range, y_range in zip(x_range_pxan, y_range_pxan):
                    mask = mask | ( (all_x_coords >= x_range[0]) & (all_x_coords < x_range[1]) & (all_y_coords >= y_range[0]) & (all_y_coords < y_range[1]) )

                det_hist += np.count_nonzero(mask, axis=0)
                time_hist[start:end] = np.count_nonzero(mask, axis=1)

        print(f'total number of hits {num_ts*num_det_elem} == {np.sum(sky_hist.flatten())}')

        # note that we transpose sky_hist, since histogram2d has rows and columns flipped
        return sky_hist.T, det_hist, time_hist

    def _check_histograms(func):
        @wraps(func)
        def wrapper(self, hits, *args, **kwargs):
            if hits == 'total':
                if self._sky_hist is None:
                    self._set_histograms(True)
                if self._sky_hist_rem is None:
                    self._set_histograms(False)
            
            elif hits == 'kept':
                if self._sky_hist is None:
                    self._set_histograms(True)
            
            elif hits == 'removed':
                if self._sky_hist_rem is None:
                    self._set_histograms(False)

            else:
                raise ValueError(f'"hits" = {hits} is not one of "total", "kept", or "removed"')

            return func(self, hits, *args, **kwargs)
        return wrapper

    # USER FUNCTIONS

    @_check_histograms
    def sky_histogram(self, hits='kept', convolve=True, norm_time=False, path_or_buf=None):
        """
        Get a 2D histogram of the resultant hitmap. 

        Parameters
        ------------------------------
        hits : str; default 'kept'
            One of 'kept', 'removed', or 'total' hits. 
        convolve : bool; default True
            Whether to convolve the hitmap. 
        norm_time : bool; default False
            `True` for hits/px per total scan duration. `False` for hits/px.
        path_or_buf : str, file handle or None; default None
            File path or object, if `None` is provided the result is returned.
            The file is saved as a csv with the header and columns representing the bin edges (in deg). 

        Returns
        -------------------------
        None or (2d float array, Quantity array)
            If `path_or_buf` is `None`, returns a 2D histogram of the hits as well as the bin edges.
            Otherwise returns `None`.
        """

        if hits == 'kept':
            sky_hist = self._sky_hist.to_numpy()
        elif hits == 'removed':
            sky_hist = self._sky_hist_rem.to_numpy()
        else:
            sky_hist = self._sky_hist.to_numpy() + self._sky_hist_rem.to_numpy()

        pixel_size = self._param['pixel_size']
        
        # convolution
        if convolve:
            ang_res = self._module.ang_res.value
            if isiterable(ang_res):
                raise ValueError('Unable to convolve since ang_res is not constant.')
            stddev = (ang_res/pixel_size)/np.sqrt(8*np.log(2))
            kernel = Gaussian2DKernel(stddev)
            sky_hist = convolve_fft(sky_hist, kernel, boundary='fill', fill_value=0)
    
        # normalize time
        if norm_time:
            total_time = self._sky_pattern.scan_duration.value
            sky_hist = sky_hist/total_time
        
        # return
        max_pixel = self._prep['max_pixel']
        bin_edges = np.linspace(-max_pixel, max_pixel, 2*max_pixel+1)[:-1]*pixel_size

        _shape = np.shape(sky_hist)
        assert((_shape[0] == _shape[1]) and (_shape[0] == bin_edges.size))

        if path_or_buf is None:
            return sky_hist, bin_edges*self._param_units['pixel_size']
        else:
            sky_hist = pd.DataFrame(sky_hist, index=bin_edges, columns=bin_edges)
            sky_hist.to_csv(path_or_buf, index=True, header=True)
                
    @_check_histograms
    def det_histogram(self, hits='kept', norm_pxan=True, norm_time=False, path=None):
        """
        Get a histogram of the number of hits per detector element on a pixel on the sky (specified in `pxan_list` or `pxan_lim`).

        Parameters
        ------------------------------
        hits : str; default 'kept'
            One of 'kept', 'removed', or 'total' hits. 
        norm_pxan : bool; default True
            Whether to average the hits by dividing the total hits by the number of pixels. 
        norm_time : bool; default False
            `True` for hits/px per total scan duration. `False` for hits/px.
        path_or_buf : str, file handle or None; default None
            File path or object, if `None` is provided the result is returned. The file is saved as a csv.

        Returns
        -------------------------
        None or (float array, int array)
            If `path_or_buf` is `None`, returns the counts for each detector element as well as the corresponding detector number.
            `NaN` values mean that detector element has been filtered off. Otherwise returns `None`.
        """
        assert(self._pxan)

        if hits == 'kept':
            det_hist = self._det_hist
        elif hits == 'removed':
            det_hist = self._det_hist_rem
        else:
            det_hist = self._det_hist + self._det_hist_rem
        
        # divide by number of pixels in pixel analysis
        if norm_pxan:
            num_pxan = self._prep['num_pxan']
            det_hist = det_hist/num_pxan
    
        # normalize time
        if norm_time:
            total_time = self._sky_pattern.scan_duration.value
            det_hist = det_hist/total_time
        
        if path is None:
            return det_hist.values, det_hist.index.to_numpy()
        else:
            det_hist.to_csv(path, index=True)

    @_check_histograms
    def time_histogram(self, hits='kept', norm_pxan=True, norm_time=False, path=None):
        """
        Get a histogram of the number of hits at each time step on a pixel on the sky (specified in `pxan_list` or `pxan_lim`).

        Parameters
        ------------------------------
        hits : str; default 'kept'
            One of 'kept', 'removed', or 'total' hits. 
        norm_pxan : bool; default True
            Whether to average the hits by dividing the total hits by the number of pixels. 
        norm_time : bool; default False
            `True` for hits/px per total scan duration. `False` for hits/px.
        path_or_buf : str, file handle or None; default None
            File path or object, if `None` is provided the result is returned. The file is saved as a csv.

        Returns
        -------------------------
        None or (float array, int array)
            If `path_or_buf` is `None`, returns the counts for each time step as well as the corresponding time offset.
            `NaN` values mean that time step is out/in the speed/accleration limits. Otherwise returns `None`.
        """

        if hits == 'kept':
            time_hist = self._time_hist
            time_offset = time_hist.index.to_numpy()
            time_hist = time_hist.values
        elif hits == 'removed':
            time_hist = self._time_hist_rem
            time_offset = time_hist.index.to_numpy()
            time_hist = time_hist.values
        else:
            time_hist1 = self._time_hist
            time_hist2 = self._time_hist_rem
            time_offset = time_hist1.index.to_numpy()
            time_hist = np.nan_to_num(time_hist1) + np.nan_to_num(time_hist2)
        
        # divide by number of pixels in pixel analysis
        if norm_pxan:
            num_pxan = self._prep['num_pxan']
            time_hist = time_hist/num_pxan
    
        # normalize time
        if norm_time:
            total_time = self._sky_pattern.scan_duration.value
            time_hist = time_hist/total_time
        
        if path is None:
            return time_hist, time_offset
        else:
            pd.Series(time_hist, index=time_offset).to_csv(path, index=True)

    def pol_histogram(self, hits='kept', path=None):
        """
        TODO add documentation
        
        """
        
        if hits == 'kept':
            field_rotation = self._field_rotation[self._validity_mask]
        elif hits =='removed':
            field_rotation = self._field_rotation[~self._validity_mask]
        else:
            field_rotation = self._field_rotation
        
        all_pol, all_pol_counts = np.unique(self._module.pol.value[self._det_mask], return_counts=True)

        pol_grid, rot_grid = np.meshgrid(all_pol, field_rotation)
        total_rot =  pd.DataFrame((pol_grid + rot_grid)%90, columns=all_pol) 

        if path is None:
            return total_rot, all_pol_counts
        else:
            total_rot.to_csv(path, index=True)

    @property
    def sky_pattern(self):
        """SkyPattern : Associated `SkyPattern` object."""
        return self._sky_pattern
    
    @property
    def module(self):
        """Module : `Module` object. """
        return self._module
    
    @property
    def field_rotation(self):
        """Quantity array : Field rotation."""
        return self._field_rotation*u.deg
    
    @property
    def param(self):
        return {p: val*self._param_units[p] for p, val in self._param.items()}

    @property
    def det_mask(self):
        """ bool array : Mask for detector elements that are turned on."""
        return self._det_mask
    
    @property
    def validity_mask(self):
        """bool array : Mask for samples that are within the acceleration/speed limits."""
        return self._validity_mask


"""def accumulation(max_duration=None, num_repeat=1, intermission_time=10**u.s, **kwargs):

    # get the sky pattern and telescope pattern
    # and other info needed to create new telescope patterns

    try:
        telescope_pattern = kwargs['telescope_pattern']
    except KeyError:
        raise TypeError('missing keyword(s)')

    sky_pattern = telescope_pattern.get_sky_pattern()

    obs_param = {
        'lat': telescope_pattern._param['lat'],
        'start_ra': telescope_pattern.ra_coord[0].value,
        'start_dec': telescope_pattern.dec_coord[0].value,
        'start_lst': telescope_pattern.lst[0].value
    }

    # get number of repeats and one duration 

    pattern_time = telescope_pattern.scan_duration
    one_duration = (pattern_time + u.Quantity(intermission_time, u.s)).value 

    if not max_duration is None:
        max_duration = u.Quantity(max_duration, u.s).value
        num_repeat = math.floor(max_duration/one_duration)

    # get initial simulation stuff
    sim0 = Simulation(**kwargs)
    sky_hist, bin_edges = sim0.sky_histogram('kept')

    # loop

    SIDEREAL_TO_UT1 = 1.002737909350795

    for i in range(1, num_repeat):
        obs_param['start_lst'] = obs_param['start_lst'] + one_duration/3600*SIDEREAL_TO_UT1
        telescope_pattern = TelescopePattern(sky_pattern, **obs_param)

        kwargs['telescope_pattern'] = telescope_pattern
        sim0 = Simulation(**kwargs)

        sky_hist += sim0.sky_histogram('kept')

"""
