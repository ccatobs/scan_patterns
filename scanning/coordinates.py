import math
from math import pi, sin, cos, tan, sqrt, radians
import json
import warnings

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.coordinates import SkyCoord
from fyst_trajectories import get_fyst_site, Coordinates
from fyst_trajectories.offsets import InstrumentOffset, boresight_to_detector, detector_to_boresight
from fyst_trajectories.site import FYST_NASMYTH_PORT
from fyst_trajectories.patterns import (
    PongScanPattern, DaisyScanPattern, PongScanConfig, DaisyScanConfig,
)

def _central_diff(a, h=None, time=None):

    # get dt
    if h is None:
        h = time[1] - time[0]

    # get derivative 
    a = np.array(a)
    len_a = len(a)

    new_a = np.empty(len_a)
    new_a[0] = (a[1] - a[0])/h
    new_a[1:-1] = (a[2:len_a] - a[0:len_a-2])/(2*h)
    new_a[-1] = (a[-1] - a[-2])/h

    return new_a

FYST_LOC = get_fyst_site().location

##################
#  SKY PATTERN 
##################

class SkyPattern():
    """ Representing the path of the center of a detector array in terms of offsets. """

    # other attributes (for development)
    # _stored_units
    # _data: time_offset, x_coord, y_coord
    # _param, _param_units
    # _sample_interval
    # _repeatable

    _param_units = {'num_repeat': u.dimensionless_unscaled}
    _stored_units = {'time_offset': u.s, 'x_coord': u.deg, 'y_coord': u.deg}

    def __init__(self, data, units=None, repeatable=False, **kwargs) -> None:
        """
        Parameters
        --------------------------------
        data : str, DataFrame, or [dict of str -> sequence]
            If `str`, a file path to a csv file. If `dict` or `DataFrame`, column names map to their values. 
            Must have columns 'time_offset', 'x_coord', 'y_coord'.
        repeatable : bool; default False
            Whether this pattern can repeat itself (ends where it starts).
        units : [dict of str -> str or Unit] or None; default None
            Mapping columns in `data` with their units. All columns do not need to be mapped.
            If not provided, all angle-like units are assumed to be in degrees and all time-like units are assumed to be in seconds.
        
        Keyword Args
        ---------------------------------
        num_repeat : int; default 1 
            Number of repeats of the pattern. Must be a positive integer.
            Cannot be used with `max_scan_duration` and `repeatable` must be `True`.
        max_scan_duration : float/Quantity/str; default unit sec
            Maximum total scan time to determine number of repeats. Must be positive. 
            Cannot be used with `num_repeat` and `repeatable` must be `True`.

        Raises
        -----------------------------------
        ValueError
            "data" could not be parsed
        ValueError
            sample interval must be constant

        Examples
        -----------------------------
        >>> # if x_coord is specifically in arcseconds
        >>> SkyPattern('file.csv', units={'x_coord': 'arcsec'})
        """

        self._repeatable = repeatable 

        try:
            if isinstance(data, str):
                data = pd.read_csv(data, index_col=False, usecols=['time_offset', 'x_coord', 'y_coord'])
            else:
                data = pd.DataFrame(data)[['time_offset', 'x_coord', 'y_coord']]
        except (ValueError, KeyError) as e:
            raise ValueError(f"'data' could not be parsed: {e}")

        # convert to specified units
        if not units is None:
            for col, unit in units.items():
                data[col] = data[col]*u.Unit(unit).to(self._stored_units[col])

        # determine sample_interval 
        sample_interval_list = np.diff(data['time_offset'].to_numpy())
        if np.std(sample_interval_list)/np.mean(sample_interval_list) <= 0.01:
            self._sample_interval = np.mean(sample_interval_list)
        else:
            raise ValueError('sample interval must be constant')

        # repeating the scan
        self._param = self._clean_param(**kwargs)
        self._data = self._repeat_scan(data)

    # INITIALIZATION

    def _clean_param(self, **kwargs):
        kwarg_keys = kwargs.keys()

        # determine number of repeats
        if 'max_scan_duration' in kwarg_keys or 'num_repeat' in kwarg_keys:

            if not self.repeatable:
                warnings.warn('This is not a repeatable pattern, but you have indicated to repeat it. This may or may not repeat.')
            
            if 'max_scan_duration' in kwarg_keys and 'num_repeat' in kwarg_keys:
                raise ValueError('"max_scan_duration" and "num_repeat" cannot be inputted together')

            if 'max_scan_duration' in kwarg_keys:
                kwargs['num_repeat'] = math.nan # set as null for now, will determine once first pattern is generated
                kwargs['max_scan_duration'] = u.Quantity(kwargs['max_scan_duration'], self._stored_units['time_offset']).value
            else:
                kwargs['num_repeat'] = int(kwargs.get('num_repeat', 1))

        return kwargs

    def _repeat_scan(self, data):
        one_scan_duration = data.iloc[-1]['time_offset'] + self.sample_interval.value

        num_repeat = self._param.get('num_repeat', 1)

        # determine number of repeats
        if math.isnan(num_repeat):
            max_scan_duration = self._param.pop('max_scan_duration') # only store number of repeats, not maximum scan duration
            num_repeat = math.floor(max_scan_duration/one_scan_duration)
            self._param['num_repeat'] = num_repeat

        # repeat pattern if necessary 
        if num_repeat < 1:
            raise ValueError(f'number of repeats = {num_repeat} is less than 1')
        elif num_repeat > 1:
            warnings.warn('You have chosen to repeat this pattern. However, analytic patterns \
            (such as Pong) may not have their next location in the repeat be exactly their first original location \
            to ensure a constant time interval. This may cause spikes in higher derivates. Mitigate this by \
            initializing this object with its intended subclass using its original parameters.')

            time_offset = data['time_offset']
            x_coord = data['x_coord']
            y_coord = data['y_coord']
            data_temp = data.copy()
            for i in range(1, num_repeat):
                data_temp['time_offset'] = time_offset + one_scan_duration*i
                data_temp['x_coord'] = x_coord
                data_temp['y_coord'] = y_coord
                data = data.append(data_temp, ignore_index=True)

        return data

    # OBJECT DATA

    def save_data(self, path_or_buf=None, columns='default', include_repeats=True):
        """

        Parameters
        ----------------------
        path_or_buf : str, file handle or None; default None
            File path or object, if `None` is provided the result is returned as a dictionary.
        columns : sequence, str or [dict of str -> str/Unit/None]; default 'default'
            Columns to write. If `dict`, map column names to their desired unit and use `None` if you would like to use the standard (deg for angle-like units, sec for time-like units).
            'default' for ['time_offset', 'x_coord', 'y_coord']
            'all' for ['time_offset', 'x_coord', 'y_coord', 'distance', 'x_vel', 'y_vel', 'vel', 'x_acc', 'y_acc', 'acc', 'x_jerk', 'y_jerk', 'jerk']
        include_repeats : bool, default 'True'
            Whether to include repeats of the SkyPattern.

        Returns
        ----------------------
        None or [dict of str -> array]
            If `path_or_buf` is `None`, returns the data as a dictionary mapping column name to values. Otherwise returns `None`.

        Examples
        ---------------------
        >>> skypattern.save_data('file.csv', columns={'time_offset': 's', 'x_coord': 'arcsec', 'y_coord': None})
        """

        # replace str options
        if columns == 'default':
            columns = ['time_offset', 'x_coord', 'y_coord']
        elif columns == 'all':
            columns = ['time_offset', 'x_coord', 'y_coord', 'distance', 'x_vel', 'y_vel', 'vel', 'x_acc', 'y_acc', 'acc', 'x_jerk', 'y_jerk', 'jerk']
        
        data = pd.DataFrame()

        # generate required data
        if isinstance(columns, dict):
            for col, unit in columns.items():
                if not unit is None:
                    data[col] = getattr(self, col).to(unit).value
                else:
                    data[col] = getattr(self, col).value
        else:
            for col in columns:
                data[col] = getattr(self, col).value

        # whether to include repetitions 
        num_repeat = self._param.get('num_repeat', 1)
        if not include_repeats and num_repeat > 1:
            before_index = int(len(data.index)/num_repeat)
            data = data.iloc[:before_index]
        
        # returning
        if path_or_buf is None:
            return data.to_dict('list')
        else:
            data.to_csv(path_or_buf, index=False)

    def save_param(self, path_or_buf=None):
        """
        Parameters
        ----------------------------
        path_or_buf : str, file handle, or None; default None
            File path or object, if `None` is provided the result is returned as a dictionary.
        
        Returns
        ----------------------
        None or [dict of str -> numeric]
            If `path_or_buf` is `None`, returns the resulting json format as a dictionary. Otherwise returns `None`.
        """
        
        param_temp = self._param.copy()

        # save param_json
        if path_or_buf is None:
            return param_temp
        else:
            with open(path_or_buf, 'w') as f:
                json.dump(param_temp, f)

    # PROPERTIES
    
    def __getattr__(self, attr):
        # for easy access of properties without unit conversions
        if attr.startswith('_'):
            prop = getattr(self, attr[1:])
            if type(prop) is u.Quantity:
                return prop.value
            else:
                return prop
        else:
            raise AttributeError(f'type object "{type(self)}" has no attribute "{attr}"')

    @property
    def repeatable(self):
        """bool: Whether this pattern is repeatable or not."""
        return self._repeatable

    @property
    def param(self):
        """dict of str -> (float or Quantity): Parameters inputted by user."""
        return_param = dict()
        for p, val in self._param.items():
            return_param[p] = val if self._param_units[p] is u.dimensionless_unscaled else val*self._param_units[p]
        return return_param

    @property
    def sample_interval(self):
        """Quantity: Time interval between samples."""
        return self._sample_interval*self._stored_units['time_offset']

    @property
    def scan_duration(self):
        """Quantity: Total scan duration."""
        return self.time_offset[-1] + self.sample_interval

    @property
    def time_offset(self):
        """Quantity array: Time offsets."""
        return self._data['time_offset'].to_numpy()*self._stored_units['time_offset']
    
    @property
    def x_coord(self):
        """Quantity array: x positions."""
        return self._data['x_coord'].to_numpy()*self._stored_units['x_coord']
    
    @property
    def y_coord(self):
        """Quantity array: y positions."""
        return self._data['y_coord'].to_numpy()*self._stored_units['y_coord']

    @property
    def distance(self):
        """Quantity array: Distance of points from the center."""
        return np.sqrt(self.x_coord**2 + self.y_coord**2)

    @property
    def x_vel(self):
        """Quantity array: x velocity."""
        return _central_diff(self.x_coord.value, self.sample_interval.value)*(self._stored_units['x_coord']/self._stored_units['time_offset'])

    @property
    def y_vel(self):
        """Quantity array: y velocity."""
        return _central_diff(self.y_coord.value, self.sample_interval.value)*(self._stored_units['y_coord']/self._stored_units['time_offset'])

    @property
    def vel(self):
        """Quantity array: Total velocity."""
        return np.sqrt(self.x_vel**2 + self.y_vel**2)

    @property
    def x_acc(self):
        """Quantity array: x acceleration."""
        return _central_diff(self.x_vel.value, self.sample_interval.value)*(self._stored_units['x_coord']/self._stored_units['time_offset']**2)

    @property
    def y_acc(self):
        """Quantity array: y acceleration."""
        return _central_diff(self.y_vel, self.sample_interval.value)*(self._stored_units['y_coord']/self._stored_units['time_offset']**2)
    
    @property
    def acc(self):
        """Quantity array: Total acceleration."""
        return np.sqrt(self.x_acc**2 + self.y_acc**2)

    @property
    def x_jerk(self):
        """Quantity array: x jerk."""
        return _central_diff(self.x_acc.value, self.sample_interval.value)*(self._stored_units['x_coord']/self._stored_units['time_offset']**3)
    
    @property
    def y_jerk(self):
        """Quantity array: y jerk."""
        return _central_diff(self.y_acc.value, self.sample_interval.value)*(self._stored_units['y_coord']/self._stored_units['time_offset']**3)
    
    @property
    def jerk(self):
        """Quantity array: Total jerk."""
        return np.sqrt(self.x_jerk**2 + self.y_jerk**2)

class Pong(SkyPattern):
    """
    The Curvy Pong pattern allows for an approximation of a Pong pattern while avoiding 
    sharp turnarounds at the vertices. The Pong pattern is an analytic and close-pathed 
    pattern that is optimized for regions a few square degrees. It makes a path that 
    intends to cover each area uniformly and ends where it starts.
    
    See "The Impact of Scanning Pattern Strategies on Uniform Sky Coverage of Large Maps" 
    (SCUBA Project SC2/ANA/S210/008) for details of implementation. 
    """

    _repeatable = True
    _param_units = {
        'num_term': u.dimensionless_unscaled,
        'width': u.deg, 'height': u.deg, 'spacing': u.deg,
        'velocity': u.deg/u.s, 'angle': u.deg, 'sample_interval': SkyPattern._stored_units['time_offset'], 
        'num_repeat': u.dimensionless_unscaled,
        'period': u.s
    }

    def __init__(self, param_json=None, **kwargs) -> None:
        """
        Initialize a Pong pattern by passing a parameter file and overwriting any parameters with **kwargs:
            option1 : Pong(param_json, **kwargs) 
        or building from scratch: 
            option2 : Pong(**kwargs)

        Parameters
        ---------------------------
        param_json : str or None
            If `str` path to JSON file containing parameters. 
        
        Keyword Args
        ----------------------------
        num_term : int
            Number of terms in the triangle wave expansion. Must be positive. 
        width : float or Quantity or str; default unit deg
            Width of the field. Must be positive. 
        height : float or Quantity or str; default unit deg
            Height of the field. Must be positive. 
        spacing : float or Quantity or str; default unit deg
            Space between adjacent (parallel) scan lines in the Pong pattern. Must be positive.
        velocity : float or Quantity or str; default unit deg/s
            Target magnitude of the total scan velocity excluding turn-arounds.
            NOTE this is now the total velocity, not just the velocity of one direction.

        angle : float or Quantity or str; default 0; default unit deg
            Position angle of the box in the native coordinate system. 
        sample_interval : float or Quantity or str; default 1/400, default unit s
            Time between read-outs. Must be positive.
        num_repeat : int; default 1 
            Number of repeats of the pattern. Must be a positive integer.
            Cannot be used with `max_scan_duration`.
        max_scan_duration : float or Quantity or str; default unit sec
            Maximum total scan time to determine number of repeats. Must be positive. 
            Cannot be used with `num_repeat`.
        
        Examples
        ---------------------------
        >>> import astropy.units as u
        >>> Pong(num_term=4, width=2, height=7200*u.arcsec, spacing='500 arcsec', velocity=1/2)
        """

        # pass kwargs
        if param_json is None:
            self._param = self._clean_param(**kwargs)

        # pass parameters by json
        else:
            with open(param_json, 'r') as f:
                param = json.load(f)
           
            # overwrite any parameters
            if 'max_scan_duration' in kwargs.keys():
                param.pop('num_repeat')
            param.update(kwargs)

            self._param = self._clean_param(**param)
        
        self._sample_interval = self._param['sample_interval']
        self._data = self._generate_scan()

    def _clean_param(self, **kwargs):
        kwargs = super()._clean_param(**kwargs)
        kwargs['num_term'] = int(kwargs['num_term'])
        kwargs['width'] = u.Quantity(kwargs['width'], self._param_units['width']).value
        kwargs['height'] = u.Quantity(kwargs['height'], self._param_units['height']).value
        kwargs['spacing'] = u.Quantity(kwargs['spacing'], self._param_units['spacing']).value
        kwargs['velocity'] = u.Quantity(kwargs['velocity'], self._param_units['velocity']).value
        kwargs['angle'] = u.Quantity(kwargs.get('angle', 0), self._param_units['angle']).value
        kwargs['sample_interval'] = u.Quantity(kwargs.get('sample_interval', 1/400), self._param_units['sample_interval']).value
        return kwargs

    def _generate_scan(self):

        # unpack parameters
        width = self._param['width']
        height = self._param['height']
        spacing = self._param['spacing']
        velocity = self._param['velocity']
        sample_interval = self._param['sample_interval']

        # Compute period for num_repeat / max_scan_duration handling
        vert_spacing = sqrt(2) * spacing
        vavg = velocity / sqrt(2)

        # Use fyst-trajectories's vertex computation (identical algorithm)
        config = PongScanConfig(
            num_terms=self._param['num_term'],
            width=width,
            height=height,
            spacing=spacing,
            velocity=velocity,
            timestep=sample_interval,
            angle=self._param['angle'],
        )
        pattern = PongScanPattern(ra=0.0, dec=0.0, config=config)
        x_numvert, y_numvert, _, _ = pattern._compute_vertices()

        period = x_numvert * y_numvert * vert_spacing * 2 / vavg

        # Determine number of repeats
        num_repeat = self._param.get('num_repeat', 1)

        if math.isnan(num_repeat):
            max_scan_duration = self._param.pop('max_scan_duration')
            num_repeat = math.floor(max_scan_duration / period)
            self._param['num_repeat'] = num_repeat

        self._param['period'] = period

        # Compute duration to match original point count:
        # pongcount = ceil(period * num_repeat / sample_interval)
        # duration = (pongcount - 1) * sample_interval
        pongcount = math.ceil(period * num_repeat / sample_interval)
        duration = (pongcount - 1) * sample_interval

        # Generate offsets via fyst-trajectories
        times, x_off, y_off = pattern.generate_offsets(duration=duration)

        return pd.DataFrame({
            'time_offset': times,
            'x_coord': x_off, 'y_coord': y_off,
        })


class Daisy(SkyPattern):
    """
    The Daisy pattern is optimized for point sources and works by having the path of the camera module 
    move at constant velocity and cross the center of the map at various angles.

    See "CV Daisy - JCMT small area scanning pattern" (JCMT TCS/UN/005) for details of implementation.
    """

    _param_units = {
        'velocity': u.deg/u.s, 'start_acc': u.deg/u.s/u.s, 
        'R0': u.deg, 'Rt': u.deg, 'Ra': u.deg,
        'T': u.s, 'sample_interval': SkyPattern._stored_units['time_offset'], 'y_offset': u.deg
    }
    _repeatable = False

    def __init__(self, param_json=None, **kwargs) -> None:
        """
        Initialize a Daisy pattern by passing a parameter file and overwriting any parameters with **kwargs:
            option1 : Daisy(param_json, **kwargs) 
        or building from scratch: 
            option2 : Daisy(**kwargs)

        Parameters
        ---------------------------
        param_json : str or None
            If `str` path to JSON file containing parameters. 
        
        Keyword Args
        ----------------------------
        velocity : float or Quanity or str; default unit deg/s
            Constant velocity (CV) for scan to go at. 
        start_acc : float or Quanity or str; default unit deg/s^2
            Acceleration at start of pattern. Cannot be 0. 
        R0 : float or Quanity or str; default unit deg
            Radius R0. Must be positive.
        Rt : float or Quanity or str; default unit deg
            Turn radius. Must be positive.
        Ra : float or Quanity or str; default unit deg
            Avoidance radius. Must be non-negative. 
        T : float or Quanity or str; default unit sec
            Total time of the simulation. Must be postivie. 
        sample_interval : float or Quanity or str; default 1/400, default unit sec
            Time step. 
        y_offset : float or Quanity or str; default 0, default unit deg
            Start offset in y. 

        Examples
        ----------------------------
        >>> import astropy.units as u
        >>> Daisy(velocity=1/3*u.deg/u.s, start_acc='0.2 deg/s/s', R0=0.47, Rt=800*u.arcsec, Ra='600 arcsec', T=300)
        """

        # pass kwargs
        if param_json is None:
            self._param = self._clean_param(**kwargs)

        # pass parameters by json
        else:
            with open(param_json, 'r') as f:
                param = json.load(f)
           
            # overwrite any parameters
            param.update(kwargs)
            self._param = self._clean_param(**param)
        
        self._sample_interval = self._param['sample_interval']
        self._data = self._generate_scan()

    def _clean_param(self, **kwargs):
        kwargs['velocity'] = u.Quantity(kwargs['velocity'], self._param_units['velocity']).value
        kwargs['start_acc'] = u.Quantity(kwargs['start_acc'], self._param_units['start_acc']).value
        kwargs['R0'] = u.Quantity(kwargs['R0'], self._param_units['R0']).value
        kwargs['Rt'] = u.Quantity(kwargs['Rt'], self._param_units['Rt']).value
        kwargs['Ra'] = u.Quantity(kwargs['Ra'], self._param_units['Ra']).value
        kwargs['T'] = u.Quantity(kwargs['T'], self._param_units['T']).value
        kwargs['sample_interval'] = u.Quantity(kwargs.get('sample_interval', 1/400), self._param_units['sample_interval']).value
        kwargs['y_offset'] = u.Quantity(kwargs.get('y_offset', 0), self._param_units['y_offset']).value
        return kwargs
        
    def _generate_scan(self):

        # unpack parameters (already cleaned to floats in degrees/seconds)
        T = self._param['T']

        # Generate offsets via fyst-trajectories
        config = DaisyScanConfig(
            radius=self._param['R0'],
            velocity=self._param['velocity'],
            turn_radius=self._param['Rt'],
            avoidance_radius=self._param['Ra'],
            start_acceleration=self._param['start_acc'],
            y_offset=self._param['y_offset'],
            timestep=self._param['sample_interval'],
        )
        pattern = DaisyScanPattern(ra=0.0, dec=0.0, config=config)
        times, x_off, y_off = pattern.generate_offsets(duration=T)

        return pd.DataFrame({
            'time_offset': times,
            'x_coord': x_off, 'y_coord': y_off,
        })
 
#######################
#  TELESCOPE PATTERN 
#######################

class TelescopePattern():
    """
    Representing the path in AZ/EL coordinates of the telescope's boresight.
    """

    # other attributes (for development)
    # _stored_units
    # _instrument
    # _data
    # _param, _param_units

    _param_units = {
        'start_ra': u.deg, 'start_dec': u.deg, 'lat': u.deg, 'lon': u.deg,
        'start_hrang': u.hourangle, 
        'start_datetime': u.dimensionless_unscaled, 
        'start_lst': u.hourangle,
        'start_elev': u.deg, 'moving_up': u.dimensionless_unscaled
    }
    _stored_units = {'time_offset': u.s, 'lst': u.hourangle, 'alt_coord': u.deg, 'az_coord': u.deg}

    # INITIALIZATION

    def __init__(self, data, instrument=None, data_loc='boresight', obs_param=None, units=None, **kwargs) -> None:
        """
        Determine the motion of the telescope. Note that **kwargs can be passed as a file through `obs_param`.
            | option1: data (sky), instrument (optional), data_loc (optional); lat, start_ra, start_dec, (start_datetime or start_hrang or start_lst or [start_elev and moving_up])
            | option2: data (telescope, excludes lst), instrument (optional), data_loc (optional); lat, (start_ra or start_datetime or start_lst)
            | option3: data (telescope, includes lst), instrument (optional), data_loc (optional); lat

        Parameters
        -----------------------------
        data : str, DataFrame, [dict of str -> sequence], or SkyPattern
            If `str`, a file path to a csv file. If `dict` or `DataFrame`, column names map to their values. 
            Columns must contain 'time_offset, 'az_coord', and 'alt_coord'. If 'lst' is included as a column, certain observation parameters are not required (see options).
            Otherwise, `data` is a `SkyPattern` object. 
        units : [dict of str -> str or Unit] or None; default None
            If `data` is not `SkyPattern`, this mapping column in `data` with their units. All columns do not need to be mapped.
            If not provided, all angle-like units are assumed to be in degrees (except for hour angle and lst which are in hourangle)
            and all time-like units are assumed to be in seconds.

        instrument : Instrument or None, default None
            An `Instrument` object. 
        data_loc : str or two-tuple; default 'boresight'
            Location relative to the center of the instrument where observation parameters are applied:
                | 1. 'boresight' for boresight of the telescope 
                | 2. string indicating a module name in the instrument e.g. 'SFH' or one of the default slots in the instrument e.g. 'c', 'i1'
                | 3. tuple of (distance, theta) indicating module's offset from the center of the instrument; default unit deg

        obs_param : str
            File path to json containing all required obervation parameters (see **kwargs).

        Keyword Args
        ------------------------------
        start_ra : float/Quantity/str; default unit deg
            Starting right acension of telescope / right acension offset for sky.
        start_dec : float/Quantity/str; default unit deg
            Declination offset for sky_pattern.
        lat : float/Quantity/str; default unit deg, default FYST_LOC.lat
            Latitude of observation. 
        lon : float/Quantity/str; default unit deg, default FYST_LOC.lon
            Longitude of observation. 

        start_datetime : str or datetime; default timezone UTC
            Starting date and time of observation.
        start_hrang : float/Quantity/str; default unit hourangle
            Starting hour angle of source.
        start_lst : float/Quantity/str; default unit hourangle
            Starting local sidereal time.
        start_elev : float/Quantity/str; default unit deg
            Starting elevation of source, must be used with `moving_up`.
        moving_up : bool, default True
            Whether observation is moving towards the meridian or away, must be used with `start_elev`. 

        Example
        -------------------------------
        >>> sky_pattern = SkyPattern(sky_pattern.csv)
        >>> telescope_pattern1 = TelescopePattern(sky_pattern, data_loc='i1', start_ra='5 hourangle', start_dec='-60 deg', start_elev=30)

        """

        # --- Observation Parameters ---

        # pass by obs_param
        if not obs_param is None:
            with open(obs_param, 'r') as f:
                param = json.load(f)

            # overwrite parameters FIXME start_
            param.update(kwargs)

        else:
            param = kwargs

        # --- sky_pattern or data ---

        # sky_pattern has been passed
        if isinstance(data, SkyPattern):
                
            self._param = self._clean_param_sky_pattern(**param)

            self._sample_interval = data.sample_interval.value
            self._data = pd.DataFrame({'time_offset': data.time_offset.value})
            self._data['lst'] = self._get_lst(data)
            az1, alt1 = self._from_sky_pattern(data)

            self._data['az_coord'] = az1
            self._data['alt_coord'] = alt1

        # data has been passed
        else:
            if isinstance(data, str):
                try: 
                    self._data = pd.read_csv(data, index_col=False)
                except ValueError:
                    raise ValueError('could not parse "data"')
            else:
                try:
                    self._data = pd.DataFrame(data)
                except ValueError:
                    raise ValueError('could not parse "data"')

            if not units is None:
                for col, unit in units.items():
                    self._data[col] = self._data[col]*u.Unit(unit).to(self._stored_units[col])
                
            if 'lst' in self._data.columns:
                self._param = self._clean_param_telescope_data(False, **param)
            else:
                self._param = self._clean_param_telescope_data(True, **param)

            try:
                self._data[['time_offset', 'lst', 'az_coord', 'alt_coord']]
            except KeyError:
                self._data[['time_offset', 'az_coord', 'alt_coord']]
                self._data['lst'] = self._get_lst()

            # determine sample_interval 
            sample_interval_list = np.diff(self.time_offset.value)
            if np.std(sample_interval_list)/np.mean(sample_interval_list) <= 0.01:
                sample_interval = np.mean(sample_interval_list)
                self._sample_interval = sample_interval
            else:
                raise ValueError('sample_interval must be constant')

        # --- Instrument and module --- 

        self._instrument = instrument
        dist, theta = self._true_module_loc(data_loc)
        
        # az_coord and alt_coord are for the module's location
        # we need the az/alt coordinates of the boresight

        if not (math.isclose(dist, 0) and math.isclose(theta, 0)):
            az0, alt0 = self._transform_to_boresight(self.az_coord.value, self.alt_coord.value, dist, theta)
            self._data['az_coord'] = az0
            self._data['alt_coord'] = alt0
        else:
            self._data['az_coord'] = self._norm_angle(self.az_coord.value)
        
        if len(self.alt_coord.value[(self.alt_coord.value < 0) | (self.alt_coord.value > 90)]) > 0:
            warnings.warn('elevation has values outside of 0 to 90 range')
        elif len(self.alt_coord.value[(self.alt_coord.value < 30) | (self.alt_coord.value > 75)]) > 0:
            warnings.warn('elevation has values outside of 30 to 75 range')

    @classmethod
    def from_trajectory(cls, trajectory, instrument=None, data_loc='boresight', **kwargs):
        """Create TelescopePattern from a fyst-trajectories Trajectory.

        Parameters
        ----------
        trajectory : fyst_trajectories.Trajectory
            A trajectory generated by fyst-trajectories (e.g., from ConstantElScanPattern).
            Must have a start_time set. The trajectory must have uniform timesteps
            (scan_patterns requires a constant sample interval). ``trajectory.times``
            should be seconds from zero.
        instrument : Instrument, optional
            Instrument for detector offset calculations.
        data_loc : str, optional
            Module location for scanning center. Default 'boresight'.
        **kwargs
            Additional keyword arguments passed to TelescopePattern.__init__.
        """
        if trajectory.start_time is None:
            raise ValueError("Trajectory must have start_time set")

        site = get_fyst_site()

        # Compute absolute times and LST
        abs_times = trajectory.start_time + TimeDelta(trajectory.times * u.s)
        lst = abs_times.sidereal_time('apparent', longitude=site.location.lon)
        lst_hourangle = lst.to(u.hourangle).value

        data = pd.DataFrame({
            'time_offset': trajectory.times,
            'az_coord': trajectory.az,
            'alt_coord': trajectory.el,
            'lst': lst_hourangle,
        })

        tp = cls(data, instrument=instrument, data_loc=data_loc, **kwargs)
        tp.scan_flag = getattr(trajectory, 'scan_flag', None)
        return tp

    def _clean_param_sky_pattern(self, **kwargs):
        kwarg_keys = kwargs.keys()
        new_kwargs = dict()

        # required
        new_kwargs['lat'] = u.Quantity(kwargs.pop('lat', FYST_LOC.lat), u.deg).value
        new_kwargs['lon'] = u.Quantity(kwargs.pop('lon', FYST_LOC.lon), u.deg).value
        new_kwargs['start_ra'] = u.Quantity(kwargs.pop('start_ra'), u.deg).value
        new_kwargs['start_dec'] = u.Quantity(kwargs.pop('start_dec'), u.deg).value

        # choose between
        if np.count_nonzero( ['start_hrang' in kwarg_keys, 'start_datetime' in kwarg_keys, 'start_lst' in kwarg_keys, 'start_elev' in kwarg_keys] ) != 1:
            raise TypeError('need one (and only one) of start_datetime, start_hrang, start_lst, or start_elev')

        if 'start_hrang' in kwarg_keys:
            new_kwargs['start_hrang'] = u.Quantity(kwargs.pop('start_hrang'), u.hourangle).value
        elif 'start_datetime' in kwarg_keys:
            new_kwargs['start_datetime'] = pd.Timestamp(kwargs.pop('start_datetime')).to_pydatetime()
        elif 'start_lst' in kwarg_keys:
            new_kwargs['start_lst'] = u.Quantity(kwargs.pop('start_lst'), u.hourangle).value
        elif 'start_elev' in kwarg_keys:
            new_kwargs['start_elev'] = u.Quantity(kwargs.pop('start_elev'), u.deg).value
            new_kwargs['moving_up'] = kwargs.pop('moving_up', True)

        if kwargs:
            raise TypeError(f'Unrecognized observation parameters: {kwargs.keys()}')

        return new_kwargs

    def _clean_param_telescope_data(self, need_lst, **kwargs):
        kwarg_keys = kwargs.keys()
        new_kwargs = dict()

        # required
        new_kwargs['lat'] = u.Quantity(kwargs.pop('lat', FYST_LOC.lat), u.deg).value
        new_kwargs['lon'] = u.Quantity(kwargs.pop('lon', FYST_LOC.lon), u.deg).value

        if need_lst:

            # choose between
            if np.count_nonzero( ['start_ra' in kwarg_keys, 'start_datetime' in kwarg_keys, 'start_lst' in kwarg_keys] ) != 1:
                raise TypeError('need one (and only one) of start_ra, start_datetime, or start_lst')

            if 'start_ra' in kwarg_keys:
                new_kwargs['start_ra'] = u.Quantity(kwargs.pop('start_ra'), u.deg).value
            elif 'start_datetime' in kwarg_keys:
                new_kwargs['start_datetime'] = pd.Timestamp(kwargs.pop('start_datetime')).to_pydatetime()
            elif 'start_lst' in kwarg_keys:
                new_kwargs['start_lst'] = u.Quantity(kwargs.pop('start_lst'), u.hourangle).value

        if kwargs:
            raise TypeError(f'Unrecognized observation parameters: {kwargs.keys()}')

        return new_kwargs

    def _from_sky_pattern(self, sky_pattern):

        param = self.param

        if 'start_datetime' in self._param:
            # Use fyst-trajectories for accurate conversion (spherical offsets,
            # apparent sidereal time, precession/nutation).
            site = get_fyst_site()
            coords = Coordinates(site)

            start_time = Time(param['start_datetime'])
            dt_seconds = sky_pattern.time_offset.to(u.s).value
            obstimes = start_time + TimeDelta(dt_seconds * u.s)

            x_offsets = sky_pattern.x_coord.to(u.deg).value
            y_offsets = sky_pattern.y_coord.to(u.deg).value
            ra_center = param['start_ra'].to(u.deg).value
            dec_center = param['start_dec'].to(u.deg).value

            # Convert sky offsets (RA/Dec plane) to individual RA/Dec positions
            center = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg)
            positions = center.spherical_offsets_by(
                x_offsets * u.deg, y_offsets * u.deg,
            )
            ra_arr = positions.ra.deg
            dec_arr = positions.dec.deg

            az, el = coords.radec_to_altaz(ra_arr, dec_arr, obstime=obstimes)
            return az, el

        # Fall back to manual formula for hour-angle/LST/elevation starts
        if max(abs(sky_pattern.x_coord.value)) > 10:
            warnings.warn('This is a larger pattern and the conversion between x and y deltas and RA/DEC may be slightly off.')

        # get alt/az
        start_dec = param['start_dec'].to(u.rad).value
        hour_angle = self.lst - (sky_pattern.x_coord/cos(start_dec) + param['start_ra']) # FIXME fine for small regions, but consider checking out https://docs.astropy.org/en/stable/coordinates/matchsep.html for larger regions

        hour_angle_rad = hour_angle.to(u.rad).value
        dec_rad = (sky_pattern.y_coord + param['start_dec']).to(u.rad).value
        lat_rad = param['lat'].to(u.rad).value

        alt_rad = np.arcsin( np.sin(dec_rad)*sin(lat_rad) + np.cos(dec_rad)*cos(lat_rad)*np.cos(hour_angle_rad) )

        cos_az_rad = (np.sin(dec_rad) - np.sin(alt_rad)*sin(lat_rad)) / (np.cos(alt_rad)*cos(lat_rad))
        cos_az_rad[cos_az_rad > 1] = 1
        cos_az_rad[cos_az_rad < -1] = -1

        az_rad = np.arccos( cos_az_rad )
        mask = np.sin(hour_angle_rad) > 0
        az_rad[mask] = 2*pi - az_rad[mask]

        return np.degrees(az_rad), np.degrees(alt_rad)

    def _get_lst(self, sky_pattern=None):

        if not sky_pattern is None:
            extra_ra_offset = sky_pattern.x_coord[0]
            extra_dec_offset = sky_pattern.y_coord[0]
        else:
            extra_ra_offset = 0*u.deg
            extra_dec_offset = 0*u.deg

        param = self.param

        # given a starting datetime
        if 'start_datetime' in param.keys():
            #start_datetime = Time(param['start_datetime'], location=(param['lat'], 0*u.deg))
            start_datetime = Time(param['start_datetime'], location=(param['lon'], param['lat']))
            start_lst = start_datetime.sidereal_time('apparent')

        # given a starting hourangle
        elif 'start_hrang' in param.keys() and not sky_pattern is None:
            start_lst = param['start_hrang'] + extra_ra_offset + param['start_ra']
        
        # given a starting lst
        elif 'start_lst' in param.keys():
            start_lst = param['start_lst']

        # given a starting elevation
        elif 'start_elev' in param.keys() and not sky_pattern is None:

            # determine possible hour angles
            alt_rad = param['start_elev'].to(u.rad).value
            dec_rad = (param['start_dec'] + extra_dec_offset).to(u.rad).value
            lat_rad = param['lat'].to(u.rad).value

            try:
                start_hrang_rad = math.acos((sin(alt_rad) - sin(dec_rad)*sin(lat_rad)) / (cos(dec_rad)*cos(lat_rad)))
            except ValueError:
                max_el = math.floor(math.degrees(math.asin(cos(0)*cos(dec_rad)*cos(lat_rad) + sin(dec_rad)*sin(lat_rad))))
                min_el = math.ceil(math.degrees(math.asin(cos(pi)*cos(dec_rad)*cos(lat_rad) + sin(dec_rad)*sin(lat_rad))))
                raise ValueError(f'Elevation = {param["start_elev"]} is not possible at provided ra, dec, and latitude. Min elevation is {min_el} and max elevation is {max_el} deg.')

            # choose hour angle
            if param['moving_up']:
                start_hrang_rad = -start_hrang_rad

            # starting sidereal time
            start_lst = start_hrang_rad*u.rad + extra_ra_offset + param['start_ra']
        
        elif 'start_ra' in param.keys() and sky_pattern is None:
            lat_rad = param['lat'].to(u.rad).value
            alt_rad = self.alt_coord[0].to(u.rad).value
            az_rad = self.az_coord[0].to(u.rad).value

            dec_rad = math.asin( sin(lat_rad)*sin(alt_rad) + cos(lat_rad)*cos(alt_rad)*cos(az_rad) )
            hrang_rad = math.acos( (sin(alt_rad) - sin(dec_rad)*sin(lat_rad)) / (cos(dec_rad)*cos(lat_rad)) )

            if sin(az_rad) > 0:
                hrang_rad = 2*pi - hrang_rad

            start_lst = hrang_rad*u.rad + param['start_ra']

        # find sidereal time
        SIDEREAL_TO_UT1 = 1.002737909350795
        return (self.time_offset.value/3600*SIDEREAL_TO_UT1*u.hourangle + start_lst).to(u.hourangle).value

    # TRANSFORMATIONS

    def _norm_angle(self, az):
        # normalize azimuth values so that:
        # 1. starting azimuth is between 0 and 360
        # 2. azmiuth values are between start-180 to start+180 (centered around the start)
        # formula = ((value - low) % diff) + low 

        lowest_az = az[0]%360 - 180
        return (az - lowest_az)%(360) + lowest_az 

    def _true_module_loc(self, module):
        # Gets the (dist, theta) of the module from the boresight (not necessarily central tube)

        if module == 'boresight':
            return 0, 0

        # passed by module identifier or instrument slot name
        if isinstance(module, str):
            try:
                return self.instrument.get_module_location(module, from_boresight=True).value
            except AttributeError as e:
                raise AttributeError(f'"instrument" is of type {type(self.instrument)}')
            except ValueError:
                try:
                    return self.instrument.get_slot_location(module, from_boresight=True).value
                except KeyError:
                    raise ValueError(f'{module} is not an existing module name or instrument slot')
        else:
            return self.instrument.location_from_boresight(module[0], module[1]).value

    def _transform_to_boresight(self, az1, alt1, dist, theta):
        # Use fyst-trajectories's spherical offset (inverse direction).
        # Same convention mapping as _transform_from_boresight:
        #   dx = dist * cos(theta), dy = dist * sin(theta), field_rotation = -el_bore
        # The field_rotation depends on the unknown boresight elevation, so we
        # iterate: guess el_bore, invert, update guess, repeat.
        theta_rad = math.radians(theta)
        dx_arcmin = dist * 60.0 * math.cos(theta_rad)
        dy_arcmin = dist * 60.0 * math.sin(theta_rad)
        offset = InstrumentOffset(dx=dx_arcmin, dy=dy_arcmin)

        az1 = np.asarray(az1, dtype=float)
        alt1 = np.asarray(alt1, dtype=float)

        # Initial guess: boresight elevation ~ detector elevation
        bore_el = alt1.copy()

        for _ in range(20):
            bore_az, bore_el_new = detector_to_boresight(
                az1, alt1, offset, field_rotation=-bore_el,
            )
            max_delta = np.max(np.abs(bore_el_new - bore_el))
            bore_el = bore_el_new
            if max_delta < 1e-12:
                break

        if np.any(bore_el < 0):
            warnings.warn('elevation has values below 0')

        return self._norm_angle(np.asarray(bore_az)), np.asarray(bore_el)

    def _transform_from_boresight(self, az0, alt0, dist, theta):
        # Use fyst-trajectories's spherical offset (forward direction).
        # scan_patterns convention: (dist, theta) where theta is measured from
        # the cross-elevation axis in the Nasmyth focal plane. The sky-frame
        # position angle is (theta + elevation). Mapping to fyst-trajectories:
        #   dx = dist * cos(theta), dy = dist * sin(theta), field_rotation = -el
        theta_rad = math.radians(theta)
        dx_arcmin = dist * 60.0 * math.cos(theta_rad)
        dy_arcmin = dist * 60.0 * math.sin(theta_rad)
        offset = InstrumentOffset(dx=dx_arcmin, dy=dy_arcmin)

        det_az, det_el = boresight_to_detector(az0, alt0, offset, field_rotation=-alt0)

        if np.any(np.asarray(det_el) < 0):
            warnings.warn('elevation has values below 0')

        return self._norm_angle(np.asarray(det_az)), np.asarray(det_el)

    # METHODS

    def view_module(self, module, includes_instr_offset=False):
        """
        Get a TelescopePattern object representing the AZ/EL coordinates of the
        provided module. 

        Parameters
        ------------------------
        module : str, two-tuple, or InstrumentOffset
            | 1. string indicating a module name in the instrument e.g. 'SFH'
            | 2. string indicating one of the default slots in the instrument e.g. 'c', 'i1'
            | 3. tuple of (distance, theta) indicating module's offset from the center of the instrument, default unit deg
            | 4. fyst-trajectories ``InstrumentOffset`` (dx/dy in arcminutes, converted to polar internally)
        includes_instr_offset : bool, default False
            if "module" parameter is a tuple of (distance, theta), this includes the instrument offset

        Returns
        -------------------------
        TelescopePattern 
            A TelescopePattern object where the "boresight" is the path of the provided module.
        """
        
        if isinstance(module, InstrumentOffset):
            dist = math.sqrt(module.dx**2 + module.dy**2) / 60.0
            theta = math.degrees(math.atan2(module.dy, module.dx))
        elif includes_instr_offset:
            assert(len(module) == 2)
            dist, theta = u.Quantity(module[0], u.deg).value, u.Quantity(module[1], u.deg).value
        else:
            dist, theta = self._true_module_loc(module)
            
        az1, alt1 = self._transform_from_boresight(self.az_coord.value, self.alt_coord.value, dist, theta)

        data = {'time_offset': self.time_offset.value, 'lst': self.lst.value, 'az_coord': az1, 'alt_coord': alt1}

        return TelescopePattern(data, lat=self.param['lat'])

    def get_sky_pattern(self, ra_c=None, dec_c=None) -> SkyPattern:
        """
        Parameters
        ----------
        ra_c, dec_c
            Reference coordinates where the pixel (0, 0) corresponds
            If set to None, the first data point is used

        Returns
        -------------------------
        SkyPattern
            A SkyPattern object for the boresight.
        """

        start_dec = self.dec_coord[0].to(u.rad).value 

        sc = SkyCoord(self.ra_coord, self.dec_coord)

        if ra_c is None:
            ra_c = self.ra_coord[0]
        else:
            ra_c = u.Quantity(ra_c, u.deg)
        if dec_c is None:
            dec_c = self.dec_coord[0]
        else:
            dec_c = u.Quantity(dec_c, u.deg)
        sc0 = SkyCoord(ra_c, dec_c)

        # updated following the URL below
        # Still be careful when the dec is high, if the result will be
        # stored with the wcs with a certain projection
        dra, ddec = sc0.spherical_offsets_to(sc)

        data = {
            'time_offset': self.time_offset.value, 
            'x_coord': dra.value,
            'y_coord': ddec.value
        }

        if max(abs(data['x_coord'])) > 10:
            warnings.warn('This is a larger pattern and the conversion between x and y deltas and RA/DEC may be slightly off.')

        return SkyPattern(data=data)

    def save_param(self, param_json=None):
        """
        Save observation parameters.

        Parameters
        ----------------------------
        path_or_buf : str, file handle, or None; default None
            File path or object, if `None` is provided the result is returned as a dictionary.
        
        Returns
        ----------------------
        None or dict
            If `path_or_buf` is `None`, returns the resulting json format as a dictionary. Otherwise returns `None`.
        """

        param_temp = self._param.copy()
        if 'start_datetime' in param_temp.keys():
            param_temp['start_datetime'] = param_temp['start_datetime'].strftime('%Y-%m-%d %H:%M:%S %z') 

        # save param_json
        if param_json is None:
            return param_temp
        else:
            with open(param_json, 'w') as f:
                json.dump(param_temp, f)

    def save_data(self, path_or_buf=None, columns='default'):
        """
        Save kinematics data of the boresight. 

        Parameters
        ----------------------
        path_or_buf : str, file handle or None; default None
            File path or object, if `None` is provided the result is returned as a dictionary.
        columns : sequence, str or [dict of str -> str/Unit/None]; default 'default'
            Columns to write. If `dict`, map column names to their desired unit and use `None` if you would like to use the standard.
            'default' for ['time_offset', 'lst', 'az_coord', 'alt_coord', 'az_vel', 'alt_vel'].
            'all' for ['time_offset', 'az_coord', 'alt_coord', 'az_vel', 'alt_vel', 'vel', 'az_acc', 'alt_acc', 'acc', 'az_jerk', 'alt_jerk', 'jerk', 'lst', 'hour_angle', 'para_angle', 'rot_angle', 'ra_coord', 'dec_coord'].
        
        Returns
        ----------------------
        None or [dict of str -> array]
            If `path_or_buf` is `None`, returns the data as a dictionary mapping column name to values. Otherwise returns `None`.

        Examples
        ---------------------
        >>> telescope_pattern.save_data('file.csv', columns={'time_offset': 'sec', 'alt_coord': 'arcsec', 'az_coord': None})
        """

        # replace str options
        if columns == 'default':
            columns = ['time_offset', 'lst', 'az_coord', 'alt_coord', 'az_vel', 'alt_vel']
        elif columns == 'all':
            columns = ['time_offset', 'az_coord', 'alt_coord', 'az_vel', 'alt_vel', 'vel', 'az_acc', 'alt_acc', 'acc', 'az_jerk', 'alt_jerk', 'jerk', 'lst', 'hour_angle', 'para_angle', 'rot_angle', 'ra_coord', 'dec_coord']
        
        data = pd.DataFrame()

        # generate required data
        if isinstance(columns, dict):
            for col, unit in columns.items():
                if not unit is None:
                    data[col] = getattr(self, col).to(unit).value
                else:
                    data[col] = getattr(self, col).value
        else:
            for col in columns:
                data[col] = getattr(self, col).value
        
        # returning
        if path_or_buf is None:
            return data.to_dict('list')
        else:
            data.to_csv(path_or_buf, index=False)

    # ATTRIBUTES
    
    @property
    def param(self):
        """dict: Parameters inputted by user."""
        return_param = dict()
        for p, val in self._param.items():
            return_param[p] = val if self._param_units[p] is u.dimensionless_unscaled else val*self._param_units[p]
        return return_param

    @property
    def instrument(self):
        """ Insturment: Insturment object."""
        return self._instrument

    @instrument.setter
    def instrument(self, value):
        self._instrument = value

    @property
    def scan_duration(self):
        """Quantity: Total scan duration."""
        return self.time_offset[-1] + self.sample_interval

    @property
    def sample_interval(self):
        """Quantity: Time between samples."""
        return self._sample_interval*self._stored_units['time_offset']

    # Other Motions/Time

    @property
    def time_offset(self):
        """Quantity array: Time offsets."""
        return self._data['time_offset'].to_numpy()*self._stored_units['time_offset']
    
    @property
    def lst(self):
        """Quantity array: Local sidereal time."""
        return self._data['lst'].to_numpy()*self._stored_units['lst']

    @property
    def ra_coord(self):
        """Quantity array: Right ascension."""
        return self._norm_angle((self.lst - self.hour_angle).to(u.deg).value)*u.deg

    @property
    def dec_coord(self):
        """Quantity array: Declination."""
        lat_rad = radians(self._param['lat'])
        alt_coord_rad = self.alt_coord.to(u.rad).value
        az_coord_rad = self.az_coord.to(u.rad).value

        dec_rad = np.arcsin( sin(lat_rad)*np.sin(alt_coord_rad) + cos(lat_rad)*np.cos(alt_coord_rad)*np.cos(az_coord_rad) )
        return np.degrees(dec_rad)*u.deg

    @property
    def hour_angle(self):
        """Quantity array: Hour angle."""
        lat_rad = radians(self._param['lat'])
        alt_coord_rad = self.alt_coord.to(u.rad).value
        az_coord_rad = self.az_coord.to(u.rad).value
        dec_rad = self.dec_coord.to(u.rad).value

        hrang_rad = np.arccos( (np.sin(alt_coord_rad) - np.sin(dec_rad)*sin(lat_rad)) / (np.cos(dec_rad)*cos(lat_rad)) )
        
        mask = np.sin(az_coord_rad) > 0
        hrang_rad[mask] = 2*pi - hrang_rad[mask]

        return (hrang_rad*u.rad).to(u.hourangle)

    @property
    def para_angle(self):
        """Quantity array: Parallactic angle."""
        dec_rad = self.dec_coord.to(u.rad).value
        hour_angle_rad = self.hour_angle.to(u.rad).value
        lat_rad = radians(self._param['lat'])

        para_angle_deg = np.degrees(np.arctan2( 
            np.sin(hour_angle_rad), 
            np.cos(dec_rad)*tan(lat_rad) - np.sin(dec_rad)*np.cos(hour_angle_rad) 
        ))
        return self._norm_angle(para_angle_deg)*u.deg

    @property
    def rot_angle(self):
        """Quantity array: Field rotation (nasmyth_sign * elevation + parallactic angle)."""
        nasmyth_sign = 1 if FYST_NASMYTH_PORT == "right" else -1
        return self._norm_angle((nasmyth_sign * self.alt_coord + self.para_angle).value)*u.deg

    # Azimuthal/Elevation Motion

    @property
    def az_coord(self):
        """Quantity array: Azimuth coordinates (in terms of East of North)."""
        return self._data['az_coord'].to_numpy()*self._stored_units['az_coord']
    
    @property
    def alt_coord(self):
        """Quantity array: Elevation coordinates."""
        return self._data['alt_coord'].to_numpy()*self._stored_units['alt_coord']

    @property
    def az_vel(self):
        """Quantity array: Azimuth velocity."""
        return _central_diff(self.az_coord.value, self.sample_interval.value)*(self._stored_units['az_coord']/self._stored_units['time_offset'])

    @property
    def alt_vel(self):
        """Quantity array: Elevation velocity."""
        return _central_diff(self.alt_coord.value, self.sample_interval.value)*(self._stored_units['alt_coord']/self._stored_units['time_offset'])

    @property
    def vel(self):
        """Quantity array: Total velocity."""
        return np.sqrt(self.az_vel**2 + self.alt_vel**2)

    @property
    def az_acc(self):
        """Quantity array: Azimuth acceleration."""
        return _central_diff(self.az_vel.value, self.sample_interval.value)*(self._stored_units['az_coord']/self._stored_units['time_offset']**2)

    @property
    def alt_acc(self):
        """Quantity array: Elevation acceleration."""
        return _central_diff(self.alt_vel.value, self.sample_interval.value)*(self._stored_units['alt_coord']/self._stored_units['time_offset']**2)
    
    @property
    def acc(self):
        """Quantity array: Total acceleration."""
        return np.sqrt(self.az_acc**2 + self.alt_acc**2)

    @property
    def az_jerk(self):
        """Quantity array: Azimuth jerk."""
        return _central_diff(self.az_acc.value, self.sample_interval.value)*(self._stored_units['az_coord']/self._stored_units['time_offset']**3)
    
    @property
    def alt_jerk(self):
        """Quantity array: Elevation jerk."""
        return _central_diff(self.alt_acc.value, self.sample_interval.value)*(self._stored_units['alt_coord']/self._stored_units['time_offset']**3)
    
    @property
    def jerk(self):
        """Quantity array: Total jerk."""
        return np.sqrt(self.az_jerk**2 + self.alt_jerk**2)
