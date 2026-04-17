import math
from math import sin, cos, tan
import warnings

import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.utils import isiterable
from .coordinates import FYST_LOC, _nasmyth_sign

class Observation():

    def __init__(self, dec, hrang_range=None, datetime_range=None, ra=None, freq=None, lat=FYST_LOC.lat) -> None:
        """
        Initialize an observation of a particular source(s) for a particular time range. 

        Parameters
        ------------------------
        dec : angle-like or iterable, default unit deg
            Declination(s) of your source.
        hrang_range : (angle-like, angle-like) or None, default unit hourangle
            (start, end) range of hour angles to consider. If both hrang_range and datetime_range is None, this is by default the full -12 to 12 range. 
            Note that start must be less than end. Cannot be larger than -12 to 12 range.
            Cannot be used with datetime_range. 
        datetime_range : (str or datetime-like, str or datetime-like) or None, default None. 
            (start, end) range of datetimes to consider. 
            Cannot be used with hrang_range. Must include ra. 
        ra : angle-like or iterable, default unit deg
            Right ascensions(s) of your source. Required if datetime_range is passed.
        freq : time-like, default unit seconds
            The timestep between points. By default, the will be enough so that there are 1500 points. 
        lat : angle-like, default unit deg, default FYST_LOC.lat
            Latitude of observation. 
        """

        # declination(s)

        if not isiterable(dec):
            dec = [dec]
        else:
            dec = list(dec)
        
        for i, val in enumerate(dec):
            dec[i] = u.Quantity(val, u.deg).value

        self._dec = dec

        # latitude
        self._lat = u.Quantity(lat, u.deg).value

        # hrang vs datetime

        if (not datetime_range is None) and (not hrang_range is None):
            raise TypeError('one (and only one) of datetime_range or hrang_range must be provided')
        
        elif not datetime_range is None:
            self.datetime_axis = True
            datetime_start = pd.Timestamp(datetime_range[0])
            datetime_end = pd.Timestamp(datetime_range[1])

            if datetime_end <= datetime_start:
                raise ValueError('datetime_end must be greater than datetime_start')

            # get freq 
            if freq is None:
                freq = ((datetime_end - datetime_start)/1500).to_pytimedelta()
            else:
                freq = TimeDelta(u.Quantity(freq, u.s)).to_datetime()
            self._freq = freq.total_seconds()

            # get ra 
            if ra is None:
                raise TypeError('"ra" must be provided with "datetime_range"')

            if not isiterable(ra):
                ra = [ra]
            else:
                ra = list(ra)
            
            if len(ra) != len(dec):
                raise ValueError('"ra" must be the same length as "dec"')

            for i, val in enumerate(ra):
                ra[i] = u.Quantity(val, u.deg).value
            
            self._ra = ra

            # get datetime range
            datetime_range = pd.date_range(datetime_start, datetime_end, freq=freq, inclusive='left')
            self._datetime_range = datetime_range.to_pydatetime()
        
        else:
            self.datetime_axis = False

            if hrang_range is None:
                hrang_range = (-12, 12)

            hrang_start = u.Quantity(hrang_range[0], u.hourangle).value
            hrang_end = u.Quantity(hrang_range[1], u.hourangle).value

            if hrang_end <= hrang_start:
                raise ValueError('hrang_end must be greater than hrang_start')
            elif hrang_end - hrang_start > 24:
                raise ValueError('hrang_range cannot be more than 24 hours apart.')

            # get freq 
            if freq is None:
                freq = ((pd.Timedelta(hours=hrang_end) - pd.Timedelta(hours=hrang_start))/1500).to_pytimedelta()
                freq = (hrang_end - hrang_start)/1500*3600
            else:
                freq = TimeDelta(u.Quantity(freq, u.s)).to_datetime().total_seconds()

            SIDEREAL_TO_UT1 = 1.002737909350795
            self._freq = freq/SIDEREAL_TO_UT1

            # get hrang range 
            self._hrang_range = np.arange(hrang_start, hrang_end, freq/3600)

    def filter(self, min_elev=30, max_elev=75, min_rot_rate=0):
        """
        Filter the time ranges when 

        Parameters
        --------------------------
        min_elev : angle-like, default unit deg, default 30 deg
            Minimum elevation to consider. 
        max_elev : angle-like, default unit deg, default 75 deg
            Maximum elevation to consider. 
        min_rot_rate : angular velocity-like, default unit deg, default 0 deg/s
            Minimum absolute field rotation rate to consider. 

        Return
        -------------
        dict 
            In the format of {index of source : list of tuples}. 
            Each tuple is a range of valid times, in datetime.datetime if datetime_range was provided, otherwise in hour angles. 
        """

        min_elev = u.Quantity(min_elev, u.deg).value
        max_elev = u.Quantity(max_elev, u.deg).value
        min_rot_rate = u.Quantity(min_rot_rate, u.deg/u.s).value

        # choose which time range to show
        if self.datetime_axis:
            timestamps = self.datetime_range
        else:
            timestamps = self.get_hrang_range().value

        ranges = dict()

        # loop over each source
        for i, dec in enumerate(self.dec.value):

            # apply conditions
            alt = self.get_elevation(i).value
            rot_rate = self.get_rot_rate(i).value
            mask = (alt >= min_elev) & (alt <= max_elev) & (abs(rot_rate) >= min_rot_rate)

            # get valid ranges

            valid_ranges = []
            current_range = []

            currently_valid = False
            for j, valid in enumerate(mask):
                if not valid == currently_valid: 
                    if valid: # started new range 
                        current_range.append(timestamps[j])
                    else: # ended range
                        current_range.append(timestamps[j-1])
                        valid_ranges.append(current_range)
                        current_range = []
                    currently_valid = valid
            
            if len(current_range) == 1:
                current_range.append(timestamps[-1])
                valid_ranges.append(tuple(current_range))

            # choose label
            if self.datetime_axis:
                ra = self.ra[i].to(u.hourangle).value
                label = f'({ra}h {dec}N): '
            else: 
                label = f'(dec={dec}N): '

            print(label, valid_ranges)
            ranges[i] = valid_ranges
        
        return ranges

    def norm_angle(self, angle):
        angle = np.array(angle).copy()

        # make values between -180 and 180
        angle = (angle + 180)%360 - 180

        # smooth out 
        angle_diff = np.diff(angle, append=math.nan)
        angle_diff_indices = [i for i, x in enumerate(abs(angle_diff)) if x > 350]

        for i in angle_diff_indices:
            diff = -360 if angle_diff[i] < 0 else 360
            angle[i+1:] = angle[i+1:] - diff

        return angle

    def get_rot_rate(self, i):
        rot_angle = self.get_rot_angle(i).value
        freq_hr = self.freq.to(u.hour).value
        return np.diff(rot_angle, append=math.nan)/freq_hr*(u.deg/u.hour)

    def get_elevation(self, i):
        lat_rad = self.lat.to(u.rad).value
        hrang_rad = self.get_hrang_range(i).to(u.rad).value
        dec_rad = self.dec[i].to(u.rad).value
        alt_rad = np.arcsin(sin(dec_rad)*sin(lat_rad) + cos(dec_rad)*cos(lat_rad)*np.cos(hrang_rad))
        return np.degrees(alt_rad)*u.deg
    
    def get_para_angle(self, i):
        lat_rad = self.lat.to(u.rad).value
        hrang_rad = self.get_hrang_range(i).to(u.rad).value
        dec_rad = self.dec[i].to(u.rad).value
        para_angle_rad = np.arctan2(
            np.sin(hrang_rad),
            (cos(dec_rad)*tan(lat_rad) - sin(dec_rad)*np.cos(hrang_rad))
        )
        return np.degrees(para_angle_rad)*u.deg
    
    def get_rot_angle(self, i):
        return self.norm_angle(
            self.get_para_angle(i).value + _nasmyth_sign() * self.get_elevation(i).value
        ) * u.deg

    def get_hrang_range(self, i=None):
        # get hour angle representation
        if self.datetime_axis:
            datetime_range = Time(self.datetime_range, location=(self.lat, 0*u.deg)) # arbitrary longitude value
            lst = datetime_range.sidereal_time('apparent')
            return (lst - self.ra[i]).to(u.hourangle)
        else:
            return self._hrang_range*u.hourangle

    @property
    def dec(self):
        return self._dec*u.deg
    
    @property 
    def lat(self):
        return self._lat*u.deg

    @property
    def freq(self):
        return self._freq*u.s

    @property
    def datetime_range(self):
        try:
            return self._datetime_range
        except AttributeError:
            return None

    @property
    def ra(self):
        try:
            return self._ra*u.deg
        except AttributeError:
            return None
    

