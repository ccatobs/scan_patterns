import copy
from math import pi, sin, cos, radians
import math
import warnings

import numpy as np
import pandas as pd
from astropy.utils import isiterable
import astropy.units as u
from astropy.convolution import Gaussian2DKernel, convolve_fft
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates
import matplotlib.patches as patches

from scanning.coordinates import SkyPattern, TelescopePattern

"""
TODO
- add bin plots

"""

# Modules and Instruments

def plot_module(mod, mode=None, path=None, show_plot=True):
    """
    Plot the pixel positions and default polarizations of a given module. 

    Parameters
    -----------------------------
    mod : Module
        A Module object. 
    mode : str or None, default None
        None for just the pixel positions.
        'pol' to include default orientations.
        'rhomubs' to include rhombus info.
        'wafer' to include wafer info. 
    path : str or None, default None
        If not None, saves the image to the file path. 
    show_plot : bool, default True
        Whether to display the resulting figure.
    """

    fig_mod = plt.figure('Module', figsize=(8, 8))
    ax_mod = plt.subplot2grid((1, 1), (0, 0))
    x_deg = mod.x.value
    y_deg = mod.y.value

    if mode is None:
        plot_mod = ax_mod.scatter(x_deg, y_deg, s=1)
    else:

        # mode
        if mode == 'pol':
            categories = np.unique(mod.pol.value)
            num_def = len(categories)
            y_label = 'Default Orientation [deg]'
            tick_locs = (categories + 7.5)*(num_def-1)/num_def
            c = mod.pol.value
        elif mode == 'rhombus':
            categories = np.unique(mod.rhombus)
            num_def = len(categories)
            y_label = 'Rhombus'
            tick_locs = (categories + 0.5)*(num_def-1)/num_def
            c = mod.rhombus
        elif mode == 'wafer':
            categories = np.unique(mod.wafer)
            num_def = len(categories)
            y_label = 'Wafer'
            tick_locs = (categories + 0.5)*(num_def-1)/num_def
            c = mod.wafer
        else:
            raise ValueError(f'mode={mode} is not valid')
        
        # determine which colormap to use
        if num_def <= 9:
            cmap = plt.cm.get_cmap('Set1', num_def)
        else:
            cmap = plt.cm.get_cmap('hsv', num_def)

        # plot
        plot_mod = ax_mod.scatter(x_deg, y_deg, c=c, cmap=cmap, s=1) # FIXME make pixel sizes proportional

        # color bar
        cbar = plt.colorbar(plot_mod, fraction=0.046, pad=0.04)
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(categories)
        cbar.ax.set_ylabel(y_label)

    ax_mod.set_aspect('equal', 'box')
    ax_mod.set(xlabel='x offset (deg)', ylabel='y offset (deg)', title='Map of Detector Array')
    ax_mod.grid()
    fig_mod.tight_layout()

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()

def instrument_config(instrument, path=None, show_plot=True):
    """
    Save a plot of the instrument configuration. 

    Parameters
    ------------------------
    instrument : Instrument
        An Instrument object. 
    path : str or None, default None
        If not None, saves the image to the file path. 
    show_plot : bool, default True
        Whether to display the resulting figure.
    """

    fig = plt.figure('Insturment', figsize=(8, 8))
    ax = plt.subplot2grid((1, 1), (0, 0))

    # default slots 
    n=100
    radius = 1.3/2
    for slot_name in instrument.slots.keys():

        # circle of diameter 1.3
        circle1_x = np.array([cos(2*pi/n*x)*radius for x in range(0,n+1)])
        circle1_y = np.array([sin(2*pi/n*x)*radius for x in range(0,n+1)])

        # apply offset
        slot_x, slot_y = instrument.get_slot_location(slot_name, from_boresight=True, polar=False).value
        circle2_x = circle1_x + slot_x
        circle2_y = circle1_y + slot_y

        ax.plot(circle2_x, circle2_y, ls='dashed', color='black')
        
    # modules 
    for identifier in instrument._modules.keys():
        mod = instrument.get_module(identifier, with_mod_rot=True, with_instr_rot=True, with_instr_offset=True, with_mod_offset=True)
        plt.scatter(mod.x.value, mod.y.value, s=0.5, label=identifier)
        
    ax.grid()
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.set(xlabel='x offset from boresight [deg]', ylabel='y offset from boresight [deg]', title='Instrument Configuration')

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()

# Sky Patterns and Telescope Patterns

def sky_path(pattern, module=None, path=None, show_plot=True):
    """
    Trace the path of the pattern on the sky. 

    Parameters
    ------------------------
    pattern : SkyPattern or TelescopePattern
        If SkyPattern object, trace the path in x and y offsets. 
        If TelescopePattern object, trace the path in RA/DEC. 
    module : str or two-tuple or [list of str or two-tuple] or None, default None
        Only applicable if pattern is TelescopePattern. Trace the paths of the listed modules/location. 
        If `None`, use boresight. 
        | 1. string indicating a module name in the instrument e.g. 'SFH'
        | 2. string indicating one of the default slots in the instrument e.g. 'c', 'i1'
        | 3. tuple of (distance, theta) indicating module's offset from the center of the instrument, default unit deg
    path : str or None, default None
        If not None, saves the image to the file path. 
    show_plot : bool, default True
        Whether to display the resulting figure.
    """

    fig_coord = plt.figure('coordinates', figsize=(8, 8))
    ax_coord = plt.subplot2grid((1, 1), (0, 0))

    # pattern is SkyPattern

    if isinstance(pattern, SkyPattern):
        ax_coord.plot(pattern.x_coord.value, pattern.y_coord.value)
        ax_coord.set(xlabel='x offset [deg]', ylabel='y offset [deg]', title=f'Sky Path')
        ax_coord.set_aspect('equal', 'box')

    # pattern is TelescopePattern
    elif isinstance(pattern, TelescopePattern):
        ax_coord.set(xlabel='RA [deg]', ylabel='DEC [deg]', title=f'Sky Path')
        if module is None:
            module = ['boresight']
        elif not isiterable(module):
            module = [module]

        for m in module:
            if m == 'boresight' or m == (0, 0):
                temp = pattern
            else:
                temp = pattern.view_module(m)

            ax_coord.plot(temp.ra_coord.value, temp.dec_coord.value, label=m, linewidth=0.5)
        
        ax_coord.invert_xaxis()
        ax_coord.legend(loc='upper right')
        ax_coord.set_aspect(1/cos(pattern.dec_coord[0].to(u.rad).value))
    else:
        raise TypeError('pattern is not of correct type')
    
    ax_coord.grid()

    fig_coord.tight_layout()

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()

def telescope_path(telescope_pattern, module=None, path=None, show_plot=True):
    """
    Trace the path of the pattern in AZ/EL. 

    Parameters
    ------------------------
    pattern : TelescopePattern
        A TelescopePattern object.  
    module : str or two-tuple or [list of str or two-tuple] or None, default None
        Trace the paths of the listed modules/location. 
        If `None`, use boresight. 
        | 1. string indicating a module name in the instrument e.g. 'SFH'
        | 2. string indicating one of the default slots in the instrument e.g. 'c', 'i1'
        | 3. tuple of (distance, theta) indicating module's offset from the center of the instrument, default unit deg
    path : str or None, default None
        If not None, saves the image to the file path. 
    show_plot : bool, default True
        Whether to display the resulting figure.
    """

    if module is None:
        module = ['boresight']
    elif not isiterable(module):
        module = [module]
    
    fig_coord = plt.figure('coordinates', figsize=(8, 8))
    ax_coord = plt.subplot2grid((1, 1), (0, 0))

    for m in module:
        if m == 'boresight' or m == (0, 0):
            temp = telescope_pattern
        else:
            temp = telescope_pattern.view_module(m)

        ax_coord.plot(temp.az_coord.value, temp.alt_coord.value, label=m)

    alt0 = telescope_pattern.alt_coord[0].to(u.rad).value
    ax_coord.set_aspect(1/cos(alt0))
    ax_coord.set(xlabel='Azimuth [deg]', ylabel='Elevation [deg]', title=f'Telescope Path')
    ax_coord.grid()
    ax_coord.legend(loc='upper right')

    fig_coord.tight_layout()

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()
    
def telescope_kinematics(telescope_pattern, module=None, plots=['coord', 'vel', 'acc', 'jerk'], path=None, show_plot=True):
    """
    Plot time vs. azimuthal and elevation kinematics. 

    Parameters
    ------------------------
    pattern : TelescopePattern
        A TelescopePattern object.  
    module : str or two-tuple or None, default None
        Select the module to use. If `None`, use boresight, otherwise:
        | 1. string indicating a module name in the instrument e.g. 'SFH'
        | 2. string indicating one of the default slots in the instrument e.g. 'c', 'i1'
        | 3. tuple of (distance, theta) indicating module's offset from the center of the instrument, default unit deg
    plots : list of str, default ['coord', 'vel', 'acc', 'jerk']
        List of which time plots to include. By default, includes all. 
    path : str or None, default None
        If not None, saves the image to the file path. 
    show_plot : bool, default True
        Whether to display the resulting figure.
    """

    if not module is None:
        telescope_pattern = telescope_pattern.view_module(module)

    num_plots = len(plots)
    i = 0 

    fig_motion = plt.figure('motion', figsize=(8, 2*num_plots))

    # time vs. azmiuth/elevation
    if 'coord' in plots:
        ax_coord = plt.subplot2grid((num_plots, 1), (i, 0))
        start_az = telescope_pattern.az_coord.value[0]
        start_alt = telescope_pattern.alt_coord.value[0]

        ax_coord.plot(telescope_pattern.time_offset.value, (telescope_pattern.az_coord.value - start_az)*cos(radians(start_alt)), label=f'AZ from {round(start_az, 2)} deg')
        ax_coord.plot(telescope_pattern.time_offset.value, telescope_pattern.alt_coord.value - start_alt, label=f'EL from {round(start_alt, 2)} deg')
        ax_coord.legend(loc='upper right')
        ax_coord.set(xlabel='Time Offset [s]', ylabel='Position [deg]', title=f'Time vs. Position')
        ax_coord.grid()
        i += 1

    # velocity
    if 'vel' in plots:
        ax_vel = plt.subplot2grid((num_plots, 1), (i, 0))
        ax_vel.plot(telescope_pattern.time_offset.value, telescope_pattern.vel.value, label='Total', c='black', ls='dashed', alpha=0.25)
        ax_vel.plot(telescope_pattern.time_offset.value, telescope_pattern.az_vel.value, label='AZ')
        ax_vel.plot(telescope_pattern.time_offset.value, telescope_pattern.alt_vel.value, label='EL')
        ax_vel.legend(loc='upper right')
        ax_vel.set(xlabel='Time offset [s]', ylabel='Velocity [deg/s]', title=f'Time vs. Velocity')
        ax_vel.grid()
        i += 1

    # acceleration
    if 'acc' in plots:
        ax_acc = plt.subplot2grid((num_plots, 1), (i, 0))
        ax_acc.plot(telescope_pattern.time_offset.value, telescope_pattern.acc.value, label='Total', c='black', ls='dashed', alpha=0.25)
        ax_acc.plot(telescope_pattern.time_offset.value, telescope_pattern.az_acc.value, label='AZ')
        ax_acc.plot(telescope_pattern.time_offset.value, telescope_pattern.alt_acc.value, label='EL')
        ax_acc.legend(loc='upper right')
        ax_acc.set(xlabel='Time offset [s]', ylabel='Acceleration [deg/s^2]', title=f'Time vs. Acceleration')
        ax_acc.grid()
        i += 1

    # jerk
    if 'jerk' in plots:
        ax_jerk = plt.subplot2grid((num_plots, 1), (i, 0))
        ax_jerk.plot(telescope_pattern.time_offset.value, telescope_pattern.jerk.value, label='Total', c='black', ls='dashed', alpha=0.25)
        ax_jerk.plot(telescope_pattern.time_offset.value, telescope_pattern.az_jerk.value, label='AZ')
        ax_jerk.plot(telescope_pattern.time_offset.value, telescope_pattern.alt_jerk.value, label='EL')
        ax_jerk.legend(loc='upper right')
        ax_jerk.set(xlabel='Time Offset (s)', ylabel='Jerk [deg/s^2]', title=f'Time vs. Jerk')
        ax_jerk.grid()
        i += 1

    fig_motion.tight_layout()

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()
    
def sky_kinematics(pattern, module=None, plots=['coord', 'vel', 'acc', 'jerk'], path=None, show_plot=True):
    """
    Plot time vs. x and y kinematics. 

    Parameters
    ------------------------
    pattern : SkyPattern or TelescopePattern
        A SkyPattern or TelescopePattern object.
    module : str or two-tuple or None, default None
        Select the module to use. If `None`, use boresight, otherwise:
        | 1. string indicating a module name in the instrument e.g. 'SFH'
        | 2. string indicating one of the default slots in the instrument e.g. 'c', 'i1'
        | 3. tuple of (distance, theta) indicating module's offset from the center of the instrument, default unit deg
    plots : list of str, default ['coord', 'vel', 'acc', 'jerk']
        List of which time plots to include. By default, includes all. 
    path : str or None, default None
        If not None, saves the image to the file path. 
    show_plot : bool, default True
        Whether to display the resulting figure.
    """

    if isinstance(pattern, TelescopePattern):
        if module is None:
            pattern = pattern.get_sky_pattern()
        else:
            pattern = pattern.view_module(module).get_sky_pattern()

    num_plots = len(plots)
    i = 0 

    fig_motion = plt.figure('motion', figsize=(8, 2*num_plots))

    # time vs. azmiuth/elevation
    if 'coord' in plots:
        ax_coord = plt.subplot2grid((num_plots, 1), (i, 0))
        ax_coord.plot(pattern.time_offset.value, pattern.x_coord.value, label=f'x offset [deg]')
        ax_coord.plot(pattern.time_offset.value, pattern.y_coord.value, label=f'y offset [deg]')
        ax_coord.legend(loc='upper right')
        ax_coord.set(xlabel='Time Offset [s]', ylabel='Position [deg]', title=f'Time vs. Position')
        ax_coord.grid()
        i += 1

    # velocity
    if 'vel' in plots:
        ax_vel = plt.subplot2grid((num_plots, 1), (i, 0))
        ax_vel.plot(pattern.time_offset.value, pattern.vel.value, label='Total', c='black', ls='dashed', alpha=0.25)
        ax_vel.plot(pattern.time_offset.value, pattern.x_vel.value, label='x')
        ax_vel.plot(pattern.time_offset.value, pattern.y_vel.value, label='y')
        ax_vel.legend(loc='upper right')
        ax_vel.set(xlabel='Time offset [s]', ylabel='Velocity [deg/s]', title=f'Time vs. Velocity')
        ax_vel.grid()
        i += 1

    # acceleration
    if 'acc' in plots:
        ax_acc = plt.subplot2grid((num_plots, 1), (i, 0))
        ax_acc.plot(pattern.time_offset.value, pattern.acc.value, label='Total', c='black', ls='dashed', alpha=0.25)
        ax_acc.plot(pattern.time_offset.value, pattern.x_acc.value, label='x')
        ax_acc.plot(pattern.time_offset.value, pattern.y_acc.value, label='y')
        ax_acc.legend(loc='upper right')
        ax_acc.set(xlabel='Time offset [s]', ylabel='Acceleration [deg/s^2]', title=f'Time vs. Acceleration')
        ax_acc.grid()
        i += 1

    # jerk
    if 'jerk' in plots:
        ax_jerk = plt.subplot2grid((num_plots, 1), (i, 0))
        ax_jerk.plot(pattern.time_offset.value, pattern.jerk.value, label='Total', c='black', ls='dashed', alpha=0.25)
        ax_jerk.plot(pattern.time_offset.value, pattern.x_jerk.value, label='x')
        ax_jerk.plot(pattern.time_offset.value, pattern.y_jerk.value, label='y')
        ax_jerk.legend(loc='upper right')
        ax_jerk.set(xlabel='Time Offset (s)', ylabel='Jerk [deg/s^2]', title=f'Time vs. Jerk')
        ax_jerk.grid()
        i += 1

    fig_motion.tight_layout()

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()

# Observation

def filter_observation(obs, plot='elevation', min_elev=30, max_elev=75, min_rot_rate=0, path=None, show_plot=True):
    """TODO"""

    if plot not in ['elevation', 'airmass', 'para_angle', 'rot_angle', 'rot_rate']:
        raise ValueError(f'"plot" = {plot} is not a valid choice')

    # plotting 
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot2grid((1, 1), (0, 0), fig=fig)

    cm = plt.cm.get_cmap('tab20').colors

    if len(obs.dec) > 10:
        warnings.warn('colors may be difficult to distinguish if more than 10 sources')

    # conditions
    min_elev = u.Quantity(min_elev, u.deg).value
    max_elev = u.Quantity(max_elev, u.deg).value
    min_rot_rate = u.Quantity(min_rot_rate, u.deg/u.hour).value

    # LOOP OVER EACH SOURCE 

    max_abs_rot_rate = 0
    for i, dec in enumerate(obs.dec.value):

        # apply conditions
        alt = obs.get_elevation(i).value
        mask_min_elev = alt >= min_elev

        alt = alt[mask_min_elev]
        rot_rate = obs.get_rot_rate(i).value[mask_min_elev]
        mask_max_elev = (alt <= max_elev) & (abs(rot_rate) > min_rot_rate)

        # x_values and label for legend
        if obs.datetime_axis:
            x_values = obs.datetime_range[mask_min_elev]
            hrang_range = obs.get_hrang_range(i).value[mask_min_elev]
            ra = obs.ra[i].to(u.hourangle).value
            label = f'({ra}h {dec}N)'
        else:
            x_values = hrang_range = obs.get_hrang_range().value[mask_min_elev]
            label = f'dec={dec}N'

        mask_hrang = hrang_range < 0

        # make plot
        if plot == 'elevation':
            ax.scatter(x_values[mask_max_elev], alt[mask_max_elev], label=label, color=cm[2*i], s=1)
            ax.scatter(x_values[~mask_max_elev], alt[~mask_max_elev], color=cm[2*i+1], s=1)
        elif plot == 'airmass':
            airmass = 1/np.cos(pi/2 - np.radians(alt))
            ax.scatter(x_values[mask_max_elev], airmass[mask_max_elev], label=label, color=cm[2*i], s=1)
            ax.scatter(x_values[~mask_max_elev], airmass[~mask_max_elev], color=cm[2*i+1], s=1)
        elif plot == 'para_angle':
            para_angle = obs.norm_angle(obs.get_para_angle(i).value)[mask_min_elev]
            ax.scatter(x_values[mask_max_elev], para_angle[mask_max_elev], label=label, color=cm[2*i], s=1)
            ax.scatter(x_values[~mask_max_elev], para_angle[~mask_max_elev], color=cm[2*i+1], s=1)
        elif plot == 'rot_angle':
            rot_angle = obs.get_rot_angle(i).value[mask_min_elev]
            ax.scatter(x_values[mask_max_elev], rot_angle[mask_max_elev], label=label, color=cm[2*i], s=1)
            ax.scatter(x_values[~mask_max_elev], rot_angle[~mask_max_elev], color=cm[2*i+1], s=1)
        elif plot == 'rot_rate':
            ax.scatter(x_values[mask_max_elev], rot_rate[mask_max_elev], label=label, color=cm[2*i], s=1)
            ax.scatter(x_values[~mask_max_elev], rot_rate[~mask_max_elev], color=cm[2*i+1], s=1)
        
            max_abs_rot_rate = max( max_abs_rot_rate, np.max(abs(rot_rate[mask_max_elev])) )

    #  handle xticks 

    if obs.datetime_axis:
        xlabel = f'Time starting from {obs.datetime_range[0].strftime("%Y-%m-%d %H:%M:%S %z")}'
        xlim=(obs.datetime_range[0], obs.datetime_range[-1])
        ax.set(xlim=xlim)
        fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    else:
        xlabel = 'Hourangle [hours]'
        xticks = [i for i in range(-12, 13, 3)]
        xtick_labels = [f'{t + 24}h' if t < 0 else f'{t}h' for t in xticks]
        ax.set(xticks=xticks, xticklabels=xtick_labels)

    # other settings

    # secondary axis for azimuthal scale factor

    def transform(x):
        new_x = 1/np.sin(np.arccos(1/x))
        mask = np.isinf(new_x)
        new_x[mask] = math.nan
        return new_x

    def inverse(x):
        new_x = 1/np.cos(np.arcsin(1/x)) 
        mask = np.isinf(new_x)
        new_x[mask] = math.nan
        return new_x

    # elevation
    if plot == 'elevation':
        ax.set(
            title='Elevation', xlabel=xlabel,
            ylabel='Elevation [deg]', ylim=(min_elev, 87)
        )
        ax.set_yticks(np.append(ax.get_yticks(), max_elev))
        ax.axhline(max_elev, ls='dashed', color='black')

        # Suppress div-by-zero from matplotlib's secondary_yaxis transform eval.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ax_elev_right = ax.secondary_yaxis('right', functions=( lambda x: 1/np.cos(np.radians(x)), lambda x: np.degrees(np.arccos(1/x)) ))
        ax_elev_right.set(ylabel='Azimuthal Scale Factor')
        ax_elev_right.set_yticks([1.2, 2, 3, 4, 5, 6, 15])

    # airmass
    elif plot == 'airmass':
        ax.set_yscale('function', functions=(lambda x: np.log10(x), lambda x: 10**x))
        max_airmass = 1/cos(pi/2 - radians(min_elev))
        ax.set(
            title='Airmass', xlabel=xlabel,
            ylabel='Airmass', ylim=(1.001, max_airmass)
        )
        ax.invert_yaxis()
        ax.axhline(1/cos(pi/2 - radians(max_elev)), ls='dashed', color='black')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ax_airmass_right = ax.secondary_yaxis('right', functions=(transform, inverse))
        ax_airmass_right.set(ylabel='Azimuthal Scale Factor')
        ax_airmass_right.set_yticks([1.2, 1.5, 2, 2.5, 3, 4])
    
    # parallactic angle
    elif plot == 'para_angle':
        ax.set(title='Parallactic Angle', xlabel=xlabel, ylabel='Parallactic Angle [deg]')

    # rotation angle
    elif plot == 'rot_angle':
        ax.set(title='Field Rotation Angle', xlabel=xlabel, ylabel='Field Rotation Angle [deg]')

    # rotation rate
    elif plot == 'rot_rate':
        max_abs_rot_rate = max(max_abs_rot_rate, 60)
        ax.set(
            title='Field Rotation Angle Rate', xlabel=xlabel, 
            ylabel='Field Rotation Angle Rate [deg/hr]', ylim=(-max_abs_rot_rate, max_abs_rot_rate)
        )
        ax.axhline(min_rot_rate, ls=':', color='black')
        ax.axhline(-min_rot_rate, ls=':', color='black')
        ax.set_yticks(np.append(ax.get_yticks(), [-min_rot_rate, min_rot_rate]))

    # final touchups
    if plot != 'rot_hist':
        ax.legend(loc='lower right')
        ax.grid()

    fig.tight_layout()
        
    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()

def field_rotation_hist(obs, min_elev=30, max_elev=75, min_rot_rate=0, path=None, show_plot=True):
    """TODO"""
    ax_ncols = 2
    ax_nrows = math.ceil(len(obs.dec)/ax_ncols)
    fig, ax = plt.subplots(ax_nrows, ax_ncols, sharex=True, sharey=True, figsize=(8, 3*ax_nrows))
    cm = plt.cm.get_cmap('tab20').colors

    # conditions
    min_elev = u.Quantity(min_elev, u.deg).value
    max_elev = u.Quantity(max_elev, u.deg).value
    min_rot_rate = u.Quantity(min_rot_rate, u.deg/u.hour).value

    # LOOP OVER EACH SOURCE 

    for i, dec in enumerate(obs.dec.value):

        # apply conditions
        alt = obs.get_elevation(i).value
        mask_min_elev = alt >= min_elev

        alt = alt[mask_min_elev]
        rot_rate = obs.get_rot_rate(i).value[mask_min_elev]
        mask_max_elev = (alt <= max_elev) & (abs(rot_rate) > min_rot_rate)

        # x_values and label for legend
        if obs.datetime_axis:
            hrang_range = obs.get_hrang_range(i).value[mask_min_elev]
            ra = obs.ra[i].to(u.hourangle).value
            label = f'({ra}h {dec}N)'
        else:
            hrang_range = obs.get_hrang_range().value[mask_min_elev]
            label = f'dec={dec}N'

        mask_hrang = hrang_range < 0

        # make plot
        ax_hist_row = math.floor(i/ax_ncols)
        ax_hist_col = i%ax_ncols

        rot_angle = obs.get_rot_angle(i).value[mask_min_elev]
        a1 = rot_angle[mask_hrang & mask_max_elev]%90
        a2 = rot_angle[~mask_hrang & mask_max_elev]%90
        total_num = len(a1) + len(a2)

        if ax_nrows == 1 and ax_ncols == 1:
            ax_hist = ax
        elif ax_nrows == 1:
            ax_hist = ax[ax_hist_col]
        else:
            ax_hist = ax[ax_hist_row, ax_hist_col]

        ax_hist.hist(a1, bins=range(0, 91, 1), color=cm[2*i+1], label='hourangle < 0', weights=np.full(len(a1), 1/total_num))
        ax_hist.hist(a2, bins=range(0, 91, 1), color=cm[2*i], label='hourangle >= 0', histtype='step', weights=np.full(len(a2), 1/total_num), linewidth=2)
        ax_hist.set(xlabel='Rotation Angle (mod 90) [deg]', ylabel='Fraction of Time', title=label)
        ax_hist.legend(loc='upper right')
        ax_hist.grid()
        ax_hist.xaxis.set_tick_params(labelbottom=True)
        ax_hist.yaxis.set_tick_params(labelbottom=True)
        ax_hist.set_xticks(range(0, 91, 15))    

    fig.tight_layout()
        
    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()

# Simulation 

def hitmap(sim, convolve=True, norm_time=False, kept_hits=True, removed_hits=False, total_hits=False, path=None, show_plot=True, with_rot0=True):
    """
    Get a hitmap. 

    Parameters
    ------------------------------
    sim : Simulation
        A simulation object. 
    convolve : bool; default True
        Whether to convolve the hitmap. 
    norm_time : bool; default False
        `True` for hits/px per total scan duration. `False` for hits/px.
    kept_hits removed_hits total_hits : bool or int; default True False False respectively
        Whether to plot seperate hitmaps for kept/rem/total hits (`True`) or not (Fal`se). 
        If an integer is passed, the corresponding plot will show, with the color bar maximum set to this interger.
    with_rot0 : bool; default True
        Include initial field rotation into side plot of detector elements. 
    path : str or None; default None
        If not `None`, saves the image to the file path. 
    show_plot : bool; default True
        Whether to display the resulting figure.
    """

    # initialize figure
    num_hitmaps = np.count_nonzero([kept_hits, removed_hits, total_hits])
    if num_hitmaps == 0:
        raise ValueError('at least one of kept_hits, removed_hits, or total_hits must be plotted')
    
    fig, ax = plt.subplots(1, num_hitmaps+1, sharex=True, sharey=True, figsize=((num_hitmaps+1)*4, 4))

    if not norm_time:
        hit_per_str = 'px'
    else:
        hit_per_str = 'px/s'

    # KEPT HITS 

    ax_i = 1

    if kept_hits:
        sky_hist, hitmap_range = sim.sky_histogram('kept', convolve=convolve, norm_time=norm_time)
        hitmap_range = hitmap_range.value
        ax[ax_i].set_title(f'Kept hits/{hit_per_str}')

        min_pixel_deg = min(hitmap_range)
        max_pixel_deg = max(hitmap_range)

        vmax = None if kept_hits is True else kept_hits
        pcm = ax[ax_i].imshow(sky_hist, extent=[min_pixel_deg, max_pixel_deg, min_pixel_deg, max_pixel_deg], vmin=0, vmax=vmax, interpolation='nearest', origin='lower')
        ax[ax_i].set_aspect('equal', 'box')
        ax[ax_i].set(xlabel='x offset (deg)', ylabel='y offset (deg)')

        ax[ax_i].axvline(x=0, c='black')
        ax[ax_i].axhline(y=0, c='black')

        divider = make_axes_locatable(ax[ax_i])
        cax = divider.append_axes("bottom", size="3%", pad=0.5)
        fig.colorbar(pcm, cax=cax, orientation='horizontal')

        ax_i += 1
    
    if removed_hits:
        sky_hist, hitmap_range = sim.sky_histogram('removed', convolve=convolve, norm_time=norm_time)
        hitmap_range = hitmap_range.value
        ax[ax_i].set_title(f'Removed hits/{hit_per_str}')

        min_pixel_deg = min(hitmap_range)
        max_pixel_deg = max(hitmap_range)

        vmax = None if removed_hits is True else removed_hits
        pcm = ax[ax_i].imshow(sky_hist, extent=[min_pixel_deg, max_pixel_deg, min_pixel_deg, max_pixel_deg], vmin=0, vmax=vmax, interpolation='nearest', origin='lower')
        ax[ax_i].set_aspect('equal', 'box')
        ax[ax_i].set(xlabel='x offset (deg)', ylabel='y offset (deg)')

        ax[ax_i].axvline(x=0, c='black')
        ax[ax_i].axhline(y=0, c='black')

        divider = make_axes_locatable(ax[ax_i])
        cax = divider.append_axes("bottom", size="3%", pad=0.5)
        fig.colorbar(pcm, cax=cax, orientation='horizontal')

        ax_i += 1
    
    if total_hits:
        sky_hist, hitmap_range = sim.sky_histogram('total', convolve=convolve, norm_time=norm_time)
        hitmap_range = hitmap_range.value
        ax[ax_i].set_title(f'Total hits/{hit_per_str}')

        min_pixel_deg = min(hitmap_range)
        max_pixel_deg = max(hitmap_range)

        vmax = None if total_hits is True else total_hits
        pcm = ax[ax_i].imshow(sky_hist, extent=[min_pixel_deg, max_pixel_deg, min_pixel_deg, max_pixel_deg], vmin=0, vmax=vmax, interpolation='nearest', origin='lower')
        ax[ax_i].set_aspect('equal', 'box')
        ax[ax_i].set(xlabel='x offset (deg)', ylabel='y offset (deg)')

        ax[ax_i].axvline(x=0, c='black')
        ax[ax_i].axhline(y=0, c='black')

        divider = make_axes_locatable(ax[ax_i])
        cax = divider.append_axes("bottom", size="3%", pad=0.5)
        fig.colorbar(pcm, cax=cax, orientation='horizontal')

        ax_i += 1

    # DETECTOR AND SKY PATTERN

    mask_det = sim.det_mask
    x_mod = sim.module.x.value[mask_det]
    y_mod = sim.module.y.value[mask_det]
    rot0 = sim.field_rotation[0].to(u.rad).value

    if len(x_mod) > 15:
        s_size = 0.05
    else:
        s_size = 10

    ax[0].plot(sim.sky_pattern.x_coord.value, sim.sky_pattern.y_coord.value, color='blue', alpha=0.5)

    if with_rot0:
        ax[0].scatter(x_mod*cos(rot0) - y_mod*sin(rot0), x_mod*sin(rot0) + y_mod*cos(rot0), s=s_size, color='red')
        ax[0].set_title('Pixel Positions (incl. initial field rotation)')
    else:
        ax[0].scatter(x_mod, y_mod, s=s_size, color='red')
        ax[0].set_title('Pixel Positions')

    ax[0].set_aspect('equal', 'box')
    ax[0].set(xlabel='x offset (deg)', ylabel='y offset (deg)')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("bottom", size="3%", pad=0.5)
    cax.axis('off')
    
    fig.tight_layout()

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()

def pxan_detector(sim, norm_pxan=True, norm_time=False, kept_hits=True, removed_hits=False, total_hits=False, path=None, show_plot=True):
    """
    Get a hitmap of the number of hits per detector element on a pixel on the sky (specified in `pxan_list` or `pxan_lim`).

    Parameters
    ------------------------------
    sim : Simulation
        A simulation object. 
    norm_pxan : bool; default True
        Whether to average the hits by dividing the total hits by the number of pixels. 
    norm_time : bool; default False
        `True` for hits/px per total scan duration. `False` for hits/px.
    kept_hits removed_hits total_hits : bool or int; default True False False respectively
        Whether to plot seperate hitmaps for kept/rem/total hits (`True`) or not (Fal`se). 
        If an integer is passed, the corresponding plot will show, with the color bar maximum set to this interger.
    path : str or None; default None
        If not `None`, saves the image to the file path. 
    show_plot : bool; default True
        Whether to display the resulting figure.
    """

    # initialize figure
    num_hitmaps = np.count_nonzero([kept_hits, removed_hits, total_hits])
    if num_hitmaps == 0:
        raise ValueError('at least one of kept_hits, removed_hits, or total_hits must be plotted')
    
    fig, ax = plt.subplots(1, num_hitmaps+1, sharex=True, sharey=True, figsize=((num_hitmaps+1)*4, 4))

    # parameters

    if norm_time:
        hit_per_str = 'px'
    else:
        hit_per_str = 'px/s'

    cm = plt.cm.get_cmap('viridis')

    mask_det = sim.det_mask
    x_mod = sim.module.x.value[mask_det]
    y_mod = sim.module.y.value[mask_det]
    #rot0 = math.radians(sim._field_rotation[0])

    # KEPT HITS

    ax_i = 1

    if kept_hits:
        det_hist, _ = sim.det_histogram(hits='kept', norm_pxan=norm_pxan, norm_time=norm_time)
        det_hist = det_hist[~np.isnan(det_hist)]

        vmax = None if kept_hits is True else kept_hits
        sc = ax[ax_i].scatter(x_mod, y_mod, c=det_hist, cmap=cm, vmin=0, vmax=vmax, s=5)
        ax[ax_i].set_aspect('equal', 'box')
        ax[ax_i].set(xlabel='x offset (deg)', ylabel='y offset (deg)', title=f'Kept hits/{hit_per_str}')

        divider = make_axes_locatable(ax[ax_i])
        cax1 = divider.append_axes("bottom", size="3%", pad=0.5)
        fig.colorbar(sc, cax=cax1, orientation='horizontal')

        ax_i += 1
    
    if removed_hits:
        det_hist, _ = sim.det_histogram(hits='removed', norm_pxan=norm_pxan, norm_time=norm_time)
        det_hist = det_hist[~np.isnan(det_hist)]

        vmax = None if removed_hits is True else removed_hits
        sc = ax[ax_i].scatter(x_mod, y_mod, c=det_hist, cmap=cm, vmin=0, vmax=vmax, s=5)
        ax[ax_i].set_aspect('equal', 'box')
        ax[ax_i].set(xlabel='x offset (deg)', ylabel='y offset (deg)', title=f'Removed hits/{hit_per_str}')

        divider = make_axes_locatable(ax[ax_i])
        cax1 = divider.append_axes("bottom", size="3%", pad=0.5)
        fig.colorbar(sc, cax=cax1, orientation='horizontal')

        ax_i += 1
    
    if total_hits:
        det_hist, _ = sim.det_histogram(hits='total', norm_pxan=norm_pxan, norm_time=norm_time)
        det_hist = det_hist[~np.isnan(det_hist)]

        vmax = None if total_hits is True else total_hits
        sc = ax[ax_i].scatter(x_mod, y_mod, c=det_hist, cmap=cm, vmin=0, vmax=vmax, s=5)
        ax[ax_i].set_aspect('equal', 'box')
        ax[ax_i].set(xlabel='x offset (deg)', ylabel='y offset (deg)', title=f'Total hits/{hit_per_str}')

        divider = make_axes_locatable(ax[ax_i])
        cax1 = divider.append_axes("bottom", size="3%", pad=0.5)
        fig.colorbar(sc, cax=cax1, orientation='horizontal')

        ax_i += 1

    # pattern it self
    x_coord = sim.sky_pattern.x_coord.value
    y_coord = sim.sky_pattern.y_coord.value

    ax[0].plot(x_coord, y_coord)
    ax[0].set_aspect('equal', 'box')
    ax[0].set(xlabel='x offset (deg)', ylabel='y offset (deg)', title='Path of Instrument Module')
    divider = make_axes_locatable(ax[0])
    cax4 = divider.append_axes("bottom", size="3%", pad=0.5)
    cax4.axis('off')

    fig.tight_layout()

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()

def pxan_time(sim, bin=1, norm_pxan=True, norm_time=False, kept_hits=True, removed_hits=False, total_hits=False, path=None, show_plot=True):
    """
    Get a histogram of the number of hits per timestamp on a pixel on the sky (specified in `pxan_list` or `pxan_lim`).

    Parameters
    ------------------------------
    sim : Simulation
        A simulation object. 
    bin : float, Quantity, or str; default 1, default unit seconds
        Desired time bin. 
    norm_pxan : bool; default True
        Whether to average the hits by dividing the total hits by the number of pixels. 
    norm_time : bool; default False
        `True` for hits/px per total scan duration. `False` for hits/px.
    kept_hits removed_hits total_hits : bool; default True False False respectively
        Whether to plot seperate hitmaps for kept/rem/total hits (`True`) or not (Fal`se). 
    path : str or None; default None
        If not `None`, saves the image to the file path. 
    show_plot : bool; default True
        Whether to display the resulting figure.
    """

    bin = u.Quantity(bin, u.s).value

    # initialize figure
    num_hitmaps = np.count_nonzero([kept_hits, removed_hits, total_hits])
    if num_hitmaps == 0:
        raise ValueError('at least one of kept_hits, removed_hits, or total_hits must be plotted')
    
    fig, ax = plt.subplots(1, num_hitmaps, sharex=True, sharey=True, figsize=(num_hitmaps*8, 4))

    if num_hitmaps == 1:
        ax = [ax]

    # parameters

    if not norm_time:
        hit_per_str = 'px'
    else:
        hit_per_str = 'px/s'

    # KEPT HITS

    ax_i = 0

    if kept_hits:
        time_hist, time_offset = sim.time_histogram(hits='kept', norm_pxan=norm_pxan, norm_time=norm_time)
        mask_nan = ~np.isnan(time_hist)
        time_hist = time_hist[mask_nan]
        time_offset = time_offset[mask_nan]

        scan_duration = sim.sky_pattern.scan_duration.value
        bin_edges = np.arange(0, scan_duration + bin, bin)

        ax[ax_i].hist(time_offset, bins=bin_edges, weights=time_hist)
        ax[ax_i].set(xlabel='Time Offset (sec)', ylabel='Number of Hits', title=f'Kept hits/{hit_per_str}')

        ax_i += 1

    if removed_hits:
        time_hist, time_offset = sim.time_histogram(hits='removed', norm_pxan=norm_pxan, norm_time=norm_time)
        mask_nan = ~np.isnan(time_hist)
        time_hist = time_hist[mask_nan]
        time_offset = time_offset[mask_nan]

        scan_duration = sim.sky_pattern.scan_duration.value
        bin_edges = np.arange(0, scan_duration + bin, bin)

        ax[ax_i].hist(time_offset, bins=bin_edges, weights=time_hist)
        ax[ax_i].set(xlabel='Time Offset (sec)', ylabel='Number of Hits', title=f'Removed hits/{hit_per_str}')

        ax_i += 1
    
    if total_hits:
        time_hist, time_offset = sim.time_histogram(hits='total', norm_pxan=norm_pxan, norm_time=norm_time)
        mask_nan = ~np.isnan(time_hist)
        time_hist = time_hist[mask_nan]
        time_offset = time_offset[mask_nan]

        scan_duration = sim.sky_pattern.scan_duration.value
        bin_edges = np.arange(0, scan_duration + bin, bin)

        ax[ax_i].hist(time_offset, bins=bin_edges, weights=time_hist)
        ax[ax_i].set(xlabel='Time Offset (sec)', ylabel='Number of Hits', title=f'total hits/{hit_per_str}')

        ax_i += 1
    
    
    fig.tight_layout()

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()

def pxan_polarization(sim, norm_pxan=True, norm_time=False, kept_hits=True, removed_hits=False, total_hits=False, path=None, show_plot=True):
    """
    Get a histogram of how many times each detector element, grouped by their initial pixel positions, hit that region

    Parameters
    ------------------------------
    sim : Simulation
        A simulation object. 
    norm_pxan : bool; default True
        Whether to average the hits by dividing the total hits by the number of pixels. 
    norm_time : bool; default False
        `True` for hits/px per total scan duration. `False` for hits/px.
    kept_hits removed_hits total_hits : bool; default True False False respectively
        Whether to plot seperate hitmaps for kept/rem/total hits (`True`) or not (Fal`se). 
    path : str or None; default None
        If not `None`, saves the image to the file path. 
    show_plot : bool; default True
        Whether to display the resulting figure.
    """

    # initialize figure
    num_hitmaps = np.count_nonzero([kept_hits, removed_hits, total_hits])
    if num_hitmaps == 0:
        raise ValueError('at least one of kept_hits, removed_hits, or total_hits must be plotted')
    
    fig, ax = plt.subplots(1, num_hitmaps, sharex=True, sharey=True, figsize=(num_hitmaps*8, 4))

    if num_hitmaps == 1:
        ax = [ax]

    ax_i = 0
    pol = sim.module.pol.value[sim.det_mask]
    all_pol = np.unique(pol)
    width = all_pol[1] - all_pol[0] - 1

    if kept_hits:
        det_hist, _ = sim.det_histogram(hits='kept', norm_pxan=norm_pxan, norm_time=norm_time)

        for p in all_pol:
            mask = pol == p
            num_p = np.sum(det_hist[mask])
            ax[ax_i].bar(p, height=num_p, width=width)

        ax[ax_i].set(xlabel='Polarization Geometry [deg]', ylabel='# of hits', title='Kept Hits')
        ax[ax_i].set_xticks(all_pol)

        ax_i +=1

    if removed_hits:
        det_hist, _ = sim.det_histogram(hits='removed', norm_pxan=norm_pxan, norm_time=norm_time)

        for p in all_pol:
            mask = pol == p
            num_p = np.sum(det_hist[mask])
            ax[ax_i].bar(p, height=num_p, width=width)

        ax[ax_i].set(xlabel='Polarization Geometry [deg]', ylabel='# of hits', title='Kept Hits')
        ax[ax_i].set_xticks(all_pol)

        ax_i +=1

    if total_hits:
        det_hist, _ = sim.det_histogram(hits='total', norm_pxan=norm_pxan, norm_time=norm_time)

        for p in all_pol:
            mask = pol == p
            print(det_hist)
            num_p = np.sum(det_hist[mask])
            ax[ax_i].bar(p, height=num_p, width=width)

        ax[ax_i].set(xlabel='Polarization Geometry [deg]', ylabel='# of hits', title='Kept Hits')
        ax[ax_i].set_xticks(all_pol)

        ax_i +=1
    
    fig.tight_layout()

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()

def polarization_histogram(sim, kept_hits=True, removed_hits=False, total_hits=False, stacked=False, path=None, show_plot=True):
    """
    Get a histogram of which polarization geometries have been captured.
     Parameters
    ------------------------------
    sim : Simulation
        A simulation object. 
    kept_hits removed_hits total_hits : bool; default True False False respectively
        Whether to plot seperate hitmaps for kept/rem/total hits (`True`) or not (False`). 
    stacked : bool; default False
        Whether to display histogram as stacked bars (`True`) or overlaying bars (`False`). 
    path : str or None; default None
        If not `None`, saves the image to the file path. 
    show_plot : bool; default True
        Whether to display the resulting figure.
    """

    # initialize figure
    num_hitmaps = np.count_nonzero([kept_hits, removed_hits, total_hits])
    if num_hitmaps == 0:
        raise ValueError('at least one of kept_hits, removed_hits, or total_hits must be plotted')
    
    fig, ax = plt.subplots(1, num_hitmaps, sharex=True, sharey=True, figsize=(num_hitmaps*8, 4))

    if num_hitmaps == 1:
        ax = [ax]

    ax_i = 0

    if kept_hits:
        pol_and_rot, all_pol_counts = sim.pol_histogram('kept')
        all_pol = np.array(pol_and_rot.columns)
        num_ts = len(pol_and_rot.index)

        if stacked:
            pol_and_rot = pol_and_rot.to_numpy()
            all_pol_counts = np.tile(all_pol_counts, (num_ts, 1))
            ax[ax_i].hist(pol_and_rot, bins=90, range=(0, 90), weights=all_pol_counts, stacked=True, label=[f'{p}$^\circ$' for p in all_pol], density=True)
        else:
            for ip, p in enumerate(all_pol):
                ax[ax_i].hist(pol_and_rot.iloc[:, ip], bins=90, range=(0, 90), weights=[all_pol_counts[ip]]*num_ts, label=f'{p}$^\circ$', histtype='step')

        ax[ax_i].grid()
        ax[ax_i].set(xlabel='Polarization Geometry + Field Rotation [deg]', ylabel='Number of Hits', title='Kept Hits', xlim=(0, 90))
        ax[ax_i].legend(loc='upper right', title='Initial Polarization')

        ax_i += 1

    if removed_hits:
        pol_and_rot, all_pol_counts = sim.pol_histogram('removed')
        all_pol = np.array(pol_and_rot.columns)
        num_ts = len(pol_and_rot.index)
        
        if stacked:
            pol_and_rot = pol_and_rot.to_numpy()
            all_pol_counts = np.tile(all_pol_counts, (num_ts, 1))
            ax[ax_i].hist(pol_and_rot, bins=90, range=(0, 90), weights=all_pol_counts, stacked=True, label=[f'{p}$^\circ$' for p in all_pol], density=True)
        else:
            for ip, p in enumerate(all_pol):
                ax[ax_i].hist(pol_and_rot.iloc[:, ip], bins=90, range=(0, 90), weights=[all_pol_counts[ip]]*num_ts, label=f'{p}$^\circ$', histtype='step')

        ax[ax_i].grid()
        ax[ax_i].set(xlabel='Polarization Geometry + Field Rotation [deg]', ylabel='Number of Hits', title='Removed Hits', xlim=(0, 90))
        ax[ax_i].legend(loc='upper right', title='Initial Polarization')

        ax_i += 1

    if total_hits:
        pol_and_rot, all_pol_counts = sim.pol_histogram('total')
        all_pol = np.array(pol_and_rot.columns)
        num_ts = len(pol_and_rot.index)

        if stacked:
            pol_and_rot = pol_and_rot.to_numpy()
            all_pol_counts = np.tile(all_pol_counts, (num_ts, 1))
            ax[ax_i].hist(pol_and_rot, bins=90, range=(0, 90), weights=all_pol_counts, stacked=True, label=[f'{p}$^\circ$' for p in all_pol])
        else:
            for ip, p in enumerate(all_pol):
                ax[ax_i].hist(pol_and_rot.iloc[:, ip], bins=90, range=(0, 90), weights=[all_pol_counts[ip]]*num_ts, label=f'{p}$^\circ$', histtype='step')

        ax[ax_i].grid()
        ax[ax_i].set(xlabel='Polarization Geometry + Field Rotation [deg]', ylabel='Number of Hits', title='Total Hits', xlim=(0, 90))
        ax[ax_i].legend(loc='upper right', title='Initial Polarization')

    fig.tight_layout()

    # saving
    if not path is None:
        plt.savefig(path)
        print(f'Saved to {path}.')
    
    # displaying
    if show_plot:
        plt.show()
    else:
        plt.close()
