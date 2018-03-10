from __future__ import print_function, absolute_import, division
from builtins import range  # overload range to ensure python3 style
import matplotlib
matplotlib.use('AGG')
from .CBcm import make_color_maps
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import ImageGrid
from . import snapshot
from . import manipulate as man
from . import utils
from . import measure
from multiprocess import Pool
import itertools
import datetime
today = datetime.date.today().strftime('%b%d')


class KeyboardInterruptError(Exception):
    pass


def plot_single(name,
                settings=None,
                folder="./",
                base="./",
                panel_mode="xy",
                log_scale=True,
                in_min=-1,
                in_max=2.2,
                com=False,
                plotCompanionCOM=False,
                plotDiskCOM=False,
                plotPotMin=False,
                first_only=False,
                gal_num=-1,
                colormap='BuOr',
                xlen=20,
                ylen=20,
                zlen=20,
                colorbar='None',
                parttype='stars',
                NBINS=512):
    """
    Plot a single snapshot
    Kwargs:
      name: Snapshot name
      folder: output dir
      base: where the snapshots are
      panel_mode: Project that you want
          xy=plot only xy (default)
          three=plot three panel
          small=plot small three panel
      log_scale: log scale of density
      in_min: Minimum for colormap
      in_max: Maximum for colormap
      com: Offset by disk center of mass?
      first_only: Only plot the first galaxy
      colormap: ame of colorblind safe color map (CBcm.py)
                          or standard matplotlib colormap
      xlen: Gives x length (in kpc)
      ylen: Gives y length (in kpc)
      zlen: Gives z length (in kpc)
      colorbar: Colorbar mode (None (Default), Single... )
      parttype: particle type (Gas, Halo, Stars...)
    """

    output_dir = folder + "plot"
    snapbase = name
    fname = base + snapbase
    outname = output_dir + name + ".png"
    if settings is None:
        settings = utils.make_settings(panel_mode = panel_mode,
                                       log_scale = log_scale,
                                       in_min = in_min,
                                       in_max = in_max,
                                       com = com,
                                       first_only = first_only,
                                       gal_num = gal_num,
                                       plotCompanionCOM = plotCompanionCOM,
                                       plotDiskCOM = plotDiskCOM,
                                       plotPotMin = plotPotMin,
                                       xlen = xlen,
                                       ylen = ylen,
                                       zlen = zlen)

    try:
        settings['colormap'] = make_color_maps()[0][colormap]
    except KeyError:
        settings['colormap'] = colormap

    settings['colorbar'] = colorbar
    settings['NBINS'] = NBINS

    part_names = ['gas',
                  'halo',
                  'stars',
                  'bulge',
                  'sfr',
                  'other']

    if type(parttype) == list:
        print("Must choose one particle type")
        return 0
    if type(parttype) == int:
        if parttype > -1:
            settings['parttype'] = part_names[parttype]
        else:
            print("Must choose one particle type")
            return 0
    elif type(parttype) == str:
        if parttype in part_names:
            settings['parttype'] = parttype
    else:
        print("Not a valid particle type")
        return 0

    plot_stars(snapshot.Snapshot(fname).bin_snap(settings), outname, settings)


def plot_loop(snaps,
              settings=None,
              folder="./",
              output_dir="./",
              panel_mode="xy",
              log_scale=True,
              in_min=-1,
              in_max=2.2,
              com=False,
              gal_num=-1,
              first_only=False,
              plotCompanionCOM=False,
              colormap='BuOr',
              xlen=20,
              ylen=20,
              zlen=20,
              colorbar='None',
              parttype='stars',
              NBINS=512,
              snapbase="snap_"):
    """
    Plot a range of snapshots
    Args:
      snaps: The snapshots you want to plot
      If argument is a integer than take a range from zero to arg.
      Else if arg. is a tuple, list, or numpy array then:
        1 element: range(0, snaps)
        2 elements: range(snaps[0], snaps[1])
        3 elements: range(snaps[0], snaps[1], snaps[2])
        NOTE: The last snapshot is added to make these inclusive lists
        more than 3 elements: use snaps
        See utils.list_snapshots() for more info

    Kwargs:
      settings: settings dictionary (see utils.make_settings())
      name: Snapshot name
      folder:  where the snapshots are
      output_dir: output directory
      panel_mode: Project that you want
          xy=plot only xy (default)
          three=plot three panel
          small=plot small three panel
      log_scale: log scale of density
      in_min: Minimum for colormap
      in_max: Maximum for colormap
      com: Offset by disk center of mass?
      first_only: Only plot the first galaxy
      colormap: ame of colorblind safe color map (CBcm.py)
                          or standard matplotlib colormap
      xlen: Gives x length (in kpc)
      ylen: Gives y length (in kpc)
      zlen: Gives z length (in kpc)
      colorbar: Colorbar mode (None (Default), Single... )
      parttype: particle type (Gas, Halo, Stars...)
    """
    import re

    output_name = output_dir+"plot"

    if settings is None:
        settings = utils.make_settings(panel_mode=panel_mode,
                                       log_scale=log_scale,
                                       in_min=in_min,
                                       in_max=in_max,
                                       com=com,
                                       first_only=first_only,
                                       gal_num=gal_num,
                                       plotCompanionCOM=plotCompanionCOM,
                                       xlen=xlen,
                                       ylen=ylen,
                                       zlen=zlen,
                                       colorbar=colorbar,
                                       NBINS=NBINS)
    try:
        settings['colormap'] = make_color_maps()[0][colormap]
    except KeyError:
        settings['colormap'] = colormap

    part_names = ['gas',
                  'halo',
                  'stars',
                  'bulge',
                  'sfr',
                  'other']

    if hasattr(parttype, '__iter__'):
        print("Must choose one particle type")
        return 0
    elif isinstance(parttype, int):
        if parttype > -1:
            settings['parttype'] = part_names[parttype]
        else:
            print("Must choose one particle type")
            return 0
    elif isinstance(parttype, str):
        if parttype in part_names:
            settings['parttype'] = parttype
    else:
        print("Not a valid particle type")
        return 0

    # Turn list of snapnumbers into names if not already
    snaps = utils.list_snapshots(snaps, folder, snapbase)
    settings_array = []
    for i, s in enumerate(snaps):
        settings_array.append(settings.copy())
        settings_array[i]['filename'] = s
        n = re.search(snapbase+'([0-9]{3})', s).group(1)
        settings_array[i]['outputname'] = output_name + n + ".png"

    pool = Pool()
    pool.map(plot_stars_helper, settings_array)


def plot_stars_helper(settings):
    """
    Helper function for multiprocessing pool
    """
    fname = settings['filename']
    outname = settings['outputname']
    snap = snapshot.Snapshot(fname)
    if snap is None:
      raise IOError("Snapshot {:s} not found.".format(fname))

    plot_stars(snap.bin_snap(settings), outname, settings)


def plot_panel(axis, perspective, bin_dict, settings, axes=[0, 1]):
    """
    Plot a single projection to an axis

    Args:
      axis: Matplotlib axis to plot to
      perspective: Entry in bin_dict
      bin_dict: Dictionary with projections
      settings: Settings dictionary

    Kwargs:
      axes: Give the axes used. X then Y, with 0=x, 1=y, 2=z

    """
    im_func = settings['im_func']
    Z = bin_dict[perspective]

    centerx = (bin_dict['{:s}x'.format(perspective)][:-1] +
               bin_dict['{:s}x'.format(perspective)][1:]) / 2
    centery = (bin_dict['{:s}y'.format(perspective)][:-1] +
               bin_dict['{:s}y'.format(perspective)][1:]) / 2
    centerX, centerY = np.meshgrid(centerx, centery, indexing='ij')  # ij indexing to match with hist2d

    Zmin = settings['in_min']
    Zmax = settings['in_max']
    plotCompanionCOM = settings['plotCompanionCOM']
    cmap = settings['colormap']
    extent = [settings['xlen'], settings['ylen'], settings['zlen']]
    extent = [extent[i] for i in axes]

    # Scale the data for pretty plots
    if ((Zmax == 0.0) & (Zmin == 0.0)):
            Zmin = Z[Z > -np.inf].min()
            Zmax = Z.max()

    #we likely have NaN values, ignore those and keep them as NaNs
    with np.errstate(invalid='ignore'):
      Z[Z < Zmin] = Zmin
      Z[Z > Zmax] = Zmax

    if im_func is not None:
        im = axis.pcolormesh(centerX, centerY, im_func(Z), vmin=Zmin, vmax=Zmax,
                             cmap=cmap)
    else:
        im = axis.pcolormesh(centerX, centerY, Z, vmin=Zmin, vmax=Zmax,
                             cmap=cmap)

    axis.set_xlim([-extent[0], extent[0]])
    axis.set_ylim([-extent[1], extent[1]])

    #axis.axis([-extent[0], extent[0], -extent[1], extent[1]])

    if plotCompanionCOM:
        companionCOM = bin_dict['companionCOM']
        if (np.abs(companionCOM[axes[0]]) < extent[0]
                and np.abs(companionCOM[axes[1]]) < extent[1]):
            axis.plot(companionCOM[axes[0]], companionCOM[axes[1]], marker='o')

    return im


def plot_stars(binDict,
               outname,
               settings,
               returnOnly=False):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Get the settings
    panels = settings['panel_mode']
    scale = settings['log_scale']
    cbarmode = settings['colorbar']
    co = utils.Conversions(UnitMass_in_g=settings['UnitMass_in_g'],
                           UnitVelocity_in_cm_per_s=settings['UnitVelocity_in_cm_per_s'],
                           UnitLength_in_cm=settings['UnitLength_in_cm'])

#for mass weighted histogram
#size in units of scale length

    if panels == "starsgas":
        #Want to plot stars next to gas
        snaptime = binDict['snaptime']

        fig, (ax1, ax2) = plt.subplots(1, 2,
                                       figsize=(20.0, 10.0),
                                       sharey=True)

        plot_panel(ax1, 'Z2', binDict, settings, axes=[0, 1])

        ax1.set_xlabel('X [kpc]', fontsize=25)
        ax1.set_ylabel('Y [kpc]', fontsize=25)

        ax1.annotate('Stars', xy=(.5, .8), xycoords='axes fraction',
                     textcoords='axes fraction',
                     xytext=(.5, .8), fontsize='larger')

        plot_panel(ax2, 'G', binDict,  settings, axes=[0, 1])

        ax2.set_xlabel('X [kpc]', fontsize=25)

        ax2.annotate('Gas', xy=(.5, .8), xycoords='axes fraction',
                     textcoords='axes fraction',
                     xytext=(.5, .8), fontsize='larger')

        fig.suptitle("t="+str(round(snaptime*co.UnitTime_in_Gyr*1000.0, 1)) +
                     "Myr", fontsize=25)

    if panels == "three":
        #Want all perspectives
        snaptime = binDict['snaptime']

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3,
                                            figsize=(15.0, 5.0))

        im = plot_panel(ax1, 'Z2', binDict, settings, axes=[0, 1])

        ax1.set_xlabel('X [kpc]', fontsize=15)
        ax1.set_ylabel('Y [kpc]', fontsize=15)

        im = plot_panel(ax2, 'H', binDict,  settings, axes=[0, 2])

        ax2.set_xlabel('X [kpc]', fontsize=15)
        ax2.set_ylabel('Z [kpc]', fontsize=15)

        im = plot_panel(ax3, 'H2', binDict,  settings, axes=[1, 2])

        ax3.set_xlabel('Y [kpc]', fontsize=15)
        ax3.set_ylabel('Z [kpc]', fontsize=15)

        fig.suptitle("t="+str(round(snaptime*co.UnitTime_in_Gyr*1000.0, 1)) +
                     "Myr", fontsize=25)

    if panels == "xy":
      # Single panel
        snaptime = binDict['snaptime']

        fig = plt.figure(1, figsize=(20.0, 10.0))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(1, 1),
                         label_mode='L',
                         cbar_pad=0.05,
                         cbar_mode=cbarmode,
                         cbar_location='right')

        grid[0].set_xlabel('X [kpc]', fontsize=25)
        grid[0].set_ylabel('Y [kpc]', fontsize=25)

        im = plot_panel(grid[0], 'Z2', binDict, settings, axes=[0, 1])

        fig.suptitle("t="+str(round(snaptime*co.UnitTime_in_Gyr*1000.0, 1)) +
                     "Myr", fontsize=25)

        cbar = grid.cbar_axes[0].colorbar(im)
        if scale:
            cbar.set_label_text('Log[M$_{\odot}$ pc$^{-2}$]')
        else:
            cbar.set_label_text('M$_{\odot}$ pc$^{-2}$')

    if panels == "small":
      # Smaller edge-on panels next to a face-on panel
        snaptime = binDict['snaptime']

        fig = plt.figure(1, figsize=(20.0, 10.0))
        ax1 = fig.add_subplot(111)

        im = plot_panel(ax1, 'Z2', binDict,  settings, axes=[0, 1])

        ax1.set_xlabel('X [kpc]', fontsize=25)
        ax1.set_ylabel('Y [kpc]', fontsize=25)
        ax1.set_aspect(1.)
        divider = make_axes_locatable(ax1)
        ax_x = divider.append_axes("top", size=1.2, pad=0.1, sharex=ax1)
        ax_y = divider.append_axes("right", size=1.2, pad=0.1, sharey=ax1)

        settings_x = settings
        settings_x['plotCompanionCOM'] = False
        binDict['H'] = binDict['H'].T
        im = plot_panel(ax_x, 'H', binDict, settings_x, axes=[0, 2])

       # ax_x.axis([-lengthX,lengthX,-lengthZ,lengthZ])
        #ax_x.set_xlabel('X [kpc]', fontsize=25)
        ax_x.get_xaxis().set_visible(False)
        ax_x.set_ylabel('Z [kpc]', fontsize=25)
        im = plot_panel(ax_y, 'H2', binDict,  settings_x, axes=[1, 2])

     #   ax_y.axis([-lengthY,lengthY,-lengthZ,lengthZ])
       # ax_y.set_xlabel('Y [kpc]', fontsize=25)
        ax_y.get_yaxis().set_visible(False)
        ax_y.set_xlabel('Z [kpc]', fontsize=25)

        fig.suptitle("t="+str(round(snaptime*co.UnitTime_in_Gyr*1000.0, 1)) +
                     "Myr", fontsize=25)

    if not returnOnly:
        plt.savefig(outname, bbox_inches='tight')
        fig.clf()
        plt.close()
    return fig


def plot_multi_ring_offsets(snaps,
                            num_centers=1,
                            filename='./',
                            outname='./snap',
                            first_only=True,
                            com=True,
                            plot=False,
                            contours=20,
                            axis=None):
    """
    Compute centers over a range snaps using multiple rings for the disk
    Todo: roll functionality into other functions
    """
    if isinstance(snaps, int):
        snap_ids = range(0, snaps+1)
    elif hasattr(snaps, '__iter__'):
        if len(snaps) == 1:
            snap_ids = range(0, snaps+1)
        elif len(snaps) == 2:
            snap_ids = range(snaps[0], snaps[1]+1)
        elif len(snap) == 3:
            snap_ids = range(snaps[0], snaps[1]+1, snaps[2])
        else:
            snap_ids = snaps

    snapbase = 'snap_'
    settings = utils.make_settings()
    settings['first_only'] = first_only
    settings['com'] = com

    if axis:
        ax1 = axis
    else:
        plt.clf()
        fig = plt.figure(1, figsize=(10.0, 10.0))
        ax1 = fig.add_subplot(111)

    (times,
     bar_dist,
     disk_dist,
     bar_offset) = measure.loop_centers(snap_ids,
                                        settings=settings,
                                        plot=plot,
                                        contours=contours,
                                        filename=filename,
                                        output=outname,
                                        num_centers=num_centers)

    for i in range(num_centers):
        ax1.plot(times, bar_offset[:, i], label='%d' % (i))
        ax1.set_ylabel('Bar displacement \n from disk center [kpc]',
                       fontsize=10)
        ax1.set_xlabel('Time [Gyr]')

    if not axis:  # only plot if we werent given an axis
        ax1.legend(fancybox=True, loc='upper right')
        plt.savefig(outname+today+'.png',
                    bbox_inches='tight')
        fig.clf()
        plt.close()


def compare_centers(folders,
                    names,
                    parttype='stars',
                    begin=0,
                    end=40,
                    Rd=0,
                    snapbase='snap_',
                    outname='',
                    outfolder='/home/pardy/plots/',
                    first_only=True,
                    com=True,
                    contours=20,
                    axis=None,
                    returnOnly=False,
                    settings=None,
                    plot_settings='-',
                    plot=False,
                    measure_fourier=False):
    """
    Function is being deprecated... functionality too limited and convoluted
    Compare the bar-disk offset of a few different sets of snapshots
    """

    if settings is None:
        settings = utils.make_settings(first_only=first_only,
                                       com=com,
                                       xlen=20,
                                       ylen=20,
                                       in_min=0,
                                       parttype=parttype,
                                       snapbase=snapbase)
    if returnOnly:
        offsetDict = {}
    else:
        if axis:
            ax1 = axis
        else:
            plt.clf()
            fig = plt.figure(1, figsize=(10.0, 10.0))
            ax1 = fig.add_subplot(111)

    Rd = [Rd]*len(folders)
    plot_settings = [plot_settings]*len(folders)
    parttype = [parttype]*len(folders)

    if not all(isinstance(x, str) for x in plot_settings):
        print("ALL ELEMENTS IN plot_settings MUST BE STRINGS")
        return

    snap_ids = range(begin, end+1)
    for folder, name, Rad, ptype, style in zip(folders,
                                               names,
                                               Rd,
                                               parttype,
                                               plot_settings):

        settings['filename'] = folder

        (times,
         bar_dist,
         disk_dist,
         bar_offset) = measure.loop_centers(snap_ids,
                                            settings=settings,
                                            Rd=Rad,
                                            plot=plot,
                                            contours=contours,
                                            measure_fourier=measure_fourier,
                                            parttype=parttype,
                                            filename=folder)




        if returnOnly:
            offsetDict[name] = bar_offset
        else:
            ax1.plot(times, bar_offset, linestyle=style, label=name)

    if not returnOnly:
        ax1.set_ylabel('Bar displacement \n from disk center [kpc]',
                       fontsize=10)
        ax1.set_xlabel('Time [Gyr]')
        ax1.legend(fancybox=True, loc='upper right')
        if not axis:  # only plot if we werent given an axis
            plt.savefig(outfolder+outname+today+'.png',
                        bbox_inches='tight')
            fig.clf()
            plt.close()
    else:
        return  times, offsetDict


def plot_loop_centers(snaps,
                      filename='./',
                      output='./snap',
                      Rd=0,
                      first_only=True,
                      com=True,
                      plot=False,
                      contours=20,
                      settings=None,
                      measure_fourier=False,
                      axis=None,
                      parttype='stars',
                      snapbase='snap_'):

    """
    Plotting wrappers around measure.loop_centers

    times, bar_dist, disk_dist, bar_offset = plot_tools.loop_centers(0, 40)

    Args:
          snaps (see plot_loop)

    Kwargs:
        filename: Path to directory
        output: Path to output and any prefix to filename
        Rd: The half mass radius of the disk (if 0 uses out ellipse)
        first_only: Only compute the centers on the first (most massive) galaxy?
        com: Subtract the center of mass from all of the measurements?
        plot: Make a plot of the contours for each snapshot?
        contours: How many contours per snapshot.
        settings: Change any of the settings from plot_tools.utils.make_settings()
        measure_fourier: Measure the Fourier modes in the bar? (Currently broken)
        axis: Add plot to an already created axis.
        parttype: Which particle type to use for plotting.
        snapbase: Prefix for snapshot names

    """

    if settings is None:
        settings = utils.make_settings(first_only=first_only,
                                       com=com,
                                       xlen=20,
                                       ylen=20,
                                       in_min=0,
                                       parttype=parttype,
                                       snapbase=snapbase,
                                       filename=filename)

    snaps = utils.list_snapshots(snaps, settings['filename'], setting['snapbase'])
    if axis:
        ax1 = axis[0]
        if len(axis) < 2:
            ax2 = None
        else:
            ax2 = axis[1]
    else:
        plt.clf()
        fig = plt.figure(1, figsize=(20.0, 20.0))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

    outname = output+"_distances.png"

    (times,
     bar_dist,
     disk_dist,
     bar_offset) = measure.loop_centers(snaps,
                                        settings=settings,
                                        Rd=Rd,
                                        plot=plot,
                                        contours=contours,
                                        measure_fourier=measure_fourier,
                                        parttype=parttype,
                                        output=output)

    if not returnOnly:
        ax1.plot(times, bar_dist, label="Bar")
        ax1.plot(times, disk_dist, label="Disk")
        ax1.set_ylabel('Displacement from \n halo center [kpc]', fontsize=10)
        if ax2:
            ax1.set_xticks([])
        else:
            ax1.set_xlabel('Time [Gyr]', fontsize=20)
        ax1.legend(fancybox=True, loc='upper right')
        if ax2:
            ax2.plot(times, bar_offset)
            ax2.set_ylabel('Bar displacement \n from disk center [kpc]',
                           fontsize=10)
            ax2.set_xlabel('Time [Gyr]', fontsize=20)

    if not returnOnly:
        if not axis:
            plt.savefig(outname, bbox_inches='tight')
            fig.clf()
            plt.close()

    return times, bar_dist, disk_dist, bar_offset


def plot_loop_fourier(snaps,
                      modes,
                      outname='./fourierModes.png',
                      folder='./',
                      maxAmp=False,
                      settings=None,
                      axis=None,
                      first_only=False,
                      com=False,
                      parttype='stars'):
    """
    Plot the Fourier modes for a range of snapshots.
    Wrapper to measure.loop_fourier()
    """
    if settings is None:
        settings = utils.make_settings(first_only=first_only,
                                       com=com,
                                       xlen=15,
                                       ylen=15,
                                       parttype=parttype,
                                       filename=folder,
                                       snapbase='snap_'
                                       )
    snaps = utils.list_snapshots(snaps, settings['filename'], setting['snapbase'])

    if axis:
        ax1 = axis
    else:
        plt.clf()
        fig = plt.figure(1, figsize=(20.0, 20.0))
        ax1 = fig.add_subplot(111)
    if not hasattr(modes, '__iter__'):
        modes = [modes]

    amps = measure.loop_fourier(snaps, modes,
                                settings,
                                max_amp=maxAmp)

    amp = np.zeros(len(amps)*len(modes)).reshape(len(amps), len(modes))
    times = np.zeros(len(amps))

    for i in range(len(amps)):
        amp[i, :] = amps[i][0]
        times[i] = amps[i][1]

    for i, m in enumerate(modes):
        ax1.plot(times, amp[:, i], label="A"+str(m))

    ax1.set_xlabel('Time [Gyr]', fontsize=20)
    # Only measure one mode
    if len(modes) == 1:
        if maxAmp:
            ax1.set_ylabel('A'+str(modes[0]), fontsize=20)
        else:
            ax1.set_ylabel('<A'+str(modes[0])+'>', fontsize=20)
    else:  # Measure more than one
        if maxAmp:
            ax1.set_ylabel('Amp', fontsize=20)
        else:
            ax1.set_ylabel('<Amp>', fontsize=20)
            plt.legend()
    if not axis:
        plt.savefig(outname, bbox_inches='tight')
        fig.clf()
        plt.close()


def plot_contours(bin_dict,
                  measurements,
                  bar_ind,
                  disk_ind,
                  pot_cent,
                  settings,
                  axis=None):
    """
    Plot a snapshot projection and the ellipses
    """

    ell_arts = []
    outname = settings['outputname']+".png"
    outfile = settings['outputname']+".txt"

    lengthX = settings['xlen']
    lengthY = settings['ylen']

    eccs = measurements['eccs']
    majors = measurements['majors']
    minors = measurements['minors']
    axes_ratios = measurements['axes_ratios']
    xCenters = measurements['xCenters']
    yCenters = measurements['yCenters']
    ellipses = measurements['ellipses']
    angles = measurements['angles']

    if axis is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
    else:
        ax = axis

    im = plot_panel(ax, 'Z2', bin_dict, settings)

    if axis is None:
        f = open(outfile, 'w')
        f.write("#Angle   Major    Minor   Eccentricity")
        f.write("min-maj-ratio Xcent    Ycent  \n")
        f.write("\n")

    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Y [kpc]')
    for i, (e,
            major,
            minor,
            ecc,
            axis_ratio,
            xCenter,
            yCenter,
            angle) in enumerate(zip(ellipses,
                                    majors,
                                    minors,
                                    eccs,
                                    axes_ratios,
                                    xCenters,
                                    yCenters,
                                    angles)):
        if i == bar_ind:
            e.set_color('red')
        if i == disk_ind:
            e.set_color('green')

        if (i==0) or ((major != majors[i-1]) and (minor != minors[i-1])):
            ell_art = ax.add_artist(e)
            ell_arts.append(ell_art)

        if axis is None:
            f.write("%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f  \n" %
                   (angle, major, minor, ecc, axis_ratio, xCenter, yCenter))

    x_pot = pot_cent[0]
    y_pot = pot_cent[1]

    if axis is None:
        f.write("#Potential minimum: %10.5f %10.5f" % (x_pot, y_pot))
    if settings['plotPotMin']:
        if np.abs(x_pot) < lengthX and np.abs(y_pot) < lengthY:
            ax.plot(x_pot, y_pot, marker='o', markersize=5)
        else:
            print(np.abs(x_pot), np.abs(y_pot))

    diff = np.sqrt((xCenters[bar_ind]-yCenters[bar_ind])**2
                   + (xCenters[disk_ind]-yCenters[disk_ind])**2)

    if axis is None:
        f.write("#Center difference from bar to largest ellipse:   %10.5f"
                % diff)
        f.close()
        plt.savefig(outname, bbox_inches='tight')
        plt.close()
        fig.clf()

    return ell_arts, im


def plot_orbit(sims, names, styles=['-'],
               colors=['#332288'],
               output='./',
               axes=None):
    """
    Plot the relative distances and velocities between two galaxies in a simulation.
    """

    sims, names = utils.check_args(sims, names)

    if axes is None:
      fig, ax = plt.subplots(2, 1, figsize=(5,10), sharex=True)
    else:
      ax = axes

    for sim, name, color, style in zip(sims, names, cycle(colors), cycle(styles)):
        distances, velocities, times = sim.measure_separation()

        ax[0].plot(times, distances, style, color=color, label=name)
        ax[1].plot(times, velocities, style, color=color, label=name)

    if axes is None:
      ax[0].set_ylabel('Separation [kpc]')
      ax[1].set_ylabel('Relative Velocity (km s$^{-1}$)')
      ax[1].set_xlabel('Time (Gyr)')
      ax[0].legend()
      ax[1].legend()
      plt.savefig(output+'orbits.pdf', dpi=600)
      plt.clf()
      plt.close()
