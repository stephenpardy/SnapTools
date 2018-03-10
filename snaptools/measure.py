import numpy as np
from . import manipulate as man
from . import snapshot
from multiprocess import Pool
import itertools
from . import utils


def find_centers_helper(settings):
    """
    Helper function for finding centers using multiprocessing
    """

    Rd = settings['Rd']
    num_contours = settings['num_contours']
    plot = settings['plot']
    num_centers = settings['num_centers']
    measure_fourier = settings['measure_fourier']

    try:
        settings = settings
        try:
            snap = snapshot.Snapshot(settings['filename'])
        except KeyError:
            raise StandardError
        return snap.find_centers(settings,
                                 Rd=Rd,
                                 numcontours=num_contours,
                                 plot=plot,
                                 num_centers=num_centers,
                                 measure_fourier=measure_fourier)

    except KeyboardInterrupt:
        raise KeyboardInterruptError()


def loop_centers(snaps,
                 settings=None,
                 Rd=0,
                 plot=False,
                 contours=20,
                 measure_fourier=False,
                 parttype='stars',
                 filename='./',
                 output='./snap',
                 num_centers=1):

    """
    Measure the centers of the bar, disk, and halo over a range of snapshots.

    times, bar_dist, disk_dist, bar_offset = measure.loop_centers(0, 40)

    Args:
          snaps: The snapshots to measure (array of strings)

    Kwargs:
        settings: Change any of the settings from utils.make_settings()
        Rd: The half mass radius of the disk (if 0 uses out ellipse)
        plot: Make a plot of the contours for each snapshot?
        contours: How many contours per snapshot.
        measure_fourier: Measure the Fourier modes in the bar? (Currently broken)
        parttype: Which particle type to use for making the measurements.

    """
    import re
    if settings is None:
        settings = utils.make_settings(first_only=True,
                                 com=True,
                                 xlen=20,
                                 ylen=20,
                                 in_min=0,
                                 parttype=parttype,
                                 filename=filename,
                                 outputname=output)


    pool = Pool()
    snapbase = settings['snapbase']
    settings['Rd'] = Rd
    settings['num_contours'] = contours
    settings['plot'] = plot
    settings['num_centers'] = num_centers
    settings['measure_fourier'] = measure_fourier

    settings_array = []
    for i, s in enumerate(snaps):
        settings_array.append(settings.copy())
        settings_array[i]['filename'] = s
        n = re.search(snapbase+'([0-9]{3})', s).group(1)
        settings_array[i]['outputname'] = output + n + 'distances'

    nsnaps = len(snaps)
    # Run measurements using pool.map
    # try and except statements are to ensure proper ctrl+c termination

    try:
        pos = pool.map(find_centers_helper, settings_array)
    except KeyboardInterrupt:
        print('got ^C while pool mapping, terminating the pool')
        pool.terminate()
        print('pool is terminated')
    except exception as e:
        print('got exception: %r, terminating the pool' % (e,))
        pool.terminate()
        print('pool is terminated')

    bar = np.zeros((len(pos), 2))
    halo = np.zeros((len(pos), 2))
    if num_centers > 1:
      disk = np.zeros((len(pos), 2, num_centers))
    else:
      disk = np.zeros((len(pos), 2))
    times = np.zeros(len(pos))
    for i in range(len(pos)):
        bar[i, :] = pos[i]['barCenter']
        halo[i, :] = pos[i]['haloCenter']
        times[i] = pos[i]['time']
        if num_centers > 1:
            disk[i, :, :] = pos[i]['diskCenters']
        else:
            disk[i, :] = pos[i]['diskCenters']


    #Define all distances from halo center
    bar_dist = np.sqrt((halo[:, 0]-bar[:, 0])**2 +
                       (halo[:, 1]-bar[:, 1])**2)

    bar_offset = np.empty((len(snaps), num_centers))
    if num_centers > 1:
        disk_dist = np.empty((len(snaps), num_centers))
        for i in range(num_centers):
            bar_offset[:, i] = np.sqrt((bar[:, 0]-disk[:, 0, i])**2 +
                                        (bar[:, 1]-disk[:, 1, i])**2)
            disk_dist[:, i] = np.sqrt((halo[:, 0]-disk[:, 0, i])**2 +
                                      (halo[:, 1]-disk[:, 1, i])**2)
    else:
        disk_dist = np.sqrt((halo[:, 0]-disk[:, 0])**2 +
                            (halo[:, 1]-disk[:, 1])**2)
        bar_offset = np.sqrt((bar[:, 0]-disk[:, 0])**2 +
                             (bar[:, 1]-disk[:, 1])**2)

    return {'times': times,
            'bar_dist': bar_dist,
            'disk_dist': disk_dist,
            'bar_offset': bar_offset,
            'halo_pos': halo,
            'disk_pos': disk,
            'bar_pos': bar}


def fourier_mode_helper(settings, Rd, modes, max_amp, use_offset):
    """
    Helper function for measuring fourier modes in multiprocessing
    """
    try:
        snap = snapshot.Snapshot(settings['filename'])
        am = snap.fourier_modes(settings, use_offset=use_offset)
        amp = np.zeros(len(modes))
        for j, m in enumerate(modes):
            # Measure modes from 1.5 -> 2.5 Rd
            inner_ind = np.floor((Rd*1.5)/(settings['xlen']/float(settings['NBINS'])))
            if inner_ind < 0:
                inner_ind = 0
            outer_ind = np.ceil((Rd*2.5)/(settings['xlen']/float(settings['NBINS'])))
            if outer_ind > settings['NBINS']:
                outer_ind = settings['NBINS']

            if max_amp:
                amp[j] = np.max(am[inner_ind:outer_ind, m])  # 8kpc with 360 bins and 20kpc total width
            else:
                amp[j] = np.mean(am[inner_ind:outer_ind, m])

        return (amp, snap.header['time'])
    except KeyboardInterrupt:
        raise KeyboardInterruptError()


def loop_fourier(snaps,
                 modes,
                 settings,
                 Rd=1.47492,
                 folder='./',
                 offsets=None,
                 max_amp=False,
                 parttype='stars'):
    """
    Measure the Fourier modes for a range of snapshots
    """

    pool = Pool()
    snapbase = 'snap_'

    settings_array = []
    for i, s in enumerate(snaps):
        settings_array.append(settings.copy())
        settings_array[i]['filename'] = s

    nsnaps = len(snaps)

    if offsets is None:
        use_offset = False
    else:
        use_offset = True
        argd = utils.check_args(snaps, offsets=offsets)
        offsets = argd['offsets']

        for i, offset in enumerate(offsets):
            settings_array[i]['offset'] = offset

    try:
        amps = pool.map(fourier_mode_helper, settings_array,
                        [Rd]*nsnaps, [modes]*nsnaps, [max_amp]*nsnaps,
                        [use_offset]*nsnaps)
    except KeyboardInterrupt:
        print('got ^C while pool mapping, terminating the pool')
        pool.terminate()
        print('pool is terminated')
    except exception as e:
        print('got exception: %r, terminating the pool' % (e,))
        pool.terminate()
        print('pool is terminated')

    return amps

