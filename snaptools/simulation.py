import os
from . import utils
from . import plot_tools
from . import measure
from . import snapshot
from functools import partial
from multiprocess import Pool
import itertools
import numpy as np
import re
import numbers


class Simulation(object):
    """
    This class holds a folder with snapshots belonging to a single simulation
    and makes it easy to apply functions over all of those snapshots (plotting,
    measuring quantities, or printing info). Initialize by supplying a folder.
    """
    def __init__(self, folder, snaps=None, snapbase='snap_', snapext='hdf5'):
        """
        Default behavior is to get all the snapshots in a folder, more complicated
        behavior is governed by utils.list_snapshots()
        """
        self.folder = os.path.realpath(folder)
        self.snapbase = snapbase
        # Default behavior is to collect everything that matches the snapbase
        if snaps is None:
            self.snaps = [folder+f for f in os.listdir(folder)
                            if re.search("^"+snapbase+"[0-9]+.*\.%s$" % snapext, f)]
            self.snaps = np.sort(self.snaps)
        #otherwise will use only certain range or given numbers
        else:
           self.snaps = utils.list_snapshots(snaps, folder, snapbase)

        self.nsnaps = len(self.snaps)
        self.settings = utils.make_settings()


    def plot(self, **kwargs):
        """
        Wrapper to plot_tools.plot_loop
        """
        plot_tools.plot_loop(self.snaps, settings=self.settings, **kwargs)


    def measure_centers(self, **kwargs):
        """
        Wrapper to measure.loop_centers
        """
        centers = measure.loop_centers(self.snaps,
                                       settings=self.settings,
                                       **kwargs)
        self.times = centers['times']
        self.bar_dist = centers['bar_dist']
        self.disk_dist = centers['disk_dist']
        self.bar_offset = centers['bar_offset']
        self.halo_pos = centers['halo_pos']
        self.disk_pos = centers['disk_pos']
        self.bar_pos = centers['bar_pos']
        return centers


    def measure_fourier(self, modes, **kwargs):
        """
        Wrapper to measure.loop_fourier
        """
        amps = measure.loop_fourier(self.snaps,
                                    modes,
                                    self.settings,
                                    **kwargs)
        amp = np.zeros(len(amps)*len(modes)).reshape(len(amps), len(modes))
        times = np.zeros(len(amps))

        for i in range(len(amps)):
            amp[i, :] = amps[i][0]
            times[i] = amps[i][1]

        self.times = times
        self.amp = amp

        return times, amp


    def get_stats(self):
        """
        Work in progress
        """
        snap = snapshot.Snapshot(self.snaps[0])
        time_begin = snap.header['time']
        snap = snapshot.Snapshot(self.snaps[-1])
        time_end = snap.header['time']


    def measure_centers_of_mass(self, indices=None):
        """
        Get the center of masses of the galaxies in the snapshots.
        """
        def center_of_mass_(snapname, indices=None):
            try:
                snap = snapshot.Snapshot(snapname)
                if indices is None:
                    indices = snap.split_galaxies('stars')
                com1s, com2s = snap.measure_com('stars', indices)
                return com1s
            except KeyboardInterrupt:
                pass

        center_of_mass = partial(center_of_mass_, indices=indices)
        return np.array(self.apply_function(center_of_mass))


    def measure_separation(self, indices=None):
        """
        Get relative velocity and radial separation between two galaxies.
        """
        def centers_(snapname, indices=None):
            try:
                snap = snapshot.Snapshot(snapname)
                if indices is None:
                    indices = snap.split_galaxies('stars')
                coms = snap.measure_com('stars', indices)
                v1 = snap.vel['stars'][indices[0], :].mean(axis=0)
                v2 = snap.vel['stars'][indices[1], :].mean(axis=0)
                time = snap.header['time']
                return [coms[0], coms[1], v1, v2, time]
            except KeyboardInterrupt:
                pass


        centers = partial(centers_, indices=indices)
        cents = self.apply_function(centers)
        nsnaps = len(self.snaps)

        distances = np.empty(nsnaps)
        velocities = np.empty(nsnaps)
        times = np.empty(nsnaps)

        for i, cent in enumerate(cents):
            distances[i] = np.sqrt(np.sum( (cent[0] - cent[1])**2 ))
            velocities[i] = np.sqrt(np.sum( (cent[2] - cent[3])**2 ))
            times[i] = cent[4]

        return distances, velocities, times


    def apply_function(self, function, *args):
        """
        Map a user supplied function over the snapshots.
        Uses pathos.multiprocessing (https://github.com/uqfoundation/pathos.git).
        """
        pool = Pool()

        try:
            val = pool.map(function, self.snaps)
            return val
        except KeyboardInterrupt:
            print('got ^C while pool mapping, terminating the pool')
            pool.terminate()
            print('pool is terminated')
        except Exception as e:
            print('got exception: %r, terminating the pool' % (e,))
            pool.terminate()
            print('pool is terminated')


    def print_settings(self):
        """
        Print the current settings
        """
        for key, val in self.settings.items():
            print("{0}: {1}".format(key, val))


    def set_settings(self, **kwargs):
        """
        Set simulation-wide settings
        """
        for name, val in kwargs.items():
            if name not in self.settings.keys():
                print("WARNING! {:s} is not a default setting!".format(name))
            self.settings[name] = val


    def get_snapshot(self, num=None, lazy=True):
        """
        Return snapshot
        """
        if num is None:
            for i, s in enumerate(self.snaps):
                print("snap %i: %s" % (i, s))
            num = int(input('Select snapshot:'))
        try:
            return snapshot.Snapshot(self.snaps[num], lazy=lazy)
        except IndexError:
            print('Not a valid snapshot')
            self.get_snapshot(None)
        except TypeError:
            return None


    def __repr__(self):
        return "Simulation located at {0} with {1} snapshots".format(self.folder, self.nsnaps)



