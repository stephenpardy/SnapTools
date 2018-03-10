from collections import defaultdict
import numpy as np
import os
from .snapshot import Snapshot
import h5py
from . import utils

"""
This file contains the Snapshot classes
These inherit modules from their superclass Snapshot
Next two classes are for particular filetypes
"""


# Convienent names for header attributes
# Add more here if you need more.
# Will use the name in the HDF5 file if not specified
HEAD_ATTRS = {'NumPart_ThisFile': 'npart',
              'NumPart_Total': 'nall',
              'NumPart_Total_HighWord': 'nall_highword',
              'MassTable': 'massarr',
              'Time': 'time',
              'Redshift': 'redshift',
              'BoxSize': 'boxsize',
              'NumFilesPerSnapshot': 'filenum',
              'Omega0': 'omega0',
              'OmegaLambda': 'omega_l',
              'HubbleParam': 'hubble',
              'Flag_Sfr': 'sfr',
              'Flag_Cooling': 'cooling',
              'Flag_StellarAge': 'stellar_age',
              'Flag_Metals': 'metals',
              'Flag_Feedback': 'feedback',
              'Flag_DoublePrecision': 'double'}
# Defines convienent standard names for misc. data blocks
# Todo: add function for arbitrary datablocks

DATABLOCKS = {"Coordinates": "pos",
              "Velocities": "vel",
              "ParticleIDs": "ids",
              "Potential": "pot",
              "Masses": "masses",
              "InternalEnergy": "U",
              "Density": "RHO",
              "Volume": "VOL",
              "Center-of-Mass": "CMCE",
              "Surface Area": "AREA",
              "Number of faces of cell": "NFAC",
              "ElectronAbundance": "NE",
              "NeutralHydrogenAbundance": "NH",
              "SmoothingLength": "HSML",
              "StarFormationRate": "SFR",
              "StellarFormationTime": "AGE",
              "Metallicity": "Z",
              "Acceleration": "ACCE",
              "VertexVelocity": "VEVE",
              "MaxFaceAngle": "FACA",
              "CoolingRate": "COOR",
              "MachNumber": "MACH",
              "DM Hsml": "DMHS",
              "DM Density": "DMDE",
              "PSum": "PTSU",
              "DMNumNgb": "DMNB",
              "NumTotalScatter": "NTSC",
              "SIDMHsml": "SHSM",
              "SIDMRho": "SRHO",
              "SVelDisp": "SVEL",
              "GFM StellarFormationTime": "GAGE",
              "GFM InitialMass": "GIMA",
              "GFM Metallicity": "GZ  ",
              "GFM Metals": "GMET",
              "GFM MetalsReleased": "GMRE",
              "GFM MetalMassReleased": "GMAR"}


MISC_DATABLOCKS = DATABLOCKS  # backwards compatibility

def load_dataset(filenames, group, variable):
    dataset = None

    for filename in filenames:
        with h5py.File(filename) as f:
            if dataset is None:
                dataset = f[group][variable][()]
            else:
                dataset = np.append(dataset, f[group][variable][()], axis=0)

    return dataset


class SnapLazy(Snapshot):
    """
    lazydict implementation of HDF5 snapshot
    """
    def __init__(self, fname, **kwargs):
        pass

    def init(self, fname, part_names=None, **kwargs):
        from functools import partial
        from . import lazydict

        if part_names is None:
            part_names = ['gas',
                          'halo',
                          'stars',
                          'bulge',
                          'sfr',
                          'other']

        self.part_names = part_names

        self.settings = utils.make_settings(**kwargs)
        self.bin_dict = None

        if not isinstance(fname, list):
            fname = [fname]

        self.filename = fname

        #load header only
        with h5py.File(fname[0], 'r') as s:
            self.header = {}
            #header_keys = s['Header'].attrs.keys()
            for head_key, head_val in s['Header'].attrs.items():
                if head_key in HEAD_ATTRS.keys():
                    self.header[HEAD_ATTRS[head_key]] = head_val
                else:
                    self.header[head_key] = head_val
            # setup loaders
            for i, part in enumerate(part_names):
                if self.header['nall'][i] > 0:
                    for key in s['PartType%d' % i].keys():
                        try:
                            attr_name = DATABLOCKS[key]
                        except KeyError:
                            attr_name = key
                        if attr_name not in self.__dict__.keys():
                            self.__dict__[attr_name] = lazydict.MutableLazyDictionary()
                        self.__dict__[attr_name][part] = partial(load_dataset, self.filename,
                                                                 "PartType%d" % i, key)

            if any(self.header['massarr']):
                wmass, = np.where(self.header['massarr'])
                for i in wmass:
                    part = self.part_names[i]
                    npart = self.header['nall'][i]
                    mass = self.header['massarr'][i]
                    if 'masses' not in self.__dict__.keys():
                        self.__dict__['masses'] = lazydict.MutableLazyDictionary()

                    # here we are keeping things lazy by defining a function
                    # that will make our array only when needed
                    # the inner lambda function takes two arguments - a number of particles (n) and a mass (m)
                    # the outer function (partial) freezes the arguments for the current particle type
                    # the outer function is necessary or else arguments will be overwritten by other types
                    self.masses[part] = partial(lambda n, m: np.ones(n)*m, npart, mass)


class SnapHDF5(Snapshot):
    """
    Snapshots in HDF5
    snap = SnapHDF5('mycoolsnapshot.hdf5')
    Note: Must have matching header and data.
    Todo: Gracefully handle mismatches between header and data
    """
    def __init__(self, fname, **kwargs):
        """
        This method is purposefully empty. __new__ method of parent class will call init().
        """
        pass

    def init(self, fname, **kwargs):
        """Read from an HDF5 file
        """
        self.settings = utils.make_settings(**kwargs)
        self.bin_dict = None
        self.filename = fname

        with h5py.File(fname, 'r') as s:
            self.header = {}
            #header_keys = s['Header'].attrs.keys()
            for head_key, head_val in s['Header'].attrs.items():
                if head_key in HEAD_ATTRS.keys():
                    self.header[HEAD_ATTRS[head_key]] = head_val
                else:
                    self.header[head_key] = head_val

            part_names = ['gas',
                          'halo',
                          'stars',
                          'bulge',
                          'sfr',
                          'other']
            self.pos = {}
            self.vel = {}
            self.ids = {}
            self.masses = {}
            self.pot = {}
            self.misc = {}
            for i, n in enumerate(self.header['npart']):
                if n > 0:
                    group = 'PartType%s' % i
                    part_name = part_names[i]
                    for key in s[group].keys():
                        if key == 'Coordinates':
                            self.pos[part_name] = s[group]['Coordinates'][()]
                        elif key == 'Velocities':
                            self.vel[part_name] = s[group]['Velocities'][()]
                        elif key == 'ParticleIDs':
                            self.ids[part_name] = s[group]['ParticleIDs'][()]
                        elif key == 'Potential':
                            self.pot[part_name] = s[group]['Potential'][()]
                        elif key == 'Masses':
                            self.masses[part_name] = s[group]['Masses'][()]
                        # If we find a misc. key then add it to the misc variable (a dict)
                        elif key in MISC_DATABLOCKS.keys():
                            if part_name not in self.misc.keys():
                                self.misc[part_name] = {}
                            self.misc[part_name][MISC_DATABLOCKS[key]] = s[group][key][()]
                        # We have an unidentified key, throw it in with the misc. keys
                        else:
                            if part_name not in self.misc.keys():
                                self.misc[part_name] = {}
                            self.misc[part_name][key] = s[group][key][()]
                    # If we never found the masses key then make one
                    if part_name not in self.masses.keys():
                        self.masses[part_name] = (np.ones(n) * self.header['massarr'][i])
