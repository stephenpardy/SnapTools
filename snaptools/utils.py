import sys
import numpy as np
from tempfile import mkstemp
from shutil import move, copy
import os
import warnings

def list_snapshots(snapids, folder, base):

    if isinstance(snapids, int):
        snaps = range(snapids+1)
        convert = lambda s: folder+base+str(s).zfill(3)
        snaps = map(convert, snaps)
    elif hasattr(snapids, '__iter__'):
        if all(isinstance(x, int) for x in snapids):
            if len(snapids) == 1:
                snaps = range(snapids[0]+1)
            elif len(snapids) == 2:
                snaps = range(snapids[0], snapids[1]+1)
            elif len(snapids) == 3:
                snaps = range(snapids[0], snapids[1], snapids[2])
                if snapids[1] == snaps[-1]+snapids[2]:
                    snaps.append(snapids[1])
            else:
                snaps = snapids
            convert = lambda s: folder+base+str(s).zfill(3)
            snaps = map(convert, snaps)
        elif all(isinstance(x, str) for x in snapids):
            snaps = snapids
    elif isinstance(snapids, str):
        snaps = snapids

    return snaps

def make_settings(**kwargs):
    #default settings
    settings = {'panel_mode': "xy",
                'log_scale': True,
                'in_min': -1,
                'in_max': 2.2,
                'com': False,
                'first_only': False,
                'gal_num': -1,
                'colormap': 'viridis',
                'xlen': 30,
                'ylen': 30,
                'zlen': 30,
                'colorbar': 'None',
                'NBINS': 512,
                'plotCompanionCOM': False,
                'plotPotMin': False,
                'parttype': 'stars',
                'filename': '',
                'outputname': '',
                'snapbase': 'snap_',
                'offset': [0, 0, 0],
                'im_func': None,
                'halo_center_method':'pot',
                'UnitMass_in_g':1.989e43,  # 1.e10 solar masses
                'UnitVelocity_in_cm_per_s':1e5,  # 1 km/s
                'UnitLength_in_cm':3.085678e21}

    for name, val, in kwargs.items():
        if name not in settings.keys():
            warnings.warn("WARNING! %s is not a default setting" % name, RuntimeWarning)
        settings[name] = val

        #Temporary backwards compatible check for first_only
        if name == 'first_only':
            warnings.warn("first_only is being deprecated", DeprecationWarning)
            if val:
                print("first_only is set, setting gal_num = 0")
                settings['gal_num'] = 0

    return settings


def check_args(base_val, *args):
    # This function is mostly broken and likely unneccassary
    # Done this way because of https://hynek.me/articles/hasattr/
    y = getattr(base_val, '__iter__', None)
    if y is None:  # ensure we can loop over these
        base_val = [base_val]

    new_args = []

    for i, val in enumerate(args):
        y = getattr(val, '__iter__', None)
        if y is None:
            val = [val]*len(base_val)
            new_args.append(val)
        elif len(val) != len(base_val):
            raise IndexError("IF YOU PROVIDE A LIST FOR ONE ARGUMENT YOU MUST PROVIDE ONE FOR ALL OF THEM")

    return base_val, new_args


def read_offsets(fname):
    (angle,
     major,
     minor,
     ecc,
     axis_ratio,
     xCenter,
     yCenter) = np.loadtxt(fname,
                           skiprows=2,
                           unpack=True,
                           comments='Pot')
    measurements = {}
    measurements['axes_ratios'] = axis_ratio
    measurements['angles'] = angle
    measurements['majors'] = major
    measurements['minors'] = minor
    measurements['xCenters'] = xCenter
    measurements['yCenters'] = yCenter
    measurements['eccs'] = ecc
    return measurements


def replace(file_path, pattern, subst, tag=None):
    #Create temp file
    fh, abs_path = mkstemp()
    new_file = open(abs_path, 'w')
    old_file = open(file_path)
    for line in old_file:
        if tag:
            if len(line.split()) > 0 and line.split()[0] == tag:
                new_file.write(line.replace(pattern, subst))
            else:
                new_file.write(line)
        else:
            new_file.write(line.replace(pattern, subst))

    #close temp file
    new_file.close()
    os.close(fh)
    old_file.close()
    #Remove original file
    os.remove(file_path)
    #Move new file
    move(abs_path, file_path)
