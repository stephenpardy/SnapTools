import numpy as np
import os
from . import snapshot_io
import h5py
import sys


def load_snap(fname):
    if os.path.exists(fname):
        curfname = fname
    elif os.path.exists(fname + ".hdf5"):
        curfname = fname + ".hdf5"
    elif os.path.exists(fname + ".0.hdf5"):
        curfname = fname + ".0.hdf5"
    else:
        print("[error] file not found : %s" % fname)
        return None

    if h5py.is_hdf5(curfname):
        snap = snapshot_io.SnapHDF5(curfname)
    else:
        snap = snapshot_io.SnapBinary(curfname)

    return snap


def open_snap(fname, block, parttype=2, shift=False):
    import readsnap as rs
    if os.path.exists(fname):
        curfname = fname
    elif os.path.exists(fname + ".hdf5"):
        curfname = fname + ".hdf5"
    elif os.path.exists(fname + ".0.hdf5"):
        curfname = fname + ".0.hdf5"
    else:
        print("[error] file not found : %s" % fname)
        sys.exit()

    if block.upper().ljust(4) == "POT ":
        f = open(curfname, 'rb')
        dt = np.dtype((np.float32))
        head = rs.snapshot_header(curfname)
        offset = np.sum(head.npart) * 4 * 3 * 2 + np.sum(head.npart) * 4
        if shift:
            offset += np.sum(head.npart) * 4
        offset += 268 + 8 * 3
        if parttype >= 0:
            add_offset = np.sum(head.npart[0:parttype])
            actual_curpartnum = head.npart[parttype]
        else:
            actual_curpartnum = np.sum(head.npart)
            add_offset = np.int32(0)

        f.seek(offset + add_offset * np.dtype(dt).itemsize, os.SEEK_CUR)
        data = np.fromfile(f, dtype=dt, count=actual_curpartnum)
        f.close()
    elif block.upper() == "HEAD":
        data = rs.snapshot_header(curfname)
    else:
        data = rs.read_block(curfname, block.upper().ljust(4),
                             parttype=parttype,
                             arepo=0).astype('float64')

    return data


def save_snap(fname, head, pos, vel, ids, U=[], pot=[]):
    f = open(fname, 'wb')
    np.int32(256).tofile(f)
    head.npart.tofile(f)
    head.massarr.tofile(f)
    head.time.tofile(f)
    head.redshift.tofile(f)
    head.sfr.tofile(f)
    head.feedback.tofile(f)
    head.nall.tofile(f)
    head.cooling.tofile(f)
    head.filenum.tofile(f)
    head.boxsize.tofile(f)
    head.omega0.tofile(f)
    head.omega_l.tofile(f)
    head.hubble.tofile(f)
    bytesleft = 256 - 6 * 4 - 6 * 8 - 8 - 8\
        - 2 * 4 - 6 * 4 - 4 - 4 - 8 - 8 - 8 - 8
    #    bytesleft=256-6*4 - 6*8 - 8 - 8 - 2*4-6*4
    np.int16([0] * (bytesleft / 2)).tofile(f)

    np.int32(256).tofile(f)
    np.int32(np.sum(head.npart) * 3 * 4).tofile(f)
    pos.tofile(f)
    np.int32(np.sum(head.npart) * 3 * 4).tofile(f)
    np.int32(np.sum(head.npart) * 3 * 4).tofile(f)

    vel.tofile(f)

    np.int32(np.sum(head.npart) * 3 * 4).tofile(f)
    np.int32(np.sum(head.npart) * 4).tofile(f)
    ids.tofile(f)

    np.int32(np.sum(head.npart) * 4).tofile(f)

    if len(U) > 0:
        np.int32(head.npart[0] * 4).tofile(f)
        U.tofile(f)

        np.int32(head.npart[0] * 4).tofile(f)

    if len(pot) > 0:
        np.int32(np.sum(head.npart) * 4).tofile(f)
        pot.tofile(f)

        np.int32(np.sum(head.npart) * 4).tofile(f)

    f.close()
