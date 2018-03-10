import numpy as np

"""
Contains routines for manipulating and deriving quantities from data
These functions work directly with subquantities
(e.g. not the snapshots themselves)
And should strive to be as general as possible.
"""


def bin_particles(p1,
                  p2,
                  extents1_in,
                  extents2_in,
                  mass,
                  BINS,
                  scale):
    """
    Given a list of coordinates in 2D, return the binned map

    Parameters:
        p1 - positions in first dimension
        p2 - positions in second dimension
        extents1: length in first dimension - or an array with extents
        extents2: length in second dimension - or an array with extents
            (histograms from -length to length)
        mass: vector of masses corresponding to each bin_particles
        BINS: Number of bins
        scale: Log scale? True/False
    """
    if not hasattr(extents1_in, '__iter__'):
        length1 = 2.0*extents1_in
        extents1 = [-extents1_in, extents1_in]
    else:
        if len(extents1_in) != 2:
            print("extents must be number or length 2 array")
            return None
        else:
            extents1 = extents1_in
            length1 = extents1_in[1] - extents1_in[0]


    if not hasattr(extents2_in, '__iter__'):
        length2 = 2.0*extents2_in
        extents2 = [-extents2_in, extents2_in]
    else:
        if len(extents2_in) != 2:
            print("extents must be number or length 2 array")
            return None
        else:
            extents2 = extents2_in
            length2 = extents2_in[1] - extents2_in[0]


    Z2, ind1, ind2 = np.histogram2d(p1, p2, range=[extents1,
                                                   extents2],
                                    weights=mass * 1E10,
                                    bins=BINS, normed=False)
    # Turn number into density (in units Msun/pc^2)
    Z2 = Z2 / (length2 / BINS * 1E3 * length1 / BINS * 1E3)
    # Log scale the data
    if scale:
        Z2[Z2 <= 0] = np.nan
        Z2[Z2 > 0] = np.log10(Z2[Z2 > 0])

    return Z2, ind1, ind2


def measure_fourier(r, theta, length, BINS_r, BINS_theta):

    Z2, x, y = np.histogram2d(r, theta, range=[[0, length],
                                               [-np.pi, np.pi]],
                              bins=[BINS_r, BINS_theta],
                              normed=False)

    Z2 = Z2.astype('float64')
    sinrange = np.zeros((BINS_theta, 10))
    cosrange = np.zeros((BINS_theta, 10))

    theta_range = np.linspace(-np.pi, np.pi, BINS_theta)
    for i in range(10):
        cosrange[:, i] = np.cos(i*theta_range)
        sinrange[:, i] = np.sin(i*theta_range)

    I0 = np.zeros(BINS_r)
    for i in range(BINS_r):
        I0[i] = 2*np.mean(Z2[i, :])

    Imc = np.zeros((BINS_r, 10))
    Ims = np.zeros((BINS_r, 10))
    for j in range(10):
        for i in range(BINS_r):
            Ims[i, j] = 2*np.mean(Z2[i, :]*sinrange[:, j])
            Imc[i, j] = 2*np.mean(Z2[i, :]*cosrange[:, j])

    am = np.zeros((BINS_r, 10))
    for j in range(10):
        am[I0 > 0, j] = np.sqrt(Imc[I0 > 0, j]**2
                                + Ims[I0 > 0, j]**2)/I0[I0 > 0]

    return am


def fit_contours(density,
                 settings,
                 numcontours=20,
                 plot=False):
    """
    Fit density map with contours

    Args:
        density: 2D density map (e.g. from histogram2d)
        settings: Standard settings dictionary

    kwargs:
        numcontours: Number of contours to fit. (Default 20)
        plot: Store the ellipses as matplotlib patches for plotting. (Default False)
    """

    from . import EllipseFitter

    contours = numcontours

    measurements = {}

    measurements['angles'] = np.array([np.NaN]*contours)
    measurements['majors'] = np.array([np.NaN]*contours)
    measurements['minors'] = np.array([np.NaN]*contours)
    measurements['xCenters'] = np.array([np.NaN]*contours)
    measurements['yCenters'] = np.array([np.NaN]*contours)
    measurements['eccs'] = np.array([np.NaN]*contours)
    measurements['axes_ratios'] = np.array([np.NaN]*contours)
    if plot:
        from matplotlib.patches import Ellipse
        measurements['ellipses'] = []

    minimum = settings['in_min']
    maximum = settings['in_max']

    # Fit ellipses starting with the largest, and going to the smallest
    for i, level in enumerate(np.linspace(minimum, maximum, contours)):

        #we expect some NaNs here, ignore them.
        with np.errstate(invalid='ignore'):
            iso_contour = density > level
        # Somewhat arbitrary cutoff based on testing.
        # If we have less than some number of pixels than break.
        if np.sum(iso_contour) < 9:
            break

        angle, major, minor, xCenter, yCenter =\
            EllipseFitter.EllipseFitter(iso_contour)

        ecc = np.sqrt(1.-minor**2/major**2)
        axis_ratio = minor/major
        measurements['axes_ratios'][i] = axis_ratio
        measurements['angles'][i] = angle

        yCenter = grid_y_to_spatial(yCenter,
                                    settings['ylen'],
                                    settings['NBINS'])
        xCenter = grid_x_to_spatial(xCenter,
                                    settings['xlen'],
                                    settings['NBINS'])
        minor = grid_length_to_spatial(minor,
                                       settings['xlen'],
                                       settings['NBINS'])
        major = grid_length_to_spatial(major,
                                       settings['xlen'],
                                       settings['NBINS'])

        measurements['majors'][i] = major
        measurements['minors'][i] = minor
        measurements['xCenters'][i] = xCenter
        measurements['yCenters'][i] = yCenter
        measurements['eccs'][i] = ecc

        if plot:
            measurements['ellipses'].append(
                Ellipse([xCenter, yCenter],
                        major,
                        minor,
                        angle=-angle,  # to plot correctly
                        fill=False))

    return measurements


def rotation_curve(pos, vel, Vel_BINS=512, hmin=0, hmax=20):
    """Take position and velocity and return rotation curve
    """

    xp = pos[:, 0]
    yp = pos[:, 1]
    vx = vel[:, 0]

    r = np.sqrt(xp**2+yp**2)
    density, binEdges = np.histogram(r, bins=Vel_BINS, range=[hmin, hmax])
    rad = 0.5*(binEdges[1:]+binEdges[:-1])
    vel = np.zeros(rad.shape)
    rad_indices = np.digitize(r, rad)
    for i in range(len(rad)):
        vel[i] = np.mean(abs(vx[rad_indices == i]))

    return rad, vel


def volume_density(pos,
                   NBINS=500,
                   normalize_radius=False,
                   scale_length=0,
                   hmin=-1,
                   hmax=2,
                   log_scale=True,
                   equalmass=False):
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    N = len(x)
    r = np.sqrt(x**2+y**2+z**2)
    if equal_mass:
        r = np.sort(r)
        Nperbin = N//NBINS
        vol = np.zeros(NBINS)
        density = np.zeros(NBINS)+float(N)/float(NBINS)
        radius = np.zeros(NBINS)
        for i in range(NBINS):
            vol[i] = 4./3.*np.pi*(np.mean(r[(i + 1)*Nperbin:
                                            (i + 2)*Nperbin])**3
                                  - np.mean(r[i*Nperbin:(i + 1)*Nperbin])**3)
            radius[i] = np.mean(r[i*Nperbin:(i + 1)*Nperbin])
    else:
        if log_scale:
            r = np.log10(r)
        density, binEdges = np.histogram(r, bins=NBINS, range=[hmin, hmax])
        if log_scale:
            binEdges = 10**binEdges
        radius = 0.5*(binEdges[1:]+binEdges[:-1])

        vol = 4./3.*np.pi*(binEdges[1:]**3-binEdges[:-1]**3)
    density = density/vol
    if scale_length:
        density = density/(N/scale_length**3)
        if normalize_radius:
            radius = radius/scale_length

    return radius, density


def surface_density(pos,
                    NBINS=500,
                    normalize_radius=False,
                    scale_length=0,
                    hmin=-1,
                    hmax=1.7,
                    log_scale=True):
    x = pos[:, 0]
    y = pos[:, 1]
    N = len(x)
    r = np.sqrt(x**2+y**2)
    if log_scale:
        r = np.log10(r)
    density, binEdges = np.histogram(r, bins=NBINS, range=[hmin, hmax])
    if log_scale:
        binEdges = 10**(binEdges)

    radius = 0.5*(binEdges[1:]+binEdges[:-1])

    area = 4.*np.pi*(binEdges[1:]**2-binEdges[:-1]**2)

    density = density/area
    if scale_length:
        density = density/(N/scale_length**3)
        if normalize_radius:
            radius = radius/scale_length

    return radius, density


"""
Convienence functions for ellipse fitting
"""

def spatial_to_grid(coord, length, BINS):
    return (coord+length)/length/2*BINS


def grid_to_spatial(coord, length, BINS):
    return coord/BINS*length*2.-length

# for backwards compatibility

def spatial_to_gridx(coord, lengthX, BINS):
    return spatial_to_grid(coord, lengthX, BINS)


def spatial_to_gridy(coord, lengthY, BINS):
    return spatial_to_grid(coord, lengthY, BINS)


def grid_x_to_spatial(coord, lengthX, BINS):
    return grid_to_spatial(coord, lengthX, BINS)


def grid_y_to_spatial(coord, lengthY, BINS):
    return grid_to_spatial(coord, lengthY, BINS)


def grid_z_to_spatial(coord, lengthZ, BINS):
    return grid_to_spatial(coord, lengthY, BINS)


def grid_length_to_spatial(length, lengthX, BINS):
    #ASSUMES X = Y = Z for now
    #CAN MAKE THIS GENERAL LATER IF IT IS REQUIRED
    return length/BINS*lengthX*2.


def combine_snaps(s1, s2, names=['s1', 's2'], id_file=None):
    """
    Combine two snapshots together
    """
    from . import snapshot as sn  # to avoid cyclical imports
    part_names = ['gas',
                  'halo',
                  'stars',
                  'bulge',
                  'sfr',
                  'other']

    snap = sn.Snapshot()
    max_id = np.sum(s1.header['nall'])
    if id_file is not None:
        try:
            id_f = h5py.File(id_file)
            grp1 = id_f.create_group(names[0])
            grp2 = id_f.create_group(names[1])
        except:
            print("Problem opening file {}".format(id_file))
            id_file = None

    # Grab the number of particles in galaxy 1
    # this will be the starting point for IDs in galaxy 2

    # Copy the main info first, using numbers from each header
    for i, (n1, n2) in enumerate(zip(s1.header['nall'],
                                     s2.header['nall'])):

        p = part_names[i]
        pos = np.empty((n1+n2, 3))
        vel = np.empty((n1+n2, 3))
        ids = np.empty((n1+n2))
        masses = np.empty((n1+n2))
        # Potentials are handled like misc. datablocks (see below)

        if n1 > 0:
            pos[:n1, 0] = s1.pos[p][:, 0]
            pos[:n1, 1] = s1.pos[p][:, 1]
            pos[:n1, 2] = s1.pos[p][:, 2]
            vel[:n1, 0] = s1.vel[p][:, 0]
            vel[:n1, 1] = s1.vel[p][:, 1]
            vel[:n1, 2] = s1.vel[p][:, 2]

            ids[:n1] = s1.ids[p]
            # Record these to a file
            if id_file is not None:
                dset = grp1.create_dataset(p, s1.ids[p].shape)
                dset[:] = s1.ids[p]

            masses[:n1] = s1.masses[p]

        if n2 > 0:
            pos[n1:, 0] = s2.pos[p][:, 0]
            pos[n1:, 1] = s2.pos[p][:, 1]
            pos[n1:, 2] = s2.pos[p][:, 2]
            vel[n1:, 0] = s2.vel[p][:, 0]
            vel[n1:, 1] = s2.vel[p][:, 1]
            vel[n1:, 2] = s2.vel[p][:, 2]

            # Add on the highest id number from the first galaxy
            # ids for second galaxy will start at max(id_gal1)
            # That means that DM ids for gal2 will be all higher than gal2
            ids[n1:] = s2.ids[p] + max_id
            if id_file is not None:
                dset = grp2.create_dataset(p, s2.ids[p].shape)
                dset[:] = s2.ids[p] + max_id

            masses[n1:] = s2.masses[p]

        # This is a slightly messy way to assign these, but:
        # end result is that you will have an array with potentials
        # from the snapshots that have them
        # Note: it doesnt really make sense to do this
        # Because adding another galaxy will change the potential
        pot = np.empty((n1+n2))
        if p in s2.pot.keys() and len(s2.pot[p]) == n2:
            pot[n1:] = s2.pot[p]
        else:
            pot = pot[:n1]
        if p in snap.pot.keys() and len(snap.pot[p]) == n1:
            pot[:n1] = snap.pot[p]
        else:
            pot = pot[n1:]

        snap.pos[p] = pos
        snap.vel[p] = vel
        snap.ids[p] = ids
        snap.masses[p] = masses
        if len(pot) > 0:
            snap.pot[p] = pot

        # Now copy the misc info

        misc = {}
        if p in s1.misc.keys():
            misc1 = s1.misc[p].keys()
        else:
            misc1 = []
        if p in s2.misc.keys():
            misc2 = s2.misc[p].keys()
        else:
            misc2 = []

        if (len(misc1) > 0) and (len(misc2) == 0):
            for m in misc1:
                # make it the same size as it was
                sz = s1.misc[p][m].shape
                misc[m] = np.empty(sz)
                # iterate through the second dimension if necessary
                if len(sz) > 1:
                    for j in range(sz[1]):
                        misc[m][:, j] = s1.misc[p][m][:, j]
                else:
                    misc[m][:] = s1.misc[p][m]
        elif (len(misc1) == 0) and (len(misc2) > 0):
            for m in misc2:
                # make it the same size as it was
                sz = s2.misc[p][m].shape
                misc[m] = np.empty(sz)
                # iterate through the second dimension if necessary
                if len(sz) > 1:
                    for j in range(sz[1]):
                        misc[m][:, j] = s2.misc[p][m][:, j]
                else:
                    misc[m][:] = s2.misc[p][m]
        else:  # both here
            for m in misc1:  # Get all the keys in snap1
                if m in misc2:  # and if they are also in snap2 then combine them
                    sz = list(s1.misc[p][m].shape)
                    sz[0] = n1+n2
                    misc[m] = np.empty(tuple(sz))
                    if len(sz) > 1:
                        for j in range(sz[1]):
                            misc[m][:n1, j] = s1.misc[p][m][:, j]
                            misc[m][n1:, j] = s2.misc[p][m][:, j]
                    else:
                        misc[m][:n1] = s1.misc[p][m]
                        misc[m][n1:] = s2.misc[p][m]

        snap.misc[p] = misc

    # wait until the end to copy the header info

    head1 = s1.header
    head2 = s2.header

    head_attrs = ['npart',
                  'nall',
                  'nall_highword',
                  'massarr',
                  'time',
                  'redshift',
                  'boxsize',
                  'filenum',
                  'omega0',
                  'omega_l',
                  'hubble',
                  'sfr',
                  'cooling',
                  'stellar_age',
                  'metals',
                  'feedback',
                  'double']

    # Do a quick check for all the attributes that we want in an HDF5 header
    # If we don't find a suitable value in either header than arbitrarily set to 0

    for attr in head_attrs:
        if attr not in(snap.header.keys()):
            if attr in (head2.keys()):
                snap.header[attr] = head2[attr]
            else:
                snap.header[attr] = 0

    snap.header['npart'] = head1['npart'] + head2['npart']
    snap.header['nall'] = head1['nall'] + head2['nall']
    try:
        snap.header['nall_highword'] = head1['nall_highword'] + head2['nall_highword']
    except:
        snap.header['nall_highword'] = head1['nall'] + head2['nall']

    snap.header['massarr'] = [0, 0, 0, 0, 0, 0]

    if id_file is not None:
        id_f.close()

    return snap
