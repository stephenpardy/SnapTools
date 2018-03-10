from . import manipulate as man
from . import utils
import numpy as np
import copy
import warnings
import os


"""
Base class for snapshots. If called with a filename, will return one of two supported subclasses.
If called with no filename, it will return an empty object.
This class contains all snapshot methods. These methods should act directly on a snapshot.
"""


class Snapshot(object):

    def __new__(cls, filename=None, lazy=True, part_names=None, **kwargs):
        """
        Factory method for calling proper subclass or empty object
        """
        if filename is not None:

            from . import snapshot_io
            import h5py

            multi = False

            if os.path.exists(filename):
                curfilename = filename
            elif os.path.exists(filename + ".hdf5"):
                curfilename = filename + ".hdf5"
            elif os.path.exists(filename + ".0.hdf5"):
                if lazy:
                    # allow multi-part files only in the lazy eval mode
                    import glob
                    filelist = list(sorted(glob.glob(filename+".[0-9].hdf5")))
                    multi = True

                curfilename = filename + ".0.hdf5"
            else:
                raise IOError("[error] file not found : %s" % filename)

            if h5py.is_hdf5(curfilename):
                if lazy:
                    snapclass = super(Snapshot, cls).__new__(snapshot_io.SnapLazy)
                    if multi:
                        snapclass.init(filelist, part_names, **kwargs)  # replaces standard __init__ method
                    else:
                        snapclass.init(curfilename, part_names, **kwargs)
                    return snapclass
                else:
                    snapclass = super(Snapshot, cls).__new__(snapshot_io.SnapHDF5)
                    snapclass.init(curfilename)  # replaces standard __init__ method
                    return snapclass
            else:
                raise RuntimeError("Filetype is not HDF5. Other file types are not implemented in this version of SnapTools.")

        else:
            return super(Snapshot, cls).__new__(cls)

    def __init__(self):
        """
        Create empty snapshot object
        """
        self.filename = None
        self.pos = {}
        self.vel = {}
        self.ids = {}
        self.masses = {}
        self.pot = {}
        self.misc = {}
        self.header = {}
        self.settings = utils.make_settings()
        self.bin_dict = None


    def set_settings(self, **kwargs):
        """
        Set the settings used by analysis and plotting tools.
        """
        for name, val in kwargs.items():
            if name not in self.settings.keys():
                print("WARNING! {} is not a default setting!".format(name))
            self.settings[name] = val


    def split_galaxies(self, ptype, mass_list=None):
        """Split galaxies based on particles that have the same mass
        Args:
            ptype: A string or iterable of particle types
        kwargs:
            mass_list: masses of particles in each galaxy
                        should be size NxM where N is number of gals and M is number length of ptype
        """
        if (getattr(ptype, '__iter__', None) is None) or (isinstance(ptype, (str, bytes))):
            ptype = [ptype]

        indices = []

        nlast = 0

        for i, p in enumerate(ptype):
            mass = self.masses[p]
            if mass_list is not None:
                for j, m in enumerate(mass_list):
                    if i == 0:
                        indices.append(np.where(mass == m)[0])
                        nlast = len(mass)
                    else:
                        indices[j] = np.append(indices[j], np.where(mass == m)[0] + nlast)
                        nlast = len(mass)

            else:
                unq = np.unique(mass)
                if i == 0:
                    ngals = len(unq)

                # Add negatives to make list same size as galaxy list.
                #Assume that larger galaxies have more parts for now
                while len(unq) < ngals:
                        unq = np.insert(unq, 0, -1.0)

                masses = np.argsort([np.sum(mass == u)*u for u in unq])[::-1]
                #invert the order so that the largest galaxy is first, as was the case in the old code
                for j, m in enumerate(masses):
                    if i == 0:
                        indices.append(np.where(mass == unq[m])[0])
                        nlast = len(mass)
                    else:
                        indices[j] = np.append(indices[j], np.where(mass == unq[m])[0] + nlast)
                        nlast = len(mass)

        return indices



    def measure_com(self, ptype, indices_list):
        """
        Measure the centers of mass of galaxies in the snapshot
        Args:
            ptype: A string or list of particle types to draw positions from
            indices_list: An iterable item containing indices of each galaxy.
                          If multiple ptypes are given then the indices should
                          index the combined position vector in the proper order.
        """

        if ((getattr(ptype, '__iter__', None) is not None) and
            (not isinstance(ptype, (str, bytes)))):
            pos = np.append(*[self.pos[k] for k in ptype], axis=0)
        else:
            pos = self.pos[ptype]

        centers = []
        for indices in indices_list:  # change this logic
            #First check to see if we have a list lists or just one galaxy
            y = getattr(indices, '__iter__', None)
            if y is None:
                centers = np.mean(pos[indices_list], axis=0)
                break

            centers.append(np.mean(pos[indices], axis=0))

        return np.array(centers)


    def center_of_mass(self, parttype):
        """
        DEPRECATED
        Given a particle type return center of mass
            and indices splitting first galaxy from the rest.

        """
        warnings.warn("center_of_mass is being deprecated", DeprecationWarning)
        # There needs to be a snazzy way of splitting up galaxies!
        # We cannot rely on each galaxy having different mass particles
        # Especially when we run more massive mergers

        idgal1 = np.where(self.masses[parttype] ==
                          self.masses[parttype][0])[0]
        idgal2 = np.where(self.masses[parttype] !=
                          self.masses[parttype][0])[0]
        if len(idgal1) < len(idgal2):
            idgal1, idgal2 = idgal2, idgal1

        if len(idgal2) > 0:
            com1 = np.array([np.mean(self.pos[parttype][idgal1, 0]),
                             np.mean(self.pos[parttype][idgal1, 1]),
                             np.mean(self.pos[parttype][idgal1, 2])])
            com2 = np.array([np.mean(self.pos[parttype][idgal2, 0]),
                             np.mean(self.pos[parttype][idgal2, 1]),
                             np.mean(self.pos[parttype][idgal2, 2])])
        else:
            com1 = [np.mean(self.pos[parttype][:, 0]),
                    np.mean(self.pos[parttype][:, 1]),
                    np.mean(self.pos[parttype][:, 2])]
            com2 = [0, 0, 0]

        return com1, com2, idgal1, idgal2


    def tree_potential_center(self, ptypes=['halo'], offset=[0.0, 0.0, 0.0], gal_num=0):
        """
        Use a tree to find the center of the potential
        kwargs:
            ptypes: list-like container of particle types
            offset: offset of simulation
            gal_num: which galaxy to test. By default will test the first galaxy.
        """
        raise NotImplementedError("This method currently requires pNbody which does not have a python3 version.")

    def find_centers(self,
                     settings=None,
                     measure_fourier=False,
                     num_centers=1,
                     Rd=0,
                     numcontours=20,
                     plot=False,
                     axis=None,
                     return_im=False):
        """
        Compute the halo, disk, and bar centers of a snapshot.
        """
        # Bin the snapshot and run the main measurements
        if settings is None:
            settings = self.settings

        bin_dict = self.bin_snap(settings)
        Z2 = bin_dict['Z2']
        measurements = man.fit_contours(Z2,
                                    settings,
                                    numcontours=numcontours,
                                    plot=plot)

        # Now convert these to the centers of interest
        majors = measurements['majors']
        minors = measurements['minors']
        ecc = measurements['eccs']
        xCenters = measurements['xCenters']
        yCenters = measurements['yCenters']

        indices = self.split_galaxies('halo', mass_list=None)
        com = self.measure_com('halo', indices[settings['gal_num']])

        indices_stars = self.split_galaxies('stars', mass_list=None)
        com_stars = self.measure_com('stars', indices_stars[settings['gal_num']])

        # If we have info on the potential then use the particle with lowest energy
        if settings['halo_center_method'] == 'pot':
            (x_pot, y_pot, z_pot) = self.potential_center(com,
                                                          indices[settings['gal_num']],
                                                          offset=com_stars)
        # All particles
        elif settings['halo_center_method'] == 'tree_all':
            # Dynamical center of all particles
            (x_pot, y_pot, z_pot) = self.tree_potential_center(ptypes=['gas',
                                                                       'halo',
                                                                       'stars',
                                                                       'sfr'],
                                                               offset=com_stars,
                                                               gal_num=settings['gal_num'])
        # Just halo particles
        elif settings['halo_center_method'] == 'tree_halo':
            # Center of halo
            (x_pot, y_pot, z_pot) = self.tree_potential_center(ptypes=['halo'],
                                                               offset=com_stars,
                                                               gal_num=settings['gal_num'])

        else:
            # else we will use the center of mass
            (x_pot, y_pot, z_pot) = np.array(com) - np.array(com_stars)
        if num_centers > 1:
            if num_centers > numcontours:
                num_centers = numcontours

            delta_centers = int(numcontours/num_centers)
            re_ind = [i*delta_centers for i in range(num_centers)]
        else:
            if Rd:
                re = 1.67835 * Rd  # this is from a numeric solution to the
                                   # disk mass formula
                                   # Mdtot*(1-Exp[-r/Rd]*(1+r/Rd))
                re_ind = np.abs((majors+minors)/2. - re).argmin()
            else:
                re_ind = 0   # If we are not given the scalelength
                             # then just take the outer piece

        xcent_disk = xCenters[re_ind]
        ycent_disk = yCenters[re_ind]
        # Define the bar as the outermost ellipse where the eccentricity is above 0.5
        if len(np.where(ecc > 0.5)[0]) > 0:
            bar_ind = np.max(np.where(ecc > 0.5)[0])
            xcent = xCenters[bar_ind]
            ycent = yCenters[bar_ind]
        else:
            xcent = ycent = 0
            bar_ind = -1

        if plot:
            from snaptools import plot_tools
            ell_arts, im = plot_tools.plot_contours(bin_dict,
                                                    measurements,
                                                    bar_ind,
                                                    re_ind,
                                                    [x_pot, y_pot],
                                                    settings,
                                                    axis=axis)
        cent_dict = {}
        cent_dict['barCenter'] = np.array([xcent, ycent])
        cent_dict['diskCenters'] = np.array([xcent_disk, ycent_disk])
        cent_dict['haloCenter'] = np.array([x_pot, y_pot])
        cent_dict['time'] = self.header['time']
        if plot:
            if return_im:
                return cent_dict, ell_arts, im
            else:
                return cent_dict, ell_arts
        else:
            return cent_dict


    def fourier_modes(self,
                      settings=None,
                      use_offset=False,
                      BINS_theta=360):
        """
        Measure the radial fourier modes in a number of bins.
        """

        if settings is None:
            settings = self.settings

        lengthX = settings['xlen']
        parttype = settings['parttype']
        BINS_r = settings['NBINS']
        pos = self.pos[parttype]

        com1, com2, gal1id, gal2id = self.center_of_mass(parttype)

        if use_offset:
            center = settings['offset']
            x_cent, y_cent, z_cent = center
        else:
            x_cent, y_cent, z_cent = com1

        px2 = pos[gal1id, 0] - x_cent
        py2 = pos[gal1id, 1] - y_cent
        r = np.sqrt(px2**2 + py2**2)
        # Y and X are reversed by definition in np.arctan2
        theta = np.arctan2(py2, px2)

        return man.measure_fourier(r, theta, lengthX, BINS_r, BINS_theta)


    def potential_center(self, com1, idgal, offset=[0, 0, 0]):
        """
        Measure the center of the dark matter potential
        """
        r = np.sqrt(np.sum((self.pos['halo'][idgal, :]-com1)**2, axis=1))
        pot = (self.pot['halo'][idgal[r < 100]])

        ke = (0.5*self.masses['halo'][idgal[r < 100]] *
              np.sum(self.vel['halo'][idgal[r < 100], :]**2, axis=1))
        # Binding energy is the sum of the grav. potential and the kinetic energies
        binding_energy = ke+0.5*pot*self.masses['halo'][idgal[r < 100]]
        most_bound = np.argsort(binding_energy)[:100]

        # Take the center to be the center of mass of these 100 most bound particles

        (x_pot,
         y_pot,
         z_pot) = (self.pos['halo'][idgal[r < 100][most_bound], :].mean(axis=0))-offset

        return x_pot, y_pot, z_pot


    def get_stripped_particles(self):
        pass


    def bin_snap(self, settings=None):
        """
        Create 2D density projection of snapshot in one or more projections.

        kwargs:
            settings: settings dictionary
                    if None then use self.settings
        """

        if settings is None:
            settings = self.settings

        bin_dict = {}
        lengthX = settings['xlen']
        lengthY = settings['ylen']
        lengthZ = settings['zlen']
        BINS = settings['NBINS']
        ptype = settings['parttype']
        panels = settings['panel_mode']
        head = self.header
        if ((getattr(ptype, '__iter__', None) is not None) and  # add additional check due to python3
            (not isinstance(ptype, (str, bytes)))):
            pos = np.append(*[self.pos[k] for k in ptype], axis=0)
            mass = np.append(*[self.masses[k] for k in ptype])
        else:
            pos = self.pos[ptype]
            mass = self.masses[ptype]

        scale = settings['log_scale']

    #size in units of scale length
        Zmin = settings['in_min']
        Zmax = settings['in_max']
        if settings['com'] or (settings['gal_num'] > -1):
            indices = self.split_galaxies(ptype, mass_list=None)

        # User supplied offsets
        if any(settings['offset']):
            x_cent, y_cent, z_cent = settings['offset']
        # Offset from com of ptype
        elif settings['com']:
            if settings['gal_num'] < 0:
                x_cent, y_cent, z_cent = self.measure_com(ptype, indices[0])
            else:
                x_cent, y_cent, z_cent = self.measure_com(ptype, indices[settings['gal_num']])
        # Don't offset
        else:
            x_cent = y_cent = z_cent = 0

        if settings['first_only'] or (settings['gal_num'] > -1):
            mass = mass[indices[settings['gal_num']]]
            px = (pos[indices[settings['gal_num']], 0] - x_cent).T
            py = (pos[indices[settings['gal_num']], 1] - y_cent).T
            pz = (pos[indices[settings['gal_num']], 2] - z_cent).T
        else:
            px = pos[:, 0] - x_cent
            py = pos[:, 1] - y_cent
            pz = pos[:, 2] - z_cent

        if settings['plotCompanionCOM']:
            #currently only records second galaxy
            if not (settings['com'] or (settings['gal_num'] > -1)):
                indices = self.split_galaxies(ptype, mass_list=None)

            #currently will plot gal_num + 1, but change this

            bin_dict['companionCOM'] = [np.mean(pos[indices[settings['gal_num'] + 1], 0]
                                                - x_cent),
                                        np.mean(pos[indices[settings['gal_num'] + 1], 1]
                                                - y_cent),
                                        np.mean(pos[indices[settings['gal_num'] + 1], 2]
                                                - z_cent)]


        # All panelmodes need this perspective
        Z2, x, y = man.bin_particles(px, py, lengthX,
                                     lengthY, mass, BINS, scale)
        bin_dict['Z2'] = Z2
        bin_dict['Z2x'] = x
        bin_dict['Z2y'] = y

        # Need other perspectives
        if (panels == "three") or (panels == "small"):
            H, x, z = man.bin_particles(px, pz, lengthX,
                                        lengthZ, mass, BINS, scale)
            bin_dict['H'] = H
            bin_dict['Hx'] = x
            bin_dict['Hy'] = z
            H2, y, z = man.bin_particles(py, pz, lengthY,
                                         lengthZ, mass, BINS, scale)
            bin_dict['H2'] = H2
            bin_dict['H2x'] = y
            bin_dict['H2y'] = z

        if (panels == "starsgas") and (ptype != 'gas'):
            # Need both stars and gas
            # call this again with different parttype
            settings_copy = copy.deepcopy(settings)
            settings_copy['parttype'] = 'gas'
            settings_copy['panel_mode'] = 'xy'
            gasDict = self.bin_snap(settings_copy)
            bin_dict['G'] = gasDict['Z2']
            bin_dict['Gx'] = gasDict['Z2x']
            bin_dict['Gy'] = gasDict['Z2y']

        bin_dict['snaptime'] = head['time']
        return bin_dict

    def to_cube(self,
                filename='snap',
                theta=0,
                lengthX=15,
                lengthY=15,
                BINS=512,
                parttype='stars',
                first_only=False,
                com=False,
                write=True):

        """
        Write snapshot to a fits cube (ppv)

        Kwargs:
            filename: Filename stub to save to disk.
                      Will append '_cube.fits'.
        """

        if write:
            from astropy.io import fits
        theta *= (np.pi / 180.)
    #    head=rs.snapshot_header(filename)
        pos2 = self.pos[parttype]
        vel2 = self.vel[parttype]
        mass2 = self.masses[parttype]

        if first_only:
            com1, com2, gal1id, gal2id = self.center_of_mass(parttype)
            pos2 = pos2[gal1id, :]
            vel2 = vel2[gal1id, :]
            mass2 = mass2[gal1id]
            if com:
                pos2 -= com1

        if theta:  # first check to see if this is even necessary
            rotation_matrix = [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]]
            for i in range(len(pos2[:, 0])):
                pos_vector = np.array([[pos2[i, 0]],
                                       [pos2[i, 1]],
                                       [pos2[i, 2]]])
                vel_vector = np.array([[vel2[i, 0]],
                                       [vel2[i, 1]],
                                       [vel2[i, 2]]])
                pos_rotated = np.dot(rotation_matrix, pos_vector)
                vel_rotated = np.dot(rotation_matrix, vel_vector)
                pos2[i, :] = pos_rotated.T[0]
                vel2[i, :] = vel_rotated.T[0]

        px2 = pos2[:, 0]
        py2 = pos2[:, 1]
        velz = vel2[:, 2]
        galvel = np.median(velz)
        velz -= galvel
        #    pz2=pos2[:,zax]
        H, Edges = np.histogramdd((px2, py2, velz),
                                  range=((-lengthX, lengthX),
                                         (-lengthY, lengthY),
                                         (-200, 200)),
                                  weights=mass2 * 1E10,
                                  bins=(BINS, BINS, 100),
                                  normed=False)


        if write:
            hdu = fits.PrimaryHDU()
            hdu.header['BITPIX'] = -64
            hdu.header['NAXIS'] = 3
            hdu.header['NAXIS1'] = 512
            hdu.header['NAXIS2'] = 512
            hdu.header['NAXIS3'] = 100
            hdu.header['CTYPE3'] = 'VELOCITY'
            #hdu.header['CTYPE3'] = 'VELO-LSR'
            hdu.header['CRVAL3'] = 0.0000000000000E+00
            hdu.header['CDELT3'] = 0.4000000000000E+04
            hdu.header['CRPIX3'] = 0.5000000000000E+02
            hdu.header['CROTA3'] = 0.0000000000000E+00
            hdu.data = H.T
            hdu.writeto(filename + '_cube.fits', clobber=True)
        else:
            return H, Edges


    def to_fits(self,
                filename='snap',
                theta=0,
                lengthX=15,
                lengthY=15,
                BINS=512,
                first_only=False,
                com=False,
                parttype='stars'):
        """
        Write snapshot to a fits map
        """

        from astropy.io import fits
        theta *= (np.pi / 180.)
    #    head=rs.snapshot_header(filename)

        pos2 = self.pos[parttype]
        mass2 = self.masses[parttype]

        if first_only:
            com1, com2, gal1id, gal2id = self.center_of_mass(parttype)
            pos2 = pos2[gal1id, :]
            mass2 = mass2[gal1id]
            if com:
                pos2 -= com1

        if theta:  # first check to see if this is even necessary
            rotation_matrix = [[np.cos(theta), 0, np.sin(theta)],
                               [0, 1, 0],
                               [-np.sin(theta), 0, np.cos(theta)]]
            for i in range(len(pos2[:, 0])):
                vector = np.array([[pos2[i, 0]], [pos2[i, 1]], [pos2[i, 2]]])
                rotated = np.dot(rotation_matrix, vector)
                pos2[i, :] = rotated.T[0]

        px2 = pos2[:, 0]
        py2 = pos2[:, 1]
        # pz2=pos2[:,zax]

        Z2, x, y = np.histogram2d(px2, py2,
                                  range=[[-lengthX, lengthX],
                                         [-lengthY, lengthY]],
                                  weights=mass2 * 1E10,
                                  bins=BINS,
                                  normed=False)

        fits.writeto(filename + '_map.fits', Z2, clobber=True)

    def to_velfield(self,
                    filename='snap',
                    lengthX=15,
                    lengthY=15,
                    BINS=512,
                    first_only=False,
                    com=False,
                    parttype='stars',
                    write=True,
                    axes=[0, 1]):
        """
        Write snapshot to a velocity field.

        Note:This is a pesudo first moment map
        where each pixel contains the average velocity
        in the y direction.
        """

        from astropy.io import fits
        from scipy.stats import binned_statistic_2d

        pos2 = self.pos[parttype]
        vel2 = self.vel[parttype]

        if first_only:
            com1, com2, gal1id, gal2id = self.center_of_mass(parttype)
            px2 = pos2[gal1id, axes[0]]
            py2 = pos2[gal1id, axes[1]]
            vy2 = vel2[gal1id, axes[1]]
            if com:
                px2 -= com1[axes[0]]
                py2 -= com1[axes[1]]
        else:
            px2 = pos2[:, axes[0]]
            py2 = pos2[:, axes[1]]
            vy2 = vel2[:, axes[1]]

        (Z2,
         xedges,
         yedges,
         binnum) = binned_statistic_2d(px2, py2, vy2,
                                       statistic='mean',
                                       range=[[-lengthX, lengthX],
                                              [-lengthY, lengthY]],
                                       bins=BINS)

        hdu = fits.PrimaryHDU()
        hdu.header['BITPIX'] = -64
        hdu.header['NAXIS'] = 2
        hdu.header['NAXIS1'] = 512
        hdu.header['NAXIS2'] = 512

        if write:
            hdu.data = Z2.T
            hdu.writeto(filename + '_velfield.fits', clobber=True)
        else:
            return Z2, xedges, yedges

    def velocity_anisotropy(self,
                            parttype):
        """
        Measure velocity anisotropy of a snapshot
        """
        vel = self.vel[parttype]
        pos = self.pos[parttype]
        vx = vel[:, 0]
        vy = vel[:, 1]
        vz = vel[:, 2]
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        vr = (x*vx + y*vy + z*vz)/r
        vtheta = (-y*vx + x*vy)/np.sqrt(x**2 + y**2)
        vphi = (x*(vx*z - x * vz) + y*(vy*z-y*vz))/(r*np.sqrt(x**2 + y**2))
        Beta = 1-(np.std(vtheta)**2+np.std(vphi)**2)/(2*np.std(vr)**2)
        return Beta


    def measure_concentration(self, center=np.array([0, 0, 0])):
        """
        Measure the concentration parameter - uses separation between 20% and 80% of the light
        """
        com1, com2, idgal1, idgal2 = self.center_of_mass('stars')
        r = np.sort(np.sqrt(np.sum( (self.pos['stars'][idgal1, :] -
                                     com1 - center)**2, axis=1)))
        pmf = np.cumsum(r)
        tot_light = np.sum(r)
        where_80perc = np.argmin(np.abs(pmf - 0.8*tot_light))
        where_20perc = np.argmin(np.abs(pmf - 0.2*tot_light))
        c = 5.0*np.log10(r[where_80perc]/r[where_20perc])
        return c


    def measure_asymmetry(self):
        """
        Measure the asymmetry parameter - rotates projection by 180 degrees
        """
        if self.bin_dict is None:
            self.bin_dict = self.bin_snap()

        surface_density = self.bin_dict['Z2']
        centerx = (self.bin_dict['Z2x'][:-1] + self.bin_dict['Z2x'][1:]) / 2
        centery = (self.bin_dict['Z2y'][:-1] + self.bin_dict['Z2y'][1:]) / 2
        centerX, centerY = np.meshgrid(centerx, centery, indexing='ij')
        A = np.nansum(np.abs(surface_density - np.rot90(surface_density, k=2))/np.nansum(np.abs(surface_density) ))
        return A


    def measure_gini(self, n=256):
        """
        Measure gini coefficient - measure of inequality
        """
        #if self.bin_dict is None:
        self.bin_dict = self.bin_snap()

        centerx = (self.bin_dict['Z2x'][:-1] + self.bin_dict['Z2x'][1:]) / 2
        centery = (self.bin_dict['Z2y'][:-1] + self.bin_dict['Z2y'][1:]) / 2
        centerX, centerY = np.meshgrid(centerx, centery, indexing='ij')

        centerR = np.sqrt(centerX**2 + centerY**2)
        rsort = np.argsort(centerR.ravel())
        stat, bin_edges, binnumber = stats.binned_statistic(centerR.ravel()[rsort],
                                                            self.bin_dict['Z2'].ravel()[rsort],
                                                            bins=n)
        avg_brightness = np.cumsum(stat)/(np.arange(n) + 1)

        brighter_pixels = np.where(avg_brightness*0.2-stat > 0)[0]
        #rp = bin_edges[brighter_pixels[0]]
        mu_rp = stat[brighter_pixels[0]]

        surface_density = self.bin_dict['Z2']

        surface_density = surface_density[surface_density > mu_rp]

        XBar = np.nanmean(surface_density)
        sorted_values = np.sort(surface_density.flatten())
        n = len(sorted_values)
        gini = 1.0/(XBar*n*(n-1))*np.nansum((2.0*np.arange(n)-n-1.0)*sorted_values)
        return gini


    def measure_m20(self, center=np.array([0., 0., 0.])):
        """
        Measure the M20 coefficient
        """
        if self.bin_dict is None:
            self.bin_dict = self.bin_snap()

        surface_density = self.bin_dict['Z2']
        centerx = (self.bin_dict['Z2x'][:-1] + self.bin_dict['Z2x'][1:]) / 2
        centery = (self.bin_dict['Z2y'][:-1] + self.bin_dict['Z2y'][1:]) / 2
        centerX, centerY = np.meshgrid(centerx, centery, indexing='ij')

        good_ind = np.where(np.isfinite(surface_density))
        sort_ind = np.argsort(surface_density[good_ind].ravel())[::-1]
        xs = centerX[good_ind].ravel()[sort_ind]
        ys = centerY[good_ind].ravel()[sort_ind]
        sorted_values = surface_density[good_ind].ravel()[sort_ind]

        fi = np.cumsum(sorted_values)
        ftot = np.sum(sorted_values)

        twenty_perc = np.nanargmin(np.abs(fi - 0.2*ftot))
        Mi = (sorted_values*((xs - center[0])**2 + (ys - center[1])**2))
        M20 = np.log10(np.sum(Mi[:twenty_perc])/np.sum(Mi))
        return M20


    def make_id_file(self, filename, mass_list=None):
        import h5py
        """
        Make a list of ids to split galaxies apart
        Args:
            filename: filename of hdf5 file to store ids in
        kwargs:
        Takes one of a few different list to know how to split galaxies
        Currently supported: mass_list = [gal1, gal2, gal3...]
                Provide a list of names of galaxies sorted by mass (lowest first).
        """
        part_names = ['gas',
                      'halo',
                      'stars',
                      'bulge',
                      'sfr',
                      'other']

        id_file = h5py.File(filename, 'w')
        grps = []

        if mass_list is not None:
            # First make a list of groups that we will fill with IDS
            for gal in mass_list:
                grps.append(id_file.create_group(gal))
            # Now cycle through parts and add them to the groups
            for npart, ptype in zip(self.header['nall'], part_names):
                if npart > 0:
                    unq = np.unique(self.masses[ptype])
                    # Add nans to make list same size as galaxy list. Assume that larger galaxies have more parts for now
                    while len(unq) < len(mass_list):
                        unq = np.insert(unq, 0, -1.0)
                    masses = np.argsort([np.sum(self.masses[ptype] == u)*u for u in unq])
                    for i, gal in enumerate(mass_list):
                        ids = self.ids[ptype][self.masses[ptype] == unq[masses[i]]]
                        if len(ids) > 0:
                            dset = grps[i].create_dataset(ptype, ids.shape)
                            dset[:] = ids
        id_file.close()


    def save(self, fname, userblock_size=0,
             part_names=['gas', 'halo', 'stars', 'bulge', 'sfr', 'other']):
        """
        Save a snapshot object to an hdf5 file. Overload base case
        Note: Must have matching header and data.
        Todo: Gracefully handle mismatches between header and data
        """
        import h5py

        # A list of header attributes, their key names, and data types
        head_attrs = {'npart': (np.int32, 'NumPart_ThisFile'),
                      'nall': (np.uint32, 'NumPart_Total'),
                      'nall_highword': (np.uint32, 'NumPart_Total_HighWord'),
                      'massarr': (np.float64, 'MassTable'),
                      'time': (np.float64, 'Time'),
                      'redshift': (np.float64, 'Redshift'),
                      'boxsize': (np.float64, 'BoxSize'),
                      'filenum': (np.int32, 'NumFilesPerSnapshot'),
                      'omega0': (np.float64, 'Omega0'),
                      'omega_l': (np.float64, 'OmegaLambda'),
                      'hubble': (np.float64, 'HubbleParam'),
                      'sfr': (np.int32, 'Flag_Sfr'),
                      'cooling': (np.int32, 'Flag_Cooling'),
                      'stellar_age': (np.int32, 'Flag_StellarAge'),
                      'metals': (np.int32, 'Flag_Metals'),
                      'feedback': (np.int32, 'Flag_Feedback'),
                      'double': (np.int32, 'Flag_DoublePrecision')}
        datablocks = {"pos": "Coordinates",
                      "vel": "Velocities",
                      "pot": "Potential",
                      "masses": "Masses",
                      "ids": "ParticleIDs",
                      "U": "InternalEnergy",
                      "RHO": "Density",
                      "VOL": "Volume",
                      "CMCE": "Center-of-Mass",
                      "AREA": "Surface Area",
                      "NFAC": "Number of faces of cell",
                      "NE": "ElectronAbundance",
                      "NH": "NeutralHydrogenAbundance",
                      "HSML": "SmoothingLength",
                      "SFR": "StarFormationRate",
                      "AGE": "StellarFormationTime",
                      "Z": "Metallicity",
                      "ACCE": "Acceleration",
                      "VEVE": "VertexVelocity",
                      "FACA": "MaxFaceAngle",
                      "COOR": "CoolingRate",
                      "MACH": "MachNumber",
                      "DMHS": "DM Hsml",
                      "DMDE": "DM Density",
                      "PTSU": "PSum",
                      "DMNB": "DMNumNgb",
                      "NTSC": "NumTotalScatter",
                      "SHSM": "SIDMHsml",
                      "SRHO": "SIDMRho",
                      "SVEL": "SVelDisp",
                      "GAGE": "GFM StellarFormationTime",
                      "GIMA": "GFM InitialMass",
                      "GZ": "GFM Metallicity",
                      "GMET": "GFM Metals",
                      "GMRE": "GFM MetalsReleased",
                      "GMAR": "GFM MetalMassReleased"}

        # Open the file
        with h5py.File(fname, 'w', userblock_size=userblock_size) as f:
            # First write the header
            grp = f.create_group('Header')
            for key, val in self.header.items():
                # If we have a name and dtype, use those
                if key in head_attrs.keys():
                    grp.attrs.create(head_attrs[key][1], val,
                                     dtype=head_attrs[key][0])
                # Otherwise simply use the name we read in
                else:
                    grp.attrs.create(key, val)
            # create the groups for all particles in the snapshot
            grps = [f.create_group('PartType{:d}'.format(i))  if n > 0 else None
                    for i, n in enumerate(self.header['nall'])]
            # iterate through datablocks first

            for attr_name, attr in self.__dict__.items():
                x = getattr(attr, 'states', None)
                if (x is None) and (attr_name not in datablocks):  # only want lazy-dict things
                    continue
                for p, val in attr.items():  # then through particle types

                    i = part_names.index(p)
                    try:
                        dset = grps[i].create_dataset(datablocks[attr_name], val.shape,
                                                      dtype=val.dtype)
                    except KeyError:
                        dset = grps[i].create_dataset(attr_name, val.shape, dtype=val.dtype)
                    dset[:] = val


    def write_csv(self, gal_num=-1, ptypes=['stars'], stepsize=100, columns=['pos', 'vel']):
        """
        Write a csv version of the snapshot. Helpful for paraview or sharing simple versions with collaborators.
        Particles are sorted by ID number, so that the same particles are always in the same row of the file.
        kwargs:
            gal_num: Galaxy number to save. -1 for all galaxies.
            ptypes: Particle types to save.
            stepsize: Save every nth particle.
            columns: Which properties to save. Properties must be in every particle type you request.
        """
        if gal_num < 0:
            idgal = list(range(len(self.ids[ptypes[0]])))
        else:
            idgal = self.split_galaxies(ptypes[0], mass_list=None)[gal_num]

        all_data = {}
        allids = self.ids[ptypes[0]][idgal]
        for column in columns:
            # Get an arbitrary name, first try in non-misc properties, then try in misc props
            try:
                all_data[column] = self.__dict__[column][ptypes[0]][idgal]
            except KeyError:
                all_data[column] = self.misc[ptypes[0]][column][idgal]

            for ptype in ptypes[1:]:
                # check to make sure we have particles
                if (ptype in self.pos.keys()) and (len(self.pos[ptype]) > 0):
                    if gal_num < 0:
                        idgal = list(range(len(self.ids[ptype])))
                    else:
                        idgal = self.split_galaxies(ptype, mass_list=None)[gal_num]
                #first construc list of all ids
                allids = np.concatenate((self.ids[ptype][idgal], allids), axis=0)
                # construct lists of other properties
                try:
                    all_data[column] = np.concatenate((self.__dict__[column][ptypes[0]][idgal],
                                                       all_data[column]), axis=0)
                except KeyError:
                    all_data[column] = np.concatenate((self.misc[ptypes[0]][column][idgal],
                                                       all_data[column]), axis=0)

        s = np.argsort(allids)

        header = []
        for column_name, column_data in all_data.items():
            if len(column_data.shape) == 1:
                header.append(column_name)
            else:
                header.append("{0}x,{0}y,{0}z".format(column_name))
        header = ','.join(header)

        with open(self.filename+".csv", "w") as f:
            f.write(header+'\n')
            for si in s[::stepsize]:
                column_text = ''
                for _, column_data in all_data.items():
                    if len(column_data.shape) > 1:
                        column_text += "{:3.3f},{:3.3f},{:3.3f},".format(*column_data[si, :])
                    else:
                        column_text += "{:g},".format(column_data[si])
                #Strip trailing comma and add newline break
                f.write(column_text[:-1]+"\n")

    def __repr__(self):
        if not self.header:  # empty dict evaluates to False
            return "Empty Snapshot"
        else:
            if self.filename is None:
                return str(self.header)
            else:
                if isinstance(self.filename, list):
                    return """Multi-part snapshot files: {:s}
--------------------------------------------------
Header: {:s}""".format(str(self.filename), str(self.header))
                else:
                    return """Snapshot file - {:s}
-------------------------------------------------
Header: {:s}""".format(self.filename, str(self.header))
