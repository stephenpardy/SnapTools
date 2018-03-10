from __future__ import print_function, absolute_import, division
import unittest
import numpy as np
from snaptools import EllipseFitter

from snaptools import manipulate as man
from snaptools import utils
from snaptools import snapshot

class BaseEllipseCase(unittest.TestCase):
    def setUp(self):

        N = 10000

        x1 = -5
        dx = 10

        y1 = -10
        dy = 5

        xlen = 50
        ylen = 50
        NBINS = 200

        x = np.random.normal(loc=x1, scale=dx, size=N)
        y = np.random.normal(loc=y1, scale=dy, size=N)

        self.true_centers = np.array([x.mean(), y.mean()])

        snap = snapshot.Snapshot()  # empty snapshot
        snap.masses['stars'] = np.ones(N)
        snap.pos['stars'] = np.zeros((N, 3))
        snap.pos['stars'][:, 0] = x
        snap.pos['stars'][:, 1] = y

        snap.masses['halo'] = np.ones(N)
        snap.pos['halo'] = np.zeros((N, 3))
        snap.pos['halo'][:, 0] = x
        snap.pos['halo'][:, 1] = y

        snap.header['time'] = 0

        Z, binx, biny = np.histogram2d(x, y, range=[[-xlen, xlen],[-ylen, ylen]],
                                       bins=NBINS)
        xbins, ybins = np.meshgrid(binx, biny, indexing='ij')

        self.snap = snap
        self.xlen = xlen
        self.ylen = ylen
        self.NBINS = NBINS


class SingleEllipseTest(BaseEllipseCase):
    def runTest(self):

        settings = utils.make_settings(xlen=self.xlen, ylen=self.ylen, NBINS=self.NBINS,
                                       log_scale=False, in_min=1, in_max=3,
                                       halo_center_method='com')
        cent_dict = self.snap.find_centers(settings=settings,
                                           numcontours=1, plot=False,
                                           return_im=False)
        assert np.allclose(cent_dict['diskCenters'], self.true_centers, atol=0.5)


if __name__ == '__main__':
    unittest.main()