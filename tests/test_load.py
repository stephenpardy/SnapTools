import unittest
import numpy as np
from snaptools import snapshot

class BaseSnapshotCase(unittest.TestCase):
    def setUp(self):
        self.snap = snapshot.Snapshot('tests/galaxies.hdf5')
        if self.snap is None:
            raise IOError('Must have the file galaxies.hdf5')


class HeaderTest(BaseSnapshotCase):
    def runTest(self):
        good_header = {'npart': np.array([0, 40000, 20000, 0, 0, 0], dtype='int'),
                       'nall': np.array([0, 40000, 20000, 0, 0, 0], dtype='int'),
                       'massarr': np.array([ 0.,  0.00104634,  0.00023252,  0.,  0.,  0.], dtype='float32')}
        for key, val in good_header.items():
            assert np.allclose(val, self.snap.header[key])


class SplitGalaxiesTest(BaseSnapshotCase):
    def runTest(self):
        indices = self.snap.split_galaxies('stars')
        self.assertEquals(len(indices), 1)
        self.assertEquals(len(indices[0]), 20000)


class CenterOfMassTest(BaseSnapshotCase):
    def runTest(self):
        first_gal = np.arange(10000, dtype=int)
        second_gal = np.arange(10000, 20000, dtype=int)
        com = self.snap.measure_com('stars', [first_gal, second_gal])
        assert np.allclose(com[0], [-9.39480209e+01,  -3.41116142e+01,  -1.63059831e-02])
        assert np.allclose(com[1], [9.40026627e+01,   3.41231651e+01,  -5.44914752e-02])


if __name__ == '__main__':
    unittest.main()