from snaptools import snapshot

class TestSnapshot():

    @classmethod
    def setup_class(self):
        """ setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        self.snap = snapshot.Snapshot('tests/galaxies0.hdf5')
        if self.snap is None:
            raise IOError('Must have the file galaxies.hdf5')


    def test_move(self):

        #shift entire snapshot over
        for i, ptype in enumerate(self.snap.part_names):
            if self.snap.header['nall'][i] > 0:
                self.snap.pos[ptype] += 100
                self.snap.vel[ptype] += 10

        self.snap.save('tests/galaxies1.hdf5')