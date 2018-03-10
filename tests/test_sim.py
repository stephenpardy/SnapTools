import numpy as np
from snaptools import simulation

class TestSimulation():

    @classmethod
    def setup_class(self):
        """ load Simulation object
        """
        self.sim = simulation.Simulation('tests/', snapbase='galaxies')
        self.first_gal = np.arange(10000, dtype=int)
        self.second_gal = np.arange(10000, 20000, dtype=int)

    def test_sim(self):
        """
        Test basics
        """
        assert self.sim.nsnaps == 2


    def test_settings(self):
        """
        Test that we can set settings properly - also use these settings for later tests
        """
        self.sim.set_settings(halo_center_method='com')  # don't have pot or a tree method for this test
        assert self.sim.settings['halo_center_method'] == 'com'


    def test_com(self):
        """
        Test the center of mass measuring
        """
        assert np.allclose(self.sim.measure_centers_of_mass(indices=[self.first_gal, self.second_gal]),
                            np.array([[-9.39480209e+01, -3.41116142e+01, -1.63059831e-02],
                                      [ 6.05205584e+00, 6.58883972e+01, 9.99835892e+01]],
                                     dtype=np.float32))


    def test_measure_separation(self):
        separation = self.sim.measure_separation(indices=[self.first_gal, self.second_gal])


    def test_measure_centers(self):
        centers = self.sim.measure_centers()

