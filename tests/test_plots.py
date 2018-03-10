import numpy as np
from snaptools import plot_tools
from snaptools import simulation
import os

class TestPlots():

    @classmethod
    def setup_class(self):
        """ load Simulation object
        """
        self.sim = simulation.Simulation('tests/', snapbase='galaxies')
        self.first_gal = np.arange(10000, dtype=int)
        self.second_gal = np.arange(10000, 20000, dtype=int)

    def test_plot_loop(self):
        figs = plot_tools.plot_loop(self.sim.snaps)
        assert figs is not None