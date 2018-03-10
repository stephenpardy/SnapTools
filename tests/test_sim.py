from snaptools import simulation

class TestSimulation():

    @classmethod
    def setup_class(self):
        """ setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        self.sim = simulation.Simulation('tests/', snapbase='galaxies')

    def test_sim(self):
        assert self.sim.nsnaps == 2
