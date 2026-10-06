import numpy as np

from nanonet import NanoparticleNetwork
from nanonet.sweeps.parameter_sweep import connection_radius_for_N


def test_connection_radius_is_fixed_across_N():
    for N in [200, 400, 600, 800]:
        assert connection_radius_for_N(N) == 0.15


def test_higher_N_increases_mean_degree_at_fixed_radius():
    def mean_degree(N):
        net = NanoparticleNetwork(
            N=N,
            connection_radius=0.15,
            mu_a=6.0,
            std_a=3.0,
            node_resistance_ohm=3.5e9,
        ).build(seed=41)
        return np.mean([degree for _, degree in net.G.degree()])

    assert mean_degree(800) > mean_degree(200)
