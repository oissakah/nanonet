import networkx as nx
import numpy as np

from nanonet import NanoparticleNetwork
from nanonet.analysis import (
    count_edge_disjoint_pathways,
    effective_resistance,
)


def _series_network():
    net = NanoparticleNetwork(
        L=1.0,
        N=3,
        mu_a=0.0,
        std_a=0.0,
        edge_k=1.0,
        node_resistance_ohm=10.0,
    )
    net.positions = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    net.G = nx.Graph()
    for node, pos in enumerate(net.positions):
        net.G.add_node(
            node, pos=pos, Va=0.0, Vth=0.0,
            R_node=10.0, activated=False,
        )
    net.G.add_edge(0, 1, R_edge=5.0, resistance=5.0, distance=0.5)
    net.G.add_edge(1, 2, R_edge=5.0, resistance=5.0, distance=0.5)
    net.source_nodes = [0]
    net.drain_nodes = [2]
    net.source_node = 0
    net.drain_node = 2
    net.n_junctions = 3
    return net


def test_current_conservation_and_effective_resistance():
    net = _series_network()
    solution = net.solve_active_network({0, 1, 2}, 2.0)

    # Each connection is 5 + 1/2*10 = 10 ohm; series total = 20 ohm.
    assert np.isclose(solution["total_current_A"], 0.1)
    assert np.isclose(solution["source_current_A"], 0.1)
    assert np.isclose(solution["drain_current_A"], 0.1)
    assert solution["current_balance_error_A"] < 1e-12
    assert np.isclose(effective_resistance(net, {0, 1, 2}, 1.0), 20.0)


def test_canonical_laplacian_is_symmetric_positive_semidefinite():
    net = _series_network()
    system = net.build_active_laplacian({0, 1, 2})
    matrix = system["laplacian"].toarray()
    assert np.allclose(matrix, matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert eigenvalues.min() >= -1e-12


def test_activation_and_resistance_are_decoupled():
    net = NanoparticleNetwork(
        N=120,
        mu_a=7.0,
        std_a=2.0,
        node_resistance_ohm=3.5e9,
    ).build(seed=41)
    va = np.array([net.G.nodes[n]["Va"] for n in net.G.nodes()])
    resistance = np.array([net.G.nodes[n]["R_node"] for n in net.G.nodes()])
    assert np.std(va) > 0
    assert np.allclose(resistance, 3.5e9)


def test_edge_disjoint_pathway_count():
    net = NanoparticleNetwork(N=4, node_resistance_ohm=0.0)
    net.positions = np.array([[0, 0], [0.5, 0.25], [0.5, -0.25], [1, 0]], float)
    net.G = nx.Graph()
    for node, pos in enumerate(net.positions):
        net.G.add_node(node, pos=pos, Va=0.0, Vth=0.0, R_node=0.0)
    for i, j in [(0, 1), (1, 3), (0, 2), (2, 3)]:
        net.G.add_edge(i, j, R_edge=1.0, resistance=1.0, distance=1.0)
    net.source_nodes = [0]
    net.drain_nodes = [3]
    assert count_edge_disjoint_pathways(net, {0, 1, 2, 3}) == 2
