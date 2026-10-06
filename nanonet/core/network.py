"""
nanonet.core.network
--------------------
Canonical voltage-gated Kirchhoff model for nanoparticle necklace networks.

Activation and resistance are deliberately decoupled:
* Va_i (also exposed as legacy Vth) controls whether junction i is active.
* R_node is a fixed junction resistance independent of Va_i.
* Each active connection uses R_edge + 0.5*(R_i + R_j), with electrode
  junction resistance omitted.
* Source, drain, device, and edge currents are reconstructed from one
  symmetric conductance-weighted circuit Laplacian.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class NanoparticleNetwork:
    """Spatial graph representation of a voltage-activated resistor network."""

    R_MIN_OHM = 1.0

    def __init__(
        self,
        L: float | tuple[float, float] = 1.0,
        N: int = 500,
        fv: float = 0.0,
        mu_a: float = 4.0,
        std_a: float = 1.0,
        *,
        connection_radius: float = 0.15,
        edge_k: float = 2.0e10,
        node_resistance_ohm: float = 3.5e9,
        source_frac: float = 0.15,
        drain_frac: float = 0.15,
        va_min: float = 0.0,
        va_max: float = 20.0,
        void_radius: float = 0.08,
        strict_N: bool = False,
        node_r_scale: float | None = None,
        r_floor: float | None = None,
    ):
        if isinstance(L, (int, float)):
            self.domain = (float(L), float(L))
        else:
            self.domain = (float(L[0]), float(L[1]))

        self.N = int(N)
        self.fv = float(fv)
        self.requested_void_fraction = float(fv)
        self.achieved_void_fraction = 0.0
        self.mu_a = float(mu_a)
        self.std_a = float(std_a)
        self.connection_radius = float(connection_radius)
        self.edge_k = float(edge_k)

        if node_r_scale is not None:
            warnings.warn(
                "node_r_scale is deprecated. It is now interpreted as a fixed "
                "junction resistance in ohms; use node_resistance_ohm instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            node_resistance_ohm = float(node_r_scale)
        if r_floor is not None:
            warnings.warn(
                "r_floor is deprecated and ignored because junction resistance "
                "is independent of activation voltage.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.node_resistance_ohm = float(node_resistance_ohm)
        self.node_r_scale = self.node_resistance_ohm
        self.r_floor = 0.0

        self.source_frac = float(source_frac)
        self.drain_frac = float(drain_frac)
        self.va_min = float(va_min)
        self.va_max = float(va_max)
        self.void_radius = float(void_radius)
        self.strict_N = bool(strict_N)

        self.G: nx.Graph = nx.Graph()
        self.positions: np.ndarray | None = None
        self.source_nodes: list[int] = []
        self.drain_nodes: list[int] = []
        self.source_node: int | None = None
        self.drain_node: int | None = None
        self.voids: list[dict] = []
        self.n_junctions = 0

        self._distribution_config: dict | None = None
        self._discrete_va: list[float] | None = None

    def build(self, seed: int | None = None) -> "NanoparticleNetwork":
        self._place_voids(seed)
        if self.strict_N:
            self._build_strict_N(seed)
        else:
            self._generate_positions(seed)
            self._build_graph(seed)
            self._prune_low_degree()
        self._identify_electrodes()
        return self

    def _sample_positions(self, n: int, rng: np.random.Generator) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        max_trials = max(10_000, n * 20_000)
        trials = 0
        while len(pts) < n and trials < max_trials:
            p = (
                float(rng.uniform(0.0, self.domain[0])),
                float(rng.uniform(0.0, self.domain[1])),
            )
            if not self.voids or not self._point_in_void(p):
                pts.append(p)
            trials += 1
        if len(pts) < n:
            raise RuntimeError(
                f"Could only place {len(pts)}/{n} nodes outside voids. "
                "Reduce fv or void_radius."
            )
        return pts

    def _build_strict_N(self, seed: int | None) -> None:
        from scipy.spatial import cKDTree

        target = self.N
        rng_pos = np.random.default_rng(seed)
        rng_va = np.random.default_rng((seed + 1000) if seed is not None else None)
        pts = list(self._sample_positions(target, rng_pos))

        graph = nx.Graph()
        pos_arr = np.asarray(pts, dtype=float)
        for _ in range(20):
            pos_arr = np.asarray(pts, dtype=float)
            graph = nx.Graph()
            for i, p in enumerate(pos_arr):
                va = self._sample_vth(rng_va)
                graph.add_node(
                    i, pos=p, Va=va, Vth=va,
                    R_node=self.node_resistance_ohm, activated=False,
                )

            tree = cKDTree(pos_arr)
            for i, j in tree.query_pairs(
                r=self.connection_radius, output_type="ndarray"
            ):
                if self.voids and self._segment_crosses_void(pos_arr[i], pos_arr[j]):
                    continue
                dist = float(np.linalg.norm(pos_arr[i] - pos_arr[j]))
                r_edge = self.edge_k * dist
                graph.add_edge(
                    int(i), int(j), resistance=r_edge, R_edge=r_edge,
                    distance=dist, length=dist, activated=False,
                )

            while True:
                low = [n for n, degree in graph.degree() if degree < 2]
                if not low:
                    break
                graph.remove_nodes_from(low)

            survivors = sorted(graph.nodes())
            if len(survivors) == target:
                break
            deficit = target - len(survivors)
            kept = [tuple(pos_arr[i]) for i in survivors]
            pts = kept + self._sample_positions(deficit, rng_pos)

        survivors = sorted(graph.nodes())
        self.positions = pos_arr[survivors] if survivors else np.empty((0, 2))
        mapping = {old: new for new, old in enumerate(survivors)}
        self.G = nx.relabel_nodes(graph.subgraph(survivors).copy(), mapping)
        for node in self.G.nodes():
            self.G.nodes[node]["pos"] = self.positions[node]
        self.n_junctions = self.G.number_of_nodes()
        if self.n_junctions != target:
            raise RuntimeError(
                f"strict_N requested {target} nodes but generated {self.n_junctions}."
            )

    def _place_voids(self, seed: int | None) -> None:
        self.requested_void_fraction = float(self.fv)
        if self.fv <= 0.0:
            self.voids = []
            self.achieved_void_fraction = 0.0
            return

        r = float(self.void_radius)
        area = self.domain[0] * self.domain[1]
        n_v = int(round(self.fv * area / (np.pi * r * r)))
        if n_v < 1:
            self.voids = []
            self.achieved_void_fraction = 0.0
            return

        rng = np.random.default_rng(seed)
        buf = 0.02
        lo_x, hi_x = buf + r, self.domain[0] - buf - r
        lo_y, hi_y = buf + r, self.domain[1] - buf - r
        if hi_x <= lo_x:
            lo_x = hi_x = self.domain[0] / 2.0
        if hi_y <= lo_y:
            lo_y = hi_y = self.domain[1] / 2.0

        self.voids = [
            {
                "center": (
                    float(rng.uniform(lo_x, hi_x)) if hi_x > lo_x else float(lo_x),
                    float(rng.uniform(lo_y, hi_y)) if hi_y > lo_y else float(lo_y),
                ),
                "radius": r,
            }
            for _ in range(n_v)
        ]
        self.achieved_void_fraction = self._compute_achieved_void_fraction()

    def _compute_achieved_void_fraction(self, grid_size: int = 700) -> float:
        """Deterministic geometric-union estimate; overlaps are counted once."""
        if not self.voids:
            return 0.0
        width, height = self.domain
        xs = (np.arange(grid_size) + 0.5) * width / grid_size
        ys = (np.arange(grid_size) + 0.5) * height / grid_size
        xg, yg = np.meshgrid(xs, ys, indexing="xy")
        inside = np.zeros((grid_size, grid_size), dtype=bool)
        for void in self.voids:
            cx, cy = void["center"]
            radius = void["radius"]
            inside |= (xg - cx) ** 2 + (yg - cy) ** 2 <= radius ** 2
        return float(np.mean(inside))

    def _point_in_void(self, point) -> bool:
        x, y = point
        return any(
            (x - v["center"][0]) ** 2 + (y - v["center"][1]) ** 2
            < v["radius"] ** 2
            for v in self.voids
        )

    def _segment_crosses_void(self, p1, p2) -> bool:
        p1 = np.asarray(p1, float)
        p2 = np.asarray(p2, float)
        direction = p2 - p1
        for void in self.voids:
            center = np.asarray(void["center"], float)
            radius = float(void["radius"])
            if np.allclose(direction, 0.0):
                hit = np.linalg.norm(p1 - center) <= radius
            else:
                t = np.clip(
                    np.dot(center - p1, direction) / np.dot(direction, direction),
                    0.0, 1.0,
                )
                hit = np.linalg.norm(p1 + t * direction - center) < radius
            if hit:
                return True
        return False

    def _generate_positions(self, seed: int | None) -> None:
        rng = np.random.default_rng(seed)
        if not self.voids:
            self.positions = rng.random((self.N, 2)) * np.asarray(self.domain)
        else:
            self.positions = np.asarray(self._sample_positions(self.N, rng), dtype=float)

    @staticmethod
    def _sample_Va_distribution(dist_config: dict, rng: np.random.Generator) -> float:
        kind = str(dist_config.get("type", "normal")).lower()
        if kind == "uniform":
            return float(rng.uniform(dist_config.get("min", 0.0), dist_config.get("max", 10.0)))
        if kind in {"normal", "gaussian"}:
            mean = float(dist_config.get("mean", 8.0))
            std = float(dist_config.get("std", 2.0))
            vmin = float(dist_config.get("min", 0.0))
            vmax = float(dist_config.get("max", np.inf))
            if std <= 0:
                return float(np.clip(mean, vmin, vmax))
            for _ in range(10_000):
                value = float(rng.normal(mean, std))
                if vmin <= value <= vmax:
                    return value
            raise RuntimeError("Could not sample the requested truncated normal.")
        if kind == "lognormal":
            return float(rng.lognormal(dist_config.get("mean", 2.0), dist_config.get("sigma", 0.5)))
        if kind == "exponential":
            return float(dist_config.get("offset", 0.0) + rng.exponential(dist_config.get("scale", 5.0)))
        if kind == "gamma":
            return float(
                dist_config.get("offset", 0.0)
                + rng.gamma(dist_config.get("shape", 2.0), dist_config.get("scale", 3.0))
            )
        if kind == "weibull":
            return float(dist_config.get("scale", 8.0) * rng.weibull(dist_config.get("shape", 2.0)))
        if kind == "bimodal":
            mode1 = dist_config.get("mode1")
            mode2 = dist_config.get("mode2")
            if mode1 is None or mode2 is None:
                raise ValueError("Bimodal distribution requires mode1 and mode2.")
            chosen = mode1 if rng.random() < dist_config.get("weight", 0.5) else mode2
            return NanoparticleNetwork._sample_Va_distribution(chosen, rng)
        raise ValueError(f"Unknown activation-voltage distribution: {kind}")

    def _sample_vth(self, rng: np.random.Generator) -> float:
        if self._discrete_va is not None:
            return float(rng.choice(self._discrete_va))
        if self._distribution_config is not None:
            return self._sample_Va_distribution(self._distribution_config, rng)
        return self._sample_Va_distribution(
            {
                "type": "normal",
                "mean": self.mu_a,
                "std": self.std_a,
                "min": self.va_min,
                "max": self.va_max,
            },
            rng,
        )

    def _build_graph(self, seed: int | None) -> None:
        from scipy.spatial import cKDTree

        if self.positions is None:
            raise RuntimeError("Positions must be generated before graph construction.")
        rng = np.random.default_rng((seed + 1000) if seed is not None else None)
        self.G = nx.Graph()

        for i, pos in enumerate(self.positions):
            va = self._sample_vth(rng)
            self.G.add_node(
                i, pos=pos, Va=va, Vth=va,
                R_node=self.node_resistance_ohm, activated=False,
            )

        tree = cKDTree(self.positions)
        for i, j in tree.query_pairs(r=self.connection_radius, output_type="ndarray"):
            if self.voids and self._segment_crosses_void(self.positions[i], self.positions[j]):
                continue
            distance = float(np.linalg.norm(self.positions[i] - self.positions[j]))
            resistance = self.edge_k * distance
            self.G.add_edge(
                int(i), int(j), resistance=resistance, R_edge=resistance,
                distance=distance, length=distance, activated=False,
            )

    def _prune_low_degree(self) -> None:
        while True:
            low = [node for node, degree in self.G.degree() if degree < 2]
            if not low:
                break
            self.G.remove_nodes_from(low)

        remaining = sorted(self.G.nodes())
        if remaining:
            self.positions = self.positions[remaining]
            mapping = {old: new for new, old in enumerate(remaining)}
            self.G = nx.relabel_nodes(self.G, mapping, copy=True)
            for node in self.G.nodes():
                self.G.nodes[node]["pos"] = self.positions[node]
        else:
            self.positions = np.empty((0, 2))
        self.n_junctions = self.G.number_of_nodes()

    def _identify_electrodes(self) -> None:
        width = self.domain[0]
        source_limit = self.source_frac * width
        drain_limit = (1.0 - self.drain_frac) * width
        self.source_nodes = [
            node for node in self.G.nodes()
            if self.positions[node][0] <= source_limit
        ]
        self.drain_nodes = [
            node for node in self.G.nodes()
            if self.positions[node][0] >= drain_limit
        ]
        self.source_node = self.source_nodes[0] if self.source_nodes else None
        self.drain_node = self.drain_nodes[0] if self.drain_nodes else None

    def generate_network(
        self,
        seed=None,
        node_Vth=None,
        node_Va=None,
        edge_k=None,
        node_resistance_ohm=None,
        node_r_scale=None,
        r_floor=None,
    ):
        """Backward-compatible generator using the corrected physics."""
        config = node_Va if node_Va is not None else node_Vth
        if config is not None:
            if isinstance(config, dict):
                self._distribution_config = dict(config)
                self.mu_a = float(config.get("mean", self.mu_a))
                self.std_a = float(config.get("std", self.std_a))
                self.va_min = float(config.get("min", self.va_min))
                self.va_max = float(config.get("max", self.va_max))
            elif hasattr(config, "__iter__"):
                self._discrete_va = [float(value) for value in config]

        if edge_k is not None:
            self.edge_k = float(edge_k)
        if node_r_scale is not None:
            warnings.warn(
                "node_r_scale is deprecated; treating it as fixed ohms.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.node_resistance_ohm = float(node_r_scale)
        if node_resistance_ohm is not None:
            self.node_resistance_ohm = float(node_resistance_ohm)
        self.node_r_scale = self.node_resistance_ohm
        if r_floor is not None:
            warnings.warn("r_floor is ignored by the corrected model.", DeprecationWarning, stacklevel=2)

        self._place_voids(seed)
        self._generate_positions(seed)
        self._build_graph(seed)
        self._prune_low_degree()
        return self

    def set_electrodes(self, source=0, drain=None):
        self.source_node = int(source)
        self.drain_node = int(drain) if drain is not None else self.n_junctions - 1
        self.source_nodes = [self.source_node]
        self.drain_nodes = [self.drain_node]
        return self.source_nodes, self.drain_nodes

    def identify_sources_drains(
        self,
        source_frac=None,
        drain_frac=None,
        *,
        left_thresh=None,
        right_thresh=None,
    ):
        if left_thresh is not None:
            source_frac = float(left_thresh)
        if right_thresh is not None:
            drain_frac = 1.0 - float(right_thresh)
        if source_frac is not None:
            self.source_frac = float(source_frac)
        if drain_frac is not None:
            self.drain_frac = float(drain_frac)
        self._identify_electrodes()
        return self.source_nodes, self.drain_nodes

    def activated_nodes(self, applied_voltage: float) -> set[int]:
        return {
            node for node in self.G.nodes()
            if self.G.nodes[node].get("Va", self.G.nodes[node]["Vth"])
            <= float(applied_voltage)
        }

    def _bridging_nodes(self, activated_nodes: Iterable[int]) -> set[int]:
        active = set(activated_nodes)
        if not active:
            return set()
        active_sources = set(self.source_nodes) & active
        active_drains = set(self.drain_nodes) & active
        if not active_sources or not active_drains:
            return set()

        bridging: set[int] = set()
        for component in nx.connected_components(self.G.subgraph(active)):
            if component & active_sources and component & active_drains:
                bridging.update(component)
        return bridging

    def total_edge_resistance(self, i: int, j: int) -> float:
        electrodes = set(self.source_nodes) | set(self.drain_nodes)
        ri = 0.0 if i in electrodes else float(self.G.nodes[i].get("R_node", 0.0))
        rj = 0.0 if j in electrodes else float(self.G.nodes[j].get("R_node", 0.0))
        return max(
            float(self.G[i][j].get("R_edge", self.G[i][j].get("resistance", 0.0)))
            + 0.5 * (ri + rj),
            self.R_MIN_OHM,
        )

    def build_active_laplacian(self, activated_nodes: Iterable[int]) -> dict | None:
        working = self._bridging_nodes(activated_nodes)
        if not working:
            return None

        nodes = sorted(working)
        index = {node: k for k, node in enumerate(nodes)}
        matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
        edge_conductance: dict[tuple[int, int], float] = {}

        for i, j in self.G.subgraph(working).edges():
            conductance = 1.0 / self.total_edge_resistance(i, j)
            edge_conductance[(i, j)] = conductance
            ii, jj = index[i], index[j]
            matrix[ii, ii] += conductance
            matrix[jj, jj] += conductance
            matrix[ii, jj] -= conductance
            matrix[jj, ii] -= conductance

        return {
            "laplacian": csr_matrix(matrix),
            "nodes": nodes,
            "index": index,
            "working_nodes": set(working),
            "edge_conductance": edge_conductance,
            "active_sources": sorted(set(self.source_nodes) & working),
            "active_drains": sorted(set(self.drain_nodes) & working),
        }

    def solve_active_network(
        self, activated_nodes: Iterable[int], applied_voltage: float
    ) -> dict:
        empty = {
            "total_current_A": 0.0,
            "source_current_A": 0.0,
            "drain_current_A": 0.0,
            "current_balance_error_A": 0.0,
            "node_potentials": {},
            "edge_currents": {},
            "working_nodes": set(),
        }
        system = self.build_active_laplacian(activated_nodes)
        if system is None:
            return empty

        nodes = system["nodes"]
        index = system["index"]
        sources = system["active_sources"]
        drains = system["active_drains"]
        boundary = sources + drains
        boundary_set = set(boundary)
        unknown = [node for node in nodes if node not in boundary_set]

        potentials = np.zeros(len(nodes), dtype=float)
        for source in sources:
            potentials[index[source]] = float(applied_voltage)

        if unknown:
            unknown_idx = np.asarray([index[node] for node in unknown], dtype=int)
            boundary_idx = np.asarray([index[node] for node in boundary], dtype=int)
            lap = system["laplacian"].tocsr()
            lhs = lap[unknown_idx][:, unknown_idx]
            rhs = -lap[unknown_idx][:, boundary_idx] @ potentials[boundary_idx]
            try:
                solved = spsolve(lhs, rhs)
            except Exception:
                solved = np.full(len(unknown), np.nan)
            if not np.all(np.isfinite(solved)):
                failed = dict(empty)
                failed["current_balance_error_A"] = np.nan
                return failed
            potentials[unknown_idx] = solved

        raw_edge_currents: dict[tuple[int, int], float] = {}
        source_current = 0.0
        drain_current = 0.0
        source_set = set(sources)
        drain_set = set(drains)

        for (i, j), conductance in system["edge_conductance"].items():
            current = float(
                conductance * (potentials[index[i]] - potentials[index[j]])
            )
            raw_edge_currents[(i, j)] = current

            if i in source_set and j not in source_set:
                source_current += current
            elif j in source_set and i not in source_set:
                source_current -= current

            if i in drain_set and j not in drain_set:
                drain_current -= current
            elif j in drain_set and i not in drain_set:
                drain_current += current

        source_current = abs(float(source_current))
        drain_current = abs(float(drain_current))
        total_current = 0.5 * (source_current + drain_current)

        max_edge_current = max(
            (abs(value) for value in raw_edge_currents.values()), default=0.0
        )
        tolerance = max(1e-30, 1e-12 * max_edge_current)
        edge_currents = {
            edge: value
            for edge, value in raw_edge_currents.items()
            if abs(value) > tolerance
        }

        return {
            "total_current_A": total_current,
            "source_current_A": source_current,
            "drain_current_A": drain_current,
            "current_balance_error_A": abs(source_current - drain_current),
            "node_potentials": {
                node: float(potentials[index[node]]) for node in nodes
            },
            "edge_currents": edge_currents,
            "working_nodes": set(nodes),
        }

    def _solve_kirchhoff(self, activated_nodes: set, V_applied: float):
        solution = self.solve_active_network(activated_nodes, V_applied)
        return solution["total_current_A"], solution["node_potentials"]

    def _reconstruct_edge_currents(
        self, activated_nodes: set, V_applied: float, phi: dict | None = None
    ) -> dict:
        return self.solve_active_network(activated_nodes, V_applied)["edge_currents"]

    def iv_curve(
        self,
        V_start: float = 0.0,
        V_max: float = 16.0,
        V_step: float = 0.5,
    ) -> dict:
        if not self.source_nodes or not self.drain_nodes:
            raise RuntimeError("Call build() or identify_sources_drains() first.")

        voltages = []
        currents = []
        conductances = []
        source_currents = []
        drain_currents = []
        balance_errors = []
        percolation_voltage = None

        voltage = float(V_start)
        while voltage <= float(V_max) + 1e-10:
            voltage = round(voltage, 10)
            activated = self.activated_nodes(voltage)
            if percolation_voltage is None and self._bridging_nodes(activated):
                percolation_voltage = voltage
            solution = self.solve_active_network(activated, voltage)
            current = solution["total_current_A"]

            voltages.append(voltage)
            currents.append(current)
            conductances.append(current / voltage if voltage > 1e-12 else 0.0)
            source_currents.append(solution["source_current_A"])
            drain_currents.append(solution["drain_current_A"])
            balance_errors.append(solution["current_balance_error_A"])
            voltage = round(voltage + float(V_step), 10)

        return {
            "voltages": np.asarray(voltages),
            "currents": np.asarray(currents),
            "conductances": np.asarray(conductances),
            "source_currents": np.asarray(source_currents),
            "drain_currents": np.asarray(drain_currents),
            "current_balance_errors": np.asarray(balance_errors),
            "threshold_voltage": percolation_voltage,
            "percolation_voltage": percolation_voltage,
        }

    def calculate_iv_curve_kirchhoff(
        self,
        V_start: float = 0.0,
        V_max: float = 16.0,
        V_step: float = 0.5,
    ) -> dict:
        result = self.iv_curve(V_start=V_start, V_max=V_max, V_step=V_step)
        n = len(result["voltages"])
        result.update({"num_paths": np.zeros(n), "path_details": [[] for _ in range(n)]})
        return result

    def edge_currents(self, V_applied: float) -> dict:
        activated = self.activated_nodes(V_applied)
        solution = self.solve_active_network(activated, V_applied)
        return {
            "total_current": solution["total_current_A"],
            "total_current_A": solution["total_current_A"],
            "source_current_A": solution["source_current_A"],
            "drain_current_A": solution["drain_current_A"],
            "current_balance_error_A": solution["current_balance_error_A"],
            "node_potentials": solution["node_potentials"],
            "edge_currents": solution["edge_currents"],
            "activated_nodes": activated,
            "conducting_edges": len(solution["edge_currents"]),
        }

    def network_summary(self) -> dict:
        graph = self.G
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        degrees = [degree for _, degree in graph.degree()]
        values = [
            graph.nodes[node].get("Va", graph.nodes[node]["Vth"])
            for node in graph.nodes()
        ]
        return {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "n_sources": len(self.source_nodes),
            "n_drains": len(self.drain_nodes),
            "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
            "density": (
                n_edges / (n_nodes * (n_nodes - 1) / 2)
                if n_nodes > 1 else 0.0
            ),
            "is_connected": nx.is_connected(graph) if n_nodes else False,
            "mu_a_actual": float(np.mean(values)) if values else float("nan"),
            "std_a_actual": float(np.std(values)) if values else float("nan"),
            "void_fraction_requested": self.requested_void_fraction,
            "void_fraction_achieved": self.achieved_void_fraction,
            "node_resistance_ohm": self.node_resistance_ohm,
        }

    def overall_resistance(self, V_applied: float = 1.0) -> float:
        solution = self.solve_active_network(self.activated_nodes(V_applied), V_applied)
        current = solution["total_current_A"]
        return (V_applied / current) if current > 0 else float("inf")

    def save_network(self, filename):
        import pickle
        with open(filename, "wb") as stream:
            pickle.dump(self, stream)

    @classmethod
    def load_network(cls, filename):
        import pickle
        with open(filename, "rb") as stream:
            return pickle.load(stream)

    def save(self, path: str) -> None:
        self.save_network(path)

    @classmethod
    def load(cls, path: str) -> "NanoparticleNetwork":
        return cls.load_network(path)
