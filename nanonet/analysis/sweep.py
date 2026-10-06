"""Per-voltage evolution analysis using nanonet's canonical circuit solver."""

from __future__ import annotations

import csv

import networkx as nx
import numpy as np


def count_edge_disjoint_pathways(net, activated_nodes) -> int:
    """Maximum number of edge-disjoint active source-to-drain paths."""
    active = set(activated_nodes)
    sources = [n for n in net.source_nodes if n in active]
    drains = [n for n in net.drain_nodes if n in active]
    if not sources or not drains:
        return 0

    graph = net.G.subgraph(active)
    if graph.number_of_edges() == 0:
        return 0

    super_source = ("__electrode__", "source")
    super_drain = ("__electrode__", "drain")
    flow_graph = nx.DiGraph()
    flow_graph.add_nodes_from(graph.nodes())
    for i, j in graph.edges():
        flow_graph.add_edge(i, j, capacity=1)
        flow_graph.add_edge(j, i, capacity=1)

    electrode_capacity = max(1, graph.number_of_edges() + 1)
    for node in sources:
        flow_graph.add_edge(super_source, node, capacity=electrode_capacity)
    for node in drains:
        flow_graph.add_edge(node, super_drain, capacity=electrode_capacity)

    value = nx.maximum_flow_value(
        flow_graph,
        super_source,
        super_drain,
        capacity="capacity",
        flow_func=nx.algorithms.flow.shortest_augmenting_path,
    )
    return int(round(float(value)))


def conductance_matrix(net, activated_nodes: set, R_MIN: float = 1.0):
    """Return the exact conductance-weighted active-circuit Laplacian."""
    del R_MIN
    system = net.build_active_laplacian(activated_nodes)
    if system is None:
        return np.zeros((0, 0)), []
    return system["laplacian"].toarray(), system["nodes"]


def _edge_currents_at(net, activated_nodes: set, V_applied: float):
    solution = net.solve_active_network(activated_nodes, V_applied)
    return (
        solution["total_current_A"],
        solution["node_potentials"],
        solution["edge_currents"],
    )


def sweep(
    net,
    V_start: float,
    V_max: float,
    V_step: float,
    *,
    current_frac: float = 0.01,
    G_voltages=None,
    G_out_prefix: str | None = None,
    effective_resistance: bool = True,
    algebraic_connectivity: bool = True,
) -> dict:
    """Run a voltage sweep and collect topology, current, and spectral metrics."""
    if not net.source_nodes or not net.drain_nodes:
        raise ValueError("Call identify_sources_drains() (or build()) first.")

    G_voltages_set = {round(float(v), 10) for v in (G_voltages or [])}
    rows = []
    edge_currents_by_V = {}
    percolation_V = None
    percolation_pathways = 0
    percolation_active_nodes = 0
    n_total_nodes = net.G.number_of_nodes()
    n_total_edges = net.G.number_of_edges()

    voltage = float(V_start)
    while voltage <= float(V_max) + 1e-10:
        Vr = round(voltage, 10)
        activated_nodes = net.activated_nodes(Vr)
        activated_edges = [
            (i, j) for i, j in net.G.edges()
            if i in activated_nodes and j in activated_nodes
        ]

        G_act = net.G.subgraph(activated_nodes)
        connected = bool(net._bridging_nodes(activated_nodes))
        if connected and percolation_V is None:
            percolation_V = Vr
            percolation_pathways = count_edge_disjoint_pathways(net, activated_nodes)
            percolation_active_nodes = len(activated_nodes)

        comp_sizes = sorted(
            (len(component) for component in nx.connected_components(G_act)),
            reverse=True,
        )
        num_components = len(comp_sizes)
        largest_cc_nodes = comp_sizes[0] if comp_sizes else 0
        second_cc_nodes = comp_sizes[1] if len(comp_sizes) > 1 else 0
        largest_cc_fraction = (
            largest_cc_nodes / len(activated_nodes) if activated_nodes else 0.0
        )
        mean_finite_cc = (
            float(np.mean(comp_sizes[1:])) if num_components > 1 else 0.0
        )

        solution = net.solve_active_network(activated_nodes, Vr)
        total_current = solution["total_current_A"]
        phi = solution["node_potentials"]
        edge_currents = solution["edge_currents"]
        edge_currents_by_V[Vr] = edge_currents

        if edge_currents:
            mags = np.asarray([abs(current) for current in edge_currents.values()])
            max_mag = float(mags.max())
            backbone_edges = int(np.sum(mags >= current_frac * max_mag))
            current_sum = float(mags.sum())
            denom = float(np.sum(mags * mags))
            participation = (
                (current_sum * current_sum) / denom
                if current_sum > 0 and denom > 0 else 0.0
            )
            mean_mag = float(mags.mean())
            max_to_mean = max_mag / mean_mag if mean_mag > 0 else 0.0
            cv_current = float(mags.std() / mean_mag) if mean_mag > 0 else 0.0
            sorted_mags = np.sort(mags)
            n_e = len(sorted_mags)
            total_mag = float(sorted_mags.sum())
            gini_current = (
                float(
                    (
                        2.0 * np.sum(np.arange(1, n_e + 1) * sorted_mags)
                        - (n_e + 1) * total_mag
                    )
                    / (n_e * total_mag)
                )
                if total_mag > 0 else 0.0
            )
            k = max(1, int(np.ceil(0.10 * n_e)))
            top10_fraction = (
                float(sorted_mags[-k:].sum() / total_mag)
                if total_mag > 0 else 0.0
            )
        else:
            backbone_edges = 0
            participation = 0.0
            max_to_mean = 0.0
            cv_current = 0.0
            gini_current = 0.0
            top10_fraction = 0.0

        if Vr in G_voltages_set and G_out_prefix and activated_nodes:
            Gmat, active_list = conductance_matrix(net, activated_nodes)
            np.save(f"{G_out_prefix}_V{Vr:g}.npy", Gmat)
            with open(
                f"{G_out_prefix}_V{Vr:g}_nodeindex.csv", "w", newline=""
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(["matrix_index", "node_id", "is_source", "is_drain"])
                for idx, node in enumerate(active_list):
                    writer.writerow([
                        idx, node,
                        int(node in net.source_nodes),
                        int(node in net.drain_nodes),
                    ])
            with open(f"{G_out_prefix}_V{Vr:g}.csv", "w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "row_index", "col_index", "row_node", "col_node",
                    "conductance_S",
                ])
                for i in range(Gmat.shape[0]):
                    for j in range(Gmat.shape[1]):
                        if Gmat[i, j] != 0.0:
                            writer.writerow([
                                i, j, active_list[i], active_list[j], Gmat[i, j]
                            ])

        conductance = total_current / Vr if Vr > 1e-12 else 0.0

        R_eff = alg_conn = gap_ratio = np.nan
        if activated_nodes and (effective_resistance or algebraic_connectivity):
            from nanonet.analysis.spectral import (
                effective_resistance as _effective_resistance,
                spectral_metrics as _spectral_metrics,
            )
            if effective_resistance:
                R_eff = _effective_resistance(net, activated_nodes, V_test=1.0)
            if algebraic_connectivity:
                alg_conn, gap_ratio, _ = _spectral_metrics(net, activated_nodes)

        rows.append({
            "V": Vr,
            "activated_nodes": len(activated_nodes),
            "activated_edges": len(activated_edges),
            "conducting_nodes": len(phi),
            "conducting_edges": len(edge_currents),
            "source_drain_connected": int(connected),
            "total_current_A": total_current,
            "total_current_chargeconserving_A": total_current,
            "source_current_A": solution["source_current_A"],
            "drain_current_A": solution["drain_current_A"],
            "current_balance_error_A": solution["current_balance_error_A"],
            "conductance_S": conductance,
            "backbone_edges": backbone_edges,
            "participation_ratio": participation,
            "current_participation_ratio": participation,
            "N_eff": participation,
            "num_components": num_components,
            "largest_cc_nodes": largest_cc_nodes,
            "second_cc_nodes": second_cc_nodes,
            "largest_cc_fraction": largest_cc_fraction,
            "mean_finite_cc": mean_finite_cc,
            "current_cv": cv_current,
            "current_gini": gini_current,
            "current_max_to_mean": max_to_mean,
            "current_top10_fraction": top10_fraction,
            "effective_resistance_ohm": R_eff,
            "algebraic_connectivity": alg_conn,
            "spectral_gap_ratio": gap_ratio,
        })
        voltage = round(voltage + float(V_step), 10)

    for row in rows:
        row["edge_disjoint_pathways_at_Vperc"] = percolation_pathways
        row["active_nodes_at_Vperc"] = percolation_active_nodes

    fieldnames = [
        "V", "activated_nodes", "activated_edges", "conducting_nodes",
        "conducting_edges", "source_drain_connected", "total_current_A",
        "total_current_chargeconserving_A", "source_current_A",
        "drain_current_A", "current_balance_error_A", "conductance_S",
        "backbone_edges", "participation_ratio",
        "current_participation_ratio", "N_eff",
        "num_components", "largest_cc_nodes", "second_cc_nodes",
        "largest_cc_fraction", "mean_finite_cc",
        "current_cv", "current_gini", "current_max_to_mean",
        "current_top10_fraction", "effective_resistance_ohm",
        "algebraic_connectivity", "spectral_gap_ratio",
        "edge_disjoint_pathways_at_Vperc", "active_nodes_at_Vperc",
    ]

    return {
        "rows": rows,
        "fieldnames": fieldnames,
        "edge_currents_by_V": edge_currents_by_V,
        "percolation_V": percolation_V,
        "percolation_pathways": percolation_pathways,
        "percolation_active_nodes": percolation_active_nodes,
        "n_total_nodes": n_total_nodes,
        "n_total_edges": n_total_edges,
    }


def write_csv(result: dict, path: str) -> str:
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result["fieldnames"])
        writer.writeheader()
        writer.writerows(result["rows"])
    return path


def write_iv_csv(result: dict, path: str) -> str:
    """Write I-V data with current-conservation diagnostics."""
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "V", "current_A", "source_current_A", "drain_current_A",
            "current_balance_error_A", "conductance_S",
        ])
        for row in result["rows"]:
            writer.writerow([
                row["V"], row["total_current_A"], row["source_current_A"],
                row["drain_current_A"], row["current_balance_error_A"],
                row["conductance_S"],
            ])
    return path


def write_edge_currents_csv(
    net,
    result: dict,
    path: str,
    conducting_only: bool = True,
    voltages=None,
) -> str:
    """Write signed per-edge currents with positions and Vperc diagnostics."""
    del conducting_only
    pos = net.positions
    edge_currents_by_V = result["edge_currents_by_V"]
    voltage_set = (
        {round(float(v), 10) for v in voltages}
        if voltages is not None else None
    )

    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "V", "edge_disjoint_pathways_at_Vperc", "active_nodes_at_Vperc",
            "node_i", "node_j", "x_i", "y_i", "x_j", "y_j",
            "current_A", "abs_current_A",
        ])
        for Vr in sorted(edge_currents_by_V):
            if voltage_set is not None and Vr not in voltage_set:
                continue
            for (i, j), current in edge_currents_by_V[Vr].items():
                writer.writerow([
                    Vr,
                    result.get("percolation_pathways", 0),
                    result.get("percolation_active_nodes", 0),
                    i, j,
                    f"{pos[i][0]:.6f}", f"{pos[i][1]:.6f}",
                    f"{pos[j][0]:.6f}", f"{pos[j][1]:.6f}",
                    f"{current:.8e}", f"{abs(current):.8e}",
                ])
    return path
