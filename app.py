"""
Nanoparticle Network Explorer — interactive Dash web app.

Run:
    conda activate network
    python app.py
Then open http://127.0.0.1:8050 in your browser.
"""

from __future__ import annotations

import traceback

import numpy as np
import networkx as nx
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output, State, callback, no_update

from nanonet.core.network import NanoparticleNetwork
from nanonet.sweeps.parameter_sweep import (
    _fit_power_law,
    _transition_voltage_from_evo,
)
from nanonet.analysis.sweep import sweep as _sweep

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

C = {
    "source":      "#2196F3",
    "drain":       "#F44336",
    "inactive":    "#CCCCCC",
    "active_low":  "#FFB300",
    "active_high": "#E64A19",
    "conduct":     "#43A047",
    "bg":          "#F8F9FA",
    "fit":         "#7B1FA2",
    "probe":       "#E53935",
    "sidebar_bg":  "#1A237E",
    "sidebar_txt": "#FFFFFF",
    "section_hdr": "#90CAF9",
    "btn_run":     "#43A047",
    "btn_build":   "#1565C0",
}


# ---------------------------------------------------------------------------
# Conductance-weighted spectral connectivity from the canonical circuit
# ---------------------------------------------------------------------------

def _fast_alg_conn(net: NanoparticleNetwork, activated_nodes: set) -> float:
    """Smallest positive eigenvalue of the same Laplacian used for transport."""
    system = net.build_active_laplacian(activated_nodes)
    if system is None:
        return float("nan")
    matrix = system["laplacian"].toarray()
    if matrix.shape[0] < 2:
        return float("nan")
    try:
        ev = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        emax = float(ev[-1]) if len(ev) else 0.0
        if emax <= 0:
            return float("nan")
        positive = ev[ev >= 1e-9 * emax]
        return float(positive[0]) if len(positive) else float("nan")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _build_network(params: dict) -> NanoparticleNetwork:
    net = NanoparticleNetwork(
        L=params["L"],
        N=params["N"],
        fv=params["fv"],
        mu_a=params["mu_a"],
        std_a=params["std_a"],
        edge_k=params["edge_k"],
        node_resistance_ohm=params["node_resistance_ohm"],
        connection_radius=params["connection_radius"],
        source_frac=params["source_frac"],
        drain_frac=params["drain_frac"],
        strict_N=False,
    )
    net.build(seed=params["seed"])
    return net


def _run_iv_and_sweep(net: NanoparticleNetwork, params: dict):
    """Run IV curve + activation sweep (no expensive spectral).
    Algebraic connectivity computed with fast NetworkX sparse solver."""
    iv = net.iv_curve(params["V_start"], params["V_max"], params["V_step"])

    # Lightweight sweep: no dense spectral, no effective resistance
    evo = _sweep(
        net,
        params["V_start"], params["V_max"], params["V_step"],
        effective_resistance=False,
        algebraic_connectivity=False,
    )

    # Attach fast λ₂ values directly
    for row in evo["rows"]:
        V = float(row["V"])
        activated = {n for n in net.G.nodes() if net.G.nodes[n]["Vth"] <= V}
        if activated:
            row["algebraic_connectivity"] = _fast_alg_conn(net, activated)
        else:
            row["algebraic_connectivity"] = float("nan")

    trans_V = _transition_voltage_from_evo(evo)
    fit = _fit_power_law(
        np.asarray(iv["voltages"], float),
        np.asarray(iv["currents"], float),
        V_T=iv["threshold_voltage"],
        v_transition=trans_V,
        fit_window=10.0,
    )
    return iv, evo, fit


def _net_to_store(net: NanoparticleNetwork) -> dict:
    """Serialise network into JSON-safe dict for dcc.Store."""
    pos_list = net.positions.tolist()
    node_data = {
        str(n): {
            "Vth":    float(net.G.nodes[n]["Vth"]),
            "R_node": float(net.G.nodes[n].get("R_node", 0.0)),
            "pos":    pos_list[n],
        }
        for n in net.G.nodes()
    }
    edge_data = [
        {
            "u": int(u), "v": int(v),
            "R_edge":    float(net.G[u][v]["R_edge"]),
            "distance":  float(net.G[u][v]["distance"]),
        }
        for u, v in net.G.edges()
    ]
    return {
        "node_data":    node_data,
        "edge_data":    edge_data,
        "source_nodes": net.source_nodes,
        "drain_nodes":  net.drain_nodes,
        "domain":       list(net.domain),
        "n_nodes":      net.G.number_of_nodes(),
        "n_edges":      net.G.number_of_edges(),
    }


def _restore_net(net_store: dict, params: dict) -> NanoparticleNetwork:
    """Rebuild a lightweight NanoparticleNetwork from stored JSON."""
    node_data   = net_store["node_data"]
    edge_data   = net_store["edge_data"]
    source_nodes = net_store["source_nodes"]
    drain_nodes  = net_store["drain_nodes"]
    domain       = net_store["domain"]

    G = nx.Graph()
    # Sort by node index so pos_arr[n] is correct
    for n_str, nd in sorted(node_data.items(), key=lambda x: int(x[0])):
        n = int(n_str)
        G.add_node(n, Vth=nd["Vth"], R_node=nd["R_node"])
    pos_arr = np.array([node_data[str(n)]["pos"] for n in sorted(G.nodes())])

    for ed in edge_data:
        G.add_edge(ed["u"], ed["v"],
                   R_edge=ed["R_edge"], distance=ed["distance"],
                   R_edge_val=ed["R_edge"])

    net = NanoparticleNetwork.__new__(NanoparticleNetwork)
    net.G            = G
    net.positions    = pos_arr
    net.source_nodes = source_nodes
    net.drain_nodes  = drain_nodes
    net.domain       = tuple(domain)
    net.edge_k       = params["edge_k"]
    net.node_resistance_ohm = params["node_resistance_ohm"]
    net.node_r_scale = net.node_resistance_ohm
    net.r_floor      = 0.0
    net.voids        = []
    net.requested_void_fraction = float(params.get("fv", 0.0))
    net.achieved_void_fraction = float(params.get("fv", 0.0))
    net.source_node = source_nodes[0] if source_nodes else None
    net.drain_node = drain_nodes[0] if drain_nodes else None
    net.n_junctions = G.number_of_nodes()
    net.N = net.n_junctions
    return net


# ---------------------------------------------------------------------------
# Figure: network topology / current map
# ---------------------------------------------------------------------------

def _network_figure(net: NanoparticleNetwork, V_probe: float,
                    show_currents: bool = True) -> go.Figure:
    G       = net.G
    pos     = net.positions
    src_set = set(net.source_nodes)
    drn_set = set(net.drain_nodes)

    activated = {n for n in G.nodes() if G.nodes[n]["Vth"] <= V_probe}
    ec: dict = {}
    total_I = 0.0
    if show_currents:
        result  = net.edge_currents(V_probe)
        ec      = result["edge_currents"]
        total_I = result["total_current"]

    fig = go.Figure()

    # --- Inactive edges (grey, thin) ---
    xi, yi = [], []
    for u, v in G.edges():
        if (u, v) not in ec and (v, u) not in ec:
            xu, yu = pos[u]; xv, yv = pos[v]
            xi += [xu, xv, None]; yi += [yu, yv, None]
    if xi:
        fig.add_trace(go.Scatter(
            x=xi, y=yi, mode="lines",
            line=dict(color="#DDDDDD", width=0.6),
            hoverinfo="none", showlegend=False,
        ))

    # --- Conducting edges (heat-mapped by |I|) ---
    if ec:
        mags  = np.array([abs(c) for c in ec.values()])
        I_max = mags.max() if len(mags) else 1.0
        for (u, v), I in ec.items():
            xu, yu = pos[u]; xv, yv = pos[v]
            t = abs(I) / I_max if I_max > 0 else 0.0
            r  = int(255 * t)
            g_ = int(180 * (1 - t))
            fig.add_trace(go.Scatter(
                x=[xu, xv, None], y=[yu, yv, None],
                mode="lines",
                line=dict(color=f"rgb({r},{g_},30)", width=1.2 + 3.0 * t),
                hoverinfo="none", showlegend=False,
            ))

    # --- Source/drain electrode strips (shaded rectangles) ---
    W, H = net.domain
    src_x = params_for_rect(net, "source")
    drn_x = params_for_rect(net, "drain")
    for x0, x1, clr, name in [
        (0,      src_x, "rgba(33,150,243,0.12)", "Source"),
        (drn_x, W,      "rgba(244,67,54,0.12)",  "Drain"),
    ]:
        fig.add_shape(type="rect",
                      x0=x0, x1=x1, y0=0, y1=H,
                      fillcolor=clr, line=dict(width=0))

    # --- Nodes ---
    nx_list, ny_list, nc_list, ns_list, nt_list = [], [], [], [], []
    for n in G.nodes():
        x, y  = pos[n]
        vth   = G.nodes[n]["Vth"]
        is_act = n in activated
        if n in src_set:
            color, size = C["source"], 9
            label = f"Source {n}<br>Vth={vth:.2f} V"
        elif n in drn_set:
            color, size = C["drain"], 9
            label = f"Drain {n}<br>Vth={vth:.2f} V"
        elif is_act:
            frac  = vth / max(V_probe, 1e-6)
            color = C["active_high"] if frac < 0.5 else C["active_low"]
            size  = 6
            label = f"Node {n}<br>Vth={vth:.2f} V ✓"
        else:
            color, size = C["inactive"], 4
            label = f"Node {n}<br>Vth={vth:.2f} V"
        nx_list.append(x); ny_list.append(y)
        nc_list.append(color); ns_list.append(size); nt_list.append(label)

    fig.add_trace(go.Scatter(
        x=nx_list, y=ny_list, mode="markers",
        marker=dict(color=nc_list, size=ns_list,
                    line=dict(width=0.4, color="#666")),
        text=nt_list, hoverinfo="text", showlegend=False,
    ))

    n_act   = len(activated)
    n_total = G.number_of_nodes()
    I_label = f"I = {total_I:.3e} A" if show_currents else "topology only"
    fig.update_layout(
        title=dict(
            text=f"V = {V_probe:.2f} V  |  {n_act}/{n_total} active  |  {I_label}",
            font=dict(size=13),
        ),
        xaxis=dict(title="x [norm.]", scaleanchor="y", showgrid=False,
                   zeroline=False, range=[-0.03, W + 0.03]),
        yaxis=dict(title="y [norm.]", showgrid=False, zeroline=False,
                   range=[-0.03, H + 0.03]),
        plot_bgcolor=C["bg"], paper_bgcolor="white",
        margin=dict(l=40, r=10, t=50, b=40),
        hovermode="closest",
    )
    return fig


def params_for_rect(net, side: str) -> float:
    """Return the electrode boundary x-coordinate."""
    W = net.domain[0]
    if side == "source":
        # source_frac stored on original net; approximate from source_nodes positions
        if net.source_nodes:
            return float(net.positions[net.source_nodes].max(axis=0)[0]) + 0.001
        return 0.15 * W
    else:
        if net.drain_nodes:
            return float(net.positions[net.drain_nodes].min(axis=0)[0]) - 0.001
        return 0.85 * W


# ---------------------------------------------------------------------------
# Figure: I-V curve + fit
# ---------------------------------------------------------------------------

def _iv_figure(iv: dict, fit: dict, evo: dict, V_probe: float) -> go.Figure:
    V = np.array(iv["voltages"])
    I = np.array(iv["currents"])

    # Phase boundaries
    V_onset   = iv["threshold_voltage"]          # Phase I → II (current first nonzero)
    V_trans   = _transition_voltage_from_evo(evo) # Phase II → III (90% nodes activated)
    V_min_ax  = float(V[0])
    V_max_ax  = float(V[-1])

    fig = go.Figure()

    # --- Phase shading (drawn first so data sits on top) ---
    x0_ii = float(V_onset) if V_onset is not None else V_min_ax
    x1_ii = float(V_trans) if (V_trans is not None and np.isfinite(V_trans)) else V_max_ax

    # Phase I: zero-current region
    if V_onset is not None:
        fig.add_vrect(x0=V_min_ax, x1=float(V_onset),
                      fillcolor="rgba(180,180,180,0.15)", line_width=0, layer="below")
    # Phase II: fit region
    fig.add_vrect(x0=x0_ii, x1=x1_ii,
                  fillcolor="rgba(100,180,100,0.12)", line_width=0, layer="below")
    # Phase III: quasi-linear region
    if V_trans is not None and np.isfinite(V_trans) and float(V_trans) < V_max_ax:
        fig.add_vrect(x0=float(V_trans), x1=V_max_ax,
                      fillcolor="rgba(255,160,50,0.10)", line_width=0, layer="below")

    # --- Raw I-V data ---
    fig.add_trace(go.Scatter(
        x=V, y=I, mode="lines+markers", name="I-V data",
        line=dict(color="#1565C0", width=2), marker=dict(size=5),
    ))

    # --- Power-law fit overlay (Phase II only) ---
    if fit.get("success"):
        V_T, zeta, A = fit["V_T"], fit["zeta"], fit["A"]
        perc_V  = V_onset or 0.0
        v_upper = x1_ii
        v_fit   = np.linspace(max(perc_V, V_T + 0.01), v_upper, 300)
        i_fit   = A * np.maximum(v_fit - V_T, 0) ** zeta
        fig.add_trace(go.Scatter(
            x=v_fit, y=i_fit, mode="lines",
            name=f"Fit: A(V−V_T)^ζ  R²={fit['R2']:.3f}",
            line=dict(color=C["fit"], width=2.5, dash="dash"),
        ))
        # V_T vertical line
        fig.add_vline(x=V_T, line=dict(color=C["fit"], dash="dot", width=1.2))

    # Phase boundary verticals (no annotation_text — labels go into phase_annotations)
    if V_onset is not None:
        fig.add_vline(x=float(V_onset),
                      line=dict(color="#888", dash="dot", width=1))
    if V_trans is not None and np.isfinite(V_trans):
        fig.add_vline(x=float(V_trans),
                      line=dict(color="#E65100", dash="dot", width=1))
    fig.add_vline(x=V_probe,
                  line=dict(color=C["probe"], dash="dash", width=1.5))

    # All annotations — fixed paper y-coords so nothing overlaps
    phase_annotations = []

    # Phase region labels (top strip)
    if V_onset is not None:
        phase_annotations.append(dict(
            x=(V_min_ax + float(V_onset)) / 2, y=0.97,
            xref="x", yref="paper", text="Phase I",
            showarrow=False, font=dict(size=10, color="#888"),
            bgcolor="rgba(255,255,255,0.6)",
        ))
    phase_annotations.append(dict(
        x=(x0_ii + x1_ii) / 2, y=0.97,
        xref="x", yref="paper", text="Phase II (fit)",
        showarrow=False, font=dict(size=10, color="#2E7D32"),
        bgcolor="rgba(255,255,255,0.6)",
    ))
    if V_trans is not None and np.isfinite(V_trans) and float(V_trans) < V_max_ax:
        phase_annotations.append(dict(
            x=(float(V_trans) + V_max_ax) / 2, y=0.97,
            xref="x", yref="paper", text="Phase III",
            showarrow=False, font=dict(size=10, color="#E65100"),
            bgcolor="rgba(255,255,255,0.6)",
        ))

    # Vertical line labels (bottom strip, below phase labels)
    if V_onset is not None:
        phase_annotations.append(dict(
            x=float(V_onset), y=0.88,
            xref="x", yref="paper", text=f"onset<br>{V_onset:.2f} V",
            showarrow=False, font=dict(size=9, color="#888"),
            bgcolor="rgba(255,255,255,0.7)", xanchor="left",
        ))
    if V_trans is not None and np.isfinite(V_trans):
        phase_annotations.append(dict(
            x=float(V_trans), y=0.88,
            xref="x", yref="paper", text=f"90%<br>{V_trans:.2f} V",
            showarrow=False, font=dict(size=9, color="#E65100"),
            bgcolor="rgba(255,255,255,0.7)", xanchor="left",
        ))
    if fit.get("success"):
        phase_annotations.append(dict(
            x=fit["V_T"], y=0.78,
            xref="x", yref="paper", text=f"V_T<br>{fit['V_T']:.2f} V",
            showarrow=False, font=dict(size=9, color=C["fit"]),
            bgcolor="rgba(255,255,255,0.7)", xanchor="left",
        ))
    phase_annotations.append(dict(
        x=V_probe, y=0.88,
        xref="x", yref="paper", text=f"probe<br>{V_probe:.2f} V",
        showarrow=False, font=dict(size=9, color=C["probe"]),
        bgcolor="rgba(255,255,255,0.7)", xanchor="right",
    ))

    fig.update_layout(
        title=dict(text="I–V Curve", font=dict(size=13)),
        xaxis=dict(title="Applied Voltage [V]"),
        yaxis=dict(title="Current [A]", exponentformat="e"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        annotations=phase_annotations,
        plot_bgcolor=C["bg"], paper_bgcolor="white",
        margin=dict(l=60, r=15, t=55, b=45),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure: algebraic connectivity
# ---------------------------------------------------------------------------

def _alg_figure(evo: dict, V_probe: float) -> go.Figure:
    rows  = evo.get("rows", [])
    V_arr = np.array([r["V"] for r in rows], float)
    ac    = np.array([r.get("algebraic_connectivity", np.nan) for r in rows], float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=V_arr, y=ac, mode="lines+markers",
        name="λ₂",
        line=dict(color=C["conduct"], width=2),
        marker=dict(size=5),
    ))
    fig.add_vline(x=V_probe,
                  line=dict(color=C["probe"], dash="dash", width=1.5),
                  annotation_text="Probe",
                  annotation_position="top left",
                  annotation_font_size=11)

    fig.update_layout(
        title=dict(text="Conductance-weighted λ₂ vs Voltage", font=dict(size=13)),
        xaxis=dict(title="Applied Voltage [V]"),
        yaxis=dict(title="λ₂ [S]", exponentformat="e"),
        plot_bgcolor=C["bg"], paper_bgcolor="white",
        margin=dict(l=60, r=15, t=50, b=45),
    )
    return fig


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

_SB = dict(color=C["sidebar_txt"])          # sidebar base text
_LBL = dict(color=C["section_hdr"], fontSize="11px",
            fontWeight="600", marginBottom="3px", display="block")
_HDR = dict(color=C["section_hdr"], fontSize="12px", fontWeight="700",
            borderBottom=f"1px solid {C['section_hdr']}",
            paddingBottom="3px", marginBottom="10px", marginTop="14px")
_SPIN_ROW = dict(display="flex", alignItems="center", gap="4px",
                 marginBottom="8px")
_SPIN_NUM = dict(
    width="80px", textAlign="center", border="1px solid #444",
    backgroundColor="#1E3A8A", color="white",
    borderRadius="4px", padding="4px 2px", fontSize="13px",
)
_SPIN_BTN = dict(
    width="26px", height="26px", lineHeight="24px", textAlign="center",
    backgroundColor="#2D4EAA", color="white",
    border="1px solid #4A6BD6", borderRadius="4px",
    cursor="pointer", fontSize="15px", fontWeight="700",
    padding="0",
)


def _spinner(label, base_id, value, step, min_val=None, max_val=None, fmt=None):
    """Label + [−] [value display] [+] row."""
    fmt_str = fmt or ("{:.2f}" if isinstance(step, float) and step < 1 else "{:g}")
    return html.Div([
        html.Span(label, style=_LBL),
        html.Div([
            html.Button("−", id=f"{base_id}-dec", n_clicks=0, style=_SPIN_BTN),
            dcc.Input(
                id=f"{base_id}-val",
                type="number", value=value, step=step,
                min=min_val, max=max_val,
                debounce=True,
                style=_SPIN_NUM,
            ),
            html.Button("+", id=f"{base_id}-inc", n_clicks=0, style=_SPIN_BTN),
        ], style=_SPIN_ROW),
    ])


# Parameter specs: (label, base_id, default, step, min, max, fmt)
PARAMS = [
    # Network geometry
    ("Domain size L [-]",       "L",   1.0,   0.1,  0.1,  10.0,  "{:.1f}"),
    ("Junctions N",              "N",   300,   50,   10,   2000,  "{:g}"),
    ("Void fraction fv [-]",    "fv",  0.0,   0.01, 0.0,  0.5,   "{:.2f}"),
    ("Connection radius [-]",   "cr",  0.15,  0.01, 0.01, 0.5,   "{:.2f}"),
    ("Source fraction [-]",     "sf",  0.15,  0.01, 0.01, 0.49,  "{:.2f}"),
    ("Drain fraction [-]",      "df",  0.15,  0.01, 0.01, 0.49,  "{:.2f}"),
    # Activation voltage
    ("Mean μₐ [V]",             "mu",  6.0,   0.5,  0.0,  30.0,  "{:.1f}"),
    ("Std σₐ [V]",              "std", 3.0,   0.5,  0.01, 20.0,  "{:.1f}"),
    # Resistance
    ("Edge k [Ω/m]",            "ek",  2e10,  1e9,  1e6,  1e13,  "{:.2e}"),
    ("Junction resistance [Ω]", "nrs", 3.5e9, 1e8,  0.0,  1e12,  "{:.2e}"),
    # Voltage sweep
    ("V start [V]",             "vs",  0.0,   0.5,  0.0,  50.0,  "{:.1f}"),
    ("V max [V]",               "vm",  16.0,  1.0,  1.0,  100.0, "{:.1f}"),
    ("V step [V]",              "vstep",0.5,  0.1,  0.05, 5.0,   "{:.2f}"),
    # Seed
    ("Random seed",             "seed",42,    1,    0,    9999,  "{:g}"),
]

SECTION_BREAKS = {
    "L":    "Network Geometry",
    "mu":   "Activation Voltage",
    "ek":   "Resistance Model",
    "vs":   "Voltage Sweep",
    "seed": "Reproducibility",
}

sidebar_children = [
    html.Div("Nanoparticle Network Explorer",
             style=dict(fontSize="16px", fontWeight="800",
                        color="white", marginBottom="16px",
                        borderBottom="1px solid #3F51B5",
                        paddingBottom="10px")),
]

for label, bid, val, step, mn, mx, fmt in PARAMS:
    if bid in SECTION_BREAKS:
        sidebar_children.append(
            html.Div(SECTION_BREAKS[bid], style=_HDR)
        )
    sidebar_children.append(_spinner(label, bid, val, step, mn, mx, fmt))

sidebar_children += [
    html.Button("Preview Network", id="btn-build",
                style=dict(width="100%", padding="8px",
                           backgroundColor=C["btn_build"], color="white",
                           border="none", borderRadius="5px",
                           fontSize="13px", fontWeight="700",
                           cursor="pointer", marginTop="14px")),
    html.Div("(topology only — no sweep)",
             style=dict(textAlign="center", fontSize="10px",
                        color="#7986CB", marginTop="2px")),
    html.Button("Build & Run I-V Sweep", id="btn-run",
                style=dict(width="100%", padding="9px",
                           backgroundColor=C["btn_run"], color="white",
                           border="none", borderRadius="5px",
                           fontSize="13px", fontWeight="700",
                           cursor="pointer", marginTop="8px")),
    html.Div(id="status-box",
             style=dict(marginTop="10px", fontSize="11px",
                        color="#B3C5EF", minHeight="18px",
                        whiteSpace="pre-wrap", lineHeight="1.5")),
]

sidebar = html.Div(sidebar_children,
                   style=dict(width="240px", minWidth="240px",
                              padding="16px 14px",
                              backgroundColor=C["sidebar_bg"],
                              overflowY="auto", height="100vh",
                              boxSizing="border-box"))


main_area = html.Div([
    # Probe voltage slider
    html.Div([
        html.Span("Probe voltage [V]:",
                  style=dict(fontWeight="600", fontSize="13px",
                             color="#333", marginRight="12px", whiteSpace="nowrap")),
        html.Div(
            dcc.Slider(id="slider-V", min=0.0, max=16.0, step=0.5, value=8.0,
                       marks=None,
                       tooltip={"placement": "bottom", "always_visible": True}),
            style=dict(flex="1"),
        ),
    ], style=dict(display="flex", alignItems="center",
                  padding="10px 20px",
                  background="#ECEFF1", borderBottom="1px solid #CFD8DC")),

    # Fit summary
    html.Div(id="fit-summary",
             style=dict(padding="7px 20px", background="#E8F5E9",
                        fontSize="13px", borderBottom="1px solid #C8E6C9",
                        minHeight="30px", color="#1B5E20")),

    # Top row: network + IV
    html.Div([
        html.Div([dcc.Graph(id="fig-network", style=dict(height="430px"),
                            config=dict(displayModeBar=True,
                                        modeBarButtonsToRemove=["select2d","lasso2d"]))],
                 style=dict(flex="1", minWidth="0")),
        html.Div([dcc.Graph(id="fig-iv", style=dict(height="430px"),
                            config=dict(displayModeBar=False))],
                 style=dict(flex="1", minWidth="0")),
    ], style=dict(display="flex", gap="8px", padding="10px 16px 4px")),

    # Bottom row: alg connectivity + stats
    html.Div([
        html.Div([dcc.Graph(id="fig-alg", style=dict(height="330px"),
                            config=dict(displayModeBar=False))],
                 style=dict(flex="1", minWidth="0")),
        html.Div(id="network-stats",
                 style=dict(flex="1", minWidth="0",
                            padding="10px 20px", fontSize="13px", color="#333")),
    ], style=dict(display="flex", gap="8px", padding="4px 16px 12px")),

], style=dict(flex="1", overflowY="auto", height="100vh"))


app = dash.Dash(
    __name__,
    title="Nanoparticle Network Explorer",
    suppress_callback_exceptions=True,
)

app.layout = html.Div([
    sidebar,
    main_area,
    dcc.Store(id="store-net"),      # serialised network (topology)
    dcc.Store(id="store-results"),  # IV + sweep results
], style=dict(display="flex",
              fontFamily="'Inter','Segoe UI','Helvetica Neue',Arial,sans-serif"))


# ---------------------------------------------------------------------------
# Spinner callbacks (+ / − buttons update the dcc.Input value)
# ---------------------------------------------------------------------------

def _make_spinner_cb(base_id, step, min_val, max_val):
    @callback(
        Output(f"{base_id}-val", "value"),
        Input(f"{base_id}-inc", "n_clicks"),
        Input(f"{base_id}-dec", "n_clicks"),
        State(f"{base_id}-val", "value"),
        prevent_initial_call=True,
    )
    def _cb(n_inc, n_dec, cur):
        from dash import ctx
        val = float(cur if cur is not None else 0)
        if ctx.triggered_id == f"{base_id}-inc":
            val += step
        else:
            val -= step
        if min_val is not None:
            val = max(min_val, val)
        if max_val is not None:
            val = min(max_val, val)
        # round to avoid float noise
        decimals = max(0, -int(np.floor(np.log10(abs(step)))) + 1) if step > 0 else 2
        return round(val, decimals)
    return _cb


# Register one callback per spinner
for _, bid, _, step, mn, mx, _ in PARAMS:
    _make_spinner_cb(bid, step, mn, mx)


# ---------------------------------------------------------------------------
# "Build Network" callback — fast, shows topology only
# ---------------------------------------------------------------------------

def _collect_params(L, N, fv, cr, sf, df, mu, std, ek, nrs,
                    vs, vm, vstep, seed):
    return dict(
        L=float(L or 1.0), N=int(N or 300),
        fv=float(fv or 0.0),
        connection_radius=float(cr or 0.15),
        source_frac=float(sf or 0.15),
        drain_frac=float(df or 0.15),
        mu_a=float(mu or 6.0), std_a=float(std or 3.0),
        edge_k=float(ek or 2e10), node_resistance_ohm=float(nrs or 3.5e9),
        V_start=float(vs or 0.0), V_max=float(vm or 16.0),
        V_step=float(vstep or 0.5), seed=int(seed or 42),
    )


_PARAM_STATES = [
    State("L-val", "value"), State("N-val", "value"),
    State("fv-val", "value"), State("cr-val", "value"),
    State("sf-val", "value"), State("df-val", "value"),
    State("mu-val", "value"), State("std-val", "value"),
    State("ek-val", "value"), State("nrs-val", "value"),
    State("vs-val", "value"), State("vm-val", "value"),
    State("vstep-val", "value"), State("seed-val", "value"),
]


@callback(
    Output("store-net", "data"),
    Output("status-box", "children"),
    Input("btn-build", "n_clicks"),
    *_PARAM_STATES,
    prevent_initial_call=True,
)
def build_network(n_clicks, L, N, fv, cr, sf, df, mu, std, ek, nrs,
                  vs, vm, vstep, seed):
    try:
        params = _collect_params(L, N, fv, cr, sf, df, mu, std, ek, nrs,
                                 vs, vm, vstep, seed)
        net = _build_network(params)
        stored = _net_to_store(net)
        stored["params"] = params
        msg = (f"✓ Network built\n"
               f"  {stored['n_nodes']} nodes, {stored['n_edges']} edges\n"
               f"  {len(net.source_nodes)} sources, {len(net.drain_nodes)} drains")
        return stored, msg
    except Exception:
        return no_update, "Build error:\n" + traceback.format_exc(limit=4)


# ---------------------------------------------------------------------------
# "Run I-V Sweep" callback
# ---------------------------------------------------------------------------

@callback(
    Output("store-results", "data"),
    Output("status-box", "children", allow_duplicate=True),
    Output("slider-V", "min"),
    Output("slider-V", "max"),
    Output("slider-V", "step"),
    Output("slider-V", "value"),
    Output("slider-V", "marks"),
    Input("btn-run", "n_clicks"),
    State("store-net", "data"),
    *_PARAM_STATES,
    prevent_initial_call=True,
)
def run_sweep(n_clicks, net_store, L, N, fv, cr, sf, df, mu, std, ek, nrs,
              vs, vm, vstep, seed):
    try:
        params = _collect_params(L, N, fv, cr, sf, df, mu, std, ek, nrs,
                                 vs, vm, vstep, seed)

        # Always build network fresh (or reuse if params unchanged)
        if net_store is None or net_store.get("params") != params:
            net = _build_network(params)
            net_store = _net_to_store(net)
            net_store["params"] = params
            rebuilt = True
        else:
            net = _restore_net(net_store, params)
            rebuilt = False

        iv, evo, fit = _run_iv_and_sweep(net, params)
        V_trans = _transition_voltage_from_evo(evo)

        results = {
            "params":             params,
            "voltages":           iv["voltages"].tolist(),
            "currents":           iv["currents"].tolist(),
            "conductances":       iv["conductances"].tolist(),
            "threshold_voltage":  iv["threshold_voltage"],   # Phase I→II onset
            "transition_voltage": V_trans,                   # Phase II→III (90% nodes)
            "fit":                fit,
            "evo_rows":           evo["rows"],
            "n_total_nodes":      evo.get("n_total_nodes", net_store["n_nodes"]),
        }

        fit_str = (f"V_T={fit['V_T']:.2f} V, ζ={fit['zeta']:.2f}"
                   if fit.get("success") else fit.get("reason", "failed"))
        build_str = "built new network" if rebuilt else "reused existing network"
        msg = (f"✓ Sweep done ({build_str})\n"
               f"  {net_store['n_nodes']} nodes, {net_store['n_edges']} edges\n"
               f"  Fit: {fit_str}")

        V_max_f = params["V_max"]
        V_step_f = params["V_step"]
        tick_interval = max(2.0, round(V_max_f / 8 / V_step_f) * V_step_f)
        marks = {round(v, 6): str(int(v)) if v == int(v) else f"{v:.1f}"
                 for v in np.arange(0, V_max_f + V_step_f, tick_interval)}

        return (results, msg,
                params["V_start"], V_max_f, V_step_f,
                min(8.0, V_max_f), marks)

    except Exception:
        return (no_update, "Sweep error:\n" + traceback.format_exc(limit=4),
                0, 16, 0.5, 8.0, {})


# ---------------------------------------------------------------------------
# Plot update callback — fires on slider move OR new results
# ---------------------------------------------------------------------------

def _empty_fig(msg="Run simulation first"):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor=C["bg"],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=10, b=10),
        annotations=[dict(text=msg, xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=15, color="#AAAAAA"))],
    )
    return fig


@callback(
    Output("fig-network", "figure"),
    Output("fig-iv", "figure"),
    Output("fig-alg", "figure"),
    Output("fit-summary", "children"),
    Output("network-stats", "children"),
    Input("slider-V", "value"),
    Input("store-net", "data"),
    Input("store-results", "data"),
    prevent_initial_call=True,
)
def update_plots(V_probe, net_store, results):
    # If no network at all, show empty panels
    if net_store is None:
        emp = _empty_fig()
        return emp, emp, emp, "Build a network first.", ""

    params = net_store.get("params", {})
    V_probe = float(V_probe or 0.0)

    try:
        net = _restore_net(net_store, params)

        # ---- Network figure ----
        has_results = results is not None
        fig_net = _network_figure(net, V_probe, show_currents=has_results)

        # ---- IV + alg conn ----
        if not has_results:
            fig_iv  = _empty_fig("Run I-V Sweep to see results")
            fig_alg = _empty_fig("Run I-V Sweep to see results")
            summary = "Run I-V Sweep to see fit results."
            stats   = _stats_panel(net_store, None, V_probe)
            return fig_net, fig_iv, fig_alg, summary, stats

        iv = {
            "voltages":          np.array(results["voltages"]),
            "currents":          np.array(results["currents"]),
            "conductances":      np.array(results["conductances"]),
            "threshold_voltage": results["threshold_voltage"],
        }
        fit = results["fit"]
        evo = {"rows": results["evo_rows"],
               "n_total_nodes": results["n_total_nodes"]}

        fig_iv  = _iv_figure(iv, fit, evo, V_probe)
        fig_alg = _alg_figure(evo, V_probe)

        # Fit summary strip
        V_onset = results["threshold_voltage"]
        V_trans = results.get("transition_voltage")
        onset_str = f"{V_onset:.2f} V" if V_onset is not None else "—"
        trans_str  = (f"{V_trans:.2f} V"
                      if V_trans is not None and np.isfinite(V_trans) else "—")
        if fit.get("success"):
            summary = (
                f"Phase I→II onset: {onset_str}   |   "
                f"Phase II→III (90% nodes): {trans_str}   ‖   "
                f"Fit window: [{onset_str}, {trans_str}]   |   "
                f"V_T = {fit['V_T']:.3f} V   |   "
                f"ζ = {fit['zeta']:.3f}   |   "
                f"A = {fit['A']:.3e}   |   "
                f"R² = {fit['R2']:.4f}"
            )
        else:
            summary = (f"Phase I→II onset: {onset_str}   |   "
                       f"Phase II→III (90% nodes): {trans_str}   |   "
                       f"Fit not available: {fit.get('reason', 'unknown')}")

        stats = _stats_panel(net_store, results, V_probe)
        return fig_net, fig_iv, fig_alg, summary, stats

    except Exception:
        tb = traceback.format_exc(limit=4)
        err = _empty_fig(tb[:200])
        return err, err, err, f"Error: {tb[:120]}", ""


def _stats_panel(net_store: dict, results, V_probe: float):
    n_nodes = net_store.get("n_nodes", "—")
    n_edges = net_store.get("n_edges", "—")
    n_src   = len(net_store.get("source_nodes", []))
    n_drn   = len(net_store.get("drain_nodes",  []))

    # Active nodes at probe
    node_data = net_store.get("node_data", {})
    n_act = sum(1 for nd in node_data.values() if nd["Vth"] <= V_probe)
    pct   = f"{100*n_act/n_nodes:.1f}%" if isinstance(n_nodes, int) and n_nodes else "—"

    # From sweep results
    V_T_iv   = results["threshold_voltage"] if results else None
    V_trans  = results.get("transition_voltage") if results else None
    fit       = results["fit"] if results else {}
    alg_conn  = float("nan")

    if results:
        for row in results["evo_rows"]:
            if abs(float(row["V"]) - float(V_probe)) < 1e-9:
                alg_conn = row.get("algebraic_connectivity", float("nan"))
                break

    def row(label, val):
        return html.Tr([
            html.Td(label, style=dict(paddingRight="18px", paddingBottom="5px",
                                      color="#555")),
            html.Td(val,   style=dict(fontWeight="600", paddingBottom="5px")),
        ])

    return html.Div([
        html.Div("Network Statistics",
                 style=dict(fontWeight="700", fontSize="14px",
                            color="#1B5E20", borderBottom="2px solid #A5D6A7",
                            paddingBottom="4px", marginBottom="10px")),
        html.Table([
            row("Total nodes",             str(n_nodes)),
            row("Total edges",             str(n_edges)),
            row("Sources / Drains",        f"{n_src} / {n_drn}"),
            row("Active nodes @ probe",    f"{n_act} / {n_nodes}  ({pct})"),
            row("Phase I→II onset",        f"{V_T_iv:.2f} V" if V_T_iv else "—"),
            row("Phase II→III (90% nodes)",
                f"{V_trans:.2f} V"
                if V_trans is not None and np.isfinite(V_trans) else "—"),
            row("λ₂ @ probe",
                f"{alg_conn:.4e}" if (results and np.isfinite(alg_conn)) else "—"),
            row("Fit V_T",
                f"{fit['V_T']:.3f} V" if fit.get("success") else "—"),
            row("Fit ζ",
                f"{fit['zeta']:.3f}" if fit.get("success") else "—"),
            row("Fit R²",
                f"{fit['R2']:.4f}" if fit.get("success") else "—"),
        ], style=dict(borderCollapse="collapse", fontSize="13px")),
    ], style=dict(padding="12px 20px"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting at http://127.0.0.1:8050")
    app.run(debug=False, host="127.0.0.1", port=8050)
