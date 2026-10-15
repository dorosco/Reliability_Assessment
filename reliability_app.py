"""
Distribution Network Reliability - Taipy web app
-------------------------------------------------
Inputs  : nodes.csv  -> node, clients, MW
          edges.csv  -> u, v, length, lambda, Tao_RP, Tao_SW   (read by POSITION, headers are ignored)
Model   : munoz_delgado(G) -> (df_res, SAIFI, SAIDI)
          df_res columns: Node, N_RP, N_SW, D_RP, D_SW
          G node attrs : clients, MW
          G edge attrs : length, lam, tao_rp, tao_sw
Source  : node 1 (substation)

Run with:  python reliability_app.py
Requires:  pip install taipy plotly networkx pandas pulp
"""
import json
import sys

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from taipy.gui import Gui, notify, get_state_id, invoke_callback, invoke_long_callback

sys.setrecursionlimit(10_000)          # recursive layout on long feeders

# =============================================================================
# 0. Configuration
# =============================================================================
SOURCE = 1
NODES_CSV = "A4243_ELC_load_data.csv"
EDGES_CSV = "A4243_ELC_system_data.csv"

NODE_COLS = ["node", "clients", "MW"]
EDGE_COLS = ["u", "v", "length", "lambda", "Tao_RP", "Tao_SW"]
EDGE_PARAMS = ["length", "lambda", "Tao_RP", "Tao_SW"]
RES_COLS = ["Node", "N_RP", "N_SW", "D_RP", "D_SW"]
CMP_COLS = ["Node", "N_base", "N_cur", "dN", "D_base", "D_cur", "dD"]

ORANGE, RED = "#ff7f0e", "#d62728"

INDICATORS = {
    "N_RP":    ("Interruptions / yr (repair)",    "int/yr"),
    "N_SW":    ("Interruptions / yr (switching)", "int/yr"),
    "N_total": ("Interruptions / yr (total)",     "int/yr"),
    "D_RP":    ("Duration / yr (repair)",         "h/yr"),
    "D_SW":    ("Duration / yr (switching)",      "h/yr"),
    "D_total": ("Duration / yr (total)",          "h/yr"),
}
indicator_lov = [(k, v[0]) for k, v in INDICATORS.items()]
view_lov = [("current", "Current scenario"),
            ("baseline", "Baseline"),
            ("delta", "Delta (current - baseline)")]

# =============================================================================
# 1. Reliability module (replace the demo by your own import)
# =============================================================================
try:
    from reliability import munoz_delgado      # <- your module/function
    DEMO_MODE = False
except ImportError:
    DEMO_MODE = True
    print("WARNING: 'reliability.munoz_delgado' not found -> using DEMO model")

    def munoz_delgado(G, source=SOURCE):
        """Toy radial model, ONLY to test the UI. Replace with your PuLP module."""
        rows = []
        for n in G.nodes:
            path = nx.shortest_path(G, source, n) if nx.has_path(G, source, n) else [n]
            on_path = {ekey(a, b) for a, b in zip(path, path[1:])}
            n_rp = d_rp = n_sw = d_sw = 0.0
            for u, v, d in G.edges(data=True):
                f = d["lam"] * d["length"]
                if ekey(u, v) in on_path:      # fault upstream -> wait for repair
                    n_rp += f
                    d_rp += f * d["tao_rp"]
                else:                          # other branch -> restored by switching
                    n_sw += f
                    d_sw += f * d["tao_sw"]
            rows.append((n, n_rp, n_sw, d_rp, d_sw))
        df = pd.DataFrame(rows, columns=RES_COLS)
        cl = pd.Series({n: G.nodes[n].get("clients", 0) for n in G.nodes})
        tot = cl.sum() or 1
        d = df.set_index("Node")
        saifi = ((d.N_RP + d.N_SW) * cl).sum() / tot
        saidi = ((d.D_RP + d.D_SW) * cl).sum() / tot
        return df, saifi, saidi


# =============================================================================
# 2. Pure helpers (no Taipy): graph, layout, validation, comparison, figure
# =============================================================================
def ekey(u, v):
    """Undirected edge key."""
    return (u, v) if u <= v else (v, u)


def _num(x):
    x = pd.to_numeric(x, errors="coerce")
    return 0.0 if pd.isna(x) else float(x)


def _read_nodes(path):
    df = pd.read_csv(path).iloc[:, :3]
    df.columns = NODE_COLS
    return df


def _read_edges(path):
    df = pd.read_csv(path).iloc[:, :6]
    df.columns = EDGE_COLS
    return df


def build_graph(nodes, edges):
    G = nx.Graph()
    for _, r in nodes.dropna(subset=["node"]).iterrows():
        G.add_node(int(r["node"]), clients=_num(r["clients"]), MW=_num(r["MW"]))
    for _, r in edges.dropna(subset=["u", "v"]).iterrows():
        G.add_edge(int(r["u"]), int(r["v"]),
                   length=_num(r["length"]), lam=_num(r["lambda"]),
                   tao_rp=_num(r["Tao_RP"]), tao_sw=_num(r["Tao_SW"]))
    return G


def topology_hash(G):
    return hash((tuple(sorted(G.nodes)), tuple(sorted(ekey(u, v) for u, v in G.edges))))


def compute_layout(G, source=SOURCE, dx=1.6, dy=1.0):
    """Layered tree layout: x = BFS depth from the substation, y = leaves stacked,
    parents centred on their children. Loops (tie switches) are drawn as extra
    edges on top of the BFS tree. Components not reachable from the source are
    stacked below, each rooted at its lowest node id."""
    pos, row = {}, [0]
    roots = [source] if source in G else []
    roots += sorted(min(c) for c in nx.connected_components(G) if source not in c)
    for root in roots:
        tree = nx.bfs_tree(G, root)
        depth = nx.single_source_shortest_path_length(G, root)

        def place(n):
            kids = sorted(tree.successors(n))
            if kids:
                ys = [place(k) for k in kids]
                y = sum(ys) / len(ys)
            else:
                y = row[0]
                row[0] += 1
            pos[n] = (depth[n] * dx, -y * dy)
            return y

        place(root)
        row[0] += 1                      # gap between components
    return pos


def with_totals(df):
    df = df.copy()
    df["Node"] = df["Node"].astype(int)
    df["N_total"] = df["N_RP"] + df["N_SW"]
    df["D_total"] = df["D_RP"] + df["D_SW"]
    return df


def validate(nodes, edges):
    errors, warnings = [], []
    if nodes.isna().any().any():
        errors.append("Nodes table has empty cells (complete or delete those rows).")
    if edges.isna().any().any():
        errors.append("Edges table has empty cells (complete or delete those rows).")
    if errors:
        return errors, warnings
    ids = nodes["node"].astype(int)
    if ids.duplicated().any():
        errors.append("Duplicate node numbers in the nodes table.")
    if SOURCE not in set(ids):
        errors.append(f"Source node {SOURCE} (substation) is missing.")
    if (nodes[["clients", "MW"]] < 0).any().any() or (edges[EDGE_PARAMS] < 0).any().any():
        errors.append("Negative values found in the input parameters.")
    used = set(edges["u"].astype(int)) | set(edges["v"].astype(int))
    if used - set(ids):
        errors.append(f"Edges reference nodes not in the node table: {sorted(used - set(ids))}")
    if (edges["u"].astype(int) == edges["v"].astype(int)).any():
        errors.append("Self-loop edges (u = v) are not allowed.")
    if not errors:
        G = build_graph(nodes, edges)
        reach = nx.node_connected_component(G, SOURCE) if SOURCE in G else set()
        lost = sorted(set(G.nodes) - reach)
        if lost:
            warnings.append(f"Nodes not connected to the source: {lost}")
    return errors, warnings


def check_indices(res, nodes, saifi, saidi, tol=1e-6):
    """Recompute customer-weighted SAIFI/SAIDI from the node table and compare.
    ADAPT or disable if your module defines the indices differently."""
    msgs = []
    n = nodes.dropna(subset=["node"]).assign(node=lambda d: d["node"].astype(int))
    r = with_totals(res).merge(n, left_on="Node", right_on="node")
    if len(r) < len(n):
        msgs.append(f"Nodes missing from the results: {sorted(set(n['node']) - set(r['Node']))}")
    tot = r["clients"].sum()
    if tot > 0:
        for name, got, col in (("SAIFI", saifi, "N_total"), ("SAIDI", saidi, "D_total")):
            exp = (r[col] * r["clients"]).sum() / tot
            if abs(got - exp) > tol * max(1.0, abs(exp)):
                msgs.append(f"{name} from module = {got:.4f}, recomputed from node table = {exp:.4f}")
    return msgs


def _edge_dict(edges):
    e = edges.dropna(subset=["u", "v"])
    return {ekey(int(r["u"]), int(r["v"])): r[EDGE_PARAMS].astype(float).to_numpy()
            for _, r in e.iterrows()}


def changed_sets(nodes, edges, base_nodes, base_edges):
    """Nodes / edges whose parameters differ from the baseline (or are new)."""
    b = (base_nodes.dropna(subset=["node"]).assign(node=lambda d: d["node"].astype(int))
         .drop_duplicates("node").set_index("node")[["clients", "MW"]].astype(float))
    chg_n = set()
    for _, r in nodes.dropna(subset=["node"]).iterrows():
        n = int(r["node"])
        if n not in b.index or not np.allclose(
                r[["clients", "MW"]].astype(float).to_numpy(), b.loc[n].to_numpy(), equal_nan=True):
            chg_n.add(n)
    be, ce = _edge_dict(base_edges), _edge_dict(edges)
    chg_e = {k for k, v in ce.items() if k not in be or not np.allclose(v, be[k], equal_nan=True)}
    return chg_n, chg_e


def build_cmp(cur, base):
    if cur.empty or base.empty:
        return pd.DataFrame(columns=CMP_COLS)
    c = with_totals(cur).set_index("Node")[["N_total", "D_total"]]
    b = with_totals(base).set_index("Node")[["N_total", "D_total"]]
    j = b.join(c, how="outer", lsuffix="_base", rsuffix="_cur")
    j["dN"] = j["N_total_cur"] - j["N_total_base"]
    j["dD"] = j["D_total_cur"] - j["D_total_base"]
    j = j.reindex(j["dD"].abs().sort_values(ascending=False).index).reset_index()
    j.columns = ["Node", "N_base", "N_cur", "D_base", "D_cur", "dN", "dD"]
    return j[CMP_COLS].round(4)


def _segments(G, pos, keep=None):
    xs, ys = [], []
    for u, v in G.edges():
        if keep is None or ekey(u, v) in keep:
            xs += [pos[u][0], pos[v][0], None]
            ys += [pos[u][1], pos[v][1], None]
    return xs, ys


def build_figure(G, pos, cur, base, view, indicator, chg_n, chg_e, hl_n, hl_e):
    f = go.Figure()
    f.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                    plot_bgcolor="rgba(0,0,0,0)")
    if G.number_of_nodes() == 0:
        return f

    have_cur = cur is not None and not cur.empty
    have_base = base is not None and not base.empty
    if view != "current" and not have_base:
        view = "current"
    cur_t = with_totals(cur).set_index("Node") if have_cur else None
    base_t = with_totals(base).set_index("Node") if have_base else None

    # ---- edges: base lines, then modified (orange) and selected (red) overlays
    ex, ey = _segments(G, pos)
    f.add_trace(go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="skip",
                           line=dict(width=2, color="#9aa0a6")))
    for keys, color, width in ((chg_e, ORANGE, 4), (hl_e, RED, 5)):
        if keys:
            x, y = _segments(G, pos, keys)
            f.add_trace(go.Scatter(x=x, y=y, mode="lines", hoverinfo="skip",
                                   line=dict(width=width, color=color)))

    # ---- edge midpoints: hover carries the edge parameters
    mx, my, mc, mt, mlw, mlc = [], [], [], [], [], []
    for u, v, d in G.edges(data=True):
        k = ekey(u, v)
        ef = d["lam"] * d["length"]
        mx.append((pos[u][0] + pos[v][0]) / 2)
        my.append((pos[u][1] + pos[v][1]) / 2)
        mc.append(ef)
        mt.append(f"<b>Edge {u}-{v}</b>" + ("  (modified)" if k in chg_e else "")
                  + f"<br>Length: {d['length']:.3f} km<br>lambda: {d['lam']:.4f}"
                  f"<br>Exp. failures: {ef:.4f}<br>Tao_RP: {d['tao_rp']} h<br>Tao_SW: {d['tao_sw']} h")
        mlw.append(3 if k in hl_e else 2.5 if k in chg_e else 0.6)
        mlc.append(RED if k in hl_e else ORANGE if k in chg_e else "#333")
    f.add_trace(go.Scatter(x=mx, y=my, mode="markers", hovertext=mt, hoverinfo="text",
                           marker=dict(size=9, symbol="diamond", color=mc, colorscale="Blues",
                                       cmin=0, showscale=False, line=dict(width=mlw, color=mlc))))

    # ---- nodes
    ns = list(G.nodes())
    mw = np.array([G.nodes[n].get("MW", 0.0) for n in ns], dtype=float)
    sizes = 10 + 22 * mw / (mw.max() if mw.max() > 0 else 1)

    def val(t, n):
        return t.at[n, indicator] if t is not None and n in t.index else np.nan

    c = np.array([val(cur_t, n) for n in ns], dtype=float)
    b = np.array([val(base_t, n) for n in ns], dtype=float)
    label, unit = INDICATORS[indicator]

    if not have_cur and not have_base:
        marker_extra = dict(color="#1f77b4")
    elif view == "delta":
        v = np.nan_to_num(c - b)
        m = float(np.abs(v).max()) or 1.0
        marker_extra = dict(color=v, colorscale="RdBu_r", cmin=-m, cmax=m, showscale=True,
                            colorbar=dict(title=f"Delta {unit}"))
    else:
        v = b if view == "baseline" else c
        both = np.concatenate([c, b])
        vmax = float(np.nanmax(both)) if np.isfinite(both).any() else 1.0
        # same 0..max scale for current and baseline -> colors are comparable
        marker_extra = dict(color=np.nan_to_num(v), colorscale="YlOrRd", cmin=0, cmax=vmax or 1.0,
                            showscale=True, colorbar=dict(title=unit))

    def hover(n):
        t = (f"<b>Node {n}</b>" + ("  (modified)" if n in chg_n else "")
             + f"<br>Clients: {G.nodes[n].get('clients', 0):.0f}"
             f"<br>Load: {G.nodes[n].get('MW', 0):.3f} MW")
        if cur_t is not None and n in cur_t.index:
            r = cur_t.loc[n]
            t += (f"<br>-------<br><b>Current</b>"
                  f"<br>N_RP {r.N_RP:.3f} | N_SW {r.N_SW:.3f} (int/yr)"
                  f"<br>D_RP {r.D_RP:.3f} | D_SW {r.D_SW:.3f} (h/yr)")
        if base_t is not None and n in base_t.index:
            r = base_t.loc[n]
            t += (f"<br><b>Baseline</b>  N {r.N_total:.3f} int/yr | D {r.D_total:.3f} h/yr")
            if cur_t is not None and n in cur_t.index:
                t += (f"<br><b>Delta</b>  N {cur_t.at[n, 'N_total'] - r.N_total:+.3f}"
                      f" | D {cur_t.at[n, 'D_total'] - r.D_total:+.3f}")
        return t

    f.add_trace(go.Scatter(
        x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns],
        mode="markers+text", text=[str(n) for n in ns], textposition="top center",
        hovertext=[hover(n) for n in ns], hoverinfo="text",
        marker=dict(size=sizes, **marker_extra,
                    line=dict(width=[3.5 if n in hl_n else 3 if n in chg_n else 0.8 for n in ns],
                              color=[RED if n in hl_n else ORANGE if n in chg_n else "#222"
                                     for n in ns]))))
    return f


# =============================================================================
# 3. Taipy state (module-level variables = per-user state)
# =============================================================================
df_nodes = _read_nodes(NODES_CSV)
df_edges = _read_edges(EDGES_CSV)
nodes_file, edges_file = None, None

pos_json = "{}"          # cached layout (JSON-safe on purpose)
topo_hash = None         # layout is recomputed only when this changes
fig = go.Figure()

df_res = pd.DataFrame(columns=RES_COLS)
saifi, saidi = 0.0, 0.0
stale = False
idle = True
demo_mode = DEMO_MODE

has_base = False
base_nodes = pd.DataFrame(columns=NODE_COLS)
base_edges = pd.DataFrame(columns=EDGE_COLS)
base_res = pd.DataFrame(columns=RES_COLS)
base_saifi, base_saidi = 0.0, 0.0

indicator = "D_total"
view = "current"
sel_nodes, sel_edges = [], []

kpi_saifi, kpi_saidi = "**SAIFI**  \n-", "**SAIDI**  \n-"
df_cmp = pd.DataFrame(columns=CMP_COLS)
scenario_msg = "No baseline stored."


# =============================================================================
# 4. Taipy callbacks
# =============================================================================
def _selection_highlight(state):
    hl_n, hl_e = set(), set()
    nodes, edges = state.df_nodes, state.df_edges
    for i in (state.sel_nodes or []):
        if 0 <= i < len(nodes) and pd.notna(nodes.iloc[i]["node"]):
            hl_n.add(int(nodes.iloc[i]["node"]))
    for i in (state.sel_edges or []):
        if 0 <= i < len(edges) and pd.notna(edges.iloc[i]["u"]) and pd.notna(edges.iloc[i]["v"]):
            u, v = int(edges.iloc[i]["u"]), int(edges.iloc[i]["v"])
            hl_e.add(ekey(u, v))
            hl_n |= {u, v}
    return hl_n, hl_e


def refresh(state):
    """Rebuild the figure. The layout is recomputed ONLY if the topology changed."""
    G = build_graph(state.df_nodes, state.df_edges)
    h = topology_hash(G)
    if h != state.topo_hash:
        state.pos_json = json.dumps({str(n): list(p) for n, p in compute_layout(G).items()})
        state.topo_hash = h
    pos = {int(k): v for k, v in json.loads(state.pos_json).items()}
    if state.has_base:
        chg_n, chg_e = changed_sets(state.df_nodes, state.df_edges,
                                    state.base_nodes, state.base_edges)
    else:
        chg_n, chg_e = set(), set()
    hl_n, hl_e = _selection_highlight(state)
    state.fig = build_figure(G, pos, state.df_res, state.base_res, state.view,
                             state.indicator, chg_n, chg_e, hl_n, hl_e)


def _kpi_text(name, unit, cur, base, has_cur, has_b):
    if not has_cur:
        return f"**{name}**  \n-"
    s = f"**{name}** ({unit})  \n### {cur:.3f}"
    if has_b:
        pct = f" ({(cur - base) / base * 100:+.1f}%)" if base else ""
        s += f"  \nBaseline {base:.3f} | Delta {cur - base:+.3f}{pct}"
    return s


def update_summary(state):
    has_cur = not state.df_res.empty
    state.kpi_saifi = _kpi_text("SAIFI", "int/cust-yr", state.saifi, state.base_saifi,
                                has_cur, state.has_base)
    state.kpi_saidi = _kpi_text("SAIDI", "h/cust-yr", state.saidi, state.base_saidi,
                                has_cur, state.has_base)
    state.df_cmp = build_cmp(state.df_res, state.base_res) if state.has_base \
        else pd.DataFrame(columns=CMP_COLS)
    state.scenario_msg = ("Baseline stored. Modified elements are outlined in orange."
                          if state.has_base else "No baseline stored.")


def on_init(state):
    update_summary(state)
    refresh(state)


def on_change(state, var_name, value):
    if var_name in ("indicator", "view", "sel_nodes", "sel_edges"):
        refresh(state)


# ---- table editing -----------------------------------------------------------
def _apply_edit(df, payload):
    df = df.copy()
    df.at[payload["index"], payload["col"]] = pd.to_numeric(payload["value"], errors="coerce")
    return df


def _add_row(df, payload):
    i = payload["index"]
    new = pd.DataFrame([{c: np.nan for c in df.columns}])
    return pd.concat([df.iloc[:i], new, df.iloc[i:]], ignore_index=True)


def _inputs_changed(state, clear_selection=False):
    if not state.df_res.empty:
        state.stale = True
    if clear_selection:
        state.sel_nodes, state.sel_edges = [], []
    refresh(state)


def node_edit(state, var_name, payload):
    state.df_nodes = _apply_edit(state.df_nodes, payload)
    _inputs_changed(state)


def edge_edit(state, var_name, payload):
    state.df_edges = _apply_edit(state.df_edges, payload)
    _inputs_changed(state)


def node_add(state, var_name, payload):
    state.df_nodes = _add_row(state.df_nodes, payload)
    _inputs_changed(state, True)


def edge_add(state, var_name, payload):
    state.df_edges = _add_row(state.df_edges, payload)
    _inputs_changed(state, True)


def node_delete(state, var_name, payload):
    state.df_nodes = state.df_nodes.drop(payload["index"]).reset_index(drop=True)
    _inputs_changed(state, True)


def edge_delete(state, var_name, payload):
    state.df_edges = state.df_edges.drop(payload["index"]).reset_index(drop=True)
    _inputs_changed(state, True)


# ---- loading CSV files ---------------------------------------------------------
def _reset_scenarios(state):
    state.df_res = pd.DataFrame(columns=RES_COLS)
    state.saifi = state.saidi = 0.0
    state.stale = False
    state.has_base = False
    state.view = "current"
    state.sel_nodes, state.sel_edges = [], []
    update_summary(state)
    refresh(state)


def load_nodes(state):
    try:
        state.df_nodes = _read_nodes(state.nodes_file)
    except Exception as e:
        notify(state, "error", f"Could not read nodes CSV: {e}")
        return
    _reset_scenarios(state)


def load_edges(state):
    try:
        state.df_edges = _read_edges(state.edges_file)
    except Exception as e:
        notify(state, "error", f"Could not read edges CSV: {e}")
        return
    _reset_scenarios(state)


# ---- running the model (long callback: UI stays responsive during PuLP) ---------
def _compute(gui_instance, state_id, nodes, edges):
    """Runs in a background thread: NO access to `state` here."""
    G = build_graph(nodes, edges)
    res, s_ifi, s_idi = munoz_delgado(G, nodes, 1)
    #print(f"el dataframe que viene de munoz_delgado es: {res}")
    clean_res = res[RES_COLS].copy()
    clean_res.reset_index(drop=True, inplace=True)

    clean_nodes = nodes.copy()
    clean_edges = edges.copy()
    payload = {
        "res": clean_res,
        "saifi": float(s_ifi),
        "saidi": float(s_idi),
        "nodes": clean_nodes,
        "edges": clean_edges
    }

    # Pipe the dictionary safely over the thread boundary to be executed on the UI main thread
    invoke_callback(gui_instance, state_id, _update_ui_with_results, [payload])
    #return {"res": clean_res, "saifi": float(s_ifi), "saidi": float(s_idi),
    #        "nodes": clean_nodes, "edges": clean_edges}

def _update_ui_with_results(state, out):
    """Runs on the main thread: Has full access to update `state`."""
    state.df_res = out["res"]
    state.saifi, state.saidi = out["saifi"], out["saidi"]
    
    # Check if inputs were edited while the model was running
    state.stale = not (out["nodes"].equals(state.df_nodes) and out["edges"].equals(state.df_edges))
    state.idle = True
    
    for m in check_indices(out["res"], out["nodes"], out["saifi"], out["saidi"]):
        notify(state, "warning", m)
        
    update_summary(state)
    refresh(state)
    notify(state, "success", "Reliability analysis completed")


def _on_run_finished(state, status, *args):
    #print(f"DEBUG: status={status}, type_results={type(results)}, len_results={len(results)}, results={results}")
    if status is False:                       # exception inside _compute
        state.idle = True
        notify(state, "error", "The reliability analysis failed. Check the console.")
        return
    if status is not True:                    # periodic call while running
        return
    #print("......entró al _on_run_finished......")
    #if not results:
    #    print(".....NO se recibió results ......")
    #out = results[0]
    #print(".....pasó el almacenamiento de result[0] en out .....")
    #state.df_res = out["res"]
    #state.saifi, state.saidi = out["saifi"], out["saidi"]
    #print(".....almacenó los valores de saifi y saidi.....")
    # inputs edited while the model was running -> results already outdated
    #state.stale = not (out["nodes"].equals(state.df_nodes) and out["edges"].equals(state.df_edges))
    #state.idle = True
    #for m in check_indices(out["res"], out["nodes"], out["saifi"], out["saidi"]):
    #    notify(state, "warning", m)
    #update_summary(state)
    #refresh(state)
    #notify(state, "success", "Reliability analysis completed")


def run_model(state):
    errors, warns = validate(state.df_nodes, state.df_edges)
    for w in warns:
        notify(state, "warning", w)
    if errors:
        for e in errors:
            notify(state, "error", e)
        return
    state.idle = False
    notify(state, "info", "Running reliability analysis...")

    state_id = get_state_id(state)

    invoke_long_callback(state, user_function=_compute,
                         user_function_args=[gui, state_id, state.df_nodes.copy(), state.df_edges.copy()],
                         user_status_function=_on_run_finished, period=1000)


# ---- scenario management ----------------------------------------------------------
def set_baseline(state):
    if state.df_res.empty:
        notify(state, "warning", "Run the analysis first.")
        return
    if state.stale:
        notify(state, "warning", "Results are outdated. Run the analysis again before storing a baseline.")
        return
    state.base_nodes, state.base_edges = state.df_nodes.copy(), state.df_edges.copy()
    state.base_res = state.df_res.copy()
    state.base_saifi, state.base_saidi = state.saifi, state.saidi
    state.has_base = True
    update_summary(state)
    refresh(state)
    notify(state, "success", "Baseline stored. Edit parameters and run again to compare.")


def restore_baseline(state):
    if not state.has_base:
        return
    state.df_nodes, state.df_edges = state.base_nodes.copy(), state.base_edges.copy()
    state.df_res = state.base_res.copy()
    state.saifi, state.saidi = state.base_saifi, state.base_saidi
    state.stale = False
    state.sel_nodes, state.sel_edges = [], []
    update_summary(state)
    refresh(state)


def clear_baseline(state):
    state.has_base = False
    state.view = "current"
    update_summary(state)
    refresh(state)


# =============================================================================
# 5. Page
# =============================================================================
page = """
# Distribution Network Reliability

<|part|render={demo_mode}|
**Demo model in use**: `reliability.py` was not found, so a placeholder calculation is running.
|>

<|layout|columns=1 2|
<|
### Data
<|{nodes_file}|file_selector|label=Load nodes CSV|extensions=.csv|on_action=load_nodes|>
<|{edges_file}|file_selector|label=Load edges CSV|extensions=.csv|on_action=load_edges|>

### Nodes
<|{df_nodes}|table|editable|selected={sel_nodes}|on_edit=node_edit|on_add=node_add|on_delete=node_delete|height=230px|width=100%|>

### Edges
<|{df_edges}|table|editable|selected={sel_edges}|on_edit=edge_edit|on_add=edge_add|on_delete=edge_delete|height=300px|width=100%|>

### Scenario
<|Run analysis|button|on_action=run_model|active={idle}|>
<|Set current as baseline|button|on_action=set_baseline|active={idle}|>
<|Restore baseline inputs|button|on_action=restore_baseline|active={has_base}|>
<|Clear baseline|button|on_action=clear_baseline|active={has_base}|>

<|{scenario_msg}|text|>
|>

<|
<|layout|columns=1 1|
<|{kpi_saifi}|text|mode=md|>
<|{kpi_saidi}|text|mode=md|>
|>

<|layout|columns=1 1|
<|{indicator}|selector|lov={indicator_lov}|value_by_id=True|dropdown|label=Color nodes by|width=100%|>
<|{view}|selector|lov={view_lov}|value_by_id=True|dropdown|label=Show|width=100%|>
|>

<|part|render={stale}|
**Parameters changed since the last run. The results shown are outdated.**
|>

<|chart|figure={fig}|height=620px|>

<|part|render={has_base}|
### Baseline vs current (sorted by largest change in D_total)
<|{df_cmp}|table|height=250px|width=100%|>
|>

### Results per node
<|{df_res}|table|height=250px|width=100%|>
|>
|>
"""

gui = Gui(page)
if __name__ == "__main__":
    gui.run(title="Network Reliability", port="auto")
