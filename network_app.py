"""
Веб-візуалізація топології мережі з можливістю динамічної зміни параметрів.
Курсова робота з дисципліни «Алгоритмізація та програмування».

Запуск:
    pip install streamlit networkx matplotlib pandas numpy
    streamlit run network_app.py
"""

from __future__ import annotations

import io
import math
import time
from collections import deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

NODE_TYPES = {
    "router":      {"color": "#E74C3C", "size": 1100, "shape": "s"},
    "switch":      {"color": "#3498DB", "size": 800,  "shape": "o"},
    "server":      {"color": "#27AE60", "size": 950,  "shape": "D"},
    "workstation": {"color": "#F39C12", "size": 500,  "shape": "o"},
    "firewall":    {"color": "#8E44AD", "size": 1000, "shape": "h"},
}


@dataclass
class NetworkModel:
    """Контейнер для графа корпоративної мережі."""
    graph: nx.Graph = field(default_factory=nx.Graph)

    def add_node(self, name: str, node_type: str = "workstation") -> None:
        if node_type not in NODE_TYPES:
            raise ValueError(f"Невідомий тип вузла: {node_type}")
        if name in self.graph:
            raise ValueError(f"Вузол '{name}' вже існує")
        self.graph.add_node(name, type=node_type)

    def remove_node(self, name: str) -> None:
        if name not in self.graph:
            raise KeyError(f"Вузол '{name}' відсутній")
        self.graph.remove_node(name)

    def add_edge(self, u: str, v: str, bandwidth: float = 100.0,
                 latency: float = 1.0) -> None:
        if u == v:
            raise ValueError("Петлі заборонені")
        if u not in self.graph or v not in self.graph:
            raise KeyError("Один з вузлів не існує")
        if bandwidth <= 0 or latency <= 0:
            raise ValueError("Параметри ребра мають бути додатні")
        self.graph.add_edge(u, v, bandwidth=bandwidth, latency=latency,
                            weight=1000.0 / bandwidth)

    def remove_edge(self, u: str, v: str) -> None:
        if not self.graph.has_edge(u, v):
            raise KeyError(f"Ребро {u}-{v} відсутнє")
        self.graph.remove_edge(u, v)

    def update_edge(self, u: str, v: str, bandwidth: float, latency: float) -> None:
        if not self.graph.has_edge(u, v):
            raise KeyError(f"Ребро {u}-{v} відсутнє")
        if bandwidth <= 0 or latency <= 0:
            raise ValueError("Параметри ребра мають бути додатні")
        self.graph[u][v]["bandwidth"] = bandwidth
        self.graph[u][v]["latency"] = latency
        self.graph[u][v]["weight"] = 1000.0 / bandwidth


def build_default_network() -> NetworkModel:
    """Будує демонстраційну корпоративну мережу (5 відділів)."""
    m = NetworkModel()
    nodes = [
        ("FW",      "firewall"),
        ("R1",      "router"),
        ("R2",      "router"),
        ("SW_FIN",  "switch"),
        ("SW_IT",   "switch"),
        ("SW_SEC",  "switch"),
        ("SW_HR",   "switch"),
        ("SW_SRV",  "switch"),
        ("SRV_DB",  "server"),
        ("SRV_WEB", "server"),
        ("SRV_MAIL","server"),
        ("PC_FIN1", "workstation"),
        ("PC_FIN2", "workstation"),
        ("PC_IT1",  "workstation"),
        ("PC_IT2",  "workstation"),
        ("PC_SEC1", "workstation"),
        ("PC_HR1",  "workstation"),
        ("PC_HR2",  "workstation"),
    ]
    for name, t in nodes:
        m.add_node(name, t)

    edges = [
        ("FW",  "R1",      1000, 0.5),
        ("R1",  "R2",      1000, 0.6),
        ("R1",  "SW_FIN",   100, 1.0),
        ("R1",  "SW_IT",    100, 1.0),
        ("R1",  "SW_SRV",  1000, 0.4),
        ("R2",  "SW_SEC",   100, 1.0),
        ("R2",  "SW_HR",    100, 1.0),
        ("SW_FIN", "PC_FIN1", 100, 1.5),
        ("SW_FIN", "PC_FIN2", 100, 1.5),
        ("SW_IT",  "PC_IT1",  100, 1.5),
        ("SW_IT",  "PC_IT2",  100, 1.5),
        ("SW_SEC", "PC_SEC1", 100, 1.5),
        ("SW_HR",  "PC_HR1",  100, 1.5),
        ("SW_HR",  "PC_HR2",  100, 1.5),
        ("SW_SRV", "SRV_DB",  1000, 0.3),
        ("SW_SRV", "SRV_WEB", 1000, 0.3),
        ("SW_SRV", "SRV_MAIL",1000, 0.3),
    ]
    for u, v, bw, lat in edges:
        m.add_edge(u, v, bw, lat)
    return m

def bfs(graph: nx.Graph, start: str) -> Tuple[List[str], Dict[str, int]]:
    """Обхід у ширину. Повертає порядок відвідування і відстані у ребрах."""
    if start not in graph:
        raise KeyError(f"Вузол '{start}' відсутній")
    visited: Set[str] = {start}
    order: List[str] = []
    distances: Dict[str, int] = {start: 0}
    queue: deque = deque([start])
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in sorted(graph.neighbors(node)):
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return order, distances


def dfs(graph: nx.Graph, start: str) -> List[str]:
    """Обхід у глибину (ітеративно через стек)."""
    if start not in graph:
        raise KeyError(f"Вузол '{start}' відсутній")
    visited: Set[str] = set()
    order: List[str] = []
    stack: List[str] = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in sorted(graph.neighbors(node), reverse=True):
            if neighbor not in visited:
                stack.append(neighbor)
    return order


def dijkstra(graph: nx.Graph, source: str, target: Optional[str] = None
             ) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    """Алгоритм Дейкстри на бінарній купі (heapq)."""
    if source not in graph:
        raise KeyError(f"Вузол '{source}' відсутній")
    dist: Dict[str, float] = {n: math.inf for n in graph.nodes()}
    prev: Dict[str, Optional[str]] = {n: None for n in graph.nodes()}
    dist[source] = 0.0
    heap: List[Tuple[float, str]] = [(0.0, source)]
    while heap:
        d, u = heappop(heap)
        if d > dist[u]:
            continue
        if target is not None and u == target:
            break
        for v in graph.neighbors(u):
            w = graph[u][v].get("weight", 1.0)
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heappush(heap, (nd, v))
    return dist, prev


def reconstruct_path(prev: Dict[str, Optional[str]], target: str) -> List[str]:
    path: List[str] = []
    cur: Optional[str] = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))


def find_articulation_points(graph: nx.Graph) -> List[str]:
    """Алгоритм Тар'яна для пошуку точок зчленування."""
    if graph.number_of_nodes() == 0:
        return []
    visited: Set[str] = set()
    disc: Dict[str, int] = {}
    low: Dict[str, int] = {}
    parent: Dict[str, Optional[str]] = {n: None for n in graph.nodes()}
    aps: Set[str] = set()
    timer = [0]

    def ap_dfs(u: str) -> None:
        children = 0
        visited.add(u)
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v in graph.neighbors(u):
            if v not in visited:
                parent[v] = u
                children += 1
                ap_dfs(v)
                low[u] = min(low[u], low[v])
                if parent[u] is None and children > 1:
                    aps.add(u)
                if parent[u] is not None and low[v] >= disc[u]:
                    aps.add(u)
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    import sys
    sys.setrecursionlimit(10000)
    for n in graph.nodes():
        if n not in visited:
            ap_dfs(n)
    return sorted(aps)


def find_bridges(graph: nx.Graph) -> List[Tuple[str, str]]:
    """Алгоритм Тар'яна для пошуку мостів."""
    visited: Set[str] = set()
    disc: Dict[str, int] = {}
    low: Dict[str, int] = {}
    parent: Dict[str, Optional[str]] = {n: None for n in graph.nodes()}
    bridges: List[Tuple[str, str]] = []
    timer = [0]

    def br_dfs(u: str) -> None:
        visited.add(u)
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v in graph.neighbors(u):
            if v not in visited:
                parent[v] = u
                br_dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append(tuple(sorted((u, v))))
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    import sys
    sys.setrecursionlimit(10000)
    for n in graph.nodes():
        if n not in visited:
            br_dfs(n)
    return sorted(set(bridges))


def connectivity_metrics(graph: nx.Graph) -> Dict[str, float]:
    """Базові метрики зв'язності."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    if n == 0:
        return {"nodes": 0, "edges": 0, "density": 0.0,
                "avg_degree": 0.0, "components": 0,
                "diameter": 0, "is_connected": False}
    density = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0
    avg_deg = (2.0 * m) / n
    components = nx.number_connected_components(graph)
    is_connected = nx.is_connected(graph)
    if is_connected:
        diameter = nx.diameter(graph)
    else:
        largest = max(nx.connected_components(graph), key=len)
        diameter = nx.diameter(graph.subgraph(largest))
    return {"nodes": n, "edges": m, "density": density,
            "avg_degree": avg_deg, "components": components,
            "diameter": diameter, "is_connected": is_connected}


def simulate_failure(graph: nx.Graph, node: str) -> Dict[str, object]:
    """Імітація відмови вузла: будує нову копію без вузла."""
    if node not in graph:
        raise KeyError(f"Вузол '{node}' відсутній")
    g2 = graph.copy()
    g2.remove_node(node)
    components = list(nx.connected_components(g2))
    return {
        "before_components": nx.number_connected_components(graph),
        "after_components": len(components),
        "before_nodes": graph.number_of_nodes(),
        "after_nodes": g2.number_of_nodes(),
        "isolated_groups": [sorted(c) for c in components],
        "graph": g2,
    }

def render_graph(graph: nx.Graph,
                 highlighted_path: Optional[List[str]] = None,
                 removed_node: Optional[str] = None,
                 layout_seed: int = 42,
                 layout_name: str = "spring",
                 show_labels: bool = True,
                 show_edge_labels: bool = False) -> plt.Figure:
    """Малює топологію мережі через matplotlib."""
    fig, ax = plt.subplots(figsize=(11, 8))
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Граф порожній", ha="center", va="center",
                fontsize=16)
        ax.axis("off")
        return fig

    if layout_name == "spring":
        pos = nx.spring_layout(graph, seed=layout_seed, k=1.5,
                               iterations=80)
    elif layout_name == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    elif layout_name == "circular":
        pos = nx.circular_layout(graph)
    else:
        pos = nx.shell_layout(graph)

    path_edges: Set[Tuple[str, str]] = set()
    if highlighted_path and len(highlighted_path) > 1:
        for i in range(len(highlighted_path) - 1):
            a, b = highlighted_path[i], highlighted_path[i + 1]
            path_edges.add(tuple(sorted((a, b))))

    edges_normal: List[Tuple[str, str]] = []
    edges_highlight: List[Tuple[str, str]] = []
    for u, v in graph.edges():
        key = tuple(sorted((u, v)))
        if key in path_edges:
            edges_highlight.append((u, v))
        else:
            edges_normal.append((u, v))

    nx.draw_networkx_edges(graph, pos, edgelist=edges_normal,
                           ax=ax, edge_color="#95A5A6", width=1.6,
                           alpha=0.7)
    if edges_highlight:
        nx.draw_networkx_edges(graph, pos, edgelist=edges_highlight,
                               ax=ax, edge_color="#E74C3C", width=3.5,
                               alpha=0.95)

    for ntype, props in NODE_TYPES.items():
        nodes = [n for n, d in graph.nodes(data=True)
                 if d.get("type") == ntype]
        if not nodes:
            continue
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes,
                               node_color=props["color"],
                               node_size=props["size"],
                               node_shape=props["shape"],
                               edgecolors="#2C3E50",
                               linewidths=1.4, ax=ax)

    if highlighted_path:
        nx.draw_networkx_nodes(graph, pos, nodelist=highlighted_path,
                               node_color="none",
                               edgecolors="#E74C3C",
                               linewidths=3.0,
                               node_size=[NODE_TYPES[graph.nodes[n]["type"]]["size"] + 250
                                          for n in highlighted_path],
                               ax=ax)

    if show_labels:
        nx.draw_networkx_labels(graph, pos, font_size=9,
                                font_family="DejaVu Sans",
                                font_weight="bold", ax=ax)
    if show_edge_labels:
        elabels = {(u, v): f"{graph[u][v]['bandwidth']:.0f} Mbps"
                   for u, v in graph.edges()}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=elabels,
                                     ax=ax, font_size=7)

    title = "Топологія корпоративної мережі"
    if removed_node:
        title += f"  —  ВІДМОВА: {removed_node}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig

st.set_page_config(
    page_title="Аналіз корпоративної мережі",
    page_icon="🕸",
    layout="wide",
)


def get_model() -> NetworkModel:
    if "model" not in st.session_state:
        st.session_state.model = build_default_network()
    return st.session_state.model


def reset_model() -> None:
    st.session_state.model = build_default_network()


def main() -> None:
    st.title("Веб-візуалізація топології корпоративної мережі")
    st.caption(
        "Курсова робота — динамічна зміна параметрів, аналіз зв'язності, "
        "пошук найкоротших шляхів, імітація відмов вузлів."
    )

    model = get_model()
    g = model.graph

    # ── Бічна панель ────────────────────────────────────────────────
    with st.sidebar:
        st.header("Параметри")

        layout_name = st.selectbox(
            "Алгоритм розкладки",
            ["spring", "kamada_kawai", "circular", "shell"],
            index=0,
        )
        layout_seed = st.slider("Зерно розкладки (seed)", 1, 200, 42)
        show_labels = st.checkbox("Підписи вузлів", value=True)
        show_edge_labels = st.checkbox("Підписи ребер (Mbps)", value=False)

        st.divider()
        st.subheader("Скидання")
        if st.button("↻ Відновити мережу за замовчуванням",
                     use_container_width=True):
            reset_model()
            st.rerun()

    # ── Метрики ─────────────────────────────────────────────────────
    metrics = connectivity_metrics(g)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Вузлів", metrics["nodes"])
    c2.metric("Ребер", metrics["edges"])
    c3.metric("Щільність", f"{metrics['density']:.3f}")
    c4.metric("Сер. ступінь", f"{metrics['avg_degree']:.2f}")
    c5.metric("Компонент", metrics["components"],
              delta="зв'язний" if metrics["is_connected"] else "НЕЗВ'ЯЗНИЙ",
              delta_color="normal" if metrics["is_connected"] else "inverse")

    # ── Вкладки ────────────────────────────────────────────────────
    tab_view, tab_edit, tab_paths, tab_conn, tab_fail, tab_data = st.tabs([
        "📊 Візуалізація",
        "✏ Редагування",
        "🛣 Маршрути",
        "🔗 Зв'язність",
        "⚠ Імітація відмов",
        "📋 Дані",
    ])

    # ── Вкладка: Візуалізація ──────────────────────────────────────
    with tab_view:
        fig = render_graph(g, layout_seed=layout_seed,
                           layout_name=layout_name,
                           show_labels=show_labels,
                           show_edge_labels=show_edge_labels)
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        st.download_button(
            "⬇ Завантажити PNG",
            data=buf.getvalue(),
            file_name="network_topology.png",
            mime="image/png",
        )

        with st.expander("Легенда"):
            for t, p in NODE_TYPES.items():
                st.markdown(
                    f"<span style='color:{p['color']};font-size:22px'>●</span> "
                    f"&nbsp;**{t}**", unsafe_allow_html=True)

    # ── Вкладка: Редагування ───────────────────────────────────────
    with tab_edit:
        col_n, col_e = st.columns(2)

        with col_n:
            st.subheader("Вузли")
            with st.form("add_node_form"):
                name = st.text_input("Назва нового вузла")
                ntype = st.selectbox("Тип", list(NODE_TYPES.keys()))
                if st.form_submit_button("➕ Додати вузол"):
                    try:
                        model.add_node(name.strip(), ntype)
                        st.success(f"Вузол '{name}' додано")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Помилка: {ex}")

            with st.form("rm_node_form"):
                rm = st.selectbox("Вузол для видалення",
                                  sorted(g.nodes()))
                if st.form_submit_button("🗑 Видалити вузол"):
                    try:
                        model.remove_node(rm)
                        st.success(f"Вузол '{rm}' видалено")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Помилка: {ex}")

        with col_e:
            st.subheader("Ребра (з'єднання)")
            nodes = sorted(g.nodes())
            with st.form("add_edge_form"):
                u = st.selectbox("Вузол A", nodes, key="ae_u")
                v = st.selectbox("Вузол B", nodes, key="ae_v")
                bw = st.number_input("Пропускна здатність (Mbps)",
                                     min_value=1.0, max_value=10000.0,
                                     value=100.0, step=10.0)
                lat = st.number_input("Затримка (мс)", min_value=0.1,
                                      max_value=500.0, value=1.0, step=0.1)
                if st.form_submit_button("➕ Додати/оновити ребро"):
                    try:
                        if g.has_edge(u, v):
                            model.update_edge(u, v, bw, lat)
                            st.success("Параметри ребра оновлено")
                        else:
                            model.add_edge(u, v, bw, lat)
                            st.success(f"Ребро {u}-{v} додано")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Помилка: {ex}")

            edges = sorted(tuple(sorted(e)) for e in g.edges())
            if edges:
                with st.form("rm_edge_form"):
                    edge_label = st.selectbox(
                        "Ребро для видалення",
                        [f"{u} ↔ {v}" for u, v in edges])
                    if st.form_submit_button("🗑 Видалити ребро"):
                        try:
                            u, v = edge_label.split(" ↔ ")
                            model.remove_edge(u, v)
                            st.success("Ребро видалено")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Помилка: {ex}")

    # ── Вкладка: Маршрути ──────────────────────────────────────────
    with tab_paths:
        st.subheader("Пошук найкоротших шляхів")
        if g.number_of_nodes() < 2:
            st.warning("Замало вузлів для пошуку маршрутів.")
        else:
            nodes = sorted(g.nodes())
            cs, ct, cm = st.columns([1, 1, 1])
            src = cs.selectbox("Джерело", nodes, key="src")
            dst = ct.selectbox("Призначення", nodes,
                               index=min(1, len(nodes) - 1), key="dst")
            algo = cm.radio("Алгоритм",
                            ["Дейкстра (за вагою)", "BFS (за к-стю ребер)"])

            if st.button("🔍 Знайти маршрут", type="primary"):
                try:
                    if algo.startswith("Дейкстра"):
                        t0 = time.perf_counter()
                        dist, prev = dijkstra(g, src, dst)
                        dt = (time.perf_counter() - t0) * 1000
                        if math.isinf(dist[dst]):
                            st.error("Шлях не існує — граф незв'язний")
                        else:
                            path = reconstruct_path(prev, dst)
                            st.success(
                                f"Шлях знайдено за {dt:.2f} мс  "
                                f"(вартість {dist[dst]:.2f}, "
                                f"стрибків: {len(path)-1})"
                            )
                            st.code(" → ".join(path))
                            fig2 = render_graph(g, highlighted_path=path,
                                                layout_name=layout_name,
                                                layout_seed=layout_seed,
                                                show_labels=show_labels,
                                                show_edge_labels=show_edge_labels)
                            st.pyplot(fig2)
                    else:
                        t0 = time.perf_counter()
                        order, dists = bfs(g, src)
                        dt = (time.perf_counter() - t0) * 1000
                        if dst not in dists:
                            st.error("Шлях не існує")
                        else:
                            df = pd.DataFrame(
                                [(n, dists[n]) for n in order],
                                columns=["Вузол", "Дистанція (ребер)"])
                            st.success(
                                f"BFS завершено за {dt:.2f} мс. "
                                f"Відстань до '{dst}': {dists[dst]} ребер"
                            )
                            st.dataframe(df, use_container_width=True,
                                         hide_index=True)
                except Exception as ex:
                    st.error(f"Помилка: {ex}")

            st.divider()
            st.subheader("Матриця найкоротших відстаней (Дейкстра)")
            if st.checkbox("Обчислити повну матрицю"):
                nodes_list = sorted(g.nodes())
                mat = pd.DataFrame(index=nodes_list, columns=nodes_list,
                                   dtype=float)
                for s in nodes_list:
                    d, _ = dijkstra(g, s)
                    for t in nodes_list:
                        mat.loc[s, t] = (round(d[t], 2)
                                         if not math.isinf(d[t])
                                         else float("nan"))
                st.dataframe(mat, use_container_width=True)

    # ── Вкладка: Зв'язність ────────────────────────────────────────
    with tab_conn:
        st.subheader("Аналіз зв'язності та критичних компонентів")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Точки зчленування** — вузли, видалення яких "
                        "розділяє граф на компоненти.")
            try:
                aps = find_articulation_points(g)
                if aps:
                    st.warning(f"Знайдено {len(aps)}: " + ", ".join(aps))
                else:
                    st.success("Точок зчленування немає.")
            except Exception as ex:
                st.error(str(ex))

        with col_r:
            st.markdown("**Мости** — ребра, видалення яких збільшує "
                        "кількість компонент.")
            try:
                bridges = find_bridges(g)
                if bridges:
                    st.warning(f"Знайдено {len(bridges)}:")
                    st.dataframe(
                        pd.DataFrame(bridges, columns=["Вузол A", "Вузол B"]),
                        hide_index=True, use_container_width=True)
                else:
                    st.success("Мостів немає.")
            except Exception as ex:
                st.error(str(ex))

        st.divider()
        st.subheader("Обхід графа (BFS / DFS)")
        col_b, col_d = st.columns(2)
        nodes = sorted(g.nodes())
        with col_b:
            sb = st.selectbox("Стартовий вузол BFS", nodes, key="bfs_src")
            if st.button("Виконати BFS"):
                order, dists = bfs(g, sb)
                st.code(" → ".join(order))
                df = pd.DataFrame([(n, dists[n]) for n in order],
                                  columns=["Вузол", "Відстань"])
                st.dataframe(df, hide_index=True, use_container_width=True)
        with col_d:
            sd = st.selectbox("Стартовий вузол DFS", nodes, key="dfs_src")
            if st.button("Виконати DFS"):
                order = dfs(g, sd)
                st.code(" → ".join(order))

    # ── Вкладка: Імітація відмов ──────────────────────────────────
    with tab_fail:
        st.subheader("Імітація відмови вузла")
        if g.number_of_nodes() == 0:
            st.warning("Граф порожній.")
        else:
            target = st.selectbox("Вузол, що відмовляє",
                                  sorted(g.nodes()))
            if st.button("⚡ Імітувати відмову", type="primary"):
                try:
                    res = simulate_failure(g, target)
                    cA, cB = st.columns(2)
                    cA.metric("Компонент до",
                              res["before_components"])
                    cB.metric("Компонент після",
                              res["after_components"],
                              delta=res["after_components"] -
                                    res["before_components"])
                    if res["after_components"] > res["before_components"]:
                        st.error(
                            f"❌ Вузол '{target}' є критичним: "
                            f"мережу розділено на {res['after_components']} "
                            "сегментів"
                        )
                    else:
                        st.success(
                            f"✓ Мережа залишилась зв'язною "
                            f"({res['after_components']} компонент)"
                        )
                    fig3 = render_graph(res["graph"], removed_node=target,
                                        layout_name=layout_name,
                                        layout_seed=layout_seed,
                                        show_labels=show_labels)
                    st.pyplot(fig3)
                    st.write("**Утворені сегменти:**")
                    for i, group in enumerate(res["isolated_groups"], 1):
                        st.write(f"{i}. ({len(group)} вузлів) "
                                 + ", ".join(group))
                except Exception as ex:
                    st.error(f"Помилка: {ex}")

        st.divider()
        st.subheader("Аналіз стійкості: послідовне видалення вузлів")
        if st.button("Запустити аналіз"):
            results = []
            for n in sorted(g.nodes()):
                gtmp = g.copy()
                gtmp.remove_node(n)
                comp = nx.number_connected_components(gtmp) if gtmp.number_of_nodes() else 0
                results.append({
                    "Вузол": n,
                    "Тип": g.nodes[n].get("type", "?"),
                    "Ступінь": g.degree(n),
                    "Компонент після": comp,
                    "Критичний": "ТАК" if comp > 1 else "ні",
                })
            df = pd.DataFrame(results).sort_values(
                ["Критичний", "Ступінь"], ascending=[False, False])
            st.dataframe(df, hide_index=True, use_container_width=True)

    # ── Вкладка: Дані ──────────────────────────────────────────────
    with tab_data:
        st.subheader("Список вузлів")
        df_n = pd.DataFrame([
            {"Вузол": n, "Тип": d.get("type"), "Ступінь": g.degree(n)}
            for n, d in g.nodes(data=True)
        ])
        st.dataframe(df_n, hide_index=True, use_container_width=True)
        st.download_button("⬇ Експорт вузлів CSV",
                           data=df_n.to_csv(index=False).encode("utf-8"),
                           file_name="nodes.csv", mime="text/csv")

        st.subheader("Список ребер")
        df_e = pd.DataFrame([
            {"A": u, "B": v,
             "Bandwidth (Mbps)": d["bandwidth"],
             "Latency (ms)": d["latency"],
             "Weight": round(d["weight"], 3)}
            for u, v, d in g.edges(data=True)
        ])
        st.dataframe(df_e, hide_index=True, use_container_width=True)
        st.download_button("⬇ Експорт ребер CSV",
                           data=df_e.to_csv(index=False).encode("utf-8"),
                           file_name="edges.csv", mime="text/csv")


def console_demo() -> None:
    """Текстовий звіт для запуску `python network_app.py`."""
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 70)
    print("ANALIZ KORPORATYVNOI MEREZHI (consolnyi rezhym)")
    print("=" * 70)
    m = build_default_network()
    g = m.graph
    metrics = connectivity_metrics(g)
    print(f"\nNodes: {metrics['nodes']}, edges: {metrics['edges']}")
    print(f"Connected: {metrics['is_connected']}, "
          f"diameter: {metrics['diameter']}, "
          f"density: {metrics['density']:.3f}")

    print("\n--- BFS from FW ---")
    order, dists = bfs(g, "FW")
    print(" -> ".join(order))

    print("\n--- Dijkstra FW -> SRV_DB ---")
    dist, prev = dijkstra(g, "FW", "SRV_DB")
    path = reconstruct_path(prev, "SRV_DB")
    print(f"Path: {' -> '.join(path)}, cost: {dist['SRV_DB']:.2f}")

    print("\n--- Articulation points ---")
    print(", ".join(find_articulation_points(g)))

    print("\n--- Bridges ---")
    for a, b in find_bridges(g):
        print(f"  {a} -- {b}")

    fig = render_graph(g)
    fig.savefig("network_topology.png", dpi=150, bbox_inches="tight")
    print("\nDiagram saved as network_topology.png")


def _is_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _is_streamlit_runtime():
    main()
elif __name__ == "__main__":
    try:
        console_demo()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as ex:
        print(f"ERROR: {ex}")
        raise
