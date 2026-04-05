"""Classify edges into clusters based on centrality metrics."""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

METRICS = ("betweenness", "closeness", "degree")


def _find_best_k(
    X: np.ndarray, method: str, k_range: range = range(2, 11)
) -> int:
    """Select the best number of clusters automatically.

    K-Means: highest silhouette score.
    GMM: lowest BIC.
    """
    if method == "kmeans":
        best_k, best_score = 2, -1.0
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score = k, score
        return best_k

    # GMM — lowest BIC
    best_k, best_bic = 2, np.inf
    for k in k_range:
        gm = GaussianMixture(n_components=k, random_state=42)
        gm.fit(X)
        bic = gm.bic(X)
        if bic < best_bic:
            best_k, best_bic = k, bic
    return best_k


def classify_edges(
    G: nx.MultiDiGraph, method: str, n_clusters: int | None = None
) -> tuple[nx.MultiDiGraph, int, dict[str, float]]:
    """Cluster edges by their centrality metrics.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Graph with "betweenness", "closeness", and "degree" edge attributes.
    method : str
        Clustering method: "kmeans" or "gmm".
    n_clusters : int or None
        Number of clusters. If None, the best k is selected automatically.

    Returns
    -------
    tuple[nx.MultiDiGraph, int, dict[str, float]]
        The graph with a "cluster" attribute on each edge, the number of
        clusters used, and a dict of model metrics.
    """
    edge_order: list[tuple[int, int, int]] = []
    features: list[list[float]] = []

    for u, v, key, data in G.edges(keys=True, data=True):
        edge_order.append((u, v, key))
        features.append([data[m] for m in METRICS])

    X = np.array(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if n_clusters is None:
        n_clusters = _find_best_k(X_scaled, method)

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = model.fit_predict(X_scaled)
        model_metrics = {
            "inertia": float(model.inertia_),
            "silhouette_score": float(silhouette_score(X_scaled, labels)),
            "n_iter": int(model.n_iter_),
        }
    else:
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
        model_metrics = {
            "bic": float(model.bic(X_scaled)),
            "aic": float(model.aic(X_scaled)),
            "log_likelihood": float(model.score(X_scaled) * X_scaled.shape[0]),
            "n_iter": int(model.n_iter_),
        }

    for i, (u, v, key) in enumerate(edge_order):
        G[u][v][key]["cluster"] = int(labels[i])

    return G, n_clusters, model_metrics


def cluster_summary(
    G: nx.MultiDiGraph,
) -> dict[int, dict[str, dict[str, float]]]:
    """Compute per-cluster summary statistics for each centrality metric.

    Returns
    -------
    dict
        ``{cluster_id: {metric: {count, mean, std, min, max}}}``
    """
    clusters: dict[int, dict[str, list[float]]] = {}

    for _u, _v, _key, data in G.edges(keys=True, data=True):
        cid = data["cluster"]
        if cid not in clusters:
            clusters[cid] = {m: [] for m in METRICS}
        for m in METRICS:
            clusters[cid][m].append(data[m])

    summary: dict[int, dict[str, dict[str, float]]] = {}
    for cid in sorted(clusters):
        summary[cid] = {}
        for m in METRICS:
            vals = np.array(clusters[cid][m])
            summary[cid][m] = {
                "count": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    return summary


def highway_cluster_crosstab(
    G: nx.MultiDiGraph,
) -> pd.DataFrame:
    """Build a cross-tabulation between OSM highway class and cluster label,
    weighted by edge length (km).

    When a highway attribute is a list (multi-typed edge), the first value is
    used.

    Returns
    -------
    pd.DataFrame
        Rows = highway classes, columns = cluster IDs, values = total length
        in km.
    """
    records: list[dict] = []

    for _u, _v, _key, data in G.edges(keys=True, data=True):
        hw = data.get("highway", "unknown")
        if isinstance(hw, list):
            hw = hw[0]
        if hw.endswith("_link"):
            hw = hw.removesuffix("_link")
        records.append({
            "highway": hw,
            "cluster": data["cluster"],
            "length_km": data.get("length", 0.0) / 1000.0,
        })

    df = pd.DataFrame(records)
    ct = df.pivot_table(
        index="highway",
        columns="cluster",
        values="length_km",
        aggfunc="sum",
        fill_value=0.0,
    )
    ct.columns.name = "cluster"
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
    return ct
