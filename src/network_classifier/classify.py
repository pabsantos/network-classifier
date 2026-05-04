"""Classify edges into clusters based on centrality metrics."""

import numpy as np
import networkx as nx
import skfuzzy as fuzz
from minisom import MiniSom
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler

METRICS = ("betweenness", "clustering", "degree")

HC_METHODS = {
    "hc_sl": "single",
    "hc_cl": "complete",
    "hc_ward": "ward",
    "hc_al": "average",
}

# Cap the number of samples used when computing silhouette scores.
# silhouette_score is O(n^2) in memory/time via pairwise distances, so we
# sub-sample to keep runtime bounded on large road networks. sklearn
# automatically uses all samples if n < SILHOUETTE_SAMPLE_SIZE.
SILHOUETTE_SAMPLE_SIZE = 5000


def _normalize_highway(hw) -> str:
    """Fold a raw OSM highway tag into its canonical class.

    Multi-typed edges (list values) collapse to their first entry; the
    ``_link`` suffix is stripped so e.g. ``primary_link`` joins ``primary``.
    """
    if isinstance(hw, list):
        hw = hw[0]
    if hw.endswith("_link"):
        hw = hw.removesuffix("_link")
    return hw


def _per_k_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    y_true: list[str],
    wcss: float,
) -> dict[str, float]:
    """Bundle silhouette, CHI, V-measure and WCSS for one k."""
    return {
        "silhouette": float(
            silhouette_score(
                X, labels, sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=42
            )
        ),
        "chi": float(calinski_harabasz_score(X, labels)),
        "v_measure": float(v_measure_score(y_true, labels)),
        "wcss": float(wcss),
    }


def _kmeans_eval(
    X: np.ndarray,
    y_true: list[str],
    k_range: range = range(2, 11),
) -> dict[int, dict[str, float]]:
    """Compute silhouette, CHI, V-measure and WCSS for KMeans at each k."""
    out: dict[int, dict[str, float]] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        out[k] = _per_k_metrics(X, labels, y_true, wcss=float(km.inertia_))
    return out


def _fkmeans_eval(
    X: np.ndarray,
    y_true: list[str],
    k_range: range = range(2, 11),
    m: float = 2.0,
) -> dict[int, dict[str, float]]:
    """Compute silhouette, CHI, V-measure and fuzzy objective for FCM at each k.

    The fuzzy objective ``jm[-1]`` is reported in the WCSS slot since it
    plays the same role for fuzzy clustering.
    """
    out: dict[int, dict[str, float]] = {}
    for k in k_range:
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X.T, c=k, m=m, error=0.005, maxiter=1000, seed=42
        )
        labels = np.argmax(u, axis=0)
        if len(set(labels)) < 2:
            continue
        out[k] = _per_k_metrics(X, labels, y_true, wcss=float(jm[-1]))
    return out


def _classify_with_hc(
    X: np.ndarray, linkage_method: str, n_clusters: int
) -> tuple[np.ndarray, dict[str, float], dict]:
    """Hierarchical (agglomerative) clustering via sklearn.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    linkage_method : str
        Linkage criterion: "single", "complete", "ward", or "average".
    n_clusters : int
        Number of clusters.
    """
    model = AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage_method, compute_distances=True
    )
    labels = model.fit_predict(X)

    model_metrics = {
        "linkage": linkage_method,
        "silhouette_score": float(
            silhouette_score(
                X, labels, sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=42
            )
        ),
        "calinski_harabasz_score": float(calinski_harabasz_score(X, labels)),
        "n_leaves": int(model.n_leaves_),
    }

    extras: dict = {
        "hc_model": model,
    }

    return labels, model_metrics, extras


def _classify_with_fkmeans(
    X: np.ndarray,
    n_clusters: int,
    y_true: list[str],
    m: float = 2.0,
) -> tuple[np.ndarray, dict[str, float], dict]:
    """Fuzzy K-Means clustering via skfuzzy cmeans.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    n_clusters : int
        Number of clusters.
    y_true : list[str]
        Ground-truth labels (highway classes) used for per-k V-measure.
    m : float
        Fuzziness exponent (default 2.0).

    Returns
    -------
    tuple
        (labels, model_metrics, extras)
    """
    performance_per_k = _fkmeans_eval(X, y_true, m=m)

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X.T, c=n_clusters, m=m, error=0.005, maxiter=1000, seed=42
    )
    labels = np.argmax(u, axis=0)

    model_metrics = {
        "fpc": float(fpc),
        "silhouette_score": float(
            silhouette_score(
                X, labels, sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=42
            )
        ),
        "calinski_harabasz_score": float(calinski_harabasz_score(X, labels)),
        "n_iter": int(p),
    }

    extras: dict = {
        "membership": u,
        "centers": cntr,
        "performance_per_k": performance_per_k,
    }

    return labels, model_metrics, extras


def _gmm_eval(
    X: np.ndarray, k_range: range = range(2, 11)
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Compute silhouette, BIC, and AIC for GMM at each k.

    Returns ``(silhouette_scores, bics, aics)`` mappings.
    """
    sil_scores: dict[int, float] = {}
    bics: dict[int, float] = {}
    aics: dict[int, float] = {}
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k, covariance_type="full", random_state=42
        )
        labels = gmm.fit_predict(X)
        bics[k] = float(gmm.bic(X))
        aics[k] = float(gmm.aic(X))
        if len(set(labels)) < 2:
            continue
        sil_scores[k] = float(
            silhouette_score(
                X, labels, sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=42
            )
        )
    return sil_scores, bics, aics


def _classify_with_gmm(
    X: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, dict[str, float], dict]:
    """Gaussian Mixture Model clustering via sklearn.

    Per-k BIC, AIC and silhouette scores (k=2..10) are computed and
    returned in ``extras`` to support diagnostic plots.
    """
    sil_scores, bics, aics = _gmm_eval(X)

    model = GaussianMixture(
        n_components=n_clusters, covariance_type="full", random_state=42
    )
    labels = model.fit_predict(X)

    model_metrics = {
        "bic": float(model.bic(X)),
        "aic": float(model.aic(X)),
        "log_likelihood": float(model.score(X) * X.shape[0]),
        "silhouette_score": float(
            silhouette_score(
                X, labels, sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=42
            )
        ),
        "calinski_harabasz_score": float(calinski_harabasz_score(X, labels)),
        "n_iter": int(model.n_iter_),
        "converged": bool(model.converged_),
    }

    extras: dict = {
        "gmm_model": model,
        "silhouette_scores": sil_scores,
        "bics": bics,
        "aics": aics,
    }

    return labels, model_metrics, extras


def _train_som(
    X: np.ndarray, random_seed: int = 42
) -> tuple[MiniSom, int]:
    """Train a SOM with a heuristic hexagonal grid.

    Grid side is derived from the rule of thumb ``5 * sqrt(N)`` total
    neurons, where N is the number of samples. The side is clamped to
    ``[5, 40]`` to keep training time bounded.
    """
    n_samples, n_features = X.shape
    total_neurons = 5.0 * np.sqrt(n_samples)
    grid_side = int(round(np.sqrt(total_neurons)))
    grid_side = max(5, min(40, grid_side))

    som = MiniSom(
        x=grid_side,
        y=grid_side,
        input_len=n_features,
        sigma=1.0,
        learning_rate=0.5,
        topology="hexagonal",
        random_seed=random_seed,
    )
    som.pca_weights_init(X)
    num_iteration = max(1000, 10 * n_samples)
    som.train(X, num_iteration, verbose=False)
    return som, grid_side


def _classify_with_som(
    X: np.ndarray, n_clusters: int, y_true: list[str]
) -> tuple[np.ndarray, dict[str, float], dict]:
    """Two-stage SOM clustering: train SOM, then KMeans on the codebook.

    Each sample is assigned the cluster of its Best Matching Unit (BMU).
    Per-k silhouette / CHI / V-measure / WCSS at sample level (k=2..10) are
    returned in ``extras`` to support diagnostic plots.
    """
    som, grid_side = _train_som(X)

    # Codebook: flatten the (x, y, n_features) weight grid
    weights = som.get_weights()
    codebook = weights.reshape(-1, weights.shape[-1])

    if n_clusters > codebook.shape[0]:
        raise ValueError(
            f"n_clusters ({n_clusters}) exceeds number of SOM neurons "
            f"({codebook.shape[0]})"
        )

    # Cache BMU coordinates for each sample once — the SOM itself is fixed
    # across the per-k sweep, so this avoids repeating O(N·neurons) lookups
    # for every k.
    bmu_coords = np.array([som.winner(x) for x in X])

    max_k = min(10, codebook.shape[0] - 1)
    performance_per_k: dict[int, dict[str, float]] = {}
    for k in range(2, max_k + 1):
        km_k = KMeans(n_clusters=k, random_state=42, n_init="auto")
        neuron_lbls = km_k.fit_predict(codebook)
        neuron_grid = neuron_lbls.reshape(grid_side, grid_side)
        sample_lbls = neuron_grid[bmu_coords[:, 0], bmu_coords[:, 1]]
        if len(set(sample_lbls)) < 2:
            continue
        performance_per_k[k] = _per_k_metrics(
            X, sample_lbls, y_true, wcss=float(km_k.inertia_)
        )

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    neuron_labels = km.fit_predict(codebook)
    neuron_label_grid = neuron_labels.reshape(grid_side, grid_side)
    sample_labels = neuron_label_grid[bmu_coords[:, 0], bmu_coords[:, 1]]

    if len(set(neuron_labels)) > 1:
        codebook_silhouette = float(silhouette_score(codebook, neuron_labels))
        codebook_ch = float(calinski_harabasz_score(codebook, neuron_labels))
    else:
        codebook_silhouette = 0.0
        codebook_ch = 0.0

    if len(set(sample_labels)) > 1:
        sample_ch = float(calinski_harabasz_score(X, sample_labels))
    else:
        sample_ch = 0.0

    model_metrics = {
        "quantization_error": float(som.quantization_error(X)),
        "topographic_error": float(som.topographic_error(X)),
        "grid_side": int(grid_side),
        "n_neurons": int(grid_side * grid_side),
        "kmeans_silhouette": codebook_silhouette,
        "kmeans_inertia": float(km.inertia_),
        "kmeans_calinski_harabasz_score": codebook_ch,
        "calinski_harabasz_score": sample_ch,
    }

    extras = {
        "som": som,
        "neuron_label_grid": neuron_label_grid,
        "grid_side": grid_side,
        "performance_per_k": performance_per_k,
    }

    return sample_labels, model_metrics, extras


def classify_edges(
    G: nx.MultiDiGraph, method: str, n_clusters: int
) -> tuple[nx.MultiDiGraph, dict[str, float], dict]:
    """Cluster edges by their centrality metrics.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Graph with "betweenness", "clustering", and "degree" edge
        attributes.
    method : str
        Clustering method: "kmeans", "som", "fkmeans", "gmm", or one of the
        hierarchical variants (hc_sl, hc_cl, hc_ward, hc_al).
    n_clusters : int
        Number of clusters to fit.

    Returns
    -------
    tuple[nx.MultiDiGraph, dict[str, float], dict]
        The graph with a "cluster" attribute on each edge, a dict of model
        metrics, and a dict of method-specific extras (e.g. the trained SOM
        and the neuron-to-cluster grid for som; per-k silhouette/inertia
        diagnostics for kmeans/fkmeans/som).
    """
    edge_order: list[tuple[int, int, int]] = []
    features: list[list[float]] = []
    y_true: list[str] = []

    for u, v, key, data in G.edges(keys=True, data=True):
        edge_order.append((u, v, key))
        features.append([data[m] for m in METRICS])
        y_true.append(_normalize_highway(data.get("highway", "unknown")))

    X = np.array(features)
    # Betweenness is heavily right-skewed; log1p compresses the tail
    # before any downstream scaling.
    bet_idx = METRICS.index("betweenness")
    X[:, bet_idx] = np.log1p(X[:, bet_idx])

    extras: dict = {}

    if method in HC_METHODS:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        labels, model_metrics, extras = _classify_with_hc(
            X_scaled, HC_METHODS[method], n_clusters
        )
    elif method == "som":
        # SOMs work better with bounded inputs.
        X_som = MinMaxScaler().fit_transform(X)
        labels, model_metrics, extras = _classify_with_som(
            X_som, n_clusters, y_true
        )
    elif method == "gmm":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        labels, model_metrics, extras = _classify_with_gmm(X_scaled, n_clusters)
    elif method == "fkmeans":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        labels, model_metrics, extras = _classify_with_fkmeans(
            X_scaled, n_clusters, y_true
        )
    elif method == "kmeans":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        performance_per_k = _kmeans_eval(X_scaled, y_true)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = model.fit_predict(X_scaled)
        model_metrics = {
            "inertia": float(model.inertia_),
            "silhouette_score": float(
                silhouette_score(
                    X_scaled,
                    labels,
                    sample_size=SILHOUETTE_SAMPLE_SIZE,
                    random_state=42,
                )
            ),
            "calinski_harabasz_score": float(
                calinski_harabasz_score(X_scaled, labels)
            ),
            "n_iter": int(model.n_iter_),
        }
        extras["performance_per_k"] = performance_per_k
    else:
        raise ValueError(f"Unknown method: {method}")

    for i, (u, v, key) in enumerate(edge_order):
        G[u][v][key]["cluster"] = int(labels[i])

    return G, model_metrics, extras


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


def highway_cluster_v_measure(G: nx.MultiDiGraph) -> float:
    """V-measure between OSM highway class (ground truth) and cluster label.

    When a highway attribute is a list (multi-typed edge), the first value is
    used. The ``_link`` suffix is stripped so e.g. ``primary_link`` is folded
    into ``primary``.
    """
    highways: list[str] = []
    clusters: list[int] = []

    for _u, _v, _key, data in G.edges(keys=True, data=True):
        highways.append(_normalize_highway(data.get("highway", "unknown")))
        clusters.append(int(data["cluster"]))

    return float(v_measure_score(highways, clusters))
