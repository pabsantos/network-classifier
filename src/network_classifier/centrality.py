"""Compute edge centrality metrics using NetworKit."""

import networkit as nk
import networkx as nx


def compute_centrality(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Compute centrality metrics and store them as edge attributes.

    Computes edge betweenness centrality directly on edges. Local
    clustering coefficient is assigned to edges as the mean of the two
    endpoints; degree (non-normalized) is assigned as the sum of the two
    endpoints.

    Manually converts the NetworkX MultiDiGraph to a weighted NetworKit
    graph to maintain edge mapping (nk.nxadapter.nx2nk does not support
    MultiDiGraph edge traceability). A separate undirected unweighted
    graph is built for the clustering coefficient (triangles are a
    topological property, undefined on directed multigraphs).

    Parameters
    ----------
    G : nx.MultiDiGraph
        Street network graph (from OSMnx).

    Returns
    -------
    nx.MultiDiGraph
        The same graph with "betweenness", "clustering", and "degree"
        attributes on each edge. Edge "length" is already set by OSMnx.
    """
    # Step 1: Map OSM node IDs to 0-based NetworKit indices
    node_id_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
    n = len(node_id_to_idx)

    # Step 2: Build weighted directed NetworKit graph and track edge order
    nk_graph = nk.Graph(n, directed=True, weighted=True)
    edge_order: list[tuple[int, int, int]] = []

    for u, v, key, data in G.edges(keys=True, data=True):
        weight = data.get("length", 1.0)
        nk_graph.addEdge(node_id_to_idx[u], node_id_to_idx[v], w=weight)
        edge_order.append((u, v, key))

    nk_graph.indexEdges()

    # Step 3: Edge betweenness centrality
    bc = nk.centrality.Betweenness(
        nk_graph, normalized=True, computeEdgeCentrality=True
    )
    bc.run()
    edge_scores = bc.edgeScores()
    for nk_edge_id, (u, v, key) in enumerate(edge_order):
        G[u][v][key]["betweenness"] = edge_scores[nk_edge_id]

    # Step 4: Local clustering coefficient (node-level). Requires an
    # undirected unweighted simple graph — build one alongside.
    nk_undirected = nk.Graph(n, directed=False, weighted=False)
    added: set[tuple[int, int]] = set()
    for u, v, _key in G.edges(keys=True):
        u_idx, v_idx = node_id_to_idx[u], node_id_to_idx[v]
        if u_idx == v_idx:
            continue
        edge = (u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx)
        if edge in added:
            continue
        nk_undirected.addEdge(u_idx, v_idx)
        added.add(edge)

    lcc = nk.centrality.LocalClusteringCoefficient(nk_undirected)
    lcc.run()
    clustering_scores = lcc.scores()

    # Step 5: Total degree per node (in + out, non-normalized).
    # DegreeCentrality on a directed graph only counts out-edges, so
    # sum in- and out-degree explicitly.
    degree_scores = [
        nk_graph.degreeIn(i) + nk_graph.degreeOut(i) for i in range(n)
    ]

    # Step 6: Assign node metrics to edges as mean of endpoints
    for u, v, key in G.edges(keys=True):
        u_idx = node_id_to_idx[u]
        v_idx = node_id_to_idx[v]
        G[u][v][key]["clustering"] = (clustering_scores[u_idx] + clustering_scores[v_idx]) / 2
        G[u][v][key]["degree"] = degree_scores[u_idx] + degree_scores[v_idx]

    return G
