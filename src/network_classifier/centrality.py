"""Compute edge centrality metrics using NetworKit."""

import networkit as nk
import networkx as nx


def compute_centrality(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Compute centrality metrics and store them as edge attributes.

    Computes edge betweenness centrality directly on edges, and closeness
    and degree centrality on nodes (assigned to edges as the mean of the
    two endpoints).

    Manually converts the NetworkX MultiDiGraph to a weighted NetworKit
    graph to maintain edge mapping (nk.nxadapter.nx2nk does not support
    MultiDiGraph edge traceability).

    Parameters
    ----------
    G : nx.MultiDiGraph
        Street network graph (from OSMnx).

    Returns
    -------
    nx.MultiDiGraph
        The same graph with "betweenness", "closeness", and "degree"
        attributes on each edge.
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

    # Step 4: Closeness centrality (node-level)
    cc = nk.centrality.Closeness(
        nk_graph, True, nk.centrality.ClosenessVariant.Generalized
    )
    cc.run()
    closeness_scores = cc.scores()

    # Step 5: Degree centrality (node-level)
    dc = nk.centrality.DegreeCentrality(nk_graph, normalized=False)
    dc.run()
    degree_scores = dc.scores()

    # Step 6: Assign node metrics to edges as mean of endpoints
    for u, v, key in G.edges(keys=True):
        u_idx = node_id_to_idx[u]
        v_idx = node_id_to_idx[v]
        G[u][v][key]["closeness"] = (closeness_scores[u_idx] + closeness_scores[v_idx]) / 2
        G[u][v][key]["degree"] = (degree_scores[u_idx] + degree_scores[v_idx]) / 2

    return G
