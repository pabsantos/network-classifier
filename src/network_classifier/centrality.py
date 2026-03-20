"""Compute edge betweenness centrality using NetworKit."""

import networkit as nk
import networkx as nx


def compute_edge_betweenness(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Compute normalized edge betweenness centrality and store it on the graph.

    Manually converts the NetworkX MultiDiGraph to a NetworKit graph to
    maintain edge mapping (nk.nxadapter.nx2nk does not support MultiDiGraph
    edge traceability).

    Parameters
    ----------
    G : nx.MultiDiGraph
        Street network graph (from OSMnx).

    Returns
    -------
    nx.MultiDiGraph
        The same graph with a "betweenness" attribute on each edge.
    """
    # Step 1: Map OSM node IDs to 0-based NetworKit indices
    node_id_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
    n = len(node_id_to_idx)

    # Step 2: Build directed NetworKit graph and track edge insertion order
    nk_graph = nk.Graph(n, directed=True)
    edge_order: list[tuple[int, int, int]] = []

    for u, v, key in G.edges(keys=True):
        nk_graph.addEdge(node_id_to_idx[u], node_id_to_idx[v])
        edge_order.append((u, v, key))

    # Step 3: Index edges and compute betweenness
    nk_graph.indexEdges()
    bc = nk.centrality.Betweenness(
        nk_graph, normalized=True, computeEdgeCentrality=True
    )
    bc.run()

    # Step 4: Write scores back to the original graph
    edge_scores = bc.edgeScores()
    for nk_edge_id, (u, v, key) in enumerate(edge_order):
        G[u][v][key]["betweenness"] = edge_scores[nk_edge_id]

    return G
