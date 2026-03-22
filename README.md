# network-classifier

CLI tool that downloads a city's road network via [OSMnx](https://osmnx.readthedocs.io/), computes centrality metrics using [NetworKit](https://networkit.github.io/), and optionally classifies road segments into clusters using [scikit-learn](https://scikit-learn.org/). Results are exported as GraphML or GeoPackage.

## Installation

Requires Python >= 3.12.

```bash
git clone https://github.com/pabsantos/network-classifier.git
cd network-classifier

# using uv
uv sync

# using pip
pip install .
```

## Usage

```bash
# Basic: compute centrality and export
network-classifier "Curitiba, Brazil" -f gpkg -o output.gpkg

# With clustering (GMM, auto-select best k)
network-classifier "Curitiba, Brazil" -f gpkg -o output.gpkg -m gmm

# With clustering (K-Means, fixed k)
network-classifier "Curitiba, Brazil" -f gpkg -o output.gpkg -m kmeans -k 5

# GraphML output
network-classifier "Curitiba, Brazil" -f graphml -o output.graphml -m gmm -k 4
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `city` | City name (e.g. `"Curitiba, Brazil"`) | required |
| `-f, --format` | Output format: `graphml` or `gpkg` | required |
| `-o, --output` | Output file path | derived from city name |
| `-n, --network-type` | Network type: `drive`, `walk`, `bike`, `all` | `drive` |
| `-m, --method` | Clustering method: `gmm` or `kmeans` | no clustering |
| `-k, --n-clusters` | Number of clusters | auto-select best k |

When `-k` is omitted, the best k is selected automatically by testing k=2..10:
- **K-Means**: highest silhouette score
- **GMM**: lowest BIC

## Output

### Edge attributes

Each edge in the exported graph contains:

| Attribute | Description |
|---|---|
| `betweenness` | Edge betweenness centrality (normalized) |
| `closeness` | Closeness centrality (mean of endpoints) |
| `degree` | Degree centrality (mean of endpoints) |
| `cluster` | Cluster label (only when `-m` is used) |

### Terminal output

When clustering is enabled, the tool prints:

**Model metrics** -- evaluation coefficients of the fitted model:
- K-Means: Inertia, Silhouette Score, Iterations
- GMM: BIC, AIC, Log-Likelihood, Iterations

**Cluster distribution** -- per-cluster summary (mean, std, min, max) of each centrality metric.

## License

See [LICENSE](LICENSE).
