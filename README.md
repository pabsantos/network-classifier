# network-classifier

CLI tool that downloads a city's road network, computes centrality metrics, and optionally classifies road segments into clusters. Results are exported as GraphML or GeoPackage.

## Installation

Requires Python >= 3.12.

### From GitHub (via pip)

```bash
pip install git+https://github.com/pabsantos/network-classifier.git
```

### From source

```bash
git clone https://github.com/pabsantos/network-classifier.git
cd network-classifier

# using uv (recommended)
uv sync

# or using pip
pip install .

# editable mode (development)
pip install -e ".[test]"
```

## Usage

### Data source

The study area can be defined in three ways:

```bash
# By city name
network-classifier "Curitiba, Brazil" -f gpkg -o output.gpkg

# By bounding box (west,south,east,north)
network-classifier --bbox -49.39,-25.60,-49.18,-25.35 -f gpkg -o output.gpkg

# By GeoJSON polygon
network-classifier --polygon boundary.geojson -f gpkg -o output.gpkg
```

### Clustering methods

```bash
# No clustering (centrality only)
network-classifier "Curitiba, Brazil" -f gpkg -o output.gpkg

# K-Means (auto-select k)
network-classifier "Curitiba, Brazil" -f gpkg -o output.gpkg -m kmeans

# K-Means (fixed k)
network-classifier "Curitiba, Brazil" -f gpkg -o output.gpkg -m kmeans -k 5

# Fuzzy K-Means
network-classifier "Curitiba, Brazil" -f gpkg -o output.gpkg -m fkmeans -k 5

# SOM (Self-Organizing Map) + K-Means
network-classifier "Curitiba, Brazil" -f gpkg -o output.gpkg -m som -k 5

# Hierarchical - Single Linkage
network-classifier "Curitiba, Brazil" -f gpkg -o output.gpkg -m hc_sl

# Hierarchical - Complete Linkage
network-classifier "Curitiba, Brazil" -f gpkg -o output.gpkg -m hc_cl

# Hierarchical - Ward
network-classifier "Curitiba, Brazil" -f gpkg -o output.gpkg -m hc_ward

# Hierarchical - Average Linkage
network-classifier "Curitiba, Brazil" -f gpkg -o output.gpkg -m hc_al
```

When `-k` is omitted, the best k is selected automatically:
- **K-Means, Fuzzy K-Means, and SOM**: highest silhouette score (k=2..10)
- **Hierarchical**: largest gap in merge distances from the dendrogram (k=2..10)

### Arguments

| Argument | Description | Default |
|---|---|---|
| `city` | City name (e.g. `"Curitiba, Brazil"`) | - |
| `--bbox` | Bounding box: `west,south,east,north` | - |
| `--polygon` | Path to a GeoJSON file | - |
| `-f, --format` | Output format: `graphml` or `gpkg` | required |
| `-o, --output` | Output file path | derived from input name |
| `-m, --method` | Clustering method (see table below) | no clustering |
| `-k, --n-clusters` | Number of clusters | auto-select |

| Method | Flag | Description |
|---|---|---|
| K-Means | `kmeans` | Classic partitional clustering |
| Fuzzy K-Means | `fkmeans` | Fuzzy clustering via FCM |
| SOM | `som` | Self-Organizing Map + K-Means on codebook |
| Single Linkage | `hc_sl` | Hierarchical - nearest neighbor |
| Complete Linkage | `hc_cl` | Hierarchical - farthest neighbor |
| Ward | `hc_ward` | Hierarchical - minimizes within-cluster variance |
| Average Linkage | `hc_al` | Hierarchical - average distance |

## Output

### Edge attributes

Each edge in the exported graph contains:

| Attribute | Description |
|---|---|
| `betweenness` | Edge betweenness centrality (normalized) |
| `clustering` | Local clustering coefficient (mean of endpoints) |
| `degree` | Degree (sum of endpoints) |
| `cluster` | Cluster label (only when `-m` is used) |

### Model metrics

When clustering is enabled, metrics are printed to the terminal and saved to `output/model_metrics.txt`:

| Method | Metrics |
|---|---|
| K-Means | Inertia, Silhouette Score, Iterations |
| Fuzzy K-Means | FPC (Fuzzy Partition Coefficient), Silhouette Score, Iterations |
| SOM | Quantization Error, Topographic Error, Grid Side, Neurons, KMeans Silhouette/Inertia |
| Hierarchical | Linkage Method, Silhouette Score, Leaves |

### Generated plots

Plots are saved to the `output/` directory:

| File | Methods | Description |
|---|---|---|
| `*_kde.png` | All | KDE distribution per metric and cluster |
| `map.png` | All | Map of road segments colored by cluster |
| `silhouette_vs_k.png` | kmeans, fkmeans, som | Silhouette score vs k |
| `elbow.png` | kmeans, fkmeans, som | Elbow method (inertia vs k) |
| `dendrogram.png` | hc_* | Dendrogram with cut line |
| `umatrix.png` | som | U-Matrix and neuron cluster assignments |
| `highway_cluster_heatmap.png` | All | Highway class x cluster heatmap |

## Contributing

### Environment setup

```bash
git clone https://github.com/pabsantos/network-classifier.git
cd network-classifier
uv sync --extra test
```

### Running tests

```bash
uv run pytest tests/ -v
```

### Project structure

```
src/network_classifier/
  cli.py          # Entry point (argparse)
  graph.py        # Network download via OSMnx
  centrality.py   # Centrality computation via NetworKit
  classify.py     # Clustering methods
  export.py       # GraphML/GPKG/TXT export
  plots.py        # Plot generation
tests/
  conftest.py     # Shared fixtures
  test_*.py       # Tests per module
```

## License

This project is licensed under GPL-3.0. See [LICENSE](LICENSE).
