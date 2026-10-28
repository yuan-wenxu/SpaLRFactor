"""Gaussian neighborhood LR scoring, independently within each sample."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def neighborhood_graph(
    coordinates: np.ndarray,
    radius: float,
    sigma: float,
    include_self: bool = False,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return row-normalized Gaussian weights and unweighted neighbor counts.

    Coordinates, radius and sigma must all be in micrometers. Only
    neighbors at distance <= radius contribute. With include_self=False,
    isolated rows have zero total weight and zero neighbors.
    """
    radius = _positive(radius, "radius")
    sigma = _positive(sigma, "sigma")
    coordinates = np.asarray(coordinates, dtype=float)
    if (coordinates.ndim != 2 or coordinates.shape[1] != 2
            or not len(coordinates) or not np.isfinite(coordinates).all()):
        raise ValueError("Coordinates must be a nonempty finite (n, 2) array")
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("Duplicate spot coordinates within a sample")
    edges = cKDTree(coordinates).query_pairs(radius, output_type="ndarray")
    distance = np.linalg.norm(
        coordinates[edges[:, 0]] - coordinates[edges[:, 1]], axis=1,
    )
    with np.errstate(over="ignore"):
        weights = np.exp(-0.5 * (distance / sigma) ** 2)
    if np.any(weights == 0):
        raise ValueError("Gaussian weights underflow; increase sigma")
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    columns = np.concatenate([edges[:, 1], edges[:, 0]])
    graph = sparse.csr_matrix(
        (np.tile(weights, 2), (rows, columns)),
        shape=(len(coordinates), len(coordinates)),
    )
    if include_self:
        graph.setdiag(1.0)
    n_neighbors = np.diff(graph.indptr)
    totals = np.asarray(graph.sum(axis=1)).ravel()
    inverse = np.divide(
        1.0, totals, out=np.zeros_like(totals), where=totals > 0,
    )
    return (sparse.diags(inverse) @ graph).tocsr(), n_neighbors


def _prepare_pairs(pairs: pd.DataFrame, features: pd.Index) -> pd.DataFrame:
    required = ["lr_pair", "ligand_feature_id", "receptor_feature_id"]
    if not set(required).issubset(pairs.columns):
        raise ValueError(f"Pair table requires columns: {required}")
    pairs = pairs.copy().reset_index(drop=True)
    if pairs.empty or pairs[required].isna().any().any():
        raise ValueError("Pair table is empty or has missing identifiers")
    for column in required:
        pairs[column] = pairs[column].astype(str)
        if pairs[column].str.strip().eq("").any():
            raise ValueError(f"Empty identifiers in {column}")
    if pairs["lr_pair"].duplicated().any():
        raise ValueError("lr_pair identifiers must be unique")
    all_genes = []
    for column in required[1:]:
        entities = []
        for value in pairs[column]:
            subunits = [gene.strip() for gene in value.split(";")]
            if not all(subunits):
                raise ValueError(f"Empty subunit IDs in {column}: {value}")
            if any("|" in gene for gene in subunits):
                raise ValueError(
                    f"Ambiguous subunit IDs in {column}: {value}"
                )
            if len(set(subunits)) != len(subunits):
                raise ValueError(
                    f"Duplicate subunit IDs in {column}: {value}"
                )
            entities.append(";".join(subunits))
            all_genes.extend(subunits)
        pairs[column] = entities
    genes = pd.Index(pd.unique(np.asarray(all_genes, dtype=str)))
    missing = genes.difference(features)
    if len(missing):
        raise ValueError(f"Unmatched feature IDs: {missing.tolist()}")
    return pairs


def _expression(
    data: ad.AnnData, pairs: pd.DataFrame, layer: str | None,
    target_sum: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if layer is not None and layer not in data.layers:
        raise ValueError(f"Missing counts layer: {layer}")
    raw = data.X if layer is None else data.layers[layer]
    counts = sparse.csr_matrix(raw, dtype=float, copy=True)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    if not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    if not np.allclose(counts.data, np.rint(counts.data), rtol=0, atol=1e-6):
        raise ValueError("Expected raw integer counts, not transformed data")
    total_counts_per_cell = np.asarray(counts.sum(axis=1)).ravel()
    if (not np.isfinite(total_counts_per_cell).all()
            or (total_counts_per_cell <= 0).any()):
        raise ValueError(
            "Every spot must have a positive finite count library"
        )
    entities = pd.Index(pd.unique(pairs[[
        "ligand_feature_id", "receptor_feature_id",
    ]].to_numpy().ravel()))
    subunit_lists = [entity.split(";") for entity in entities]
    genes = pd.Index(pd.unique(np.asarray([
        gene for subunits in subunit_lists for gene in subunits
    ], dtype=str)))
    values = counts[:, data.var_names.get_indexer(genes)].toarray()
    values /= total_counts_per_cell[:, None]
    values *= target_sum
    np.log1p(values, out=values)
    expression = np.empty((data.n_obs, len(entities)))
    for index, subunits in enumerate(subunit_lists):
        indices = genes.get_indexer(subunits)
        expression[:, index] = values[:, indices].min(axis=1)
    ligand_indices = entities.get_indexer(pairs["ligand_feature_id"])
    receptor_indices = entities.get_indexer(pairs["receptor_feature_id"])
    return (
        expression, ligand_indices, receptor_indices, total_counts_per_cell,
    )


def score_lr(
    data: ad.AnnData,
    pairs: pd.DataFrame,
    *,
    layer: str | None = "counts",
    radius: float = 250.0,
    sigma: float = 125.0,
    include_self: bool = False,
    target_sum: float = 1e4,
) -> ad.AnnData:
    """Compute both directional scores without modifying the input AnnData.

    Expression E = log1p(counts / total_counts_per_cell * target_sum).
    Feature IDs for required subunits are separated by semicolons in
    ligand_feature_id and receptor_feature_id. Each entity uses the minimum
    E across its subunits, assuming 1:1 assembly. Complex expression is
    computed within each spot before neighborhood averaging. A zero in any
    required subunit gives zero complex expression; missing feature IDs
    raise an error.

    With row-normalized Gaussian neighborhood weights W:
        outgoing[i, p] = E[i, ligand[p]] * (W @ E[:, receptor[p]])[i]
        incoming[i, p] = E[i, receptor[p]] * (W @ E[:, ligand[p]])[i]

    The LR pair retains its biological ligand-to-receptor direction in both
    matrices. Incoming represents the focal spot as receiver. Cell IDs are
    taken from obs_names. obs['x_um'] and obs['y_um'] must contain physical
    coordinates in micrometers; radius and sigma also use micrometers.
    If obs['sample'] exists, graphs stay within each sample.
    Otherwise, the whole object is treated as a single sample.

    Returns AnnData with cells as rows and LR pairs as columns, preserving
    their input order. X contains incoming scores and layers['outgoing']
    contains outgoing scores. obs and var retain cell and pair metadata;
    obsm['spatial'] contains micrometer coordinates. Unavailable rows contain
    NaN and must be excluded before matrix factorization.
    """
    radius = _positive(radius, "radius")
    sigma = _positive(sigma, "sigma")
    target_sum = _positive(target_sum, "target_sum")
    if not data.n_obs or not data.n_vars:
        raise ValueError("Expression data must be nonempty")
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError("Spot IDs and feature IDs must be unique")
    if not data.obs.columns.is_unique or not pairs.columns.is_unique:
        raise ValueError("Observation and pair column names must be unique")
    pairs = _prepare_pairs(pairs, data.var_names)
    observations = data.obs.copy()
    if "sample" not in observations:
        samples = np.repeat("sample_0", data.n_obs)
    else:
        if observations["sample"].isna().any():
            raise ValueError("Sample labels must not contain missing values")
        samples = observations["sample"].astype(str).to_numpy()
        if len(pd.unique(samples)) != observations["sample"].nunique():
            raise ValueError("Sample labels collide when converted to strings")
    coordinate_columns = ["x_um", "y_um"]
    if not set(coordinate_columns).issubset(observations):
        raise ValueError(
            "obs must contain x_um and y_um coordinates in micrometers"
        )
    try:
        coordinates = observations[coordinate_columns].to_numpy(
            dtype=float, copy=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "x_um and y_um must be numeric micrometer coordinates"
        ) from error
    if not np.isfinite(coordinates).all():
        raise ValueError("Spatial coordinates must be finite")
    (
        expression, ligand_indices, receptor_indices, total_counts_per_cell,
    ) = _expression(data, pairs, layer, target_sum)
    observations["total_counts_per_cell"] = total_counts_per_cell
    observations["x_um"] = coordinates[:, 0]
    observations["y_um"] = coordinates[:, 1]
    all_n_neighbors = np.zeros(data.n_obs, dtype=int)
    shape = (data.n_obs, len(pairs))
    outgoing = np.full(shape, np.nan)
    incoming = np.full(shape, np.nan)
    for sample in pd.unique(samples):
        locations = np.flatnonzero(samples == sample)
        graph, n_neighbors = neighborhood_graph(
            coordinates[locations], radius, sigma, include_self,
        )
        all_n_neighbors[locations] = n_neighbors
        valid = n_neighbors > 0
        values = expression[locations]
        neighbor_values = graph @ values
        ligand = values[:, ligand_indices]
        receptor = values[:, receptor_indices]
        neighbor_ligand = neighbor_values[:, ligand_indices]
        neighbor_receptor = neighbor_values[:, receptor_indices]
        outgoing[locations[valid]] = (
            ligand[valid] * neighbor_receptor[valid]
        )
        incoming[locations[valid]] = (
            receptor[valid] * neighbor_ligand[valid]
        )
    observations["n_neighbors"] = all_n_neighbors
    observations["score_available"] = all_n_neighbors > 0

    variables = pairs.set_index("lr_pair", drop=False)
    variables.index.name = None
    return ad.AnnData(
        X=incoming,
        obs=observations,
        var=variables,
        layers={"outgoing": outgoing},
        obsm={"spatial": coordinates},
    )
