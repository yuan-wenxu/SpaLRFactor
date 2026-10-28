"""Repeated NMF fitting and pattern-count evaluation for LR score matrices."""

from __future__ import annotations

from itertools import combinations
from numbers import Integral
from typing import Iterable
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _controls(
    n_repeats: int, max_iter: int, tol: float,
    random_state: int | None, *, minimum_repeats: int,
) -> tuple[int, int, float, int | None]:
    n_repeats = _integer(n_repeats, "n_repeats", minimum_repeats)
    max_iter = _integer(max_iter, "max_iter")
    tol = float(tol)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    if random_state is not None:
        random_state = _integer(random_state, "random_state", 0)
    return n_repeats, max_iter, tol, random_state


def _prepare_matrix(data, layer: str | None):
    if isinstance(data, ad.AnnData):
        if layer is not None and layer not in data.layers:
            raise ValueError(f"Missing score layer: {layer}")
        raw = data.X if layer is None else data.layers[layer]
        spot_ids, lr_ids = data.obs_names.copy(), data.var_names.copy()
    elif layer is not None:
        raise ValueError("layer can only be specified for AnnData input")
    elif isinstance(data, pd.DataFrame):
        raw = data.to_numpy()
        spot_ids, lr_ids = data.index.copy(), data.columns.copy()
    else:
        raw = data
        spot_ids = lr_ids = None
    if np.iscomplexobj(raw):
        raise ValueError("NMF requires real-valued scores")
    if sparse.issparse(raw):
        matrix = sparse.csr_matrix(raw, dtype=float, copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        values = matrix.data
    else:
        matrix = np.array(raw, dtype=float, copy=True)
        values = matrix
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("Scores must be a nonempty 2D matrix")
    if not np.isfinite(values).all():
        raise ValueError(
            "Scores must be finite. Filter unavailable score rows before "
            "NMF; do not replace missing scores with zero."
        )
    if (values < 0).any():
        raise ValueError("NMF scores must be nonnegative")
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Scores must have a positive finite Frobenius norm")
    if spot_ids is None:
        spot_ids = pd.Index([
            f"spot_{i + 1}" for i in range(matrix.shape[0])
        ])
        lr_ids = pd.Index([f"lr_{i + 1}" for i in range(matrix.shape[1])])
    if not spot_ids.is_unique or not lr_ids.is_unique:
        raise ValueError("Spot and LR identifiers must be unique")
    return matrix, spot_ids, lr_ids, norm


def _rank(n_patterns: int, shape: tuple[int, int]) -> int:
    n_patterns = _integer(n_patterns, "n_patterns")
    if n_patterns > min(shape):
        raise ValueError("n_patterns must not exceed min(n_spots, n_LR)")
    return n_patterns


def _normalize_factors(weights, loadings):
    """Unit L2 norm per active H row, preserving W @ H exactly in scale."""
    scales = np.linalg.norm(loadings, axis=1)
    active = (scales > 0) & np.any(weights > 0, axis=0)
    loadings = np.divide(
        loadings, scales[:, None], out=np.zeros_like(loadings),
        where=scales[:, None] > 0,
    )
    weights = weights * scales[None, :]
    loadings[~active] = 0
    weights[:, ~active] = 0
    return weights, loadings, int(active.sum())


def _fit_runs(
    matrix, n_patterns, n_repeats, max_iter, tol, random_state, norm,
    *, keep_loadings,
):
    children = np.random.SeedSequence(
        random_state, spawn_key=(n_patterns,),
    ).spawn(n_repeats)
    records, repeated_loadings = [], []
    best_error, best_factors = np.inf, None
    for repeat, child in enumerate(children, start=1):
        seed = int(child.generate_state(1)[0])
        model = NMF(
            n_components=n_patterns, init="random", solver="cd",
            beta_loss="frobenius", max_iter=max_iter, tol=tol,
            random_state=seed, alpha_W=0.0, alpha_H=0.0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            weights = model.fit_transform(matrix)
        for warning in caught:
            if not issubclass(warning.category, ConvergenceWarning):
                warnings.warn(
                    str(warning.message), warning.category, stacklevel=3,
                )
        converged = not any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        )
        loadings = model.components_
        error = float(model.reconstruction_err_)
        if (not np.isfinite(error) or not np.isfinite(weights).all()
                or not np.isfinite(loadings).all()):
            raise RuntimeError("NMF produced nonfinite factors or error")
        weights, loadings, active = _normalize_factors(weights, loadings)
        records.append({
            "n_patterns": n_patterns, "repeat": repeat, "seed": seed,
            "n_iter": int(model.n_iter_), "converged": converged,
            "active_patterns": active, "reconstruction_error": error,
            "relative_error": error / norm,
        })
        if keep_loadings:
            repeated_loadings.append(loadings)
        if error < best_error:
            best_error = error
            best_factors = weights, loadings
    return best_factors, pd.DataFrame(records), repeated_loadings


def _pattern_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Match unit-norm H rows one-to-one; inactive rows contribute zero."""
    similarity = np.clip(left @ right.T, 0.0, 1.0)
    rows, columns = linear_sum_assignment(similarity, maximize=True)
    return float(similarity[rows, columns].mean())


def _warn_unconverged(runs: pd.DataFrame) -> None:
    count = int((~runs["converged"]).sum())
    if count:
        warnings.warn(
            f"{count} of {len(runs)} NMF fits did not converge. "
            "Inspect run diagnostics and consider increasing max_iter.",
            ConvergenceWarning, stacklevel=3,
        )


def evaluate_nmf(
    data,
    n_patterns: Iterable[int],
    *,
    layer: str | None = None,
    n_repeats: int = 100,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare pattern counts using repeated random-initialized NMF fits.

    Input is a spots-by-LR AnnData, DataFrame, dense array or sparse matrix.
    For AnnData, layer=None selects X (incoming scores from score_lr);
    layer='outgoing' selects the other direction. No scaling, centering,
    imputation, sampling or automatic filtering is applied.

    Each rank has n_repeats independent starts, each using at most max_iter
    coordinate-descent iterations to minimize Frobenius reconstruction
    loss. Seeds depend on random_state and rank, not the rank-list order.

    Stability is the mean matched cosine similarity of LR loading patterns
    across all pairs of runs. Hungarian matching handles pattern-label
    permutations, and unit-norm loadings remove scale ambiguity. Higher is
    more reproducible; inactive patterns contribute zero similarity. This
    measures initialization stability, not biological reproducibility.

    Relative error is ||X-WH||_F / ||X||_F, not a classification error rate.
    The summary table contains stability, reconstruction-error summaries,
    iteration counts, convergence and inactive-pattern diagnostics. Means
    and population standard deviations include unconverged runs; these are
    flagged and generate a warning. No pattern count is selected for you.

    Returns (summary, diagnostics): one row per pattern count in summary,
    and one row per fit in diagnostics. At least two repeats are required.
    """
    controls = _controls(
        n_repeats, max_iter, tol, random_state, minimum_repeats=2,
    )
    n_repeats, max_iter, tol, random_state = controls
    matrix, _, _, norm = _prepare_matrix(data, layer)
    try:
        ranks = [_rank(value, matrix.shape) for value in n_patterns]
    except TypeError as error:
        raise ValueError("n_patterns must be an iterable of integers") from error
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("n_patterns must be nonempty and contain no duplicates")
    summaries, all_runs = [], []
    for rank in ranks:
        _, runs, loadings = _fit_runs(
            matrix, rank, n_repeats, max_iter, tol, random_state, norm,
            keep_loadings=True,
        )
        similarities = [
            _pattern_similarity(left, right)
            for left, right in combinations(loadings, 2)
        ]
        best = runs.loc[runs["reconstruction_error"].idxmin()]
        summaries.append({
            "n_patterns": rank, "n_repeats": n_repeats,
            "stability": float(np.mean(similarities)),
            "stability_std": float(np.std(similarities)),
            "relative_error_mean": float(runs["relative_error"].mean()),
            "relative_error_std": float(runs["relative_error"].std(ddof=0)),
            "relative_error_min": float(best["relative_error"]),
            "reconstruction_error_mean": float(
                runs["reconstruction_error"].mean()
            ),
            "reconstruction_error_min": float(best["reconstruction_error"]),
            "converged_repeats": int(runs["converged"].sum()),
            "mean_n_iter": float(runs["n_iter"].mean()),
            "min_active_patterns": int(runs["active_patterns"].min()),
            "best_seed": int(best["seed"]),
            "best_converged": bool(best["converged"]),
        })
        all_runs.append(runs)
    diagnostics = pd.concat(all_runs, ignore_index=True)
    _warn_unconverged(diagnostics)
    result = pd.DataFrame(summaries)
    return result, diagnostics


def fit_nmf(
    data,
    n_patterns: int,
    *,
    layer: str | None = None,
    n_repeats: int = 100,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return W (spots x patterns) and H (patterns x LR) for a chosen rank.

    Fits repeated random starts and returns the run with the smallest
    Frobenius reconstruction error. Using the same input, rank, repeats,
    settings and integer random_state reproduces evaluate_nmf's best run.
    Inputs and layer selection follow evaluate_nmf; one repeat is allowed.

    W @ H approximates the input scores. Active H rows have unit L2 norm,
    with the inverse scale ambiguity absorbed into W. Inactive patterns
    have zero W columns and H rows. Patterns are ordered by decreasing
    total W weight, with matching labels in both output DataFrames. Spot
    and LR input order and identifiers are preserved.

    Only the two factor matrices are returned, without metadata in attrs.
    Unconverged runs remain eligible for selection and generate a warning.
    No files are written and no TF activation is inferred by this function.
    """
    controls = _controls(
        n_repeats, max_iter, tol, random_state, minimum_repeats=1,
    )
    n_repeats, max_iter, tol, random_state = controls
    matrix, spot_ids, lr_ids, norm = _prepare_matrix(data, layer)
    n_patterns = _rank(n_patterns, matrix.shape)
    factors, runs, _ = _fit_runs(
        matrix, n_patterns, n_repeats, max_iter, tol, random_state, norm,
        keep_loadings=False,
    )
    _warn_unconverged(runs)
    weights, loadings = factors
    order = np.argsort(-weights.sum(axis=0), kind="stable")
    pattern_ids = pd.Index([
        f"pattern_{i + 1}" for i in range(n_patterns)
    ], name="pattern")
    spot_patterns = pd.DataFrame(
        weights[:, order], index=spot_ids, columns=pattern_ids,
    )
    lr_patterns = pd.DataFrame(
        loadings[order], index=pattern_ids, columns=lr_ids,
    )
    return spot_patterns, lr_patterns
