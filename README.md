# SpaLRFactor

SpaLRFactor computes spatial ligand-receptor (LR) scores and decomposes
them into patterns using nonnegative matrix factorization (NMF).

```python
from spalrfactor import score_lr, evaluate_nmf, fit_nmf
```

## score_lr

```python
scores = score_lr(data, pairs)
```

| Parameter | Default | Description |
| --- | --- | --- |
| `data` | Required | AnnData with spots as rows and genes as columns. |
| `pairs` | Required | DataFrame with `lr_pair`, `ligand_feature_id`, and `receptor_feature_id`. |
| `layer` | `"counts"` | Raw count layer; `None` uses `data.X`. |
| `radius` | `250.0` | Maximum neighbor distance in micrometers. |
| `sigma` | `125.0` | Gaussian distance-weighting scale in micrometers. |
| `include_self` | `False` | Include the focal spot in its neighborhood. |
| `target_sum` | `10000` | Per-spot count normalization target before log1p. |

`data.obs` must contain `x_um` and `y_um`. If `sample` is present,
neighborhoods are constructed within each sample. Gene IDs in `pairs`
must match `data.var_names`. Separate required complex subunits with
semicolons; complex expression is the minimum across subunits.

Returns a spots × LR AnnData: `X` contains incoming scores (local receptor
× neighboring ligand), and `layers["outgoing"]` contains outgoing scores
(local ligand × neighboring receptor).

## evaluate_nmf

```python
scores = scores[scores.obs["score_available"]].copy()
summary, diagnostics = evaluate_nmf(scores, n_patterns=[2, 4, 6, 8, 10])
```

| Parameter | Default | Description |
| --- | --- | --- |
| `data` | Required | Spots × LR AnnData, DataFrame, dense array, or sparse matrix. |
| `n_patterns` | Required | List of candidate pattern counts. |
| `layer` | `None` | For AnnData, use `X`; `"outgoing"` selects sending scores. |
| `n_repeats` | `100` | Independent random starts per candidate; at least 2. |
| `max_iter` | `1000` | Maximum iterations per fit. |
| `tol` | `1e-4` | Convergence tolerance. |
| `random_state` | `42` | Root seed for reproducible random starts. |

Input scores must be finite and nonnegative. No score scaling or filtering
is performed by NMF.

Returns two DataFrames: `summary` contains stability, reconstruction error,
and convergence summaries for each candidate; `diagnostics` contains one
record per fit. Higher stability and lower error indicate more consistent
patterns and better reconstruction, respectively.

## fit_nmf

```python
W, H = fit_nmf(scores, n_patterns=10)
W_out, H_out = fit_nmf(scores, n_patterns=10, layer="outgoing")
```

`n_patterns` is a single chosen pattern count. All other parameters and
defaults match `evaluate_nmf`; `n_repeats=1` is also allowed.

Returns the lowest-error fit as two DataFrames: `W` (spots × patterns)
and `H` (patterns × LR), with `W @ H` approximating the input scores.
Active rows of `H` have unit L2 norm. Unconverged fits generate a warning
and remain eligible for selection.
