# DatasauRust modifications changelist

The original DatasauRust implementation was designed around built-in shapes and
four summary statistics. The Pokémon pipeline required arbitrary runtime
silhouettes, deterministic generation, and a complete five-statistic contract.

## Runtime contours

- Added runtime contour loading in `src/contour.rs`.
- Added `--shape-file <csv>` so a contour extracted from any Pokémon artwork can
  be used without adding coordinates to `src/shapes.rs`.
- Kept `--shape <built-in>` for backward compatibility.
- Made `--shape` and `--shape-file` mutually exclusive to avoid ambiguous
  targets.

## Complete statistics

- Extended `compute_stats` from four values to five:
  - mean x and mean y;
  - sample standard deviation x and sample standard deviation y; and
  - Pearson correlation.
- Updated tolerance checks, plots, console output, and manifests to report and
  validate correlation alongside the original four values.

Mean and standard deviation do not completely describe a two-dimensional
second-order distribution. Pearson correlation supplies the off-diagonal term
of the covariance matrix, ensuring every accepted dataset shares the complete
target sample covariance.

## Exact covariance projection

- Added an analytic moment projection in `src/optim.rs`.
- The projection centers the current points, whitens them using their current
  covariance, recolors them with the target covariance, and restores the target
  mean.
- Added periodic projection through `--project-every`.
- Added mandatory final projection before acceptance.

This makes the five statistics hard constraints rather than approximate terms
in the shape objective.

## Deterministic optimization

- Replaced internal `thread_rng()` calls with one seeded RNG.
- The same RNG is passed through initialization and every perturbation and
  acceptance operation.
- `--seed` therefore controls the complete run rather than only Gaussian
  initialization.

## Full-precision acceptance

- Added `--stats-tolerance` as an absolute tolerance for all five statistics.
- Acceptance uses full-precision values and never relies on rounded display
  output.
- Added final contour-distance rejection, so matching the statistics alone is
  insufficient: the result must still resemble the requested silhouette.

## Reproducibility manifest

- Added `--manifest-out` for a machine-readable JSON report.
- The manifest records:
  - target and measured statistics;
  - maximum statistics error and tolerance;
  - Pearson correlation;
  - random seed and iteration count;
  - built-in or runtime shape source;
  - point count and output path; and
  - final contour-distance metrics.

Together, these changes let the Rust optimizer consume generic extracted
contours while enforcing the same statistical contract used to validate the
Python training data and diffusion samples.
