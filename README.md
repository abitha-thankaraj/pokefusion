![CI](https://github.com/araffin/datasaurust/workflows/CI/badge.svg)

# DatasauRust

Blazingly fast implementation of the [Datasaurus](https://www.autodesk.com/research/publications/same-stats-different-graphs) paper (500x faster than the original): "Same Stats, Different Graphs: Generating Datasets with Varied Appearance and Identical Statistics through Simulated Annealing" by Justin Matejka and George Fitzmaurice.


https://user-images.githubusercontent.com/1973948/230972049-adcb8012-f25f-4df4-84ce-aafc7f58f184.mp4




## Usage

To run with plot `-p` (using gnuplot):
```
cargo run --release -- -d data/seed_datasets/Datasaurus_data.csv -p
```

With pre-defined shape:
```
cargo run --release -- -p -n 3000000 --decimals 2 --shape cat --allowed-distance 0.1
```

Starting from Gaussian noise:
```
cargo run --release -- -p -n 3000000 --decimals 2 --shape cat --allowed-distance 0.1 --gaussian
```

## Create videos

Create video and gif (use `--save-plot`):
```
pip install moviepy ffmpeg-python

python scripts/create_video.py logs/cat/ logs/cat.mp4
```

From one shape to another:
```
cargo run --release -- -p -n 2000000 --decimals 1 --shape dog --allowed-distance 0.1 --log-interval 10000 -d logs/gaussian_cat/output.csv --save-plots
```


Note: The original datasets and python code comes from http://www.autodeskresearch.com/papers/samestats

## Pokémon point-cloud pipeline

Iteration 1 adds a species-agnostic path from official transparent artwork to
class-conditional, fixed-moment 2D point clouds. The extractor receives image
bytes and one global configuration; it contains no Pokémon IDs, names, masks,
thresholds, or coordinates. Every generated CSV has 142 headerless `x,y` rows
and is analytically projected to the full-precision mean, sample covariance,
standard deviations, and Pearson correlation of
`data/seed_datasets/Datasaurus_data.csv`.

Python 3.10+ dependencies are `numpy`, `scipy`, `Pillow`, `requests`,
`PyYAML`, `contourpy`, `matplotlib`, and `torch`. Use an existing environment
that provides them. The pipeline itself does not create or mutate an
environment.

Acquire the five configured artworks, record source metadata, extract the same
generic contours, save QA previews, and generate the 5 × 16 smoke dataset with
one command:

```bash
python scripts/generate_pokemon_dataset.py \
  --config configs/pokemon_i1.yaml \
  --samples-per-species 16
```

Generate the first training volume and validate every file and manifest hash:

```bash
python scripts/generate_pokemon_dataset.py \
  --config configs/pokemon_i1.yaml \
  --samples-per-species 128
python scripts/validate_pokemon_dataset.py \
  --config configs/pokemon_i1.yaml
```

Exercise the unchanged extractor on a PokéAPI ID that is absent from the
training manifest:

```bash
python scripts/check_heldout_pokemon.py \
  --pokemon-id 7 \
  --out data/pokemon/heldout_validation.json
```

Source artwork, masks, point clouds, previews, checkpoints, and sampled runs are
ignored by Git. The small deterministic contour artifacts, source metadata,
generation manifest, and validation summary are versioned. The source metadata
contains the Pokémon Company rights notice; do not infer commercial image
redistribution rights from the PokeAPI sprites repository license.

### Runtime contours in Rust

Use a generated contour CSV instead of a built-in shape. `--shape` and
`--shape-file` are mutually exclusive. All optimizer random choices share
`--seed`; full moments are reprojected every `--project-every` accepted moves
and once at the end. `--stats-tolerance` is an absolute full-precision check,
not a rounded display comparison.

```bash
cargo run --release -- \
  --shape-file data/pokemon/contours/pikachu.csv \
  --seed 42 \
  --project-every 1000 \
  --stats-tolerance 1e-4 \
  --manifest-out logs/pikachu/manifest.json
```

### Diffusion baseline

The model trains only on point coordinates. It whitens the shared target
covariance, permutes rows on every training access, uses a
permutation-equivariant Transformer without point-index embeddings, predicts
DDPM noise, and samples through the EMA model.

```bash
python pokemon_point_diffusion.py check \
  --data-dir data/pokemon/points
python pokemon_point_diffusion.py train \
  --data-dir data/pokemon/points \
  --out runs/pokemon_i1 \
  --steps 30000
python pokemon_point_diffusion.py sample \
  --checkpoint runs/pokemon_i1/checkpoint.pt \
  --out runs/pokemon_i1/samples \
  --per-class 8
```

Sampling writes one coordinate CSV per generated cloud, `metrics.json`, a
class-labeled preview, a randomized 5 × 5 blinded preview, and its separate key.
All saved samples are reprojected to the exact target moments.

Run the Python and Rust tests with:

```bash
python -m unittest discover -s tests -v
cargo test --all-targets
```
