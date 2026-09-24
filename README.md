# EEG-Twin

CEBRA trajectory embedding and Transformer digital twin for post-cardiac-arrest
EEG, with the figures built on them.

## Requirements

Python 3.11, `pip install -r requirements.txt`. A GPU for twin training; the
rest runs on CPU.

Two ways in. They give the same results — route B just starts after the build.

### Route A — from raw

Everything downstream is generated.

| Path | Contents |
|---|---|
| `data/raw/protopnet/{MGH,ULB,YNH,BIDMC,BWH,UTW}/*.npz` | per-recording prototype-network outputs |
| `data/raw/qeeg/` | per-patient 10 s qEEG csvs (with-spike set) |
| `data/raw/offsets.csv` | time from ROSC per recording |
| `data/tables/ICARE_clinical.csv` | age, sex, vfib, ROSC, time to arrest, CPC |
| `data/tables/split_train.csv`, `split_test.csv` | patient-level split (one `patient_id` column) |

~54 GB across ~6,500 files, and the build takes hours. Drop the two split csvs
to let the build derive its own split.

### Route B — from the built dataset

| Path | Contents |
|---|---|
| `data/dataset/PPNet_data_train.npz`, `PPNet_data_test.npz` | route A's output, 1.6 GB |
| `data/tables/ICARE_clinical.csv` | as above |

`run_build_dataset` reports `SKIP — inputs missing` and the run starts at CEBRA.
This is the route to hand someone who only needs results, since the two `.npz`
files carry everything downstream reads.

## Run

```bash
python run_all.py
```

Runs every step in order, skips whatever is already built, and stops with a
plain list if an input is missing.

```bash
python run_all.py --list      # show the plan
python run_all.py --force     # rebuild everything
python run_all.py --figures   # figures only
python tests/smoke_test.py    # synthetic end-to-end check, ~5 min
```

### On Colab

Put `notebooks/run_pipeline_colab.ipynb` in the Drive folder holding the data
and open it with Colab. It detects the route, clones this repo, installs, runs
every stage, and copies results back beside the data. GPU **and high-RAM** —
the twin peaks at ~15.9 GB, above free tier's 12.7 GB.

Route A reads the raw 54 GB over the Drive mount rather than copying it, so
the build is slower there than on a local disk.

## Steps

| Step | In | Out | Time |
|---|---|---|---|
| `run_build_dataset` | `data/raw/`, `data/tables/` | `data/dataset/` | hours |
| `run_cebra_features` | `data/dataset/` | `data/cebra/prep_*.npz` | ~1 min |
| `run_train_cebra` | `data/cebra/prep_*` | `data/cebra/embeddings_*` | 6-45 min |
| `run_export_for_twin` | dataset + embeddings | `data/twin/input/` | ~1 min |
| `run_train_twin` | `data/twin/input/`, clinical | `data/twin/handoffs/`, models, fig1-7, metrics | hours, GPU |
| `run_fig_globes` | embeddings + handoffs | Figure 3 | ~1 min |
| `run_fig_twin` | embeddings + handoffs | Figure 2 | ~1 min |
| `run_fig_prototypes` | embeddings | prototype figure | ~1 min |
| `run_embedding_eval` | prep + embeddings | metrics *(optional)* | ~4 min |
| `run_retrieval_ablation` | handoffs | ablation table *(optional)* | ~3 s |

## Output

```
outputs/figures/cebra/   trajectory globes · twin in CEBRA space · prototype map
outputs/figures/twin/    fig1-7
outputs/figures/eval/    confusion · CPC · centroid distances
outputs/models/          CEBRA and twin weights, scalers, feature pipeline
outputs/metrics/twin/    13 tables + all_metrics.json
```
Figures are written as interactive HTML, 600-dpi PNG and vector PDF.

## Layout

`data/` holds anything a later step reads. `outputs/` holds terminal artifacts.
One location per artifact, no fallbacks. Both are gitignored.

```
run_all.py       the pipeline
src/             config, constants, and one module per stage
scripts/         one runner per stage
tests/           smoke test + synthetic input generator
notebooks/       Colab versions of the figure, preprocessing and twin code
docs/            pipeline diagram, class-encoding reference
data/            inputs and intermediates   (gitignored)
outputs/         figures, models, metrics   (gitignored)
```

Change hyperparameters in `src/config.py`; change which patients a figure uses
at the top of its runner.

## Notes

**Class encoding.** `constants.py` uses
`['BurstSupp','Seizure','LPD','GPD','LRDA','GRDA','Continuous','Discontinuous']`.
Index 0 is Burst Suppression: the base softmax emits 6 classes with 0 = "Other",
and preprocessing splits that class by the burst-suppression index into
BurstSupp (< 0.5), Discontinuous (0.5-0.9) and Continuous (>= 0.9). See `docs/class_encoding.md`.

**Palette.** Validated on all 28 pairs: worst normal-vision dE 20.1 (floor 15),
worst colour-blind dE 8.0 (target 8).

**Smoothing.** Gaussian over the raw segment sequence (sigma 9 bins = 45 min),
re-projected onto the sphere, then centripetal Catmull-Rom. Chosen by sweep to
keep state structure while the line stays smooth.

**Determinism.** Seed 42; the CEBRA steps reproduce the embedding bit-for-bit
on the same device.

## Reproduction

Rebuilt end to end from prototype-network outputs — dataset, CEBRA embedding,
and ~110 transformer trainings — and it lands on the published numbers:

| | Published | This pipeline |
|---|---|---|
| Twin AUROC @84h | 0.939 | 0.9394 |
| Reference transformer | 0.935 | 0.9363 |
| AUPRC | 0.892 | 0.8932 |
| Brier, raw -> temp-scaled | 0.125 -> 0.106 | 0.125 -> 0.104 |
| Temperature | 0.6 | 0.60 |
| Sensitivity / specificity @0.5 | 0.922 / 0.807 | 0.922 / 0.802 |
| TPR @ FPR<=0.05 | 0.766 | 0.762 |
| Cohort | 695 / 299 | 695 / 299 |

Cost: ~6 min to build the dataset, ~6 min for CEBRA, ~15 h for the twin on one
GPU. The ablation figure dominates that last number -- it retrains a 10-seed
ensemble per feature set. Set `TWIN_TRAIN['N_ABL']` lower in `src/config.py` to
trade Figure 1's error bars for hours.

## Limits

- `build_dataset` fills unrecognised qEEG columns with NaN silently. It now
  reports every skipped recording and why, so check that tally on a first run.

- Prototype-network training is not in this repo; its outputs are an input.
- Run on macOS/ARM (Python 3.11, numpy 1.26) and Colab Linux (Python 3.13,
  numpy 2.x, CUDA). Windows untested.
- 45 prototypes span 7 of the 8 phenotypes; none codes for Discontinuous.
