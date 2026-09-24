"""
config.py — every knob for the pipeline, in one place.

Paths default to <project>/data and <project>/outputs and can be overridden
with the CEBRA_DATA and CEBRA_OUT environment variables.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Two roots, one rule:
#   data/     anything that feeds a later step  (inputs and intermediates)
#   outputs/  anything terminal                 (figures, models, metrics)
# Every location below is the ONE place its artifact lives. No fallbacks.
DATA = Path(os.environ.get('CEBRA_DATA', ROOT / 'data'))
OUT = Path(os.environ.get('CEBRA_OUT', ROOT / 'outputs'))

# ── data/ ───────────────────────────────────────────────────────────
RAW_DIR = DATA / 'raw'                   # build-dataset inputs (ProtoPNet, qEEG, offsets)
TABLES_DIR = DATA / 'tables'             # clinical csv + split id csvs
DATASET_DIR = DATA / 'dataset'           # build-dataset out -> CEBRA features, twin export
CEBRA_DIR = DATA / 'cebra'               # CEBRA out -> figures, eval, twin export
TWIN_INPUT_DIR = DATA / 'twin' / 'input'      # twin export out -> twin training
TWIN_HANDOFF_DIR = DATA / 'twin' / 'handoffs' # twin training out -> figures, ablation

# ── outputs/ ────────────────────────────────────────────────────────
FIG_CEBRA = OUT / 'figures' / 'cebra'
FIG_TWIN = OUT / 'figures' / 'twin'
FIG_EVAL = OUT / 'figures' / 'eval'
MODELS_DIR = OUT / 'models'
METRICS_DIR = OUT / 'metrics'

for _d in (DATASET_DIR, CEBRA_DIR, TWIN_INPUT_DIR, TWIN_HANDOFF_DIR,
           FIG_CEBRA, FIG_TWIN, FIG_EVAL, MODELS_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# named files, so nothing is spelled out twice
PPNET_TRAIN = DATASET_DIR / 'PPNet_data_train.npz'
PPNET_TEST = DATASET_DIR / 'PPNet_data_test.npz'
CLINICAL_CSV = TABLES_DIR / 'ICARE_clinical.csv'
HANDOFF_STEP4 = TWIN_HANDOFF_DIR / 'twin_step4_handoff.npz'
HANDOFF_MATCH = TWIN_HANDOFF_DIR / 'twin_matching_handoff.npz'
TWIN_NPZ = {s: TWIN_INPUT_DIR / f'PPNet Data {s.capitalize()} with CEBRA COMBO V.npz'
            for s in ('train', 'test')}
prep = lambda split: CEBRA_DIR / f'prep_{split}.npz'
embeddings = lambda split: CEBRA_DIR / f'embeddings_{split}.npz'

RUN_TAG = 'combo_v'

# ── Step 1: features fed to CEBRA ───────────────────────────────────
# PPNet intermediate features (1275-d, PCA'd to 50) + the 13 qEEG features.
PREP = dict(
    RUN_TAG        = RUN_TAG,
    FEATURE_KEYS   = ['features', 'cebra_features'],
    PCA_KEY        = 'features',
    PCA_COMPONENTS = 50,
    SEED           = 42,
    BIN_SEC        = 300,          # 5-min segments
)

# ── Step 2: CEBRA hybrid training (manuscript Appendix E) ───────────
# combo_v = time objective + `predictions` + `cpc_binary` + `probabilities`.
# CPC supervises training only; it is never an input at inference.
TRAIN = dict(
    RUN_TAG            = RUN_TAG,
    LABEL_KEYS_DISC    = ['predictions', 'cpc_binary'],
    LABEL_KEYS_CONT    = ['probabilities'],
    USE_TIME_OBJECTIVE = True,
    OUTPUT_DIM         = 3,
    TIME_OFFSET        = 144,      # 144 bins x 5 min = 12 h
    BATCH_SIZE         = 1024,
    MAX_ITER           = 20000,
    TEMPERATURE        = 0.5,
    NUM_UNITS          = 16,
    LR                 = 3e-4,
    KNN_NEIGHBORS      = 10,
    SEED               = 42,
)

# ── Figures ─────────────────────────────────────────────────────────
FIG = dict(
    WINDOW  = 24,     # 2 h per waypoint
    SLERP_N = 10,
    HOUR    = 24,     # bedside "as of" hour for the twin figure
    TOP_K   = 10,     # neighbours retrieved
    N_GLOBE = 3,      # neighbours drawn on the globe
    ALPHA   = 0.75,   # combined = ALPHA*trajectory + (1-ALPHA)*feature
)


# ── Transformer twin (src/twin_model.py) ──
# Defaults are the manuscript settings. For a smoke test:
#   TWIN_TRAIN.update(N_REF=1, N_TWIN=1, N_ABL=1, N_CAL=1, MAX_EPOCHS=1, SUBSET=(2, 2))
# or, without editing this file:
#   CEBRA_TWIN_SMOKE=1 python run_all.py --with-twin
TWIN_TRAIN = dict(
    N_REF=10, N_TWIN=10, N_ABL=10, N_CAL=10,
    MAX_EPOCHS=None,      # None = the notebook's 25
    SUBSET=None,          # None = all 695/299; (n_pos, n_neg) for a smoke run
    DEVICE=None,          # None = cuda if available else cpu
)
# CEBRA_SMOKE shrinks the contrastive run so the pipeline can be exercised
# end to end on synthetic data (tests/smoke_test.py). Never for real results.
if os.environ.get('CEBRA_SMOKE'):
    TRAIN.update(MAX_ITER=200, BATCH_SIZE=128)

if os.environ.get('CEBRA_TWIN_SMOKE'):
    TWIN_TRAIN.update(N_REF=1, N_TWIN=1, N_ABL=1, N_CAL=1, MAX_EPOCHS=1, SUBSET=(2, 2))


# ── Step 0: rebuilding PPNet_data_*.npz from ProtoPNet outputs ───────
# Point these at the prototype-network artifacts
# and run scripts/run_build_dataset.py. None of them ship with this folder.
_SITES = ('MGH', 'ULB', 'YNH', 'BIDMC', 'BWH', 'UTW')
PREPROCESS = dict(
    PROTOPNET_DIRS      = [RAW_DIR / 'protopnet' / s for s in _SITES],
    BCI_CSV_DIR         = RAW_DIR / 'qeeg',
    OFFSETS_CSV         = RAW_DIR / 'offsets.csv',
    CLINICAL_CSV        = CLINICAL_CSV,
    TRAIN_IDS           = TABLES_DIR / 'split_train.csv',
    TEST_IDS            = TABLES_DIR / 'split_test.csv',
    EPOCHS_PER_SEGMENT  = 6,        # 6 x 50 s = 300 s
    SEC_PER_CHUNK       = 21600,    # 6 h
    MAX_CHUNK           = 14,       # drop chunks >= 14 (past ~84 h)
)
