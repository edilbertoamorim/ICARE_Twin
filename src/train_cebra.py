"""
train_cebra.py — multi-objective CEBRA at 5-min resolution.

Loss = sum of enabled InfoNCE terms on a single shared 3D embedding:
  - time-positive (always-on by default; 12h bidirectional)
  - one term per discrete label in LABEL_KEYS_DISC
  - one term per continuous label in LABEL_KEYS_CONT (kNN in label space)

Combinations (set RUN_TAG to match 01):
  i)   DISC=['predictions'],               CONT=[]
  ii)  DISC=['predictions','cpc_binary'],  CONT=[]
  iii) DISC=[],                            CONT=['probabilities']
  iv)  DISC=['cpc_binary'],                CONT=['probabilities']
  v)   DISC=['predictions','cpc_binary'],  CONT=['probabilities']

Note: cpc is binarized in 01 (cpc_binary: 0=good, 1=poor). Raw cpc_scores
is kept in the npz for downstream AUC analysis.
"""
import cebra
import cebra.models
import cebra.distributions
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import trange

from config import prep, embeddings, MODELS_DIR
from progress import step as _step, done, log
from config import TRAIN as _C

# ── CONFIG ──────────────────────────────────────────────────────────
RUN_TAG = _C['RUN_TAG']


LABEL_KEYS_DISC = _C['LABEL_KEYS_DISC']
LABEL_KEYS_CONT = _C['LABEL_KEYS_CONT']
USE_TIME_OBJECTIVE = _C['USE_TIME_OBJECTIVE']

OUTPUT_DIM = _C['OUTPUT_DIM']
TIME_OFFSET = _C['TIME_OFFSET']
BATCH_SIZE = _C['BATCH_SIZE']
MAX_ITER = _C['MAX_ITER']
TEMPERATURE = _C['TEMPERATURE']
NUM_UNITS = _C['NUM_UNITS']
LR = _C['LR']
KNN_NEIGHBORS = _C['KNN_NEIGHBORS']

SEED = _C['SEED']
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print(f"  device: {device}")

# ── Load preprocessed data ─────────────────────────────────────────
log('train CEBRA')
d = np.load(prep('train'), allow_pickle=True)
X_train    = d['X'].astype(np.float32)
pat_starts = d['pat_starts']
pat_ends   = d['pat_ends']
N, D = X_train.shape
print(f"  {N} samples, {D} features")

X_tensor     = torch.tensor(X_train,    dtype=torch.float32, device=device)
pat_starts_t = torch.tensor(pat_starts, dtype=torch.long,    device=device)
pat_ends_t   = torch.tensor(pat_ends,   dtype=torch.long,    device=device)

# ── Build samplers per enabled objective ───────────────────────────
def to_dense_int(arr):
    """Map arbitrary int labels to dense 0..K-1 (DiscreteEmpirical-friendly)."""
    arr = np.asarray(arr).reshape(-1)
    uniq = np.unique(arr)
    remap = {int(v): i for i, v in enumerate(uniq)}
    return np.array([remap[int(v)] for v in arr], dtype=np.int64)

disc_samplers = []
for key in LABEL_KEYS_DISC:
    arr = to_dense_int(d[key])
    print(f"  Discrete: {key}  unique={len(np.unique(arr))}")
    label_t = torch.tensor(arr, dtype=torch.long, device=device)
    dist = cebra.distributions.discrete.DiscreteEmpirical(label_t.cpu())
    disc_samplers.append((key, label_t, dist))

cont_samplers = []
for key in LABEL_KEYS_CONT:
    arr = np.asarray(d[key]).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    print(f"  Continuous: {key}  shape={arr.shape}  kNN k={KNN_NEIGHBORS}")
    nbrs = NearestNeighbors(n_neighbors=KNN_NEIGHBORS, algorithm='auto', n_jobs=-1).fit(arr)
    _, knn_idx = nbrs.kneighbors(arr)
    cont_samplers.append((key, torch.tensor(knn_idx, dtype=torch.long, device=device)))

n_obj = int(USE_TIME_OBJECTIVE) + len(disc_samplers) + len(cont_samplers)
assert n_obj > 0, "No objectives enabled. Set USE_TIME_OBJECTIVE or add labels."

# ── Model ──────────────────────────────────────────────────────────
model = cebra.models.init(
    "offset10-model",
    num_neurons=D,
    num_units=NUM_UNITS,
    num_output=OUTPUT_DIM,
).to(device)
offset = model.get_offset()

criterion = cebra.models.criterions.FixedCosineInfoNCE(temperature=TEMPERATURE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITER)

# ── Helpers ────────────────────────────────────────────────────────
def expand_idx(idx, starts_t, ends_t):
    offsets = torch.arange(-offset.left, offset.right, device=idx.device)
    expanded = idx[:, None] + offsets[None, :]
    lo = starts_t[idx][:, None]
    hi = ends_t[idx][:, None] - 1
    return expanded.clamp(min=lo, max=hi)

def get_batch(idx, X_t, starts_t, ends_t):
    return X_t[expand_idx(idx, starts_t, ends_t)].transpose(2, 1)

# ── Training ───────────────────────────────────────────────────────
_step(f'{n_obj} objectives · {MAX_ITER} iterations · batch {BATCH_SIZE} · device {device}')
print(f"  TIME_OFFSET={TIME_OFFSET} bins ({TIME_OFFSET*5} min)")
print(f"  TEMP={TEMPERATURE}  NUM_UNITS={NUM_UNITS}  LR={LR}")

pbar = trange(MAX_ITER)
for step in pbar:
    ref_idx = torch.randint(0, N, (BATCH_SIZE,), device=device)
    neg_idx = torch.randint(0, N, (BATCH_SIZE,), device=device)

    ref_emb = model(get_batch(ref_idx, X_tensor, pat_starts_t, pat_ends_t))
    neg_emb = model(get_batch(neg_idx, X_tensor, pat_starts_t, pat_ends_t))

    total_loss = 0.0
    log = {}

    if USE_TIME_OBJECTIVE:
        direction = torch.randint(0, 2, (BATCH_SIZE,), device=device) * 2 - 1
        t_pos = (ref_idx + direction * TIME_OFFSET).clamp(
            min=pat_starts_t[ref_idx], max=pat_ends_t[ref_idx] - 1)
        t_emb = model(get_batch(t_pos, X_tensor, pat_starts_t, pat_ends_t))
        l, _, _ = criterion(ref_emb, t_emb, neg_emb)
        total_loss = total_loss + l
        log['time'] = l.item()

    for key, label_t, dist in disc_samplers:
        pos = dist.sample_conditional(label_t[ref_idx].cpu()).to(device)
        emb = model(get_batch(pos, X_tensor, pat_starts_t, pat_ends_t))
        l, _, _ = criterion(ref_emb, emb, neg_emb)
        total_loss = total_loss + l
        log[key] = l.item()

    for key, knn_t in cont_samplers:
        cols = torch.randint(0, KNN_NEIGHBORS, (BATCH_SIZE,), device=device)
        pos = knn_t[ref_idx, cols]
        emb = model(get_batch(pos, X_tensor, pat_starts_t, pat_ends_t))
        l, _, _ = criterion(ref_emb, emb, neg_emb)
        total_loss = total_loss + l
        log[key] = l.item()

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 200 == 0:
        log['total'] = total_loss.item()
        pbar.set_postfix({**{k: f"{v:.2f}" for k, v in log.items()},
                          'lr': f"{scheduler.get_last_lr()[0]:.1e}"})

# ── Compute embeddings ─────────────────────────────────────────────
def compute_emb(X_arr, starts, ends, batch=4096):
    X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)
    s_t = torch.tensor(starts, dtype=torch.long, device=device)
    e_t = torch.tensor(ends,   dtype=torch.long, device=device)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_arr), batch):
            idx = torch.arange(i, min(i+batch, len(X_arr)), device=device)
            out.append(model(get_batch(idx, X_t, s_t, e_t)).cpu().numpy())
    return np.concatenate(out, axis=0)

_step('embedding train split')
X_train_emb = compute_emb(X_train, pat_starts, pat_ends)

_step('embedding test split')
d_test = np.load(prep('test'), allow_pickle=True)
X_test_emb = compute_emb(d_test['X'].astype(np.float32),
                         d_test['pat_starts'], d_test['pat_ends'])

# ── Save ───────────────────────────────────────────────────────────
np.savez(embeddings('train'), embedding=X_train_emb)
np.savez(embeddings('test'),  embedding=X_test_emb)
torch.save(model.state_dict(), MODELS_DIR / 'cebra_model.pt')

done(f'train {X_train_emb.shape}  test {X_test_emb.shape}', embeddings('train').parent)
