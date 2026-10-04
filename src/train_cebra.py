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

`fit` and `embed` are also used by cebra_tuning.py (cross-validated
hyperparameter selection on training patients); run as a script, this module
trains on the full training split with config.TRAIN and embeds both splits.
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

RUN_TAG = _C['RUN_TAG']


def default_device():
    return ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu")


def to_dense_int(arr):
    """Map arbitrary int labels to dense 0..K-1 (DiscreteEmpirical-friendly)."""
    arr = np.asarray(arr).reshape(-1)
    uniq = np.unique(arr)
    remap = {int(v): i for i, v in enumerate(uniq)}
    return np.array([remap[int(v)] for v in arr], dtype=np.int64)


def _batch(model_offset, idx, X_t, starts_t, ends_t):
    offsets = torch.arange(-model_offset.left, model_offset.right, device=idx.device)
    expanded = idx[:, None] + offsets[None, :]
    lo = starts_t[idx][:, None]
    hi = ends_t[idx][:, None] - 1
    return X_t[expanded.clamp(min=lo, max=hi)].transpose(2, 1)


def fit(X, pat_starts, pat_ends, labels, cfg, device, verbose=True, desc=None):
    """
    Train one CEBRA model.

    X                   (N, D) float features, rows grouped by patient in time order
    pat_starts/pat_ends (N,) first / one-past-last row of each row's patient
    labels              dict: label key -> (N,) or (N, k) array aligned with X
    cfg                 config.TRAIN-shaped dict (LABEL_KEYS_*, TEMPERATURE, ...)
    Returns (model, offset).
    """
    np.random.seed(cfg['SEED'])
    torch.manual_seed(cfg['SEED'])
    torch.cuda.manual_seed_all(cfg['SEED'])
    out = print if verbose else (lambda *a, **k: None)

    N, D = X.shape
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    pat_starts_t = torch.tensor(pat_starts, dtype=torch.long, device=device)
    pat_ends_t = torch.tensor(pat_ends, dtype=torch.long, device=device)
    B, K_NN = cfg['BATCH_SIZE'], cfg['KNN_NEIGHBORS']

    # ── Build samplers per enabled objective ───────────────────────────
    disc_samplers = []
    for key in cfg['LABEL_KEYS_DISC']:
        arr = to_dense_int(labels[key])
        out(f"  Discrete: {key}  unique={len(np.unique(arr))}")
        label_t = torch.tensor(arr, dtype=torch.long, device=device)
        dist = cebra.distributions.discrete.DiscreteEmpirical(label_t.cpu())
        disc_samplers.append((key, label_t, dist))

    cont_samplers = []
    for key in cfg['LABEL_KEYS_CONT']:
        arr = np.asarray(labels[key]).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        out(f"  Continuous: {key}  shape={arr.shape}  kNN k={K_NN}")
        nbrs = NearestNeighbors(n_neighbors=K_NN, algorithm='auto', n_jobs=-1).fit(arr)
        _, knn_idx = nbrs.kneighbors(arr)
        cont_samplers.append((key, torch.tensor(knn_idx, dtype=torch.long, device=device)))

    n_obj = int(cfg['USE_TIME_OBJECTIVE']) + len(disc_samplers) + len(cont_samplers)
    assert n_obj > 0, "No objectives enabled. Set USE_TIME_OBJECTIVE or add labels."

    # ── Model ──────────────────────────────────────────────────────────
    model = cebra.models.init(
        "offset10-model",
        num_neurons=D,
        num_units=cfg['NUM_UNITS'],
        num_output=cfg['OUTPUT_DIM'],
    ).to(device)
    offset = model.get_offset()

    criterion = cebra.models.criterions.FixedCosineInfoNCE(temperature=cfg['TEMPERATURE'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['LR'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['MAX_ITER'])

    def get_batch(idx):
        return _batch(offset, idx, X_tensor, pat_starts_t, pat_ends_t)

    # ── Training ───────────────────────────────────────────────────────
    if verbose:
        _step(f'{n_obj} objectives · {cfg["MAX_ITER"]} iterations · batch {B} · device {device}')
    out(f"  TIME_OFFSET={cfg['TIME_OFFSET']} bins ({cfg['TIME_OFFSET']*5} min)")
    out(f"  TEMP={cfg['TEMPERATURE']}  NUM_UNITS={cfg['NUM_UNITS']}  LR={cfg['LR']}")

    pbar = trange(cfg['MAX_ITER'], desc=desc, leave=verbose)
    for step in pbar:
        ref_idx = torch.randint(0, N, (B,), device=device)
        neg_idx = torch.randint(0, N, (B,), device=device)

        ref_emb = model(get_batch(ref_idx))
        neg_emb = model(get_batch(neg_idx))

        total_loss = 0.0
        losses = {}

        if cfg['USE_TIME_OBJECTIVE']:
            direction = torch.randint(0, 2, (B,), device=device) * 2 - 1
            t_pos = (ref_idx + direction * cfg['TIME_OFFSET']).clamp(
                min=pat_starts_t[ref_idx], max=pat_ends_t[ref_idx] - 1)
            t_emb = model(get_batch(t_pos))
            l, _, _ = criterion(ref_emb, t_emb, neg_emb)
            total_loss = total_loss + l
            losses['time'] = l.item()

        for key, label_t, dist in disc_samplers:
            pos = dist.sample_conditional(label_t[ref_idx].cpu()).to(device)
            emb = model(get_batch(pos))
            l, _, _ = criterion(ref_emb, emb, neg_emb)
            total_loss = total_loss + l
            losses[key] = l.item()

        for key, knn_t in cont_samplers:
            cols = torch.randint(0, K_NN, (B,), device=device)
            pos = knn_t[ref_idx, cols]
            emb = model(get_batch(pos))
            l, _, _ = criterion(ref_emb, emb, neg_emb)
            total_loss = total_loss + l
            losses[key] = l.item()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 200 == 0:
            losses['total'] = total_loss.item()
            pbar.set_postfix({**{k: f"{v:.2f}" for k, v in losses.items()},
                              'lr': f"{scheduler.get_last_lr()[0]:.1e}"})
    return model, offset


def embed(model, offset, X_arr, starts, ends, device, batch=4096):
    """Embed every row of X_arr (patient boundaries given by starts/ends)."""
    X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)
    s_t = torch.tensor(starts, dtype=torch.long, device=device)
    e_t = torch.tensor(ends, dtype=torch.long, device=device)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_arr), batch):
            idx = torch.arange(i, min(i + batch, len(X_arr)), device=device)
            out.append(model(_batch(offset, idx, X_t, s_t, e_t)).cpu().numpy())
    return np.concatenate(out, axis=0)


if __name__ == '__main__':
    device = default_device()
    print(f"  device: {device}")

    # ── Load preprocessed data ─────────────────────────────────────────
    log('train CEBRA')
    d = np.load(prep('train'), allow_pickle=True)
    X_train = d['X'].astype(np.float32)
    print(f"  {X_train.shape[0]} samples, {X_train.shape[1]} features")

    keys = _C['LABEL_KEYS_DISC'] + _C['LABEL_KEYS_CONT']
    model, offset = fit(X_train, d['pat_starts'], d['pat_ends'],
                        {k: d[k] for k in keys}, _C, device)

    # ── Compute embeddings ─────────────────────────────────────────────
    _step('embedding train split')
    X_train_emb = embed(model, offset, X_train, d['pat_starts'], d['pat_ends'], device)

    _step('embedding test split')
    d_test = np.load(prep('test'), allow_pickle=True)
    X_test_emb = embed(model, offset, d_test['X'].astype(np.float32),
                       d_test['pat_starts'], d_test['pat_ends'], device)

    # ── Save ───────────────────────────────────────────────────────────
    np.savez(embeddings('train'), embedding=X_train_emb)
    np.savez(embeddings('test'), embedding=X_test_emb)
    torch.save(model.state_dict(), MODELS_DIR / 'cebra_model.pt')

    done(f'train {X_train_emb.shape}  test {X_test_emb.shape}', embeddings('train').parent)
