"""
twin_model.py — the Transformer digital twin.

Mirrors notebooks/transformer_twin.ipynb cell for cell, unmodified except:
  * the Drive-mount / hardcoded-path cell is replaced by config.py paths
  * matplotlib runs headless and plt.show() is a no-op
  * seed counts, epoch cap, device and an optional patient subset come from
    config.TWIN_TRAIN so a smoke run is possible without editing the science

Reads   <out>/PPNet Data {Train,Test} with CEBRA COMBO V.npz   (the export-for-twin step)
        data/tables/ICARE_clinical.csv
Writes  <out>/twin/  twin_step4_handoff.npz · twin_matching_handoff.npz
                     twin_models.pt · twin_feature_pipeline.pkl
                     fig1..fig7.png · metrics/
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as _plt
_plt.show = lambda *a, **k: None          # headless: never block on a figure

from progress import log as _plog


def _stage(k, n, what):
    """Announce a block, so a two-hour run is not a black box."""
    _plog(f'[{k:2d}/{n}] {what}')

# ===================== cell 2 =====================
_stage(1, 27, 'paths')
from config import (TWIN_TRAIN as _TT, TWIN_INPUT_DIR, TWIN_HANDOFF_DIR,
                    CLINICAL_CSV, MODELS_DIR, METRICS_DIR, FIG_TWIN)
from progress import bar as _bar, step as _pstep

DATA_DIR = TWIN_INPUT_DIR
OUT_DIR = TWIN_HANDOFF_DIR
for _p, _w in ((DATA_DIR / 'PPNet Data Train with CEBRA COMBO V.npz', 'run the export-for-twin step first'),
               (CLINICAL_CSV, 'see data/README.md')):
    if not _p.exists():
        raise SystemExit(f'missing {_p}  ({_w})')
print('twin inputs :', DATA_DIR)
print('twin outputs:', OUT_DIR)


# ===================== cell 3 =====================
_stage(2, 27, 'imports and device')
# Library imports and compute device
import json, time, gc, random, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore'); torch.backends.cudnn.benchmark = False
_dev = _TT.get('DEVICE') or os.environ.get('CEBRA_TWIN_DEVICE')
DEVICE = torch.device(_dev) if _dev else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE)

# ===================== cell 4 =====================
_stage(3, 27, 'load features, build multi-stream arrays')
# Load frozen ProtoPNet/CEBRA/EEG/clinical features; build the per-5-min multistream with 8-class IIIC labels; fit train scalers; official 695/299 split
# qEEG temporal summary per patient: [mean, std, slope, last] over the observed 5-min segments
def compute_temporal_features(X, M):
    N, C, T = X.shape; feats = []
    for i in range(N):
        obs = np.where(M[i] > 0)[0]
        if len(obs) < 2: feats.append(np.zeros(C * 4, dtype=np.float32)); continue
        ov = X[i][:, obs]
        feats.append(np.concatenate([ov.mean(1), ov.std(1),
                     (ov[:, -1] - ov[:, 0]) / max(obs[-1] - obs[0], 1), ov[:, -1]]).astype(np.float32))
    return np.stack(feats)

# load the frozen ProtoPNet + CEBRA features (one row per 5-min EEG segment) for the official train/test patients
train_npz = {k: v for k, v in np.load(DATA_DIR / 'PPNet Data Train with CEBRA COMBO V.npz', allow_pickle=True).items()}
test_npz  = {k: v for k, v in np.load(DATA_DIR / 'PPNet Data Test with CEBRA COMBO V.npz',  allow_pickle=True).items()}
# for each patient: the row indices of its segments, time-sorted
def build_row_index(z):
    df = pd.DataFrame({'pid': z['patient_ids'], 't': z['times'].astype(np.int64), 'idx': np.arange(len(z['patient_ids']))}).sort_values(['pid', 't'])
    return df.groupby('pid')['idx'].apply(lambda s: s.values).to_dict()
TRAIN_ROWS = build_row_index(train_npz); TEST_ROWS = build_row_index(test_npz)
TRAIN_PIDS = sorted(TRAIN_ROWS.keys()); TEST_PIDS = sorted(TEST_ROWS.keys())
cpc_train = np.array([int(train_npz['cpc_scores'][TRAIN_ROWS[p][0]]) for p in TRAIN_PIDS])
cpc_test  = np.array([int(test_npz['cpc_scores'][TEST_ROWS[p][0]])  for p in TEST_PIDS])
# outcome label: good = CPC 1-2 (1), poor = CPC 3-5 (0)
y_train = (cpc_train <= 2).astype(int); y_test = (cpc_test <= 2).astype(int)

# clinical variables (age, sex, vfib, ROSC, time-to-arrest): median-impute then standardize on TRAIN
clin_df = pd.read_csv(CLINICAL_CSV, index_col=0).copy()
def _enc(s):
    if isinstance(s, str):
        if s.strip().upper() == 'M': return 1.0
        if s.strip().upper() == 'F': return 0.0
    return np.nan
clin_df['sex_M'] = clin_df['sex'].apply(_enc)
CLIN_COLS = ['age', 'sex_M', 'vfib', 'ROSC(minutes)', 'time_to_CA(seconds)']
def _row(pid):
    return (clin_df.loc[pid, CLIN_COLS].astype(np.float32).values if pid in clin_df.index else np.full(len(CLIN_COLS), np.nan, dtype=np.float32))
C_train = np.stack([_row(p) for p in TRAIN_PIDS]).astype(np.float32); C_test = np.stack([_row(p) for p in TEST_PIDS]).astype(np.float32)
_tmed = np.nanmedian(C_train, 0); _tmed = np.where(np.isnan(_tmed), 0.0, _tmed)
for A in (C_train, C_test):
    ix = np.isnan(A); A[ix] = np.take(_tmed, np.where(ix)[1])
sc_clin = StandardScaler().fit(C_train)
C_train_s = sc_clin.transform(C_train).astype(np.float32); C_test_s = sc_clin.transform(C_test).astype(np.float32)

# each patient = 1008 five-minute slots (84 h); all feature streams stacked into MS_DIM channels
SEQ_LEN = 1008; MS_DIM = 3 + 1275 + 45 + 8 + 13
# channel layout: CEBRA(3) | ProtoPNet feats(1275) | prototype acts(45) | 8-class IIIC(8) | qEEG(13)
STREAM_SLICES = {'cebra': slice(0,3), 'ppnet': slice(3,3+1275), 'acts': slice(3+1275,3+1275+45),
                 'probs': slice(3+1275+45,3+1275+45+8), 'eeg': slice(3+1275+45+8,3+1275+45+8+13)}
# build each patient into a (channels x 1008) array + a mask of which slots were actually observed
def build_multistream(pids, npz, rows):
    cebra = npz['cebra_embedding'].astype(np.float32); feats = npz['features'].astype(np.float32)
    # 8-class IIIC label one-hot per segment (block-averaged later = the prof's 'prototype label frequencies')
    acts = npz['activations'].astype(np.float32); probs = np.eye(8, dtype=np.float32)[npz['predictions'].astype(int)]
    cf = np.nan_to_num(npz['cebra_features'].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0); cf[:, 3] = np.log1p(np.clip(cf[:, 3], 0, None))
    full = np.concatenate([cebra, feats, acts, probs, cf], axis=1); times = npz['times'].astype(np.int64); X, M = [], []
    for pid in pids:
        seq = np.full((SEQ_LEN, MS_DIM), np.nan, dtype=np.float32); mask = np.zeros(SEQ_LEN, dtype=np.float32)
        for s_idx in rows[pid]:
            # drop each segment into its 5-min time slot (correctly handles recording gaps)
            slot = times[s_idx] // 300
            if slot >= SEQ_LEN: continue
            seq[slot] = full[s_idx]; mask[slot] = 1.0
        # fill unobserved slots (mask=0 there, so these values are excluded by every downstream step)
        fm = np.nan_to_num(np.nanmean(seq, 0), nan=0.0); seq = np.where(np.isnan(seq), fm, seq)
        X.append(np.nan_to_num(seq, posinf=0.0, neginf=0.0).astype(np.float32).T); M.append(mask)
    return np.stack(X).astype(np.float32), np.stack(M).astype(np.float32)
X_ms_tr, M_ms_tr = build_multistream(TRAIN_PIDS, train_npz, TRAIN_ROWS)
X_ms_te, M_ms_te = build_multistream(TEST_PIDS,  test_npz,  TEST_ROWS)
# standardize channels on TRAIN statistics only, then apply the same to test (no leakage)
sc_full = StandardScaler().fit(X_ms_tr.transpose(0,2,1).reshape(-1, MS_DIM))
def _scale(X): return sc_full.transform(X.transpose(0,2,1).reshape(-1, MS_DIM)).reshape(X.transpose(0,2,1).shape).transpose(0,2,1).astype(np.float32)
X_ms_tr_s = _scale(X_ms_tr); X_ms_te_s = _scale(X_ms_te)
def slice_streams(X, streams):
    return X[:, np.concatenate([np.arange(STREAM_SLICES[s].start, STREAM_SLICES[s].stop) for s in streams]), :]
# reference-model input = ProtoPNet + qEEG streams (clinical is fused separately)
XB_tr = slice_streams(X_ms_tr_s, ['ppnet', 'eeg']).copy(); XB_te = slice_streams(X_ms_te_s, ['ppnet', 'eeg']).copy()
N_BASE = XB_tr.shape[1]; EEG0 = N_BASE - 13; N_FEAT = N_BASE + 52
sc_tf = StandardScaler().fit(compute_temporal_features(XB_tr[:, EEG0:, :], M_ms_tr))
TF_MEAN = sc_tf.mean_.astype(np.float32); TF_STD = np.where(sc_tf.scale_ < 1e-6, 1.0, sc_tf.scale_).astype(np.float32)
# per-segment 8-class IIIC targets for the auxiliary head
IIIC_TR = X_ms_tr_s[:, STREAM_SLICES['probs'], :].transpose(0, 2, 1)
print(f'N_BASE={N_BASE} N_FEAT={N_FEAT} | TRAIN {len(TRAIN_PIDS)} pos={y_train.sum()} | TEST {len(TEST_PIDS)} pos={y_test.sum()}')

# optional patient subset — smoke testing only, set via config.TWIN_TRAIN
if _TT.get('SUBSET'):
    _n_pos, _n_neg = _TT['SUBSET']
    import numpy as _np
    def _pick(y, M, npos, nneg):
        o = _np.argsort(-M.sum(1))
        return _np.array([i for i in o if y[i] == 1][:npos] + [i for i in o if y[i] == 0][:nneg])
    # train only: the test split stays whole so downstream figures still see
    # every CPC grade
    _itr = _pick(y_train, M_ms_tr, _n_pos, _n_neg)
    XB_tr, M_ms_tr, C_train_s, y_train, IIIC_TR = (XB_tr[_itr], M_ms_tr[_itr], C_train_s[_itr],
                                                   y_train[_itr], IIIC_TR[_itr])
    X_ms_tr_s = X_ms_tr_s[_itr]; X_ms_tr = X_ms_tr[_itr]
    C_train = C_train[_itr]          # raw clinical: both handoffs write this
    TRAIN_PIDS = [TRAIN_PIDS[i] for i in _itr]
    print(f'*** SUBSET {len(y_train)} train / {len(y_test)} test — SMOKE TEST, NOT SCIENCE ***')

# ===================== cell 5 =====================
_stage(4, 27, 'reference hyperparameters')
# Reference outcome-transformer hyperparameters
CFG = dict(d_model=256, n_blocks=5, n_queries=32, n_heads=8, ffn_expand=2, dropout=0.27, drop_path=0.11,
           mixup_alpha=0.54, mask_pct=0.18, num_masks=2, label_smoothing=0.067, lr=4.9e-4, batch_size=16,
           max_epochs=25, weight_decay=0.01, ema_decay=0.999, grad_clip=1.0)
def n_batches(n, bs): return (n + bs - 1) // bs
def apply_mixup(x, c, y, alpha):
    if alpha <= 0: return x, c, y, y, 1.0
    lam = np.random.beta(alpha, alpha); perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * c + (1 - lam) * c[perm], y, y[perm], lam
def apply_time_masking(x, mask_pct, num_masks):
    B, _, T = x.shape; x = x.clone(); L = max(1, int(T * mask_pct))
    for _ in range(num_masks):
        s = torch.randint(0, max(1, T - L + 1), (B,), device=x.device)
        for i in range(B): x[i, :, s[i]:s[i] + L] = 0.0
    return x
def magnitude_warp(xb):
    B, C, T = xb.shape
    return xb * (1.0 + F.interpolate(torch.randn(B, C, 4, device=xb.device) * 0.1, size=T, mode='linear', align_corners=True))
def class_weight(y): return torch.tensor([1.0, float((y == 0).sum() / max(1, (y == 1).sum()))], device=DEVICE)
def ce_loss(cfg, logits, ya, yb, lam, cw):
    ls = cfg['label_smoothing']; K = logits.size(1); lp = F.log_softmax(logits, 1)
    def sm(t):
        s = torch.full_like(logits, ls / (K - 1)); s.scatter_(1, t.unsqueeze(1), 1.0 - ls); return s
    return (lam * (-sm(ya) * lp).sum(1) * cw[ya] + (1 - lam) * (-sm(yb) * lp).sum(1) * cw[yb]).mean()
def iiic_loss(pred, target, mask):
    mo = mask.unsqueeze(-1).float(); return ((pred - target) ** 2 * mo).sum() / mo.sum().clamp_min(1)
class EMA:
    def __init__(self, m, d): self.decay = d; self.shadow = {k: v.detach().clone() for k, v in m.state_dict().items()}
    def update(self, m):
        for k, v in m.state_dict().items():
            if self.shadow[k].dtype.is_floating_point: self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else: self.shadow[k] = v.detach().clone()
    def apply(self, m): m.load_state_dict(self.shadow)
print('helpers ready.')
if _TT.get('MAX_EPOCHS'):
    for _k in list(CFG):
        if 'epoch' in _k.lower(): CFG[_k] = int(_TT['MAX_EPOCHS'])
    _CFG2_EPOCH_CAP = int(_TT['MAX_EPOCHS'])      # CFG2 is defined later
    print('epoch cap ->', _TT['MAX_EPOCHS'])
else:
    _CFG2_EPOCH_CAP = None


# ===================== cell 6 =====================
_stage(5, 27, 'reference architecture')
# Reference outcome-transformer architecture (5-min-token cross-attention model)
# stochastic-depth regularizer
class DropPath(nn.Module):
    def __init__(self, p=0.0): super().__init__(); self.p = p
    def forward(self, x):
        if self.p == 0 or not self.training: return x
        keep = 1 - self.p; shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x / keep * (keep + torch.rand(shape, device=x.device)).floor_()
# one cross-attention block: learned queries attend over the token sequence (mask hides unobserved tokens)
class CrossAttnBlock(nn.Module):
    def __init__(self, d, n_heads, dropout, drop_path, ffn_expand):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm_attn = nn.LayerNorm(d); self.norm_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * ffn_expand), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * ffn_expand, d), nn.Dropout(dropout))
        self.dp_attn = DropPath(drop_path); self.dp_ffn = DropPath(drop_path)
    def forward(self, q, kv, kv_mask):
        a, _ = self.attn(self.norm_attn(q), kv, kv, key_padding_mask=(kv_mask == 0))
        q = q + self.dp_attn(a)
        return q + self.dp_ffn(self.ffn(self.norm_ffn(q)))
# REFERENCE outcome model: pools the whole recording into one outcome (strong, but not causal -> not the twin)
class OutcomeTransformer(nn.Module):
    def __init__(self, cfg, n_feat, n_clin=5):
        super().__init__(); d = cfg['d_model']
        self.tokenizer = nn.Linear(n_feat, d); self.token_norm = nn.LayerNorm(d)
        self.pos_embed = nn.Parameter(torch.randn(1, SEQ_LEN, d) * 0.02)
        # a few learned query tokens that summarize the 1008-token sequence
        self.queries = nn.Parameter(torch.randn(1, cfg['n_queries'], d) * 0.02)
        self.blocks = nn.ModuleList([CrossAttnBlock(d, cfg['n_heads'], cfg['dropout'], cfg['drop_path'], cfg['ffn_expand']) for _ in range(cfg['n_blocks'])])
        self.final_norm = nn.LayerNorm(d)
        self.clin_proj = nn.Sequential(nn.Linear(n_clin, d // 2), nn.GELU(), nn.Dropout(cfg['dropout']))
        head_in = d + d // 2
        self.classifier = nn.Sequential(nn.LayerNorm(head_in), nn.Dropout(cfg['dropout']), nn.Linear(head_in, d // 2), nn.GELU(), nn.Linear(d // 2, 2))
        # auxiliary head: reconstruct the 8-class IIIC label per token (regularizer)
        self.iiic_head = nn.Linear(d, 8)
    def forward(self, x, mask, clinical, return_aux=False):
        tok = self.token_norm(self.tokenizer(x.transpose(1, 2)))
        tok = tok + self.pos_embed
        q = self.queries.expand(tok.size(0), -1, -1)
        for blk in self.blocks: q = blk(q, tok, mask)
        # max-pool the query tokens -> one summary vector
        pooled = self.final_norm(q).max(1).values
        # concatenate the EEG summary with the clinical projection, then classify
        fused = torch.cat([pooled, self.clin_proj(clinical)], dim=1)
        logits = self.classifier(fused)
        if return_aux: return logits, self.iiic_head(tok)
        return logits
print('OutcomeTransformer ready.')

# ===================== cell 7 =====================
_stage(6, 27, 'reference trainer')
# Reference transformer: causal input builder, training loop, TTA + hour-by-hour prediction
CUTOFF_HOURS = [6, 12, 18, 24, 36, 48, 72, 84]; SLOTS_PER_HOUR = 12
def h2slot(h): return min(SEQ_LEN, int(round(h * SLOTS_PER_HOUR)))
# GPU version of the qEEG [mean,std,slope,last] summary over observed slots
def temp_feats_torch(eeg, mask):
    B, C, T = eeg.shape
    m = mask.unsqueeze(1).to(eeg.dtype); cnt = mask.sum(1).to(eeg.dtype).clamp(min=1.0).view(B, 1)
    mean = (eeg * m).sum(2) / cnt; std = ((eeg * eeg * m).sum(2) / cnt - mean * mean).clamp(min=0).sqrt()
    pos = torch.arange(T, device=eeg.device).view(1, T).expand(B, T); has = mask > 0
    last_idx = torch.where(has, pos, torch.full_like(pos, -1)).argmax(1); first_idx = torch.where(has, pos, torch.full_like(pos, T + 1)).argmin(1)
    gl = last_idx.clamp(min=0).view(B, 1, 1).expand(B, C, 1); gf = first_idx.clamp(max=T - 1).view(B, 1, 1).expand(B, C, 1)
    lv = eeg.gather(2, gl).squeeze(2); fv = eeg.gather(2, gf).squeeze(2)
    span = (last_idx - first_idx).to(eeg.dtype).clamp(min=1.0).view(B, 1)
    feats = torch.cat([mean, std, (lv - fv) / span, lv], dim=1)
    return torch.where((mask.sum(1) >= 2).view(B, 1), feats, torch.zeros_like(feats))
# append the qEEG temporal summary as extra (constant-over-time) channels to every token
def build_input(x_base, mask, tfm, tfs):
    tf = (temp_feats_torch(x_base[:, EEG0:, :], mask) - tfm) / tfs
    return torch.cat([x_base, tf.unsqueeze(-1).expand(-1, -1, x_base.size(2))], dim=1)
# anytime augmentation: randomly keep only the first few hours so the model is accurate at any cutoff
def apply_prefix_trunc(mb, min_h=4.0, max_h=84.0, p_full=0.35):
    B, T = mb.shape
    cut = (torch.empty(B, device=mb.device).uniform_(min_h, max_h) * SLOTS_PER_HOUR).long().clamp(1, T)
    cut = torch.where(torch.rand(B, device=mb.device) < p_full, torch.full_like(cut, T), cut)
    mt = mb * (torch.arange(T, device=mb.device).view(1, T) < cut.view(B, 1)).float()
    e = mt.sum(1) == 0
    if e.any(): mt[e] = mb[e]
    return mt
def _tt(a): return torch.tensor(a, dtype=torch.float32)
# train ONE reference model (mixup + time-masking + channel-drop + EMA + OneCycle + IIIC aux loss)
def train_seed(cfg, seed, Xtr, Mtr, Ctr, ytr, Atr, n_feat, tf_mean=None, tf_std=None, anytime_aug=False, use_temp_feats=True, channel_drop_p=0.15, iiic_weight=0.3):
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    model = OutcomeTransformer(cfg, n_feat).to(DEVICE)
    tfm = _tt(tf_mean).to(DEVICE).view(1, -1) if use_temp_feats else None
    tfs = _tt(tf_std).to(DEVICE).view(1, -1) if use_temp_feats else None
    Xt = _tt(Xtr); Mt = _tt(Mtr); Ct = _tt(Ctr); Yt = torch.tensor(ytr, dtype=torch.long); At = _tt(Atr)
    n = len(Xt); bs = cfg['batch_size']; epochs = cfg['max_epochs']
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = OneCycleLR(opt, max_lr=cfg['lr'], total_steps=n_batches(n, bs) * epochs, pct_start=0.1, anneal_strategy='cos')
    ema = EMA(model, cfg['ema_decay']); cw = class_weight(ytr); alpha = cfg['mixup_alpha']
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            ii = perm[i:i+bs]
            xb0 = Xt[ii].to(DEVICE); mb = Mt[ii].to(DEVICE); cb = Ct[ii].to(DEVICE); yb = Yt[ii].to(DEVICE); ab = At[ii].to(DEVICE)
            if anytime_aug: mb = apply_prefix_trunc(mb)
            xb = build_input(xb0, mb, tfm, tfs) if use_temp_feats else xb0
            if channel_drop_p > 0: xb = xb * (torch.rand(xb.size(0), xb.size(1), 1, device=xb.device) > channel_drop_p).float()
            xb = magnitude_warp(apply_time_masking(xb, cfg['mask_pct'], cfg['num_masks']))
            xb, cb, ya, yb2, lm = apply_mixup(xb, cb, yb, alpha)
            opt.zero_grad(set_to_none=True)
            logits, aux = model(xb, mb, cb, return_aux=True)
            loss = ce_loss(cfg, logits, ya, yb2, lm, cw) + iiic_weight * iiic_loss(aux, ab, mb)
            loss.backward()
            if cfg['grad_clip'] > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            opt.step(); sched.step(); ema.update(model)
    ema.apply(model); return model
@torch.no_grad()
# test-time augmentation: average over several lightly-augmented forward passes
def predict_tta(model, Xb, M, C, tf_mean=None, tf_std=None, use_temp_feats=True, n_aug=8, chdrop_p=0.15, mask_pct=0.10, batch=32):
    model.eval(); Xt = _tt(Xb); Mt = _tt(M); Ct = _tt(C)
    tfm = _tt(tf_mean).to(DEVICE).view(1, -1) if use_temp_feats else None
    tfs = _tt(tf_std).to(DEVICE).view(1, -1) if use_temp_feats else None
    def one(aug):
        out = []
        for i in range(0, len(Xt), batch):
            mb = Mt[i:i+batch].to(DEVICE)
            xb = build_input(Xt[i:i+batch].to(DEVICE), mb, tfm, tfs) if use_temp_feats else Xt[i:i+batch].to(DEVICE)
            if aug:
                xb = xb * (torch.rand(xb.size(0), xb.size(1), 1, device=xb.device) > chdrop_p).float()
                T = xb.size(2); ml = int(T * mask_pct); st = torch.randint(0, T - ml, (1,)).item(); xb = xb.clone(); xb[:, :, st:st+ml] = 0
            out.append(F.softmax(model(xb, mb, Ct[i:i+batch].to(DEVICE)), 1)[:, 1].cpu().numpy())
        return np.concatenate(out)
    return np.mean([one(False)] + [one(True) for _ in range(n_aug)], 0)
# causal hour-by-hour prediction: zero out everything after hour h, then predict
def predict_hourly(model, Xb, M, C, cutoffs, tf_mean=None, tf_std=None, use_temp_feats=True):
    out = {}
    for h in cutoffs:
        s = h2slot(h); Mh = M.copy()
        if s < M.shape[1]: Mh[:, s:] = 0.0
        valid = Mh.sum(1) > 0; Mf = Mh.copy(); Mf[~valid, 0] = 1.0
        p = predict_tta(model, Xb, Mf, C, tf_mean, tf_std, use_temp_feats=use_temp_feats).copy(); p[~valid] = np.nan
        out[h] = p
    return out
print('trainer ready.')

# ===================== cell 9 =====================
_stage(7, 27, 'block summary features')
# Step 3 — per-1h-block interpretable summary features (block means + within/causal-prefix trends + PCA + clinical)
# aggregate the 1008 five-min slots into 84 one-hour blocks (the twin's timestep)
BLK = 84; BSZ = SEQ_LEN // BLK
PROBS_CH = np.arange(STREAM_SLICES['probs'].start, STREAM_SLICES['probs'].stop)
EEG_CH   = np.arange(STREAM_SLICES['eeg'].start,   STREAM_SLICES['eeg'].stop)
ACTS_CH  = np.arange(STREAM_SLICES['acts'].start,  STREAM_SLICES['acts'].stop)
PPNET_CH = np.arange(STREAM_SLICES['ppnet'].start, STREAM_SLICES['ppnet'].stop)
# per-block mean of the chosen channels (over observed slots)
def _bmean(X_ms, M, ch):
    N = X_ms.shape[0]; Xc = X_ms[:, ch, :]; out = np.zeros((N, BLK, len(ch)), np.float32); bm = np.zeros((N, BLK), np.float32)
    for b in range(BLK):
        sl = slice(b*BSZ, (b+1)*BSZ); m = M[:, sl]; cnt = np.clip(m.sum(1), 1, None)
        bm[:, b] = (m.sum(1) > 0).astype(np.float32)
        out[:, b] = ((Xc[:, :, sl] * m[:, None, :]).sum(2) / cnt[:, None]).astype(np.float32)
    return out, bm
# WITHIN-block [std, slope, last] of qEEG
def _btemporal(X_ms, M, ch):
    N = X_ms.shape[0]; C = len(ch); Xc = X_ms[:, ch, :]
    std = np.zeros((N, BLK, C), np.float32); slope = np.zeros((N, BLK, C), np.float32); last = np.zeros((N, BLK, C), np.float32)
    pos = np.arange(BSZ); ari = np.arange(N)[:, None]; arc = np.arange(C)[None, :]
    for b in range(BLK):
        sl = slice(b*BSZ, (b+1)*BSZ); m = M[:, sl]; xb = Xc[:, :, sl]; cnt = np.clip(m.sum(1), 1, None)
        mean = (xb * m[:, None, :]).sum(2) / cnt[:, None]
        std[:, b] = np.sqrt(((xb*xb * m[:, None, :]).sum(2) / cnt[:, None] - mean*mean).clip(min=0))
        has = m > 0; li = np.where(has, pos[None, :], -1).argmax(1); fi = np.where(has, pos[None, :], BSZ + 1).argmin(1)
        lv = xb[ari, arc, li.clip(0)[:, None]]; fv = xb[ari, arc, fi.clip(max=BSZ-1)[:, None]]
        span = np.clip((li - fi).astype(np.float32), 1, None)[:, None]; last[:, b] = lv; slope[:, b] = (lv - fv) / span
    return np.concatenate([std, slope, last], 2)
# CAUSAL cumulative [mean,std,slope,last] of qEEG over all hours up to this block (Part A's key feature, made causal)
def _prefix_temporal(X_ms, M, ch):
    N, T = X_ms.shape[0], X_ms.shape[2]; C = len(ch); Xc = X_ms[:, ch, :]; pos = np.arange(T)
    csum = np.cumsum(Xc * M[:, None, :], 2); csq = np.cumsum(Xc * Xc * M[:, None, :], 2); ccnt = np.cumsum(M, 1)
    ari = np.arange(N)[:, None]; arc = np.arange(C)[None, :]; out = np.zeros((N, BLK, 4 * C), np.float32)
    for b in range(BLK):
        e = (b + 1) * BSZ - 1; cnt = np.clip(ccnt[:, e], 1, None)[:, None]
        mean = csum[:, :, e] / cnt; std = np.sqrt((csq[:, :, e] / cnt - mean * mean).clip(min=0))
        obs = M[:, :e + 1]; li = np.where(obs > 0, pos[:e + 1][None, :], -1).max(1); fi = np.where(obs > 0, pos[:e + 1][None, :], T + 1).min(1)
        valid = li >= 0
        lv = Xc[ari, arc, li.clip(0)[:, None]]; fv = Xc[ari, arc, fi.clip(0, T - 1)[:, None]]
        span = np.clip((li - fi).astype(np.float32), 1, None)[:, None]
        feats = np.concatenate([mean, std, (lv - fv) / span, lv], 1).astype(np.float32); feats[~valid] = 0.0; out[:, b] = feats
    return out
def _stats(mode, X_ms, M):
    if mode == 'rich':        return _btemporal(X_ms, M, EEG_CH)
    if mode == 'rich_prefix': return _prefix_temporal(X_ms, M, EEG_CH)
    return np.zeros((X_ms.shape[0], BLK, 0), np.float32)
# the 'rollable' summary = block-mean IIIC+qEEG+prototypes + PCA-32 of ProtoPNet (the part the twin forecasts)
def _roll(mode, X_ms, M, pca_ch, sp, pca):
    rich = mode in ('rich', 'rich_prefix')
    mc, bm = _bmean(X_ms, M, np.concatenate([PROBS_CH, EEG_CH] + ([ACTS_CH] if rich else [])))
    mp, _ = _bmean(X_ms, M, pca_ch)
    p32 = pca.transform(sp.transform(mp.reshape(-1, mp.shape[2]))).reshape(len(mc), BLK, 32).astype(np.float32)
    return np.concatenate([mc, p32], 2), bm
# fit the block-summary scaler + ProtoPNet PCA on TRAIN only
def fit_refs(mode, X_ms, M):
    rich = mode in ('rich', 'rich_prefix'); pca_ch = PPNET_CH if rich else np.concatenate([PPNET_CH, ACTS_CH])
    mp, bm = _bmean(X_ms, M, pca_ch); sp = StandardScaler().fit(mp[bm > 0]); pca = PCA(32, random_state=0).fit(sp.transform(mp[bm > 0]))
    roll, _ = _roll(mode, X_ms, M, pca_ch, sp, pca); stats = _stats(mode, X_ms, M)
    refs = dict(mode=mode, pca_ch=pca_ch, sp=sp, pca=pca, roll_dim=roll.shape[2], stats_dim=stats.shape[2])
    full = np.concatenate([roll, stats], 2); refs['sc'] = StandardScaler().fit(full[bm > 0]); return refs
# build the per-block Step-3 vector: summary + delta-from-prior-block + clinical; FT = the forecast target
def make_features(refs, X_ms, M, clin):
    roll, bm = _roll(refs['mode'], X_ms, M, refs['pca_ch'], refs['sp'], refs['pca']); stats = _stats(refs['mode'], X_ms, M)
    full = np.concatenate([roll, stats], 2); D = full.shape[2]
    z = (refs['sc'].transform(full.reshape(-1, D)).reshape(full.shape).astype(np.float32)) * bm[:, :, None]
    rd = refs['roll_dim']; delta = np.zeros((len(z), BLK, rd), np.float32)
    for i in range(len(z)):
        prev = None
        for b in range(BLK):
            if bm[i, b] > 0: delta[i, b] = 0.0 if prev is None else z[i, b, :rd] - z[i, prev, :rd]; prev = b
    Xf = np.concatenate([z, delta, np.repeat(clin[:, None, :], BLK, 1)], 2).astype(np.float32); FT = z[:, :, :rd].copy()
    return Xf, FT, bm
print('block builders ready (compact / rich / rich_prefix).')

# ===================== cell 11 =====================
_stage(8, 27, 'twin model and trainer')
# Step 4 — digital-twin transformer (dual-head causal model) + trainer + roll-forward simulation
# index of each patient's last observed block
def last_block(BM): return np.where(BM > 0, np.arange(BM.shape[1])[None, :], -1).max(1)
# Step-4 DIGITAL TWIN: causal block transformer with two heads (outcome + next-block forecast)
class TwinTransformer(nn.Module):
    def __init__(self, in_dim, roll_dim, d=256, nlayers=4, nheads=8, dropout=0.2, nblk=BLK):
        super().__init__()
        # learned embedding for unobserved blocks (they DO participate in this causal attention)
        self.embed = nn.Linear(in_dim, d); self.missing = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, nblk, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nheads, 2 * d, dropout, batch_first=True, activation='gelu', norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers); self.norm = nn.LayerNorm(d); self.roll_dim = roll_dim
        # primary = outcome logit; delta_head = forecast of next block's summary; iiic = aux label head
        self.outcome = nn.Linear(d, 1); self.delta_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, roll_dim)); self.iiic = nn.Linear(d, 8)
        # causal mask: block t can only attend to blocks <= t
        self.register_buffer('cmask', torch.triu(torch.full((nblk, nblk), float('-inf')), diagonal=1))
    def forward(self, x, bmask):
        h = self.embed(x); h = torch.where(bmask.unsqueeze(-1).bool(), h, self.missing) + self.pos
        h = self.norm(self.enc(h, mask=self.cmask))
        return self.outcome(h).squeeze(-1), x[:, :, :self.roll_dim] + self.delta_head(h), h, self.iiic(h)
CFG2 = dict(nlayers=4, nheads=8, dropout=0.2, lr=1e-3, weight_decay=1e-4, epochs=70, batch=32, lam_fc=1.5, ss_prob=0.6, ss_w=0.7, ss_R=18, ls=0.05, feat_drop=0.1, blk_drop=0.1, iiic_w=0.3, ema=0.998)
if _CFG2_EPOCH_CAP:
    CFG2['epochs'] = _CFG2_EPOCH_CAP
# distillation loss: match the reference teacher's soft outcome (kdbce / logit-MSE / KL)
def distill_term(out, soft, mode, T):
    soft = soft.clamp(1e-4, 1 - 1e-4); zt = torch.log(soft / (1 - soft))
    if mode == 'logitmse': return (out - zt) ** 2
    pT = torch.sigmoid(zt / T)
    if mode == 'kl':
        qS = torch.sigmoid(out / T).clamp(1e-4, 1 - 1e-4)
        return (pT * torch.log(pT / qS) + (1 - pT) * torch.log((1 - pT) / (1 - qS))) * (T * T)
    return F.binary_cross_entropy_with_logits(out / T, pT, reduction='none') * (T * T)
# train ONE twin: outcome BCE + forecast MSE + IIIC aux + distillation + scheduled-sampling roll-forward
def train_twin(cfg, seed, Xf, BM, FT, y, AUX=None, soft=None, d=256, distill_w=2.0, distill_T=2.0, distill_mode='kdbce', hard_w=1.0):
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    rd = FT.shape[2]; sdim = Xf.shape[2] - 2 * rd - 5; ds = rd + sdim
    model = TwinTransformer(Xf.shape[2], rd, d, cfg['nlayers'], cfg['nheads'], cfg['dropout']).to(DEVICE)
    Xt = _tt(Xf); Bt = _tt(BM); Ft = _tt(FT); Yt = _tt(y)
    At = _tt(AUX) if AUX is not None else None; St = _tt(soft) if soft is not None else None
    pos_w = torch.tensor([(y == 0).sum() / max(1, (y == 1).sum())], device=DEVICE); n = len(Xt); bs = cfg['batch']
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = OneCycleLR(opt, max_lr=cfg['lr'], total_steps=((n + bs - 1) // bs) * cfg['epochs'], pct_start=0.1)
    ema = EMA(model, cfg['ema'])
    for ep in range(cfg['epochs']):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            ii = perm[i:i+bs]; xb = Xt[ii].to(DEVICE); bm = Bt[ii].to(DEVICE); ft = Ft[ii].to(DEVICE); yb = Yt[ii].to(DEVICE)
            xb = xb * (torch.rand(xb.size(0), 1, xb.size(2), device=DEVICE) > cfg['feat_drop']).float()
            bm2 = bm * (torch.rand_like(bm) > cfg['blk_drop']).float(); em = bm2.sum(1) == 0; bm2[em] = bm[em]; bm = bm2
            out, fc, _, aux = model(xb, bm); yt = yb[:, None].expand(-1, BLK) * (1 - cfg['ls']) + 0.5 * cfg['ls']
            ol = F.binary_cross_entropy_with_logits(out, yt, pos_weight=pos_w, reduction='none'); ol = (ol * bm).sum() / bm.sum().clamp(min=1)
            loss = hard_w * ol
            # secondary loss: forecast head predicts the NEXT block's summary vector
            if cfg['lam_fc'] > 0:
                fmask = bm[:, :-1] * bm[:, 1:]; fl = (((fc[:, :-1] - ft[:, 1:]) ** 2).mean(-1) * fmask).sum() / fmask.sum().clamp(min=1); loss = loss + cfg['lam_fc'] * fl
            if At is not None:
                ab = At[ii].to(DEVICE); al = ((aux - ab) ** 2).mean(-1); loss = loss + cfg['iiic_w'] * (al * bm).sum() / bm.sum().clamp(min=1)
            # distillation: pull the twin's outcome toward the teacher's soft prediction
            if St is not None and distill_w > 0:
                dl = distill_term(out, St[ii].to(DEVICE), distill_mode, distill_T); loss = loss + distill_w * (dl * bm).sum() / bm.sum().clamp(min=1)
            # scheduled sampling: feed the twin its OWN forecasts and still require the right outcome (trains roll-forward)
            if cfg['ss_w'] > 0 and random.random() < cfg['ss_prob']:
                R = cfg['ss_R']; s0 = random.randint(2, max(3, BLK - R - 1)); xr = xb.clone(); bmr = bm.clone()
                with torch.no_grad():
                    for st in range(s0, min(s0 + R, BLK - 1)):
                        _, fcr, _, _ = model(xr, bmr); nm = fcr[:, st]; xr[:, st+1, :rd] = nm
                        if sdim > 0: xr[:, st+1, rd:rd+sdim] = xr[:, st, rd:rd+sdim]
                        xr[:, st+1, ds:ds+rd] = nm - xr[:, st, :rd]; bmr[:, st+1] = 1.0
                out_r, _, _, _ = model(xr, bmr); e = min(s0 + R, BLK)
                loss = loss + cfg['ss_w'] * F.binary_cross_entropy_with_logits(out_r[:, s0:e], yb[:, None].expand(-1, e - s0), pos_weight=pos_w)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); ema.update(model)
    ema.apply(model); return model
@torch.no_grad()
# Step 4 transformer outputs (per patient, per 1h block):
#   [0] outcome  (N, 84)      logit -> P(good outcome) at each block        (primary head)
#   [1] forecast (N, 84, 98)  predicted NEXT block's summary vector         (secondary head)
#   [2] hidden   (N, 84, d)   per-block hidden state = trajectory embedding (feeds Steps 5-6)
#   ([3] iiic    (N, 84, 8)   auxiliary 8-class IIIC label reconstruction, training-only)
def predict_twin(model, Xf, BM):
    model.eval(); o, fc, h, _ = model(_tt(Xf).to(DEVICE), _tt(BM).to(DEVICE)); return o.cpu().numpy(), fc.cpu().numpy(), h.cpu().numpy()
def auc(y, p): return float(roc_auc_score(y, p))
# outcome read directly at hour K (only blocks <= K observed)
def direct_prob(model, Xf, BM, K):
    bm = BM.copy(); bm[:, K:] = 0.0; lo, _, _ = predict_twin(model, Xf, bm); return 1 / (1 + np.exp(-lo[:, K - 1]))
# DIGITAL TWIN: from K observed hours, autoregressively forecast future blocks to 84 h, then read the outcome
def roll_forward(model, Xf, BM, k_obs):
    rd = model.roll_dim; sdim = Xf.shape[2] - 2 * rd - 5; ds = rd + sdim
    xf = Xf.copy(); m = np.zeros_like(BM); m[:, :k_obs] = BM[:, :k_obs]
    for b in range(k_obs, BLK):
        _, fc, _ = predict_twin(model, xf, m); nm = fc[:, b - 1]; xf[:, b, :rd] = nm
        if sdim > 0: xf[:, b, rd:rd + sdim] = xf[:, b - 1, rd:rd + sdim]
        xf[:, b, ds:ds + rd] = nm - xf[:, b - 1, :rd]; m[:, b] = 1.0
    out, _, _ = predict_twin(model, xf, m); return 1 / (1 + np.exp(-out)), m
def ens_term(models, Xf, BM, K): return np.mean([roll_forward(mdl, Xf, BM, K)[0][:, -1] for mdl in models], 0)
def ens_direct(models, Xf, BM, K): return np.mean([direct_prob(mdl, Xf, BM, K) for mdl in models], 0)
# hour-by-hour outcome AUC, read causally at each patient's last observed block <= h
def twin_hourly(mods, Xf, BM, cutoffs):
    out = np.mean([predict_twin(m, Xf, BM)[0] for m in mods], 0); res = {}
    for h in cutoffs:
        cap = min(h, BLK); obs = BM[:, :cap]; valid = obs.sum(1) > 0
        lbk = np.where(obs > 0, np.arange(cap)[None, :], -1).max(1)
        p = 1 / (1 + np.exp(-out[np.arange(len(out)), lbk.clip(0)])); res[h] = auc(y_test[valid], p[valid]) if len(np.unique(y_test[valid])) > 1 else float('nan')
    return res

# ===================== cell 13 =====================
_stage(9, 27, 'seed configuration')
# Configuration — seed counts (2 for dev, raise to 10 for final) and distillation settings
# seed counts: 2 for quick dev; raise to ~10 for the final figures
N_REF = int(_TT.get('N_REF', 10))
N_TWIN = int(_TT.get('N_TWIN', 10))
N_ABL = int(_TT.get('N_ABL', 10))
N_CAL = int(_TT.get('N_CAL', 10))
# locked twin recipe: lean on the teacher (low hard-label weight) + strong distillation
TWIN_CFG = dict(hard_w=0.3, distill_w=8, distill_mode='kdbce')
KS = CUTOFF_HOURS
# reliability-curve helper for the calibration plots
def reliab(y, p, nb=10):
    e = np.linspace(0, 1, nb + 1); idx = np.clip(np.digitize(p, e) - 1, 0, nb - 1); xs, ys = [], []
    for b in range(nb):
        m = idx == b
        if m.sum() > 0: xs.append(p[m].mean()); ys.append(y[m].mean())
    return np.array(xs), np.array(ys)

# ===================== cell 15 =====================
_stage(10, 27, 'TRAIN reference transformer')
# Train the reference transformer (distillation teacher); its hour-by-hour AUC and soft per-block targets
# train the reference-model ensemble = the distillation TEACHER (anytime-trained)
_pstep(f'reference transformer: {N_REF} seeds')
REFM = [train_seed(CFG, sd, XB_tr, M_ms_tr, C_train_s, y_train, IIIC_TR, N_FEAT, TF_MEAN, TF_STD, anytime_aug=True)
        for sd in _bar(range(7, 7 + 10 * N_REF, 10), 'reference seeds', total=N_REF, unit='seed')]
# teacher's hour-by-hour test prediction (TTA-averaged across seeds)
ref_hourly = {h: np.nanmean([predict_hourly(m, XB_te, M_ms_te, C_test_s, KS, TF_MEAN, TF_STD)[h] for m in REFM], 0) for h in KS}
covO = {h: (M_ms_te[:, :h2slot(h)].sum(1) > 0) for h in KS}
REF_HOURLY = {h: auc(y_test[covO[h]], ref_hourly[h][covO[h]]) for h in KS}
@torch.no_grad()
# teacher's soft P(good) at each block on TRAIN -> the targets the twin is distilled toward
def ref_soft_blocks(models, Xb, M, C):
    soft = np.zeros((len(Xb), BLK), np.float32)
    for b in range(BLK):
        s = h2slot(b + 1); Mh = M.copy(); Mh[:, s:] = 0.0; valid = Mh.sum(1) > 0; Mf = Mh.copy(); Mf[~valid, 0] = 1.0
        p = np.mean([predict_tta(m, Xb, Mf, C, TF_MEAN, TF_STD, n_aug=0) for m in models], 0); p[~valid] = 0.5; soft[:, b] = p
    return soft
SOFT_TR = ref_soft_blocks(REFM, XB_tr, M_ms_tr, C_train_s)
print('reference @84h', round(REF_HOURLY[84], 4))

# ===================== cell 17 =====================
_stage(11, 27, 'TRAIN digital twin')
# Build block features and train the distilled digital twin (seed ensemble)
# fit the Step-3 block features on TRAIN (causal-prefix variant)
BREF = fit_refs('rich_prefix', X_ms_tr, M_ms_tr)
Xf_tr, FT_tr, bmtr = make_features(BREF, X_ms_tr, M_ms_tr, C_train_s)
Xf_te, FT_te, bmte = make_features(BREF, X_ms_te, M_ms_te, C_test_s)
# per-block IIIC label frequencies = the twin's auxiliary target
AUXTR = _bmean(X_ms_tr, M_ms_tr, PROBS_CH)[0]; lb = last_block(bmte)
# train the distilled digital-twin ensemble (THE deliverable model)
_pstep(f'digital twin: {N_TWIN} seeds')
TW = [train_twin(CFG2, sd, Xf_tr, bmtr, FT_tr, y_train, AUX=AUXTR, soft=SOFT_TR, **TWIN_CFG)
      for sd in _bar(range(7, 7 + 10 * N_TWIN, 10), 'twin seeds', total=N_TWIN, unit='seed')]
# the twin's full-data per-block P(good) for every test patient
tw_prob = 1 / (1 + np.exp(-np.mean([predict_twin(m, Xf_te, bmte)[0] for m in TW], 0)))
# which summary-vector columns belong to each feature group (for Figure 3)
GROUPS = {'IIIC probs': slice(0, 8), 'qEEG': slice(8, 21), 'prototype acts': slice(21, 66), 'ProtoPNet PCA': slice(66, 98)}
print('twin trained', len(TW), 'seeds')

# ===================== cell 19 =====================
_stage(12, 27, 'Figure 1 — input ablation (RETRAINS PER FEATURE SET, the longest block)')
# Figure 1 — input ablation: outcome AUC + calibration per feature set
# clinical-only baseline = simple logistic regression
from sklearn.linear_model import LogisticRegression
ABL_SEEDS = list(range(7, 7 + 10 * N_ABL, 10))
# train an outcome model on a chosen subset of input streams; return its hour-by-hour AUC + test probs
def scenario(streams, use_clin, has_eeg):
    global EEG0
    if streams is None:
        p = LogisticRegression(max_iter=1000, class_weight='balanced').fit(C_train_s, y_train).predict_proba(C_test_s)[:, 1]
        return [auc(y_test, p)] * len(KS), p
    Xtr = slice_streams(X_ms_tr_s, streams); Xte = slice_streams(X_ms_te_s, streams); nf = Xtr.shape[1]; n_feat = nf + 52 if has_eeg else nf
    Ctr = C_train_s if use_clin else np.zeros_like(C_train_s); Cte = C_test_s if use_clin else np.zeros_like(C_test_s)
    e0 = EEG0; EEG0 = nf - 13
    try:
        mods = [train_seed(CFG, sd, Xtr, M_ms_tr, Ctr, y_train, IIIC_TR, n_feat, TF_MEAN, TF_STD, use_temp_feats=has_eeg, anytime_aug=True) for sd in ABL_SEEDS]
        hr = {h: np.nanmean([predict_hourly(m, Xte, M_ms_te, Cte, KS, TF_MEAN, TF_STD, use_temp_feats=has_eeg)[h] for m in mods], 0) for h in KS}
        p = np.mean([predict_tta(m, Xte, M_ms_te, Cte, TF_MEAN, TF_STD, use_temp_feats=has_eeg) for m in mods], 0)
    finally:
        EEG0 = e0
    return [auc(y_test[covO[h]], hr[h][covO[h]]) for h in KS], p
# the input combinations to compare (individual streams + ProtoPNet/CEBRA combos)
SPEC = {'ProtoPNet': (['ppnet'], False, False), 'CEBRA': (['cebra'], False, False), 'EEG': (['eeg'], False, True), 'Clinical': (None, False, False),
        'ProtoPNet+EEG': (['ppnet', 'eeg'], False, True), 'ProtoPNet+Clinical': (['ppnet'], True, False), 'ProtoPNet+EEG+Clinical': (['ppnet', 'eeg'], True, True),
        'CEBRA+EEG': (['cebra', 'eeg'], False, True), 'CEBRA+Clinical': (['cebra'], True, False), 'CEBRA+EEG+Clinical': (['cebra', 'eeg'], True, True)}
SC = {k: scenario(*v) for k, v in
      _bar(list(SPEC.items()), 'ablation feature sets', total=len(SPEC), unit='set')}
ROWS = [('Individual streams', ['ProtoPNet', 'CEBRA', 'EEG', 'Clinical']),
        ('ProtoPNet combinations', ['ProtoPNet', 'ProtoPNet+EEG', 'ProtoPNet+Clinical', 'ProtoPNet+EEG+Clinical']),
        ('CEBRA combinations', ['CEBRA', 'CEBRA+EEG', 'CEBRA+Clinical', 'CEBRA+EEG+Clinical'])]
CMAP = {'Individual streams': plt.cm.Greens, 'ProtoPNet combinations': plt.cm.Blues, 'CEBRA combinations': plt.cm.Purples}
fig, axes = plt.subplots(3, 2, figsize=(13, 14.5))
for r, (gname, names) in enumerate(ROWS):
    axa, axc = axes[r]; shades = np.linspace(0.5, 0.95, len(names))
    for j, name in enumerate(names):
        hauc, p = SC[name]; br = brier_score_loss(y_test, p); col = CMAP[gname](shades[j])
        axa.plot(KS, hauc, ('o:' if name == 'Clinical' else 'o-'), lw=2.2, color=col, label=f'{name} ({hauc[-1]:.3f})')
        xr, yr = reliab(y_test, p); axc.plot(xr, yr, 'o-', lw=2.2, color=col, label=f'{name} (Brier {br:.3f})')
    axa.set_ylim(0.55, 0.96); axa.set_xlabel('hours of EEG used'); axa.set_ylabel('outcome AUC'); axa.set_title(f'{gname} — AUC'); axa.legend(fontsize=8); axa.grid(alpha=0.3)
    axc.plot([0, 1], [0, 1], 'k:', lw=1); axc.set_xlabel('predicted P(good)'); axc.set_ylabel('observed frequency'); axc.set_title(f'{gname} — calibration'); axc.legend(fontsize=8); axc.grid(alpha=0.3)
fig.suptitle('Figure 1 — Input ablation: outcome AUC and calibration by feature set', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.975]); fig.savefig(FIG_TWIN / 'fig1.png', dpi=130, bbox_inches='tight'); plt.show()
print({k: round(SC[k][0][-1], 3) for k in SPEC})

# ===================== cell 21 =====================
_stage(13, 27, 'Figure 2 — per-block AUC')
# Figure 2 — per-block outcome AUC, twin vs reference transformer
# twin vs reference: hour-by-hour outcome AUC
tw_h = twin_hourly(TW, Xf_te, bmte, KS)
fig, ax = plt.subplots(figsize=(7, 5.2))
ax.plot(KS, [tw_h[h] for h in KS], 'o-', lw=2.6, label=f'digital twin ({tw_h[84]:.3f})')
ax.plot(KS, [REF_HOURLY[h] for h in KS], 's--', lw=2, color='k', alpha=0.7, label=f'reference transformer ({REF_HOURLY[84]:.3f})')
ax.set_xlabel('hours of EEG used'); ax.set_ylabel('outcome AUC'); ax.set_ylim(0.74, 0.96)
ax.set_title('Figure 2 — Outcome head: reference transformer vs digital twin'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG_TWIN / 'fig2.png', dpi=130); plt.show()
print({h: round(tw_h[h], 3) for h in KS})

# ===================== cell 23 =====================
_stage(14, 27, 'Figure 3 — forecast skill')
# Figure 3 — forecast head skill vs persistence
# only score forecasts between consecutive observed blocks
fmask = (bmte[:, :-1] * bmte[:, 1:]).astype(bool); fc = np.mean([predict_twin(m, Xf_te, bmte)[1] for m in TW], 0)
def fmse(a, b, sl): return ((a[:, :-1, sl] - b[:, 1:, sl]) ** 2).mean(-1)[fmask].mean()
# forecast skill vs persistence, per feature group (>0 beats 'next = current')
skill = {g: float(100 * (1 - fmse(fc, FT_te, sl) / fmse(FT_te, FT_te, sl))) for g, sl in {**GROUPS, 'overall': slice(None)}.items()}
fig, ax = plt.subplots(figsize=(7, 5.2))
ax.bar(range(len(skill)), list(skill.values()), color=['C2' if v > 0 else 'C3' for v in skill.values()])
ax.axhline(0, color='k', lw=0.8); ax.set_xticks(range(len(skill))); ax.set_xticklabels(list(skill.keys()), rotation=20, ha='right', fontsize=9)
ax.set_ylabel('forecast skill vs persistence (%)'); ax.set_title('Figure 3 — Forecast head vs persistence'); ax.grid(alpha=0.3, axis='y')
fig.tight_layout(); fig.savefig(FIG_TWIN / 'fig3.png', dpi=130); plt.show()
print({k: round(v, 1) for k, v in skill.items()})

# ===================== cell 25 =====================
_stage(15, 27, 'Figure 4 — calibration (TRAINS two more ensembles)')
# Figure 4 — held-out outcome calibration (temperature scaling)
# hold out a calibration fold; the teacher + twin here are trained WITHOUT it (leakage-clean)
rng = np.random.RandomState(0); perm = rng.permutation(len(y_train)); cal_i = perm[:len(y_train)//5]; fit_i = perm[len(y_train)//5:]
_pstep(f'calibration reference: {N_CAL} seeds')
RC = [train_seed(CFG, sd, XB_tr[fit_i], M_ms_tr[fit_i], C_train_s[fit_i], y_train[fit_i], IIIC_TR[fit_i], N_FEAT, TF_MEAN, TF_STD, anytime_aug=True)
      for sd in _bar(range(7, 7 + 10 * N_CAL, 10), 'calibration seeds', total=N_CAL, unit='seed')]
SOFT_FIT = ref_soft_blocks(RC, XB_tr[fit_i], M_ms_tr[fit_i], C_train_s[fit_i])
_pstep(f'calibration twin: {N_CAL} seeds')
TWC = [train_twin(CFG2, sd, Xf_tr[fit_i], bmtr[fit_i], FT_tr[fit_i], y_train[fit_i], AUX=AUXTR[fit_i], soft=SOFT_FIT, **TWIN_CFG)
       for sd in _bar(range(7, 7 + 10 * N_CAL, 10), 'calibration twin seeds', total=N_CAL, unit='seed')]
def sig(z): return 1 / (1 + np.exp(-z))
# fit one temperature on the held-out fold (grid search minimizing BCE)
def fit_T(logit, y):
    ts = np.linspace(0.5, 8.0, 301)
    bce = [-np.mean(y * np.log(np.clip(sig(logit / t), 1e-6, 1 - 1e-6)) + (1 - y) * np.log(np.clip(1 - sig(logit / t), 1e-6, 1 - 1e-6))) for t in ts]
    return float(ts[int(np.argmin(bce))])
clb = last_block(bmtr[cal_i]); cok = clb >= 0
cal_logit = np.mean([predict_twin(m, Xf_tr[cal_i], bmtr[cal_i])[0] for m in TWC], 0)[np.arange(len(cal_i)), clb.clip(0)]
T = fit_T(cal_logit[cok], y_train[cal_i][cok].astype(float))
tok = lb >= 0; te_logit = np.mean([predict_twin(m, Xf_te, bmte)[0] for m in TWC], 0)[np.arange(len(y_test)), lb.clip(0)]
p_raw = sig(te_logit)[tok]; p_cal = sig(te_logit / T)[tok]; yt = y_test[tok]
b_raw = float(brier_score_loss(yt, p_raw)); b_cal = float(brier_score_loss(yt, p_cal))
fig, ax = plt.subplots(figsize=(6, 5.4)); ax.plot([0, 1], [0, 1], 'k:', lw=1, label='perfect')
xr, yr = reliab(yt, p_raw); xc, yc = reliab(yt, p_cal)
ax.plot(xr, yr, 'o-', color='C3', label=f'raw (Brier {b_raw:.3f})'); ax.plot(xc, yc, 's-', color='C0', label=f'temp-scaled T={T:.2f} (Brier {b_cal:.3f})')
ax.set_xlabel('predicted P(good)'); ax.set_ylabel('observed frequency'); ax.set_title('Figure 4 — Outcome calibration (held-out)'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG_TWIN / 'fig4.png', dpi=130); plt.show()
print(f'Brier {b_raw:.3f} -> {b_cal:.3f} (T={T:.2f})')

# ===================== cell 27 =====================
_stage(16, 27, 'Figure 5 — roll-forward')
# Figure 5 — digital-twin roll-forward from 6 h of EEG
# roll the twin forward from the first 6 h of EEG
K_OBS = 6; obs_blk = bmte.sum(1); plast = tw_prob[np.arange(len(y_test)), lb]; e6 = bmte[:, :K_OBS].sum(1) > 0
TWIN_ROLL = np.mean([roll_forward(m, Xf_te, bmte, K_OBS)[0] for m in TW], 0); twin6 = TWIN_ROLL[:, -1]
# pick clear good / poor example patients to display
goods = np.where((y_test == 1) & (obs_blk >= 40) & e6 & (plast > 0.6) & (twin6 > 0.5))[0]
poors = np.where((y_test == 0) & (obs_blk >= 40) & e6 & (plast < 0.4) & (twin6 < 0.5))[0]
if len(goods) == 0: goods = np.where((y_test == 1) & (obs_blk >= 40) & e6)[0]
if len(poors) == 0: poors = np.where((y_test == 0) & (obs_blk >= 40) & e6)[0]
cg = int(goods[np.argmax(obs_blk[goods])]); cp = int(poors[np.argmax(obs_blk[poors])])
cg2 = int(next((g for g in goods[np.argsort(-obs_blk[goods])] if int(g) not in (cg, cp)), cg))
hh = np.arange(BLK) + 1; fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, ex in zip(axes, [cg, cp, cg2]):
    pr = np.where(bmte[ex] > 0, tw_prob[ex], np.nan); ps = TWIN_ROLL[ex]
    ax.plot(hh, pr, '-', color='C0', alpha=0.5, label='full-data prediction')
    ax.plot(hh[:K_OBS], ps[:K_OBS], '-', color='C3', lw=2.6, label=f'observed ({K_OBS} h)')
    ax.plot(hh[K_OBS-1:], ps[K_OBS-1:], '--', color='C3', lw=2.6, label='twin (rolled forward)')
    ax.axhline(0.5, color='gray', ls=':'); ax.set_xlabel('hours'); ax.set_ylim(-0.02, 1.02)
    ax.set_title(f'{TEST_PIDS[ex]} (true {"good" if y_test[ex] else "poor"})'); ax.grid(alpha=0.3)
axes[0].set_ylabel('P(good)'); axes[0].legend(fontsize=8, loc='best')
fig.suptitle('Figure 5 — Digital twin roll-forward (from 6 h of EEG)'); fig.tight_layout(); fig.savefig(FIG_TWIN / 'fig5.png', dpi=130); plt.show()
print([TEST_PIDS[i] for i in (cg, cp, cg2)])

# ===================== cell 29 =====================
_stage(17, 27, 'Figure 6 — overview spaghetti')
# Figure 6 — overview spaghetti (left) + per-window small multiples (right)
obs_blk = bmte.sum(1); plast = tw_prob[np.arange(len(y_test)), lb]
cand = np.where((y_test == 1) & (obs_blk >= 60) & (bmte[:, :6].sum(1) > 0) & (plast > 0.85))[0]
# score candidate patients by how closely the forecasts track the official curve
def _fgap(ex):
    off = tw_prob[ex]; g = []
    for K in (12, 24):
        r = roll_forward(TW[0], Xf_te[ex:ex+1], bmte[ex:ex+1], K)[0][0]
        reg = bmte[ex].astype(bool).copy(); reg[:K] = False
        if reg.sum(): g.append(float(np.abs(r[reg] - off[reg]).mean()))
    return np.mean(g) if g else 9.0
EX = int(min(cand, key=_fgap)) if len(cand) else int(np.argmax(obs_blk))
last_h = int(bmte[EX].nonzero()[0].max()) + 1
hh = np.arange(BLK) + 1
windows = [K for K in range(6, BLK, 6) if K < last_h]
small = [K for K in [6, 12, 24, 36, 48, 72] if K < last_h]
off = np.where(bmte[EX, :last_h] > 0, tw_prob[EX, :last_h], np.nan)
# ensemble roll-forward outcome trajectory for each 6-h observation window
ROLLS = {K: np.mean([roll_forward(m, Xf_te[EX:EX+1], bmte[EX:EX+1], K)[0][0] for m in TW], 0) for K in windows}
cmap = plt.cm.viridis; cnorm = lambda K: cmap(K / max(windows))
fig = plt.figure(figsize=(16, 6.8), layout='constrained')
sfL, sfR = fig.subfigures(1, 2, width_ratios=[1.0, 1.15])
axB = sfL.subplots()
for K in windows: axB.plot(hh[K-1:], ROLLS[K][K-1:], '-', color=cnorm(K), lw=1.6, alpha=0.85)
axB.plot(hh[:last_h], off, '-', color='k', lw=3, label='full-data', zorder=6)
axB.axhline(0.5, color='gray', ls=':'); axB.set_ylim(-0.02, 1.02)
axB.set_xlabel('hours', fontsize=11); axB.set_ylabel('P(good)', fontsize=11); axB.set_title('overview — forecast every 6 h')
axB.legend(fontsize=9, loc='lower right'); axB.grid(alpha=0.3)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(6, max(windows))); sm.set_array([])
sfL.colorbar(sm, ax=axB, fraction=0.046, pad=0.02).set_label('forecast started at (h)')
nrs = int(np.ceil(len(small) / 3))
axes = np.atleast_1d(sfR.subplots(nrs, 3, sharex=True, sharey=True)).ravel()
for j, K in enumerate(small):
    ax = axes[j]; ax.plot(hh[:last_h], off, '-', color='k', lw=1.4)
    ax.plot(hh[K-1:], ROLLS[K][K-1:], '-', color=cnorm(K), lw=2.6)
    ax.plot(hh[K-1], ROLLS[K][K-1], 'o', color=cnorm(K), ms=6)
    ax.axvline(K, color='gray', ls=':', lw=0.8); ax.axhline(0.5, color='gray', ls=':', lw=0.8)
    ax.set_ylim(-0.02, 1.02); ax.set_title(f'from {K} h', fontsize=10); ax.grid(alpha=0.3)
for j in range(len(small), len(axes)): axes[j].axis('off')
sfR.supxlabel('hours', fontsize=11); sfR.supylabel('P(good)', fontsize=11)
fig.suptitle(f'Figure 6 — Twin roll-forward by observation window ({TEST_PIDS[EX]}, true {"good" if y_test[EX] else "poor"})', fontsize=13)
fig.get_layout_engine().set(rect=(0, 0, 1, 0.96))
fig.savefig(FIG_TWIN / 'fig6.png', dpi=130, bbox_inches='tight'); plt.show()
print('Figure 6 patient:', TEST_PIDS[EX])

# ===================== cell 31 =====================
_stage(18, 27, 'Figure 7 — label-frequency forecast')
# Figure 7 — forecast head vs actual: IIIC label-frequency trajectories (black = actual, colored = forecast from 12h/24h)
obs_blk = bmte.sum(1); plast = tw_prob[np.arange(len(y_test)), lb]
cand = np.where((y_test == 1) & (obs_blk >= 60) & (bmte[:, :6].sum(1) > 0) & (plast > 0.85))[0]
def _fgap(ex):
    off = tw_prob[ex]; g = []
    for K in (12, 24):
        r = roll_forward(TW[0], Xf_te[ex:ex+1], bmte[ex:ex+1], K)[0][0]
        reg = bmte[ex].astype(bool).copy(); reg[:K] = False
        if reg.sum(): g.append(float(np.abs(r[reg] - off[reg]).mean()))
    return np.mean(g) if g else 9.0
EX = int(min(cand, key=_fgap)) if len(cand) else int(np.argmax(obs_blk))
last_h = int(bmte[EX].nonzero()[0].max()) + 1; hh = np.arange(BLK) + 1
# like roll_forward but returns the forecasted SUMMARY vector (not the outcome)
def roll_features(model, Xf, BM, k_obs):
    rd = model.roll_dim; sdim = Xf.shape[2] - 2 * rd - 5; ds = rd + sdim
    xf = Xf.copy(); m = np.zeros_like(BM); m[:, :k_obs] = BM[:, :k_obs]
    for b in range(k_obs, BLK):
        _, fc, _ = predict_twin(model, xf, m); nm = fc[:, b - 1]; xf[:, b, :rd] = nm
        if sdim > 0: xf[:, b, rd:rd + sdim] = xf[:, b - 1, rd:rd + sdim]
        xf[:, b, ds:ds + rd] = nm - xf[:, b - 1, :rd]; m[:, b] = 1.0
    return xf[:, :, :rd]
rd = FT_te.shape[2]; mu = BREF['sc'].mean_[:rd]; sg = BREF['sc'].scale_[:rd]
obs = bmte[EX].astype(bool)
act = np.where(obs[:, None], FT_te[EX] * sg + mu, np.nan)
# forecast of the summary from 12 h and 24 h of EEG, un-standardized to raw label frequencies
FC = {K: np.mean([roll_features(m, Xf_te[EX:EX+1], bmte[EX:EX+1], K)[0] for m in TW], 0) * sg + mu for K in (12, 24)}
LABELS = ['Burst Suppression', 'Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Continuous', 'Discontinuous']
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True, sharey=True, layout='constrained')
for j, ax in enumerate(axes.ravel()):
    ax.plot(hh[:last_h], act[:last_h, j], '-', color='k', lw=1.8, label='actual')
    for K, col in [(12, 'C0'), (24, 'C1')]:
        ax.plot(hh[K-1:last_h], FC[K][K-1:last_h, j], '--', color=col, lw=1.8, label=f'forecast from {K} h')
    ax.set_title(LABELS[j], fontsize=10); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=8)
fig.supxlabel('hours', fontsize=11); fig.supylabel('label frequency', fontsize=11)
fig.suptitle(f'Figure 7 — Forecast head reproducing label frequencies ({TEST_PIDS[EX]}, true {"good" if y_test[EX] else "poor"})', fontsize=13)
fig.savefig(FIG_TWIN / 'fig7.png', dpi=130, bbox_inches='tight'); plt.show()
print('Figure 7 patient:', TEST_PIDS[EX])


# ===================== cell 33 =====================
_stage(19, 27, 'summary metrics')
# Summary metrics and the Step-4 output structure handed to Steps 5-6
coh = bmte[:, :12].sum(1) > 0
# twin roll-forward gain at 12 h (roll-forward outcome vs direct outcome)
gain12 = auc(y_test[coh], ens_term(TW, Xf_te, bmte, 12)[coh]) - auc(y_test[coh], ens_direct(TW, Xf_te, bmte, 12)[coh])
print(f'twin @84h {tw_h[84]:.4f} | reference {REF_HOURLY[84]:.4f} | forecast {skill["overall"]:+.1f}% | Brier {b_raw:.3f}->{b_cal:.3f} | roll-forward gain @12h {gain12:+.4f}')
print()
print('Step-4 outputs for each patient (per 1h block, BLK=84) — handoff to Steps 5-6:')
print('  outcome logits      predict_twin(m, Xf, BM)[0]  shape (N, 84)      -> P(good) per block')
print('  next-block forecast predict_twin(m, Xf, BM)[1]  shape (N, 84, 98)')
print('  trajectory embedding predict_twin(m, Xf, BM)[2]  shape (N, 84, d)  -> similarity / retrieval')
print('  IIIC label stream is 8-class:', ['Burst Suppression','Seizure','LPD','GPD','LRDA','GRDA','Continuous','Discontinuous'])

# ===================== cell 35 =====================
_stage(20, 27, 'write step4 handoff + models')
import pickle
CLASS_NAMES = ['Burst Suppression', 'Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Continuous', 'Discontinuous']
def _ens_out(Xf, BM):
    O = [predict_twin(m, Xf, BM) for m in TW]
    logit = np.mean([o[0] for o in O], 0).astype(np.float32)
    return logit, (1 / (1 + np.exp(-logit))).astype(np.float32), np.mean([o[1] for o in O], 0).astype(np.float32)
def _labels(X_ms, M):
    freq, bm = _bmean(X_ms, M, PROBS_CH)
    return freq.astype(np.float32), np.where(bm > 0, freq.argmax(-1), -1).astype(np.int64)
log_tr, p_tr, fc_tr = _ens_out(Xf_tr, bmtr); log_te, p_te, fc_te = _ens_out(Xf_te, bmte)
hid_tr = predict_twin(TW[0], Xf_tr, bmtr)[2].astype(np.float32)
hid_te = predict_twin(TW[0], Xf_te, bmte)[2].astype(np.float32)
freq_tr, seq_tr = _labels(X_ms_tr, M_ms_tr); freq_te, seq_te = _labels(X_ms_te, M_ms_te)
B = dict(
    train_pids=np.array(TRAIN_PIDS), test_pids=np.array(TEST_PIDS),
    y_train=y_train.astype(np.int64), y_test=y_test.astype(np.int64), cpc_train=cpc_train.astype(np.int64), cpc_test=cpc_test.astype(np.int64),
    mask_train=bmtr.astype(np.int8), mask_test=bmte.astype(np.int8),
    outcome_prob_train=p_tr, outcome_prob_test=p_te, outcome_logit_train=log_tr, outcome_logit_test=log_te,
    forecast_train=fc_tr, forecast_test=fc_te, hidden_train=hid_tr, hidden_test=hid_te,
    summary_train=FT_tr.astype(np.float32), summary_test=FT_te.astype(np.float32),
    labelfreq_train=freq_tr, labelfreq_test=freq_te, labelseq_train=seq_tr, labelseq_test=seq_te,
    clin_norm_train=C_train_s.astype(np.float32), clin_norm_test=C_test_s.astype(np.float32),
    clin_raw_train=C_train.astype(np.float32), clin_raw_test=C_test.astype(np.float32))
_pstep('compressing step4 handoff (~127 MB, takes a few minutes)')
OUT = OUT_DIR / 'twin_step4_handoff.npz'; np.savez_compressed(OUT, **B)
torch.save({'state_dicts': [m.state_dict() for m in TW], 'in_dim': int(Xf_tr.shape[2]), 'roll_dim': int(FT_tr.shape[2]), 'cfg': CFG2, 'twin_cfg': TWIN_CFG}, MODELS_DIR / 'twin_models.pt')
pickle.dump({'BREF': BREF, 'clin_scaler': sc_clin, 'class_names': CLASS_NAMES, 'clin_cols': list(CLIN_COLS)}, open(MODELS_DIR / 'twin_feature_pipeline.pkl', 'wb'))
README = {
    'train_pids / test_pids': 'patient ids; TRAIN = retrieval database, TEST = queries (Step 6)',
    'y_* / cpc_*': 'outcome good=1/poor=0 ; CPC score 1-5', 'mask_*': '(N,84) 1 if that 1h block was observed',
    'outcome_prob_* / outcome_logit_*': '(N,84) per-block P(good) / logit, ensemble-mean (primary head)',
    'forecast_*': '(N,84,%d) next-block summary-vector forecast, ensemble-mean (secondary head)' % int(fc_tr.shape[2]),
    'hidden_*': '(N,84,%d) per-block trajectory embedding (one twin seed) -> cosine similarity, Step 5' % int(hid_tr.shape[2]),
    'summary_*': '(N,84,%d) standardized interpretable block summary (the Step-3 vector)' % int(FT_tr.shape[2]),
    'labelfreq_*': '(N,84,8) per-block IIIC label frequencies', 'labelseq_*': '(N,84) dominant IIIC label per block (-1=unobserved) -> Hamming, Step 5',
    'clin_norm_*': '(N,5) standardized clinical -> Euclidean, Step 5', 'clin_raw_*': '(N,5) raw clinical -> display top-1 case, Step 6'}
meta = dict(blocks=int(BLK), block_hours=1, hidden_dim=int(hid_tr.shape[2]), roll_dim=int(FT_tr.shape[2]), n_twin_seeds=len(TW),
            class_names=CLASS_NAMES, clinical_cols=list(CLIN_COLS),
            summary_layout={'IIIC_freq': [0, 8], 'qEEG_mean': [8, 21], 'prototype_acts': [21, 66], 'ProtoPNet_PCA': [66, 98]},
            files={'outputs': 'twin_step4_handoff.npz', 'models': 'twin_models.pt', 'pipeline': 'twin_feature_pipeline.pkl'},
            note='outcome+forecast = ensemble-mean over twin seeds; hidden = single seed (latent not averageable). Steps 5-6 read the npz only; models/pipeline are for embedding NEW patients.', readme=README)
json.dump(meta, open(OUT_DIR / 'twin_step4_handoff_meta.json', 'w'), indent=2)
d = np.load(OUT, allow_pickle=True)
print('saved', OUT.name, '(%.0f MB)' % (OUT.stat().st_size / 1e6), '+ twin_models.pt + twin_feature_pipeline.pkl + twin_step4_handoff_meta.json')
for k in d.files: print(f'  {k:<20} {str(d[k].shape):<16} {d[k].dtype}')


# ===================== cell 37 =====================
_stage(21, 27, 'write matching handoff')
CLASS_NAMES = ['Burst Suppression', 'Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Continuous', 'Discontinuous']
current_hours = list(range(6, 85, 6)); future_hours = list(range(6, 85, 6))
def _roll_traj(Xf, BM, _who=''):
    N = len(Xf); RT = np.full((N, len(current_hours), len(future_hours)), np.nan, np.float32)
    for ci, h in enumerate(_bar(current_hours, f'roll-forward {_who}',
                               total=len(current_hours), unit='hour')):
        RF = np.mean([roll_forward(m, Xf, BM, h)[0] for m in TW], 0)
        valid = BM[:, :h].sum(1) > 0
        for fi, f in enumerate(future_hours):
            if f >= h: RT[valid, ci, fi] = RF[valid, f - 1]
    return RT
def _cum(X_ms, M, ch):
    Xc = X_ms[:, ch, :]; csum = np.cumsum(Xc * M[:, None, :], 2); ccnt = np.cumsum(M, 1)
    out = np.full((len(X_ms), len(current_hours), len(ch)), np.nan, np.float32)
    for ci, h in enumerate(current_hours):
        s = h * BSZ - 1; cnt = ccnt[:, s]; v = cnt > 0
        out[v, ci] = (csum[:, :, s][v] / cnt[v, None]).astype(np.float32)
    return out
CEB_CH = np.arange(STREAM_SLICES['cebra'].start, STREAM_SLICES['cebra'].stop)
def _matchpack(X_ms, M, _who=''):
    streams = [('cebra', CEB_CH), ('protopnet', PPNET_CH), ('proto_acts', ACTS_CH),
               ('labelfreq', PROBS_CH), ('qeeg', EEG_CH)]
    return {n: _cum(X_ms, M, ch) for n, ch in
            _bar(streams, f'matching streams {_who}', total=len(streams), unit='stream')}
_pstep('rolling the twin forward from each observation hour')
rt_tr = _roll_traj(Xf_tr, bmtr, 'train'); rt_te = _roll_traj(Xf_te, bmte, 'test')
_pstep('cumulative block-means per stream, per observation hour')
mt_tr = _matchpack(X_ms_tr, M_ms_tr, 'train'); mt_te = _matchpack(X_ms_te, M_ms_te, 'test')
M2 = dict(train_pids=np.array(TRAIN_PIDS), test_pids=np.array(TEST_PIDS),
          y_train=y_train.astype(np.int64), y_test=y_test.astype(np.int64), cpc_train=cpc_train.astype(np.int64), cpc_test=cpc_test.astype(np.int64),
          current_hours=np.array(current_hours), future_hours=np.array(future_hours),
          roll_traj_train=rt_tr, roll_traj_test=rt_te, clin_train=C_train.astype(np.float32), clin_test=C_test.astype(np.float32))
for k, v in mt_tr.items(): M2[f'{k}_train'] = v.astype(np.float32)
for k, v in mt_te.items(): M2[f'{k}_test'] = v.astype(np.float32)
_pstep('compressing matching handoff (~41 MB, takes a minute)')
OUT2 = OUT_DIR / 'twin_matching_handoff.npz'; np.savez_compressed(OUT2, **M2)
meta2 = dict(
    roll_traj='shape (N, len(current_hours), len(future_hours)). roll_traj[i, ci, fi] = patient i predicted P(good) at future_hours[fi] given EEG observed up to current_hours[ci]; NaN where future<current or no EEG yet. EXAMPLE: roll_traj_train[i, current_hours.index(36), :] = patient i predicted outcome trajectory from hour 36 onward.',
    matching_features='each <name>_train/_test is (N, len(current_hours), dim): cumulative block-mean of that stream up to each current hour -> cebra(3), protopnet(1275), proto_acts(45), labelfreq(8 IIIC), qeeg(13); clin_*(5) constant per patient',
    current_hours=current_hours, future_hours=future_hours, class_names=CLASS_NAMES, clinical_cols=list(CLIN_COLS),
    workflow='for a query (patient, current_hour): match on cebra+protopnet+proto_acts/labelfreq+qeeg(+clin) at that current hour -> top-10 TRAIN patients -> plot their roll_traj rows at that current hour')
json.dump(meta2, open(OUT_DIR / 'twin_matching_handoff_meta.json', 'w'), indent=2)
d2 = np.load(OUT2, allow_pickle=True)
print('saved', OUT2.name, '(%.0f MB)' % (OUT2.stat().st_size / 1e6), '+ twin_matching_handoff_meta.json')
for k in d2.files: print(f'  {k:<18} {str(d2[k].shape):<18} {d2[k].dtype}')


# ===================== cell 39 =====================
_stage(22, 27, 'metrics — gather predictions')
# Metrics — helpers + gather test-set predictions
# We report on the OFFICIAL 695/299 test split. Positive class = GOOD outcome (CPC 1-2). The digital twin (TW ensemble)
# is the deployed model; the reference transformer is shown for comparison. Clinically the key operating point is
# predicting POOR outcome at low false-poor rate (TPR@FPR<=0.05, poor direction = the I-CARE-style metric).
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, roc_curve, confusion_matrix, matthews_corrcoef
from sklearn.linear_model import LogisticRegression as _LR
try:                                  # notebook-only; plain print outside one
    from IPython.display import display
except ImportError:
    display = print
import pandas as pd
def _sig(z): return 1 / (1 + np.exp(-z))
def _auroc(y, p): return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
def _auprc(y, p): return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan
def _brier(y, p): return float(brier_score_loss(y, p))
def _logloss(y, p): return float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))
def thr_metrics(y, p, thr):                                   # sensitivity/specificity/PPV/NPV/... at a probability threshold
    yp = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan; spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan; npv = tn / (tn + fn) if (tn + fn) else np.nan
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv and sens) else np.nan
    return dict(threshold=float(thr), sensitivity=sens, specificity=spec, ppv=ppv, npv=npv, accuracy=(tp + tn) / len(y),
                balanced_acc=0.5 * (sens + spec), f1=f1, mcc=float(matthews_corrcoef(y, yp)) if len(np.unique(yp)) > 1 else np.nan,
                youden_J=sens + spec - 1, TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))
def tpr_at_fpr(y, p, mx):                                     # max TPR (sensitivity) achievable with FPR <= mx, + its threshold
    fpr, tpr, thr = roc_curve(y, p); ok = fpr <= mx
    if not ok.any(): return np.nan, np.nan
    i = np.where(ok)[0][-1]; return float(tpr[i]), float(thr[i])
def spec_at_sens(y, p, mn):                                   # max specificity achievable with sensitivity >= mn, + its threshold
    fpr, tpr, thr = roc_curve(y, p); ok = tpr >= mn
    if not ok.any(): return np.nan, np.nan
    i = np.where(ok)[0][0]; return float(1 - fpr[i]), float(thr[i])
def youden_thr(y, p):
    fpr, tpr, thr = roc_curve(y, p); return float(thr[int(np.argmax(tpr - fpr))])
def ece(y, p, nb=10):                                         # expected calibration error
    edges = np.linspace(0, 1, nb + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, nb - 1); e = 0.0
    for b in range(nb):
        m = idx == b
        if m.sum(): e += m.sum() / len(y) * abs(p[m].mean() - y[m].mean())
    return float(e)
def cal_slope_intercept(y, p):                                # logistic recalibration: ideal slope=1, intercept=0
    if len(np.unique(y)) < 2: return np.nan, np.nan
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    m = _LR(C=1e6, solver='lbfgs', max_iter=1000).fit(z, y); return float(m.coef_[0, 0]), float(m.intercept_[0])
def boot_ci(y, p, fn, n=2000, seed=0):                        # percentile bootstrap 95% CI
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        b = rng.randint(0, len(y), len(y))
        if len(np.unique(y[b])) > 1:
            try: v.append(fn(y[b], p[b]))
            except Exception: pass
    return (float(np.nanmean(v)), float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) if v else (np.nan, np.nan, np.nan)
def boot_diff(y, pa, pb, fn, n=2000, seed=0):                 # paired bootstrap of metric(pa) - metric(pb)
    rng = np.random.RandomState(seed); v = []
    for _ in range(n):
        b = rng.randint(0, len(y), len(y))
        if len(np.unique(y[b])) > 1:
            try: v.append(fn(y[b], pa[b]) - fn(y[b], pb[b]))
            except Exception: pass
    return (float(np.nanmean(v)), float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) if v else (np.nan, np.nan, np.nan)
def core_metrics(y, p, thr=0.5):
    d = dict(N=len(y), N_good=int(y.sum()), prevalence_good=float(y.mean()), AUROC=_auroc(y, p), AUPRC=_auprc(y, p), Brier=_brier(y, p), LogLoss=_logloss(y, p))
    d.update({k: thr_metrics(y, p, thr)[k] for k in ['sensitivity', 'specificity', 'ppv', 'npv', 'accuracy', 'balanced_acc', 'f1', 'mcc']})
    return d
TW_LOGIT = np.mean([predict_twin(m, Xf_te, bmte)[0] for m in TW], 0)      # (N,84) twin ensemble logits
def twin_p_at(h):                                             # twin P(good) at cutoff h (read at last observed block <= h) + coverage mask
    cap = min(h, BLK); obs = bmte[:, :cap]; lbk = np.where(obs > 0, np.arange(cap)[None, :], -1).max(1)
    return _sig(TW_LOGIT[np.arange(len(TW_LOGIT)), lbk.clip(0)]), obs.sum(1) > 0
tw_logit_full = TW_LOGIT[np.arange(len(y_test)), lb.clip(0)]
Y = y_test.astype(int)
p_twin = _sig(tw_logit_full)                                  # twin full-recording P(good), raw
p_twin_cal = _sig(tw_logit_full / T)                          # twin full-recording P(good), temperature-scaled (held-out T)
mref = covO[84]; p_ref = ref_hourly[84]                       # reference full-recording P(good)
print('predictions gathered | twin full AUROC', round(_auroc(Y, p_twin), 4), '| reference', round(_auroc(Y[mref], p_ref[mref]), 4))

# ===================== cell 40 =====================
_stage(23, 27, 'metrics — headline performance')
# Metrics — headline test performance (full recording) + bootstrap CIs
MODELS = {'digital twin (raw)': (Y, p_twin), 'digital twin (temp-scaled)': (Y, p_twin_cal), 'reference transformer': (Y[mref], p_ref[mref])}
tblA = pd.DataFrame({name: core_metrics(y, p) for name, (y, p) in MODELS.items()}).T.round(4)
print('Table A — headline test metrics (full recording, threshold 0.5, positive=good):'); display(tblA)
ci_defs = [('AUROC', _auroc), ('AUPRC', _auprc), ('Brier', _brier),
           ('Sensitivity@0.5', lambda y, p: thr_metrics(y, p, 0.5)['sensitivity']),
           ('Specificity@0.5', lambda y, p: thr_metrics(y, p, 0.5)['specificity']),
           ('Balanced acc@0.5', lambda y, p: thr_metrics(y, p, 0.5)['balanced_acc']),
           ('TPR@FPR<=0.05 (poor)', lambda y, p: tpr_at_fpr(1 - y, 1 - p, 0.05)[0])]
tblB = pd.DataFrame({nm: dict(zip(['estimate', 'CI95_low', 'CI95_high'], boot_ci(Y, p_twin, fn))) for nm, fn in ci_defs}).T.round(4)
print('Table B — digital twin (raw), 95% bootstrap CIs:'); display(tblB)
cc = mref                                                     # patients where both models are defined (full recording)
cmp = {}
for nm, fn in [('AUROC', _auroc), ('AUPRC', _auprc), ('Brier (lower=better)', _brier)]:
    d, lo, hi = boot_diff(Y[cc], p_twin[cc], p_ref[cc], fn)
    cmp[nm] = dict(twin=round(fn(Y[cc], p_twin[cc]), 4), reference=round(fn(Y[cc], p_ref[cc]), 4), diff_twin_minus_ref=round(d, 4), diff_CI95_low=round(lo, 4), diff_CI95_high=round(hi, 4))
tblCMP = pd.DataFrame(cmp).T
print('Table B2 — twin vs reference (paired bootstrap difference):'); display(tblCMP)

# ===================== cell 41 =====================
_stage(24, 27, 'metrics — operating points')
# Metrics — clinical operating points + threshold table + confusion matrices
rowsC = []
for f in [0.01, 0.05, 0.10]:                                  # POOR-outcome direction (I-CARE style: predict poor at low false-poor rate)
    v, t = tpr_at_fpr(1 - Y, 1 - p_twin, f); rowsC.append(dict(operating_point=f'TPR @ FPR<={f}', direction='poor', value=round(v, 4), P_good_thresh=round(1 - t, 4) if not np.isnan(t) else np.nan, meaning=f'poor-outcome sensitivity at <= {f:.0%} false-poor rate'))
for s in [0.90, 0.95, 0.99]:
    v, t = spec_at_sens(1 - Y, 1 - p_twin, s); rowsC.append(dict(operating_point=f'Spec @ Sens>={s}', direction='poor', value=round(v, 4), P_good_thresh=round(1 - t, 4) if not np.isnan(t) else np.nan, meaning=f'specificity at >= {s:.0%} poor-outcome sensitivity'))
for f in [0.05, 0.10]:                                        # GOOD-outcome direction for completeness
    v, t = tpr_at_fpr(Y, p_twin, f); rowsC.append(dict(operating_point=f'TPR @ FPR<={f}', direction='good', value=round(v, 4), P_good_thresh=round(t, 4) if not np.isnan(t) else np.nan, meaning=f'good-outcome sensitivity at <= {f:.0%} false-good rate'))
tblC = pd.DataFrame(rowsC); print('Table C — clinical operating points (digital twin, full recording):'); display(tblC)
thrs = {'0.5 (default)': 0.5, 'Youden-J optimal': youden_thr(Y, p_twin), 'high-specificity (good FPR<=0.05)': tpr_at_fpr(Y, p_twin, 0.05)[1]}
tblD = pd.DataFrame({name: thr_metrics(Y, p_twin, t) for name, t in thrs.items()}).T.round(4)
print('Table D — threshold operating points (digital twin, good direction):'); display(tblD)
def conf_df(y, p, thr, lbls=('poor', 'good')):
    cm = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]); return pd.DataFrame(cm, index=[f'true {lbls[0]}', f'true {lbls[1]}'], columns=[f'pred {lbls[0]}', f'pred {lbls[1]}'])
cm05 = conf_df(Y, p_twin, 0.5); tstar = tpr_at_fpr(Y, p_twin, 0.05)[1]; cmhs = conf_df(Y, p_twin, tstar)
print('Confusion @ 0.5 (twin):'); display(cm05); print(f'Confusion @ high-specificity threshold {tstar:.3f} (twin):'); display(cmhs)
tblSW = pd.DataFrame({f'{t:.1f}': thr_metrics(Y, p_twin, t) for t in np.round(np.arange(0.1, 0.91, 0.1), 1)}).T[['sensitivity', 'specificity', 'ppv', 'npv', 'f1', 'accuracy', 'balanced_acc']].round(4)
tblSW.index.name = 'threshold (P good)'
print('Table D2 — threshold sweep (digital twin, good direction):'); display(tblSW)

# ===================== cell 42 =====================
_stage(25, 27, 'metrics — hour by hour')
# Metrics — hour-by-hour (twin + reference)
def hourly_table(prob_fn):
    rows = {}
    for h in KS:
        p, v = prob_fn(h); yv, pv = Y[v], p[v]; m = core_metrics(yv, pv)
        rows[h] = dict(N=m['N'], AUROC=m['AUROC'], AUPRC=m['AUPRC'], Brier=m['Brier'], Sensitivity=m['sensitivity'], Specificity=m['specificity'], BalancedAcc=m['balanced_acc'])
    return pd.DataFrame(rows).T.round(4)
tblE = hourly_table(twin_p_at); tblE.index.name = 'hours'
print('Table E — hour-by-hour metrics (digital twin):'); display(tblE)
tblF = hourly_table(lambda h: (ref_hourly[h], covO[h])); tblF.index.name = 'hours'
print('Table F — hour-by-hour metrics (reference transformer):'); display(tblF)

# ===================== cell 43 =====================
_stage(26, 27, 'metrics — calibration and roll-forward')
# Metrics — calibration + digital-twin (roll-forward / forecast)
tblG = pd.DataFrame({
    'held-out raw': dict(Brier=_brier(yt, p_raw), LogLoss=_logloss(yt, p_raw), ECE=ece(yt, p_raw), cal_slope=cal_slope_intercept(yt, p_raw)[0], cal_intercept=cal_slope_intercept(yt, p_raw)[1], temperature=1.0),
    'held-out temp-scaled': dict(Brier=_brier(yt, p_cal), LogLoss=_logloss(yt, p_cal), ECE=ece(yt, p_cal), cal_slope=cal_slope_intercept(yt, p_cal)[0], cal_intercept=cal_slope_intercept(yt, p_cal)[1], temperature=T),
    'deployed twin raw': dict(Brier=_brier(Y, p_twin), LogLoss=_logloss(Y, p_twin), ECE=ece(Y, p_twin), cal_slope=cal_slope_intercept(Y, p_twin)[0], cal_intercept=cal_slope_intercept(Y, p_twin)[1], temperature=1.0),
    'deployed twin temp-scaled': dict(Brier=_brier(Y, p_twin_cal), LogLoss=_logloss(Y, p_twin_cal), ECE=ece(Y, p_twin_cal), cal_slope=cal_slope_intercept(Y, p_twin_cal)[0], cal_intercept=cal_slope_intercept(Y, p_twin_cal)[1], temperature=T),
}).T.round(4)
print('Table G — calibration (held-out fold = leakage-clean; deployed = full twin):'); display(tblG)
rows_rf = {}
for K in [6, 12, 18, 24]:                                     # does rolling the digital twin forward beat the direct read at early hours?
    coh = bmte[:, :K].sum(1) > 0; yv = Y[coh]
    a_dr = _auroc(yv, ens_direct(TW, Xf_te, bmte, K)[coh]); a_rf = _auroc(yv, ens_term(TW, Xf_te, bmte, K)[coh])
    rows_rf[f'{K}h'] = dict(N=int(coh.sum()), direct_AUROC=round(a_dr, 4), rollforward_AUROC=round(a_rf, 4), gain=round(a_rf - a_dr, 4))
tblH = pd.DataFrame(rows_rf).T; print('Table H — digital-twin roll-forward vs direct (early cutoffs):'); display(tblH)
tblI = pd.DataFrame({k: {'forecast_skill_vs_persistence_%': round(v, 2)} for k, v in skill.items()}).T
print('Table I — forecast head skill vs persistence, per feature group:'); display(tblI)

# ===================== cell 44 =====================
_stage(27, 27, 'metrics — write tables')
# Metrics — save all tables to files
import json as _json
MET_DIR = METRICS_DIR / 'twin'; MET_DIR.mkdir(parents=True, exist_ok=True)
tables = {'A_headline': tblA, 'B_bootstrap_CI': tblB, 'C_clinical_operating_points': tblC, 'D_thresholds': tblD,
          'E_hourly_twin': tblE, 'F_hourly_reference': tblF, 'G_calibration': tblG, 'B2_twin_vs_reference': tblCMP, 'D2_threshold_sweep': tblSW, 'H_rollforward': tblH, 'I_forecast_skill': tblI,
          'confusion_at_0.5': cm05, 'confusion_high_spec': cmhs}
for name, df in tables.items(): df.to_csv(MET_DIR / f'metrics_{name}.csv')
allm = {name: df.reset_index().to_dict(orient='records') for name, df in tables.items()}
allm['_meta'] = dict(positive_class='good outcome (CPC 1-2)', n_test=int(len(Y)), n_good=int(Y.sum()), n_poor=int((Y == 0).sum()),
                     notes='Official 695/299 test split (runs ~0.04 optimistic vs repeated re-splits). Discrimination = deployed twin ensemble (raw probs). Calibration held-out fold is leakage-clean. TPR@FPR (poor) = I-CARE-style: sensitivity for POOR outcome at low false-poor rate.')
_json.dump(allm, open(MET_DIR / 'all_metrics.json', 'w'), indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
print('saved', len(tables), 'tables to', MET_DIR, '+ all_metrics.json')

print('\ntwin complete ->', OUT_DIR)
