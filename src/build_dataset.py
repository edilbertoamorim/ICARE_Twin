"""
build_dataset.py — the ProtoPNet aggregation (chunkify_ppnet v0.0.6),
parameterised against config.py.

The science is unchanged: 6 epochs mean-pooled to 300 s segments, base class
"Other" split by BCI into BurstSupp / Continuous / Discontinuous, CPC and
time-from-ROSC attached, chunks >= 14 dropped, then the patient-level split.

Only I/O changed: paths come from config.PREPROCESS, the clinical table is read
from data/tables/, the split column is auto-detected, and the outputs are
written straight to data/PPNet_data_{train,test}.npz — the names the rest of
this pipeline expects.

NOT part of run_all.py. Run it only to rebuild the dataset from ProtoPNet
outputs; everything downstream starts from the npz it writes.

*** NEVER EXECUTED — none of its four inputs are present in this folder. ***
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax
from joblib import Parallel, delayed
import warnings

from config import PREPROCESS as _P, DATASET_DIR
from progress import step, done, log

warnings.filterwarnings(action='ignore', message='Mean of empty slice')

# ── preflight ───────────────────────────────────────────────────────
_need = {'protopnet_dirs': _P['PROTOPNET_DIRS'], 'bci_csv_dir': _P['BCI_CSV_DIR'],
         'offsets_csv': _P['OFFSETS_CSV'], 'clinical_csv': _P['CLINICAL_CSV'],
         'train_ids': _P['TRAIN_IDS'], 'test_ids': _P['TEST_IDS']}
_missing = []
for _k, _v in _need.items():
    for _p in (_v if isinstance(_v, (list, tuple)) else [_v]):
        if not os.path.exists(_p):
            _missing.append(f'  {_k:16s} {_p}')
if _missing:
    log('cannot build the dataset — missing inputs:')
    for _m in _missing:
        log(_m, indent=1)
    log('see data/README.md. Nothing was written.')
    sys.exit(1)

N = _P['EPOCHS_PER_SEGMENT']
SEC_PER_CHUNK = _P['SEC_PER_CHUNK']
MAX_CHUNK = _P['MAX_CHUNK']

log('build dataset')
step(f"protopnet  {sum(len(list(__import__('pathlib').Path(d).glob('*.npz'))) for d in _P['PROTOPNET_DIRS'])} npz "
     f"across {len(_P['PROTOPNET_DIRS'])} sites")
step(f"qeeg       {len(list(__import__('pathlib').Path(_P['BCI_CSV_DIR']).glob('*.csv')))} csv")
step(f"segments   {N} epochs x 50 s = {N * 50} s")

cebra_cols = [
    'corrmean', 'meanskewamp', 'sdspectent', 'shanavg', 
    'thetaalphamean', 'BCI','SIQ', 'SIQ_delta', 'SIQ_beta', 
    'SIQ_alpha', 'SIQ_theta', 'lv_l5', 'spike_count_of_SSD'
]

# Directories
npz_directories = list(_P['PROTOPNET_DIRS'])
bci_csv_dir = str(_P['BCI_CSV_DIR'])

df_offset = pd.read_csv(_P['OFFSETS_CSV'])
df_offset['Extended_Ids'] = df_offset['ICARE_files'].apply(lambda r: str(r)[:-4])
offsets = dict(zip(df_offset["Extended_Ids"].values, df_offset["time_from_rosc"].values))

df_cpc = pd.read_csv(_P['CLINICAL_CSV'])   # pat_ICARE + cpc
cpc_dict = dict(zip(df_cpc['pat_ICARE'], df_cpc['cpc']))

# --- 2. The Worker Function (Runs on individual cores) ---
def process_patient_file(file, npz_dir, offsets, cpc_dict, bci_csv_dir, cebra_cols, N, SEC_PER_CHUNK):
    file_base = file.replace(".npz", "")
    patient_id = "_".join(file_base.split('_')[:2]) 
    bci_csv_path = os.path.join(bci_csv_dir, f"{patient_id}_rel10s_with_spike.csv")
    
    if not os.path.exists(bci_csv_path):
        return {'_skip': 'no qEEG csv', 'pid': patient_id}
        
    try:
        with np.load(os.path.join(npz_dir, file)) as data:
            feat = data["extracted_features"] if "extracted_features" in data else data["proto_features"]
            pred = np.atleast_1d(data["predictions"])
            acts = data['activations']
            logits = data["logits_marginlesses"]
            
            if len(feat.shape) == 1:
                feat, acts = feat.reshape(1, -1), acts.reshape(1, -1)
                logits = logits.reshape(1, -1)
            probs = softmax(logits, axis=1) 
        
        df_bci = pd.read_csv(bci_csv_path)
        df_file_bci = df_bci[df_bci['file'].str.contains(file_base, na=False)].sort_values('rel_sec')
        if df_file_bci.empty:
            return {'_skip': 'qEEG csv has no rows for this recording', 'pid': patient_id}
            
        bci_aligned = df_file_bci['BCI'].values 
        # Pre-allocate a NaN array with the strict number of required columns
        cebra_aligned = np.full((len(df_file_bci), len(cebra_cols)), np.nan)
        
        # Fill in the columns that actually exist in this specific CSV
        for col_idx, col_name in enumerate(cebra_cols):
            if col_name in df_file_bci.columns:
                cebra_aligned[:, col_idx] = df_file_bci[col_name].values
        
        start_sec = offsets[file_base] 
        
        bci_equivalent_rows = len(bci_aligned) // 5
        min_len = min(feat.shape[0], bci_equivalent_rows)
        if min_len == 0:
            return {'_skip': 'no overlapping epochs', 'pid': patient_id}
        
        feat = feat[:min_len]
        pred = pred[:min_len]
        acts = acts[:min_len]
        logits = logits[:min_len] 
        probs = probs[:min_len]   
        bci_aligned = bci_aligned[:min_len * 5] 
        cebra_aligned = cebra_aligned[:min_len * 5] 

        cutoff = (min_len // N) * N
        remainder = min_len % N

        if cutoff > 0:
            feat_c = feat[:cutoff].reshape(-1, N, feat.shape[1]).mean(axis=1)
            pred_c = stats.mode(pred[:cutoff].reshape(-1, N), axis=1, keepdims=False)[0]
            acts_c = acts[:cutoff].reshape(-1, N, acts.shape[1]).mean(axis=1)
            logits_c = logits[:cutoff].reshape(-1, N, logits.shape[1]).mean(axis=1)
            probs_c  = probs[:cutoff].reshape(-1, N, probs.shape[1]).mean(axis=1)
            bci_c  = np.nanmean(bci_aligned[:cutoff * 5].reshape(-1, N * 5), axis=1)
            
            num_features = cebra_aligned.shape[1]
            cebra_c = np.nanmean(cebra_aligned[:cutoff * 5].reshape(-1, N * 5, num_features), axis=1)
            res_c  = np.full(feat_c.shape[0], N * 50.0) 
            time_c = start_sec + (np.arange(0, cutoff, N) * 50)
        else:
            feat_c, pred_c, acts_c, logits_c, probs_c, bci_c, cebra_c, res_c, time_c = None, None, None, None, None, None, None, None, None

        if remainder > 0:
            feat_r = feat[cutoff:].mean(axis=0, keepdims=True)
            pred_r = np.array([stats.mode(pred[cutoff:], keepdims=False)[0]])
            acts_r = acts[cutoff:].mean(axis=0, keepdims=True)
            logits_r = logits[cutoff:].mean(axis=0, keepdims=True)
            probs_r  = probs[cutoff:].mean(axis=0, keepdims=True)
            bci_r  = np.array([np.nanmean(bci_aligned[cutoff * 5:])])
            cebra_r = np.nanmean(cebra_aligned[cutoff * 5:], axis=0, keepdims=True)
            res_r  = np.array([remainder * 50.0]) 
            time_r = np.array([start_sec + cutoff * 50])

            if feat_c is not None:
                feat_c = np.vstack([feat_c, feat_r])
                pred_c = np.concatenate([pred_c, pred_r])
                acts_c = np.vstack([acts_c, acts_r])
                logits_c = np.vstack([logits_c, logits_r]) 
                probs_c  = np.vstack([probs_c, probs_r])   
                bci_c  = np.concatenate([bci_c, bci_r])
                cebra_c = np.vstack([cebra_c, cebra_r]) 
                res_c  = np.concatenate([res_c, res_r])
                time_c = np.concatenate([time_c, time_r])
            else:
                feat_c, pred_c, acts_c, logits_c, probs_c, bci_c, cebra_c, res_c, time_c = feat_r, pred_r, acts_r, logits_r, probs_r, bci_r, cebra_r, res_r, time_r

        other_mask = (pred_c == 0) 
        bs_mask   = other_mask & ~np.isnan(bci_c) & (bci_c < 0.5)
        disc_mask = other_mask & ~np.isnan(bci_c) & (bci_c >= 0.5) & (bci_c < 0.9)
        cont_mask = other_mask & ~np.isnan(bci_c) & (bci_c >= 0.9)

        pred_c[bs_mask] = 0
        pred_c[cont_mask] = 6
        pred_c[disc_mask] = 7

        chunk_ids = time_c // SEC_PER_CHUNK
        pat_cpc = cpc_dict.get(patient_id, np.nan)
        
        return {
            'features': feat_c, 'predictions': pred_c, 'activations': acts_c,
            'logits': logits_c, 'probs': probs_c, 'cebra': cebra_c,
            'chunks': chunk_ids, 'resolutions': res_c, 'times': time_c,
            'pats': [patient_id] * feat_c.shape[0], 'cpc': [pat_cpc] * feat_c.shape[0]
        }
        
    except Exception as e:
        return {'_skip': f'error: {type(e).__name__}: {e}', 'pid': patient_id}

# --- 3. Parallel Execution Setup ---

# Gather all valid files to process
tasks = []
for npz_dir in npz_directories:
    if os.path.exists(npz_dir):
        for file in os.listdir(npz_dir):
            if file.endswith(".npz") and file.replace(".npz", "") in offsets:
                tasks.append((file, npz_dir))

step(f'{len(tasks)} recordings to aggregate — parallel across all cores')

# Execute parallel pool (n_jobs=-1 uses all available CPU cores)
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(process_patient_file)(file, npz_dir, offsets, cpc_dict, bci_csv_dir, cebra_cols, N, SEC_PER_CHUNK)
    for file, npz_dir in tasks
)

# --- 4. Stack, Filter, and Chronologically Sort ---
from collections import Counter as _C
_skipped = [r for r in results if r is not None and '_skip' in r]
_kept = [r for r in results if r is not None and '_skip' not in r]
step(f'aggregated {len(_kept)}/{len(tasks)} recordings')
if _skipped:
    step(f'{len(_skipped)} skipped:')
    for _why, _n in _C(r['_skip'] for r in _skipped).most_common():
        _pids = sorted({r['pid'] for r in _skipped if r['_skip'] == _why})
        _shown = ', '.join(_pids[:6]) + (f' (+{len(_pids) - 6} more)' if len(_pids) > 6 else '')
        step(f'  {_n:4d}  {_why}')
        step(f'        {_shown}')
_lost = sorted({r['pid'] for r in _skipped} - {r['pats'][0] for r in _kept})
if _lost:
    step(f'{len(_lost)} patients lost entirely (no usable recording)')
step(f'stacking 11 arrays over {len(_kept)} recordings — a couple of minutes, no output')

# Filter out None results from files that failed or were skipped
valid_results = _kept

final_time_chunks = np.concatenate([r['chunks'] for r in valid_results])
final_times       = np.concatenate([r['times'] for r in valid_results])
final_features    = np.vstack([r['features'] for r in valid_results])
final_predictions = np.concatenate([r['predictions'] for r in valid_results])
final_activations = np.vstack([r['activations'] for r in valid_results])
final_logits      = np.vstack([r['logits'] for r in valid_results]) 
final_probs       = np.vstack([r['probs'] for r in valid_results])  
final_cebra       = np.vstack([r['cebra'] for r in valid_results]) 
final_resolutions = np.concatenate([r['resolutions'] for r in valid_results])
final_patient_ids = np.array([p for r in valid_results for p in r['pats']])
final_cpc         = np.array([c for r in valid_results for c in r['cpc']])

df_sort = pd.DataFrame({'patient': final_patient_ids, 'time': final_times})
sort_idx = df_sort.sort_values(['patient', 'time']).index

valid_mask = final_time_chunks[sort_idx] < 14
final_idx = sort_idx[valid_mask]


# ── split and write, using the names the pipeline expects ───────────
valid_mask = final_time_chunks[sort_idx] < MAX_CHUNK
final_idx = sort_idx[valid_mask]

ALL = dict(features=final_features, predictions=final_predictions,
           activations=final_activations, logits=final_logits,
           probabilities=final_probs, cebra_features=final_cebra,
           chunks=final_time_chunks, times=final_times,
           resolutions=final_resolutions, patient_ids=final_patient_ids,
           cpc_scores=final_cpc)
ALL = {k: v[final_idx] for k, v in ALL.items()}
step(f'{ALL["predictions"].shape[0]:,} segments from '
     f'{len(np.unique(ALL["patient_ids"]))} patients '
     f'(chunks >= {MAX_CHUNK} dropped)')


def _ids(path):
    """Patient ids from a split csv, whatever the id column is called."""
    df = pd.read_csv(path)
    for c in ('patient_id', 'Patient', 'pat_ICARE', 'patient', 'ICARE'):
        if c in df.columns:
            return df[c].astype(str).values
    return df.iloc[:, 0].astype(str).values


step('splitting and writing')
for name, ids in (('train', _ids(_P['TRAIN_IDS'])), ('test', _ids(_P['TEST_IDS']))):
    m = np.isin(ALL['patient_ids'].astype(str), ids)
    out = {k: v[m] for k, v in ALL.items()}
    dst = DATASET_DIR / f'PPNet_data_{name}.npz'
    np.savez_compressed(dst, **out)
    done(f'{name}: {out["predictions"].shape[0]:,} segments, '
         f'{len(np.unique(out["patient_ids"]))} patients', dst)
