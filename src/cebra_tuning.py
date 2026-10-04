"""
cebra_tuning.py — choose CEBRA hyperparameters without touching the test set.

Patient-grouped, outcome-stratified K-fold cross-validation on the TRAINING
patients only (PPNet_data_train.npz; the test split is never loaded). For
each fold the CEBRA input preprocessing (median sanitising, PCA, scaling --
the same steps as preprocess.py) is refit on that fold's training patients,
CEBRA is trained on them with one grid point, and every patient is reduced to
the L2-normalised mean of its embedded segments. A logistic regression fit on
the training-fold patients is scored on the validation-fold patients.

Pre-specified selection rule: the grid point with the highest mean validation
AUROC (all hours) across folds. Every (grid point, fold) result is appended to
metrics/cebra/tuning_cv.csv as soon as it finishes, and a rerun skips what is
already there, so a disconnect loses at most one training.

The chosen values are reported, not written into config.py: update
config.TRAIN by hand, then rerun the pipeline from 'train CEBRA' onwards.
"""
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from config import PPNET_TRAIN, METRICS_DIR, PREP, TRAIN, CEBRA_TUNING
from progress import log, step, done, bar
import train_cebra as tc

OUT = METRICS_DIR / 'cebra'
CV_CSV = OUT / 'tuning_cv.csv'


def load_train():
    """Training split only, ordered by (patient, time) exactly like preprocess.py."""
    d = np.load(PPNET_TRAIN, allow_pickle=True)
    order = np.lexsort((d['times'].astype(float), d['patient_ids']))
    raw = {k: d[k][order] for k in PREP['FEATURE_KEYS']}
    meta = dict(pid=d['patient_ids'][order], times=d['times'][order].astype(float),
                predictions=d['predictions'][order], probabilities=d['probabilities'][order],
                cpc_binary=(d['cpc_scores'][order].astype(np.float32) >= 3).astype(np.int64))
    return raw, meta


def prep_fold(raw, fit_rows, apply_rows):
    """preprocess.py's transform, fit on fit_rows only: non-finite -> fit-row
    column median, PCA on PCA_KEY, StandardScaler per block, concatenated."""
    A_out, B_out = [], []
    for key in PREP['FEATURE_KEYS']:
        A = raw[key][fit_rows].astype(np.float64)
        B = raw[key][apply_rows].astype(np.float64)
        med = np.nanmedian(np.where(np.isfinite(A), A, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        for M in (A, B):
            r, c = np.where(~np.isfinite(M))
            M[r, c] = med[c]
        A, B = A.astype(np.float32), B.astype(np.float32)
        if key == PREP['PCA_KEY'] and PREP['PCA_COMPONENTS'] is not None:
            pca = PCA(n_components=PREP['PCA_COMPONENTS'], random_state=PREP['SEED'])
            A = pca.fit_transform(A); B = pca.transform(B)
        sc = StandardScaler()
        A_out.append(sc.fit_transform(A).astype(np.float32))
        B_out.append(sc.transform(B).astype(np.float32))
    return np.concatenate(A_out, 1), np.concatenate(B_out, 1)


def boundaries(pids):
    """First / one-past-last row of each row's patient (rows grouped by patient)."""
    _, first, counts = np.unique(pids, return_index=True, return_counts=True)
    starts = np.repeat(first, counts)
    return starts, starts + np.repeat(counts, counts)


def patient_means(emb, pids, keep):
    u = np.unique(pids[keep])
    X = np.stack([emb[keep & (pids == q)].mean(0) for q in u])
    return u, X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)


def score_fold(emb_fit, emb_val, meta, fit_rows, val_rows):
    """Validation AUROCs of classifiers fit on training-fold patient means."""
    good = 1 - meta['cpc_binary']
    res = {}
    for win, max_h in (('all', None), ('24h', 24)):
        f_keep = np.ones(len(fit_rows), bool) if max_h is None else meta['times'][fit_rows] < max_h * 3600
        v_keep = np.ones(len(val_rows), bool) if max_h is None else meta['times'][val_rows] < max_h * 3600
        uf, Xf = patient_means(emb_fit, meta['pid'][fit_rows], f_keep)
        uv, Xv = patient_means(emb_val, meta['pid'][val_rows], v_keep)
        lab = dict(zip(meta['pid'], good))
        yf = np.array([lab[q] for q in uf]); yv = np.array([lab[q] for q in uv])
        p = LogisticRegression(max_iter=1000).fit(Xf, yf).predict_proba(Xv)[:, 1]
        res[f'AUROC_logistic_{win}'] = roc_auc_score(yv, p)
        if win == 'all':
            k = min(15, len(yf) - 1)
            p = KNeighborsClassifier(k, metric='cosine').fit(Xf, yf).predict_proba(Xv)[:, 1]
            res['AUROC_knn_all'] = roc_auc_score(yv, p)
            res['n_val_patients'] = len(yv)
    return res


def grid_points():
    g = CEBRA_TUNING['GRID']
    for vals in itertools.product(*g.values()):
        yield dict(zip(g.keys(), vals))


def tag(gp):
    return (f"T{gp['TEMPERATURE']}_U{gp['NUM_UNITS']}_"
            + ('withCPC' if 'cpc_binary' in gp['LABEL_KEYS_DISC'] else 'noCPC'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = tc.default_device()
    log(f'CEBRA tuning: {CEBRA_TUNING["N_FOLDS"]}-fold CV on training patients only (device {device})')

    step('loading the training split (the test split is not read)')
    raw, meta = load_train()
    pids = np.unique(meta['pid'])
    y_pat = np.array([1 - meta['cpc_binary'][meta['pid'] == q][0] for q in pids])
    log(f'  {len(pids)} training patients, {len(meta["pid"]):,} segments, {int(y_pat.sum())} good outcome')

    skf = StratifiedKFold(CEBRA_TUNING['N_FOLDS'], shuffle=True, random_state=CEBRA_TUNING['SEED'])
    folds = [(pids[a], pids[b]) for a, b in skf.split(pids, y_pat)]

    points = list(grid_points())
    done_keys = set()
    if CV_CSV.exists():
        prev = pd.read_csv(CV_CSV)
        done_keys = set(zip(prev['config'], prev['fold']))
        log(f'  resuming: {len(done_keys)} of {len(points) * len(folds)} trainings already in {CV_CSV.name}')
    step(f'{len(points)} grid points x {len(folds)} folds = {len(points) * len(folds)} CEBRA trainings')

    for fi, (p_fit, p_val) in enumerate(folds):
        todo = [gp for gp in points if (tag(gp), fi) not in done_keys]
        if not todo:
            continue
        step(f'fold {fi + 1}/{len(folds)}: refitting preprocessing on {len(p_fit)} training-fold patients')
        fit_rows = np.where(np.isin(meta['pid'], p_fit))[0]
        val_rows = np.where(np.isin(meta['pid'], p_val))[0]
        X_fit, X_val = prep_fold(raw, fit_rows, val_rows)
        s_fit, e_fit = boundaries(meta['pid'][fit_rows])
        s_val, e_val = boundaries(meta['pid'][val_rows])
        for gp in bar(todo, f'fold {fi + 1} grid', total=len(todo), unit='config'):
            cfg = {**TRAIN, **gp}
            labels = {k: meta[k][fit_rows] for k in cfg['LABEL_KEYS_DISC'] + cfg['LABEL_KEYS_CONT']}
            model, offset = tc.fit(X_fit, s_fit, e_fit, labels, cfg, device, verbose=False, desc=tag(gp))
            r = score_fold(tc.embed(model, offset, X_fit, s_fit, e_fit, device),
                           tc.embed(model, offset, X_val, s_val, e_val, device),
                           meta, fit_rows, val_rows)
            row = dict(config=tag(gp), fold=fi, TEMPERATURE=gp['TEMPERATURE'], NUM_UNITS=gp['NUM_UNITS'],
                       LABEL_KEYS_DISC='+'.join(gp['LABEL_KEYS_DISC']), **r)
            pd.DataFrame([row]).to_csv(CV_CSV, mode='a', header=not CV_CSV.exists(), index=False)
            log(f'    {tag(gp)} fold {fi + 1}: validation AUROC {r["AUROC_logistic_all"]:.4f} '
                f'(saved to {CV_CSV.name})')

    step('summarising')
    cv = pd.read_csv(CV_CSV)
    summ = (cv.groupby(['config', 'TEMPERATURE', 'NUM_UNITS', 'LABEL_KEYS_DISC'])
              [['AUROC_logistic_all', 'AUROC_logistic_24h', 'AUROC_knn_all']]
              .agg(['mean', 'std']).round(4))
    summ.columns = ['_'.join(c) for c in summ.columns]
    summ = summ.reset_index().sort_values('AUROC_logistic_all_mean', ascending=False)
    summ['n_folds'] = summ.config.map(cv.groupby('config').size())
    complete = summ[summ.n_folds == len(folds)]
    best = complete.iloc[0] if len(complete) else None
    summ['selected'] = summ.config == (best.config if best is not None else None)
    summ.to_csv(OUT / 'tuning_summary.csv', index=False)
    print(summ.to_string(index=False))
    if best is None:
        log('no grid point has all folds yet; rerun to finish')
        return
    sel = dict(TEMPERATURE=float(best.TEMPERATURE), NUM_UNITS=int(best.NUM_UNITS),
               LABEL_KEYS_DISC=best.LABEL_KEYS_DISC.split('+'),
               cv_AUROC_mean=float(best.AUROC_logistic_all_mean), cv_AUROC_sd=float(best.AUROC_logistic_all_std),
               n_folds=len(folds), selection_rule='highest mean validation-fold AUROC, logistic on patient means, all hours',
               test_split_used=False)
    json.dump(sel, open(OUT / 'tuning_selected.json', 'w'), indent=2)
    log(f'selected: {sel}')
    log('next: set these in config.TRAIN, then rerun the pipeline from "train CEBRA" onwards')
    done('tuning_cv.csv + tuning_summary.csv + tuning_selected.json', OUT)


if __name__ == '__main__':
    main()
