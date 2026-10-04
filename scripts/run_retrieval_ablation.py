"""
Optional - what the CEBRA stream contributes to case retrieval.

Reproduces the manuscript's Table 4 (neighbour-vote AUROC by observation
window) and then ablates the similarity streams, so CEBRA's marginal
contribution is measured rather than asserted.

Needs only the two twin handoff files; no CEBRA training required.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.metrics import roc_auc_score

import twin as tf
from config import FIG

HOURS = (12, 24, 48, 84)
K = 25                      # neighbours in the vote (manuscript Table 4)
ALPHA = FIG['ALPHA']


def _l2(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def streams_at(twin, H, q='test'):
    """Per-stream query-vs-database cosine similarity at hour H. The database is
    always the training split; q='train' queries it against itself (for choosing k)."""
    t4, tm = twin['step4'], twin['match']
    hi = list(tm['current_hours']).index(H)
    out = {}
    for s in tf.FEATURE_STREAMS:
        T, Q = tm[f'{s}_train'][:, hi, :], tm[f'{s}_{q}'][:, hi, :]
        mu, sd = np.nanmean(T, 0), np.nanstd(T, 0) + 1e-8
        out[s] = _l2(np.nan_to_num((Q - mu) / sd)) @ _l2(np.nan_to_num((T - mu) / sd)).T
    T, Q = tm['clin_train'], tm[f'clin_{q}']
    mu, sd = T.mean(0), T.std(0) + 1e-8
    out['clin'] = _l2((Q - mu) / sd) @ _l2((T - mu) / sd).T
    out['traj'] = _l2(t4[f'hidden_{q}'][:, H - 1, :]) @ _l2(t4['hidden_train'][:, H - 1, :]).T
    # train patients with no EEG yet carry NaN and are ineligible, exactly as
    # they would be at the bedside; test patients likewise cannot be queried
    ok = ~np.isnan(tm['cebra_train'][:, hi, :]).any(axis=1)
    qok = ~np.isnan(tm[f'cebra_{q}'][:, hi, :]).any(axis=1)
    return out, ok, qok


def vote_auc(S, ok, qok, y_tr, y_te, k=K):
    # never ask for more neighbours than the eligible database holds
    k = max(1, min(k, int(ok.sum()) - 1))
    S = S.copy()
    S[:, ~ok] = -np.inf
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    return roc_auc_score(y_te[qok], y_tr[idx].mean(axis=1)[qok])


if __name__ == '__main__':
    twin = tf.load_twin()
    y_tr, y_te = twin['step4']['y_train'], twin['step4']['y_test']
    feat_keys = tf.FEATURE_STREAMS + ['clin']

    rows, Ns = {}, {}
    for H in HOURS:
        s, ok, qok = streams_at(twin, H)
        Ns[H] = int(qok.sum())
        feat = np.mean([s[k] for k in feat_keys], axis=0)
        no_cebra = np.mean([s[k] for k in feat_keys if k != 'cebra'], axis=0)
        cfg = {
            'FULL  a*traj + (1-a)*feat': ALPHA * s['traj'] + (1 - ALPHA) * feat,
            '  minus CEBRA':             ALPHA * s['traj'] + (1 - ALPHA) * no_cebra,
            '  trajectory only':         s['traj'],
            '  feature term only':       feat,
            '  CEBRA only':              s['cebra'],
            '  ProtoPNet only':          s['protopnet'],
            '  qEEG only':               s['qeeg'],
            '  clinical only':           s['clin'],
        }
        for name, S in cfg.items():
            rows.setdefault(name, {})[H] = vote_auc(S, ok, qok, y_tr, y_te)

    hdr = ' '.join(f'{"h" + str(h):>7s}' for h in HOURS)
    print(f'neighbour-vote AUROC, top-{K}, alpha={ALPHA}\n')
    print(f'{"configuration":28s} {hdr}')
    print(f'{"N (queries with EEG)":28s} ' + ' '.join(f'{Ns[h]:7d}' for h in HOURS))
    for name, v in rows.items():
        print(f'{name:28s} ' + ' '.join(f'{v[h]:7.3f}' for h in HOURS))
    d, m = rows['FULL  a*traj + (1-a)*feat'], rows['  minus CEBRA']
    print(f'\n{"CEBRA marginal":28s} ' +
          ' '.join(f'{d[h] - m[h]:+7.3f}' for h in HOURS))

    # ── choosing k, and how calibrated the neighbour vote is ─────────────
    # k is chosen on the TRAINING split only: every train patient queries the
    # rest of the database (itself excluded), and k* maximises the mean
    # leave-one-out vote AUROC over the observation windows. The test split is
    # then scored at every k, with patient-bootstrap CIs, so the choice of 25
    # can be checked rather than assumed.
    import pandas as pd
    from config import METRICS_DIR
    from progress import step, done
    out = METRICS_DIR / 'retrieval'
    out.mkdir(parents=True, exist_ok=True)

    def full_sim(s):
        return ALPHA * s['traj'] + (1 - ALPHA) * np.mean([s[k] for k in feat_keys], axis=0)

    def vote(S, ok, k):
        S = S.copy()
        S[:, ~ok] = -np.inf
        k = max(1, min(k, int(ok.sum()) - 1))
        return y_tr[np.argpartition(-S, k, axis=1)[:, :k]].mean(axis=1)

    def boot_auc(y, p, n=2000, seed=0):
        r = np.random.RandomState(seed)
        v = [roc_auc_score(y[i], p[i]) for i in (r.randint(0, len(y), len(y)) for _ in range(n))
             if len(np.unique(y[i])) > 1]
        return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)

    K_GRID = (1, 5, 10, 15, 25, 50, 100)
    step(f'top-k sweep: k in {K_GRID}, chosen by leave-one-out on train')
    sims = {H: (streams_at(twin, H), streams_at(twin, H, q='train')) for H in HOURS}
    rows_k = []
    for k in K_GRID:
        for H in HOURS:
            (s_te, ok, qok), (s_tr, _, qok_tr) = sims[H]
            S_tr = full_sim(s_tr)
            np.fill_diagonal(S_tr, -np.inf)              # a patient may not retrieve itself
            m = qok_tr & ok
            loo = roc_auc_score(y_tr[m], vote(S_tr, ok, k)[m]) if len(np.unique(y_tr[m])) > 1 else np.nan
            p = vote(full_sim(s_te), ok, k)[qok]
            lo, hi = boot_auc(y_te[qok], p)
            rows_k.append(dict(k=k, hours=H, N_test=int(qok.sum()), train_LOO_AUROC=loo,
                               test_AUROC=roc_auc_score(y_te[qok], p), test_CI95_low=lo, test_CI95_high=hi))
    tk = pd.DataFrame(rows_k).round(4)
    loo_mean = tk.groupby('k').train_LOO_AUROC.mean()
    k_star = int(loo_mean.idxmax()) if loo_mean.notna().any() else K   # all-NaN only on toy cohorts
    tk['chosen_on_train'] = tk.k == k_star
    tk.to_csv(out / 'retrieval_topk_sweep.csv', index=False)
    print(f'\ntop-k sweep (k* = {k_star}, chosen by mean train leave-one-out AUROC):')
    print(tk.pivot(index='k', columns='hours', values='test_AUROC').to_string())

    # calibration of the neighbour good-fraction: fixed bins, count per bin,
    # observed good rate with a Wilson 95% CI
    def wilson(x, n, z=1.96):
        if n == 0:
            return np.nan, np.nan
        ph = x / n; d = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / d; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
        return c - h, c + h
    EDGES = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0001])
    rows_c = []
    for k in sorted({K, k_star}):
        for H in HOURS:
            (s_te, ok, qok), _ = sims[H]
            p, y = vote(full_sim(s_te), ok, k)[qok], y_te[qok]
            b = np.clip(np.digitize(p, EDGES) - 1, 0, len(EDGES) - 2)
            for j in range(len(EDGES) - 1):
                m = b == j
                lo, hi = wilson(int(y[m].sum()), int(m.sum()))
                rows_c.append(dict(k=k, hours=H, bin=f'[{EDGES[j]:.1f}, {min(EDGES[j + 1], 1):.1f}]',
                                   n=int(m.sum()), mean_neighbour_good_fraction=p[m].mean() if m.any() else np.nan,
                                   observed_good_rate=y[m].mean() if m.any() else np.nan,
                                   observed_CI95_low=lo, observed_CI95_high=hi))
    tc = pd.DataFrame(rows_c).round(4)
    tc.to_csv(out / 'retrieval_calibration_bins.csv', index=False)
    print(f'\nneighbour-vote calibration, 24 h (k={K}):')
    print(tc[(tc.k == K) & (tc.hours == 24)].drop(columns=['k', 'hours']).to_string(index=False))
    done('retrieval_topk_sweep.csv + retrieval_calibration_bins.csv', out)
