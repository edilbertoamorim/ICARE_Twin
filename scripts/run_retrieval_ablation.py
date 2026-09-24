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


def streams_at(twin, H):
    """Per-stream query-vs-database cosine similarity at hour H."""
    t4, tm = twin['step4'], twin['match']
    hi = list(tm['current_hours']).index(H)
    out = {}
    for s in tf.FEATURE_STREAMS:
        T, Q = tm[f'{s}_train'][:, hi, :], tm[f'{s}_test'][:, hi, :]
        mu, sd = np.nanmean(T, 0), np.nanstd(T, 0) + 1e-8
        out[s] = _l2(np.nan_to_num((Q - mu) / sd)) @ _l2(np.nan_to_num((T - mu) / sd)).T
    T, Q = tm['clin_train'], tm['clin_test']
    mu, sd = T.mean(0), T.std(0) + 1e-8
    out['clin'] = _l2((Q - mu) / sd) @ _l2((T - mu) / sd).T
    out['traj'] = _l2(t4['hidden_test'][:, H - 1, :]) @ _l2(t4['hidden_train'][:, H - 1, :]).T
    # train patients with no EEG yet carry NaN and are ineligible, exactly as
    # they would be at the bedside; test patients likewise cannot be queried
    ok = ~np.isnan(tm['cebra_train'][:, hi, :]).any(axis=1)
    qok = ~np.isnan(tm['cebra_test'][:, hi, :]).any(axis=1)
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
