"""
Optional - baseline outcome models the twin should beat, scored the same way.

  clinical   logistic regression on the 5 admission variables (constant in time)
  HMM        two categorical hidden Markov models on the hourly dominant EEG-state
             label (8 ACNS classes), one fit on good- and one on poor-outcome
             training patients. At hour h a test patient's P(good) is the
             posterior from the two forward-algorithm likelihoods of its labels
             up to h plus the training prior -- causal by construction.

Each is compared with the digital twin on the same available-case cohort at
every cutoff (AUROC with patient-bootstrap CIs and a paired bootstrap of the
difference). Needs only the step-4 twin handoff.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import HANDOFF_STEP4, METRICS_DIR
from progress import log, step, done, bar

HOURS = (6, 12, 18, 24, 36, 48, 72, 84)
N_SYM = 8
STATES = (2, 3, 4, 5)       # chosen per outcome class by BIC on training patients
N_RESTARTS = 3
N_ITER = 200


def align(seq):
    """Start each sequence at its first observed block (recordings start at
    different hours); unobserved blocks after that stay -1 = missing."""
    out = np.full_like(seq, -1)
    first = np.where((seq >= 0).any(1), (seq >= 0).argmax(1), seq.shape[1])
    for i, f in enumerate(first):
        out[i, :seq.shape[1] - f] = seq[i, f:]
    return out, first


def emis(B, obs):
    """(N, K) emission likelihoods; missing observations contribute 1."""
    e = B[:, np.clip(obs, 0, None)].T
    e[obs < 0] = 1.0
    return e


def forward(pi, A, B, X):
    """Scaled forward pass. Returns per-step log scale factors (N, T); their
    cumulative sum is the log-likelihood of the first t+1 blocks."""
    N, T = X.shape
    a = pi[None, :] * emis(B, X[:, 0])
    c = a.sum(1); a /= c[:, None]
    logc = np.zeros((N, T)); logc[:, 0] = np.log(c)
    alphas = [a]
    for t in range(1, T):
        a = (a @ A) * emis(B, X[:, t])
        c = a.sum(1); a /= c[:, None]
        logc[:, t] = np.log(c); alphas.append(a)
    return np.stack(alphas, 1), logc


def fit_hmm(X, K, seed):
    """Baum-Welch for a categorical HMM with missing observations."""
    r = np.random.RandomState(seed)
    N, T = X.shape
    pi = np.full(K, 1.0 / K)
    A = 0.8 * np.eye(K) + 0.2 * r.dirichlet(np.ones(K), K)
    A /= A.sum(1, keepdims=True)
    freq = np.bincount(X[X >= 0], minlength=N_SYM) + 1.0
    B = r.dirichlet(freq, K)
    # transitions after a patient's last observed block carry no information
    last = np.where(X >= 0, np.arange(T)[None, :], -1).max(1)
    prev = -np.inf
    for _ in range(N_ITER):
        al, logc = forward(pi, A, B, X)
        be = np.ones((N, T, K))
        for t in range(T - 2, -1, -1):
            be[:, t] = (be[:, t + 1] * emis(B, X[:, t + 1])) @ A.T / np.exp(logc[:, t + 1])[:, None]
        g = al * be
        g /= g.sum(2, keepdims=True)
        xi = np.zeros((K, K))
        for t in range(T - 1):
            if not (last > t).any():
                break
            x = al[last > t, t, :, None] * A[None] * (emis(B, X[last > t, t + 1]) * be[last > t, t + 1])[:, None, :]
            xi += (x / x.sum((1, 2), keepdims=True)).sum(0)
        pi = g[:, 0].mean(0) + 1e-9; pi /= pi.sum()
        A = xi + 1e-6; A /= A.sum(1, keepdims=True)
        B = np.full((K, N_SYM), 1e-6)
        for s in range(N_SYM):
            B[:, s] += g[X == s].sum(0)
        B /= B.sum(1, keepdims=True)
        ll = logc.sum()
        if ll - prev < 1e-6 * abs(ll):
            break
        prev = ll
    return pi, A, B, logc.sum()


def best_hmm(X, who):
    """Pick the number of states by BIC, best of N_RESTARTS restarts each."""
    n_obs = int((X >= 0).sum()); best = None
    for K in bar(STATES, f'HMM ({who}) states', total=len(STATES), unit='K'):
        fits = [fit_hmm(X, K, s) for s in range(N_RESTARTS)]
        pi, A, B, ll = max(fits, key=lambda f: f[3])
        bic = -2 * ll + ((K - 1) + K * (K - 1) + K * (N_SYM - 1)) * np.log(n_obs)
        if best is None or bic < best[4]:
            best = (pi, A, B, ll, bic, K)
    return best


def loglik_upto(model, Xal, first, h):
    """log P(labels in absolute blocks < h); NaN when no EEG yet by hour h."""
    pi, A, B = model[:3]
    _, logc = forward(pi, A, B, Xal)
    n = np.clip(h - first, 0, Xal.shape[1])
    out = np.full(len(Xal), np.nan)
    ok = n > 0
    out[ok] = np.cumsum(logc, 1)[ok, n[ok] - 1]
    return out


def boot(y, a, b=None, n=2000, seed=0):
    r = np.random.RandomState(seed); v = []
    for _ in range(n):
        i = r.randint(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            v.append(roc_auc_score(y[i], a[i]) - (roc_auc_score(y[i], b[i]) if b is not None else 0))
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)


if __name__ == '__main__':
    log('baselines: clinical logistic regression and outcome-conditional HMM')
    d = np.load(HANDOFF_STEP4, allow_pickle=True)
    y_tr, y_te = d['y_train'].astype(int), d['y_test'].astype(int)

    step('clinical-only logistic regression (5 admission variables)')
    p_clin = (LogisticRegression(max_iter=1000, class_weight='balanced')
              .fit(d['clin_norm_train'], y_tr).predict_proba(d['clin_norm_test'])[:, 1])

    step('HMMs on the hourly dominant EEG-state label, one per outcome')
    Xtr, _ = align(d['labelseq_train']); Xte, first_te = align(d['labelseq_test'])
    good, poor = best_hmm(Xtr[y_tr == 1], 'good'), best_hmm(Xtr[y_tr == 0], 'poor')
    log(f'  good-outcome HMM: {good[5]} states   poor-outcome HMM: {poor[5]} states')
    prior = np.log(y_tr.mean()) - np.log(1 - y_tr.mean())

    step('scoring every cutoff on the twin\'s available-case cohort')
    mask, p_twin_blk = d['mask_test'], d['outcome_prob_test']
    rows = []
    for h in HOURS:
        cap = min(h, mask.shape[1]); obs = mask[:, :cap] > 0
        v = obs.any(1)
        lb = np.where(obs, np.arange(cap)[None, :], -1).max(1).clip(0)
        p_twin = p_twin_blk[np.arange(len(y_te)), lb]
        z = loglik_upto(good, Xte, first_te, h) - loglik_upto(poor, Xte, first_te, h) + prior
        p_hmm = 1 / (1 + np.exp(-z))
        v &= np.isfinite(p_hmm)
        y = y_te[v]
        if v.sum() < 2 or len(np.unique(y)) < 2:
            continue
        for name, p in (('digital twin', p_twin), ('HMM (EEG states)', p_hmm), ('clinical only', p_clin)):
            lo, hi = boot(y, p[v])
            dlo, dhi = boot(y, p_twin[v], p[v]) if name != 'digital twin' else (np.nan, np.nan)
            rows.append(dict(hours=h, N=int(v.sum()), model=name, AUROC=roc_auc_score(y, p[v]),
                             CI95_low=lo, CI95_high=hi,
                             twin_minus_model=(roc_auc_score(y, p_twin[v]) - roc_auc_score(y, p[v])) if name != 'digital twin' else np.nan,
                             diff_CI95_low=dlo, diff_CI95_high=dhi))
    t = pd.DataFrame(rows).round(4)
    print(t.pivot(index='hours', columns='model', values='AUROC').to_string())

    out = METRICS_DIR / 'baselines'
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / 'baselines_hourly.csv', index=False)
    np.savez(out / 'hmm_params.npz', **{f'{w}_{k}': v for w, m in (('good', good), ('poor', poor))
                                        for k, v in zip(('pi', 'A', 'B'), m[:3])})
    done('baselines_hourly.csv + hmm_params.npz', out)
