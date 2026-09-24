"""
Synthetic preprocessing inputs, shaped exactly like the real ones.

Written so the whole pipeline — including the build-dataset step, which has never run against
real data — can be exercised end to end. The numbers are noise; only the
shapes, column names, dtypes and file naming matter.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SITES = ['MGH', 'ULB', 'YNH', 'BIDMC', 'BWH', 'UTW']
CEBRA_COLS = ['corrmean', 'meanskewamp', 'sdspectent', 'shanavg', 'thetaalphamean',
              'BCI', 'SIQ', 'SIQ_delta', 'SIQ_beta', 'SIQ_alpha', 'SIQ_theta',
              'lv_l5', 'spike_count_of_SSD']
# cpc per patient; needs 1 / 3 / 5 in TEST for the Figure-3 archetypes
COHORT = [('ICARE_9001', 1, 'train'), ('ICARE_9002', 1, 'train'),
          ('ICARE_9003', 5, 'train'), ('ICARE_9004', 5, 'train'),
          ('ICARE_9005', 2, 'train'), ('ICARE_9006', 3, 'train'),
          ('ICARE_9007', 1, 'test'),  ('ICARE_9008', 3, 'test'),
          ('ICARE_9009', 5, 'test'),  ('ICARE_9010', 5, 'test')]
HOURS = 84
N_PROTO_ROWS = HOURS * 72          # 50 s epochs
N_BCI_ROWS = N_PROTO_ROWS * 5      # 10 s rows


def build(root: Path, rng=np.random.default_rng(0)):
    raw, tables = root / 'data' / 'raw', root / 'data' / 'tables'
    for s in SITES:
        (raw / 'protopnet' / s).mkdir(parents=True, exist_ok=True)
    (raw / 'qeeg').mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    offsets = []
    for i, (pid, cpc, _) in enumerate(COHORT):
        site = SITES[i % len(SITES)]
        stem = f'{pid}_2010010{i % 9 + 1}_000000'
        T = N_PROTO_ROWS

        # base softmax class 0 is "Other"; the BCI split turns it into
        # BurstSupp / Continuous / Discontinuous downstream
        pred = rng.integers(0, 6, T)
        np.savez(raw / 'protopnet' / site / f'{stem}.npz',
                 extracted_features=rng.normal(size=(T, 1275)).astype(np.float32),
                 predictions=pred.astype(np.int64),
                 activations=rng.random((T, 45)).astype(np.float32),
                 logits_marginlesses=rng.normal(size=(T, 6)).astype(np.float32))

        # a recovering patient drifts to a high BCI, a poor one stays low
        base = np.linspace(0.1, 0.95, N_BCI_ROWS) if cpc <= 2 else \
               np.linspace(0.6, 0.05, N_BCI_ROWS)
        bci = np.clip(base + rng.normal(0, 0.05, N_BCI_ROWS), 0, 1)
        df = pd.DataFrame({c: rng.normal(size=N_BCI_ROWS) for c in CEBRA_COLS})
        df['BCI'] = bci
        df.insert(0, 'file', f'{stem}.mat')
        df.insert(1, 'rel_sec', np.arange(N_BCI_ROWS) * 10)
        df.to_csv(raw / 'qeeg' / f'{pid}_rel10s_with_spike.csv', index=False)

        offsets.append({'ICARE_files': f'{stem}.mat', 'time_from_rosc': 3600})

    pd.DataFrame(offsets).to_csv(raw / 'offsets.csv', index=False)
    pd.DataFrame([{'pat_ICARE': p, 'cpc': c, 'cpc_bin': 'good' if c <= 2 else 'poor',
                   'ROSC(minutes)': 20, 'age': 60, 'sex': 'M', 'vfib': 1,
                   'time_to_CA(seconds)': 1000} for p, c, _ in COHORT]
                 ).to_csv(tables / 'ICARE_clinical.csv', index=False)
    for split in ('train', 'test'):
        pd.DataFrame({'patient_id': [p for p, _, s in COHORT if s == split]}
                     ).to_csv(tables / f'split_{split}.csv', index=False)

    return {'patients': len(COHORT), 'hours_each': HOURS,
            'train': sum(1 for *_, s in COHORT if s == 'train'),
            'test': sum(1 for *_, s in COHORT if s == 'test')}


if __name__ == '__main__':
    import sys
    print(build(Path(sys.argv[1] if len(sys.argv) > 1 else '.')))
