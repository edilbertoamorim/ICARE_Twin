"""
Full-pipeline smoke test.

Builds synthetic preprocessing inputs in a temp directory, runs run_all.py
against them, then asserts every artifact exists in the place the layout says
it should. Nothing in the real data/ or outputs/ is touched.

    python tests/smoke_test.py            # ~10 min
    python tests/smoke_test.py --keep     # leave the temp dir for inspection

The numbers it produces are meaningless — 10 synthetic patients, a 200-iteration
CEBRA run and a 1-epoch twin. This checks wiring, not science.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from make_synthetic_inputs import build   # noqa: E402

# (label, path relative to the temp root)
EXPECTED = [
    ('dataset (train)',      'data/dataset/PPNet_data_train.npz'),
    ('dataset (test)',       'data/dataset/PPNet_data_test.npz'),
    ('cebra prep',           'data/cebra/prep_train.npz'),
    ('cebra embeddings',     'data/cebra/embeddings_train.npz'),
    ('twin input npz',       'data/twin/input/PPNet Data Train with CEBRA COMBO V.npz'),
    ('twin handoff (step4)', 'data/twin/handoffs/twin_step4_handoff.npz'),
    ('twin handoff (match)', 'data/twin/handoffs/twin_matching_handoff.npz'),
    ('cebra model',          'outputs/models/cebra_model.pt'),
    ('cebra scaler',         'outputs/models/cebra_scaler.pkl'),
    ('twin weights',         'outputs/models/twin_models.pt'),
    ('twin pipeline',        'outputs/models/twin_feature_pipeline.pkl'),
    ('Fig 3 globes',         'outputs/figures/cebra/Fig3_cebra_globes.png'),
    ('Fig 2 twin',           'outputs/figures/cebra/__FIG2__'),
    ('prototype figure',     'outputs/figures/cebra/__PROTO__'),
    ('twin fig1',            'outputs/figures/twin/fig1.png'),
    ('twin fig7',            'outputs/figures/twin/fig7.png'),
    ('twin metrics',         'outputs/metrics/twin/all_metrics.json'),
]
# nothing generated may sit outside these two roots
ROOTS = ('data', 'outputs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep', action='store_true', help='keep the temp dir')
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix='eegtwin_smoke_'))
    print(f'workspace: {tmp}\n')
    print('building synthetic inputs ...', build(tmp), '\n')

    env = {**os.environ,
           'CEBRA_DATA': str(tmp / 'data'),
           'CEBRA_OUT': str(tmp / 'outputs'),
           'CEBRA_SMOKE': '1',
           'CEBRA_TWIN_SMOKE': '1'}

    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / 'run_all.py')],
                       cwd=ROOT, env=env)
    dt = time.time() - t0
    print(f'\nrun_all.py exited {r.returncode} after {dt:.0f}s\n')

    print('artifact                   where')
    print('-' * 72)
    # patient ids are cohort-dependent, so match those two by prefix
    globs = {'__FIG2__': 'Fig2_twin_cebra_*.png',
             '__PROTO__': 'Fig_prototypes_*.png'}
    bad = []
    for label, rel in EXPECTED:
        if rel.split('/')[-1] in globs:
            hits = list((tmp / rel).parent.glob(globs[rel.split('/')[-1]]))
            ok = bool(hits)
            print(f'  {"OK " if ok else "MISS"}  {label:22s} '
                  f'{hits[0].relative_to(tmp) if ok else rel}')
            if not ok:
                bad.append(rel)
            continue
        p = tmp / rel
        ok = p.exists() and p.stat().st_size > 0
        print(f'  {"OK " if ok else "MISS"}  {label:22s} {rel}')
        if not ok:
            bad.append(rel)

    stray = [p for p in tmp.rglob('*')
             if p.is_file() and p.relative_to(tmp).parts[0] not in ROOTS]
    if stray:
        print('\nfiles written outside data/ and outputs/:')
        for p in stray[:10]:
            print(f'   {p.relative_to(tmp)}')

    ok = r.returncode == 0 and not bad and not stray
    print('\n' + ('PASS — full pipeline ran, every artifact in place'
                  if ok else 'FAIL'))
    if bad:
        print(f'  {len(bad)} missing artifact(s)')

    if a.keep:
        print(f'\nkept: {tmp}')
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
