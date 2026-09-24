#!/usr/bin/env python3
"""
run_all.py — the whole runnable pipeline, one command.

    python run_all.py              # run what's needed, skip what's done
    python run_all.py --force      # recompute everything, including training
    python run_all.py --list       # show the plan and exit
    python run_all.py --figures    # only the figure steps (assumes 01/02 done)

Preflight-checks the inputs, skips any step whose outputs already exist, and
skips (with a warning, not an error) any step whose inputs are absent — so the
CEBRA half still runs if the twin handoff files are missing.

Every stage runs from here; the build-dataset stage is skipped when its inputs
are absent and the dataset already exists.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from config import (PPNET_TRAIN, PPNET_TEST, prep, embeddings, TWIN_NPZ,
                    HANDOFF_STEP4, HANDOFF_MATCH, CLINICAL_CSV, PREPROCESS,
                    FIG_CEBRA, ROOT)   # noqa: E402

PPNET = [PPNET_TRAIN, PPNET_TEST]
TWIN = [HANDOFF_STEP4, HANDOFF_MATCH]
PREP = [prep('train'), prep('test')]
CLINICAL = CLINICAL_CSV
EMB = [embeddings('train'), embeddings('test')]
FIGS = FIG_CEBRA
# what the user must supply: the preprocessing inputs. Everything else is built.
RAW = (list(PREPROCESS['PROTOPNET_DIRS']) +
       [PREPROCESS['BCI_CSV_DIR'], PREPROCESS['OFFSETS_CSV'],
        PREPROCESS['CLINICAL_CSV'], PREPROCESS['TRAIN_IDS'], PREPROCESS['TEST_IDS']])

# name, script, needs, produces (empty = always run), optional
STEPS = [
    ('build dataset',     'run_build_dataset.py',      RAW,          PPNET, False, False),
    ('CEBRA features',    'run_cebra_features.py',         PPNET,        PREP,  False, False),
    ('train CEBRA',       'run_train_cebra.py',        PREP,         EMB,   False, False),
    ('Figure 3 globes',   'run_fig_globes.py',         EMB + TWIN,   [FIGS / 'Fig3_cebra_globes.png'], False, True),
    ('Figure 2 twin',     'run_fig_twin.py',           EMB + TWIN,   [FIGS / 'Fig2_twin_cebra_ICARE_0010_h24.png'], False, True),
    ('prototype figure',  'run_fig_prototypes.py',     EMB,          [FIGS / 'Fig_prototypes_ICARE_0279.png'], False, True),
    ('embedding metrics', 'run_embedding_eval.py',     PREP + EMB,   [],    True,  False),
    ('retrieval ablation','run_retrieval_ablation.py', TWIN,         [],    True,  True),
    ('export for twin',   'run_export_for_twin.py',    PPNET + EMB,
     [TWIN_NPZ['train']], False, False),
    ('train twin',        'run_train_twin.py',         [TWIN_NPZ['train'], CLINICAL],
     [HANDOFF_STEP4], False, False),
]

# run order: the twin handoffs must exist before the figures that read them
ORDER = ['run_build_dataset', 'run_cebra_features', 'run_train_cebra',
         'run_export_for_twin', 'run_train_twin', 'run_fig_globes',
         'run_fig_twin', 'run_fig_prototypes', 'run_embedding_eval',
         'run_retrieval_ablation']
STEPS.sort(key=lambda st: ORDER.index(st[1][:-3]))
FIGURE_STEPS = ('run_fig_globes', 'run_fig_twin', 'run_fig_prototypes')


def missing(paths):
    return [p for p in paths if not Path(p).exists()]


def stale(needs, makes):
    """
    True when an output is older than an input, i.e. it was built from data
    that has since been rebuilt. Existence alone is not enough: rebuilding the
    dataset must invalidate everything downstream, or a run silently mixes a
    new dataset with an old embedding.
    """
    if not makes:
        return False
    newest_in = max((Path(p).stat().st_mtime for p in needs if Path(p).exists()),
                    default=0)
    oldest_out = min((Path(p).stat().st_mtime for p in makes if Path(p).exists()),
                     default=0)
    return oldest_out < newest_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='recompute even if outputs exist')
    ap.add_argument('--list', action='store_true', help='show the plan and exit')
    ap.add_argument('--figures', action='store_true', help='figure steps only')
    ap.add_argument('--skip-optional', action='store_true', help='skip metrics/ablation/export')
    a = ap.parse_args()

    steps = ([s for s in STEPS if s[1][:-3] in FIGURE_STEPS]
             if a.figures else list(STEPS))
    if a.skip_optional:
        steps = [s for s in steps if not s[4]]

    if not a.figures and missing(PPNET) and missing(RAW):
        def rel(p):
            try:
                return Path(p).relative_to(ROOT)
            except ValueError:
                return p
        print('No input data found. Supply either route:\n')
        print('  Route A -- build the dataset from raw (~54 GB, hours):')
        for p in missing(RAW):
            print(f'     {rel(p)}')
        print('\n  Route B -- start from the built dataset (1.6 GB, skips the build):')
        for p in PPNET + [CLINICAL]:
            print(f'     {rel(p)}')
        print('\nEverything else is generated. See "Requirements" in README.md.')
        print('Nothing was run.')
        return 1

    if a.list:
        for n, s, need, _, opt, _ in steps:
            print(f'  {n:20s} {s:26s} {"optional" if opt else "required"}'
                  f'{"   [inputs missing]" if missing(need) else ""}')
        return 0

    results, t_all = [], time.time()
    for name, script, needs, makes, _opt, skippable in steps:
        if missing(needs):
            results.append((name, 'SKIP', 0, 'inputs missing'))
            print(f'--- {name}: skipped, inputs missing')
            continue
        if makes and not a.force and not missing(makes):
            if not stale(needs, makes):
                results.append((name, 'CACHED', 0, ''))
                print(f'--- {name}: up to date')
                continue
            print(f'--- {name}: inputs are newer than outputs, rebuilding')
        print(f'\n{"=" * 62}\n== {name}\n{"=" * 62}', flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(ROOT / 'scripts' / script)],
                           cwd=ROOT, env=os.environ)
        dt = time.time() - t0
        if r.returncode:
            results.append((name, 'FAIL', dt, f'exit {r.returncode}'))
            print(f'\n{name} FAILED (exit {r.returncode}). Stopping.')
            break
        results.append((name, 'OK', dt, ''))

    print('\n' + '=' * 58)
    for name, st, dt, note in results:
        print(f'  {name:20s} {st:7s} {dt:7.0f}s  {note}')
    print(f'  {"TOTAL":20s}         {time.time() - t_all:7.0f}s')
    print('=' * 58)
    print(f'  figures: {FIGS}')
    return 1 if any(r[1] == 'FAIL' for r in results) else 0


if __name__ == '__main__':
    sys.exit(main())
