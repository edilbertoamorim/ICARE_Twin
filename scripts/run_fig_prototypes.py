"""
11_fig_prototypes.py — CEBRA manifold + ProtoPNet prototypes for one patient.

Patient-agnostic: set PID to anyone in either split.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import figures as cf
from progress import log
from constants import CLASS_NAMES

from config import RUN_TAG
PID         = 'ICARE_0279'
SPLIT       = 'train'      # manifold cloud
TOP_K       = 50           # segments averaged to place each prototype
N_HIGHLIGHT = 5
WINDOW      = 24
STEM        = None   # set from PID below

if __name__ == '__main__':
    log('prototype figure')
    run = cf.load_run(RUN_TAG)
    if PID not in run['test']['pid'] and PID not in run['train']['pid']:
        import numpy as _np
        PID = str(_np.unique(run['test']['pid'])[0])
        print(f'  configured patient not in this cohort — using {PID}')
    pos, pstate = cf.prototype_landmarks(run, SPLIT, TOP_K)
    counts = {CLASS_NAMES[c]: int((pstate == c).sum()) for c in np.unique(pstate)}
    print(f'{len(pos)} prototypes by phenotype: {counts}')
    missing = [CLASS_NAMES[c] for c in range(8) if c not in np.unique(pstate)]
    if missing:
        print(f'  no prototype codes for: {", ".join(missing)}')

    _, p = cf.find_patient(run, PID)
    src = run['test'] if PID in run['test']['pid'] else run['train']
    m = src['pid'] == PID
    mean_act = src['acts'][m].mean(axis=0)
    print(f'\n{PID} top-{N_HIGHLIGHT} prototypes by mean activation:')
    for i in np.argsort(-mean_act)[:N_HIGHLIGHT]:
        print(f'  prototype {i:2d}  {CLASS_NAMES[pstate[i]]:14s} '
              f'mean={mean_act[i]:.3f}')

    fig = cf.fig_patient_prototypes(run, PID, split=SPLIT, top_k=TOP_K,
                                    window=WINDOW, n_highlight=N_HIGHLIGHT)
    cf.export(fig, STEM or f'Fig_prototypes_{PID}')

    # globe alone, square and full-size: rotate freely, the overlay prints the
    # camera as a dict you can paste back as camera=...
    globe = cf.fig_patient_prototypes(run, PID, split=SPLIT, top_k=TOP_K,
                                      window=WINDOW, n_highlight=N_HIGHLIGHT,
                                      panel_b=False)
    cf.export(globe, f'{STEM or f"Fig_prototypes_{PID}"}_globe', formats=('html',))
    print('\nDone.')
