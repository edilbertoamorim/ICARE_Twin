"""
09_fig3_trajectory_globes.py — Figure 3, CEBRA trajectory globes.

Atlas + one globe per outcome grade (CPC 1 / 3 / 5), each the patient the twin
called most confidently and correctly at that grade -- so the panels differ by
what the MODEL believed, not by how the EEG states happened to move.

All four globes share one camera. In the interactive HTML they rotate together:
drag any globe and the other three follow, and the overlay prints the camera as
a dict you can paste into CAMERA below to lock the view for the static export.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import figures as cf
from progress import log
import twin as tf

from config import RUN_TAG
SPLIT     = 'train'          # split providing the manifold/atlas cloud
MIN_OBS   = 40               # observed 1-h blocks required of a candidate
WINDOW    = 24
SLERP_N   = 10
STEM      = 'Fig3_cebra_globes'
SINGLE_GLOBE_HTML = True
CAMERA    = dict(eye=dict(x=0.706, y=-0.617, z=0.612),
               up=dict(x=-0.4, y=0.372, z=0.838))

# (pid or None, caption, CPC grade). None -> most confidently, correctly called
# patient at that grade.
PANELS = [
    (None, 'Recovery',     1),
    (None, 'Intermediate', 3),
    (None, 'Non-recovery', 5),
]

if __name__ == '__main__':
    log('Figure 3 globes')
    run  = cf.load_run(RUN_TAG)
    twin = tf.load_twin()
    print(f"Loaded {RUN_TAG}: train {run['train']['emb'].shape}, "
          f"test {run['test']['emb'].shape}")

    panels, chosen = [], []
    for pid, caption, cpc in PANELS:
        ranked = [r for r in tf.rank_by_cpc_confidence(twin, cpc, 'test', MIN_OBS)
                  if r[0] not in chosen]
        print(f'\nCPC {cpc} — most confidently, correctly called:')
        for q, score, c in ranked[:4]:
            print(f"  {q}  score={score:+.3f}  P_final={c['p_final']:.2f}  "
                  f"P_mean={c['p_mean']:.2f}  uncert={c['uncertainty']:.2f}")
        if pid is None and not ranked:
            print(f'  -> no patient at CPC {cpc} clears MIN_OBS={MIN_OBS}; '
                  f'dropping the {caption} panel')
            continue
        pid = pid or ranked[0][0]
        chosen.append(pid)
        c = tf.confidence_profile(twin, pid)
        print(f"  -> {caption} (CPC {cpc}): {pid}  P_final={c['p_final']:.2f}")
        panels.append((pid, f'{caption} (CPC {cpc})'))

    fig = cf.fig_trajectory_globes(run, patients=panels, split=SPLIT,
                                   window=WINDOW, slerp_n=SLERP_N,
                                   camera=CAMERA)
    cf.export(fig, STEM)

    if SINGLE_GLOBE_HTML:
        print('\nStandalone globes:')
        cf.export_single_globes(run, panels, split=SPLIT, window=WINDOW,
                                slerp_n=SLERP_N, camera=CAMERA)
    print('\nDone.')
