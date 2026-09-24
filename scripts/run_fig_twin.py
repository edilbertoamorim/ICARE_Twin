"""
10_fig2_twin_in_cebra_space.py — Figure 2 (CEBRA component of the digital twin).

Two panels of the paper figure: one recovering and one non-recovering patient,
each shown with the analogues the twin actually retrieved at HOUR.

Both were picked to have a COMPLEX course -- all eight ACNS states visited, many
transitions -- so the figure demonstrates that the embedding and the retrieval
hold up on a hard trajectory, not just a clean one. Both are also patients the
twin called confidently and correctly.

Patient-agnostic: PATIENTS takes any test-split IDs, HOUR any value in
current_hours (6, 12, ... 84).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import figures as cf
from progress import log
import twin as tf

from config import RUN_TAG
HOUR     = 24          # bedside "as of" hour; must be in current_hours
TOP_K    = 10          # neighbours retrieved (globe shows the closest N_GLOBE)
N_GLOBE  = 3
ALPHA    = 0.75        # manuscript S2.8
# Per-patient camera and panel-b legend corner. The rolled-forward curves sit
# high for a recovering patient and low for a non-recovering one, so the legend
# goes to the opposite corner in each case.
CAM_GOOD = dict(eye=dict(x=-0.156, y=1.826, z=-0.802),
                up=dict(x=-0.275, y=0.367, z=0.889))
CAM_POOR = dict(eye=dict(x=0.827, y=-0.879, z=0.695),
                up=dict(x=0.204, y=0.718, z=0.665))

# (patient, role). Both traverse all 8 states; see rank_by_complexity below.
PATIENTS = [
    ('ICARE_0010', 'recovering',     CAM_GOOD, 'bottom-right'),
    ('ICARE_0077', 'non-recovering', CAM_POOR, 'top-right'),
]

def resolve(pid, want_good, twin):
    """Use the configured patient if this cohort has one; else the patient the
    twin called most confidently, and correctly, in that outcome class."""
    pids = [str(x) for x in twin['step4']['test_pids']]
    if pid in pids:
        return pid
    scored = []
    for q in pids:
        c = tf.confidence_profile(twin, q)
        if c is None or c['good'] != want_good:
            continue
        scored.append((c['p_final'] if c['good'] else 1 - c['p_final'], q))
    if not scored:
        return None
    scored.sort(reverse=True)
    print(f'  {pid} not in this cohort — using {scored[0][1]}')
    return scored[0][1]


if __name__ == '__main__':
    log('Figure 2 twin')
    run  = cf.load_run(RUN_TAG)
    twin = tf.load_twin()

    for pid, role, camera, roll_legend in PATIENTS:
        pid = resolve(pid, role == 'recovering', twin)
        if pid is None:
            print(f'  no {role} patient in this cohort — skipping')
            continue
        c = tf.confidence_profile(twin, pid)
        res = tf.retrieve_neighbors(twin, pid, HOUR, TOP_K, ALPHA)
        gf = sum(n['good'] for n in res['neighbors']) / len(res['neighbors'])
        print(f"\n{pid} ({role})  CPC={c['cpc']}  P_final={c['p_final']:.2f}  "
              f"P(good)@h{HOUR}={c['p'][min(HOUR, len(c['p'])) - 1]:.2f}")
        print(f"  pool {res['eligible']}/695 · analogue good-fraction {gf:.0%}")
        for i, nb in enumerate(res['neighbors'][:N_GLOBE], 1):
            print(f"   {i}. {nb['pid']}  sim={nb['sim']:.3f}  CPC={nb['cpc']}  "
                  f"{'GOOD' if nb['good'] else 'poor'}")

        fig = tf.fig_twin_in_cebra_space(run, twin, pid, HOUR, TOP_K, ALPHA,
                                         n_globe=N_GLOBE, camera=camera,
                                         roll_legend=roll_legend)
        cf.export(fig, f'Fig2_twin_cebra_{pid}_h{HOUR}')
    print('\nDone.')
