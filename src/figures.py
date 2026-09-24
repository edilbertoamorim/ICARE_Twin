"""
figures.py — reusable, patient-agnostic Plotly builders for the
CEBRA figures in the EEG-Twin manuscript.

Everything keys off constants.py, so the manuscript colour schema is the
single source of truth. Nothing here hardcodes a patient ID.

Typical use:
    import figures as cf
    run  = cf.load_run('combo_v')
    fig  = cf.fig_trajectory_globes(run, good_pid='ICARE_0279',
                                         poor_pid='ICARE_0231')
    cf.export(fig, 'Fig3_cebra_globes')
"""
import os
import re
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

from constants import (CLASS_NAMES, CLASS_COLORS, DISPLAY_ORDER, N_CLASSES,
                        OUTCOME_GOOD_RGB, OUTCOME_BAD_RGB)
from config import prep, embeddings, FIG_CEBRA

try:
    from progress import done as _log_written
except ImportError:
    def _log_written(msg, p=None): print(f'  {msg}')


# ── publication defaults ────────────────────────────────────────────
FONT      = 'Arial'
EXPORT_SCALE = 3          # ~600 dpi at the default pixel sizes
LABEL_RADIUS = 1.50       # state-label distance, in units of cloud radius
# Regions named in-scene on the atlas panel; the legend names all eight.
# Default = the recovery-axis anchors plus the two commonest periodic states.
ATLAS_LABEL_CLASSES = tuple(range(8))   # name every region

# One shared orientation for every globe in the paper, resolved jointly with
# the Fig-3 exemplars by optimal_camera() over every Fig-2 and Fig-3 trajectory:
#   5/8 regions forward, >=93% of the WORST trajectory exposed, both endpoints
#   of every trajectory on the visible face, regions well separated.
# All eight regions cannot face the viewer at once -- the class centroids span
# 171 deg -- so the three at the back stay named by their ring labels and
# leader lines, and read through the translucent cloud.
PAPER_CAMERA = dict(
    eye=dict(x=-0.42769188403952535, y=1.6529493873128962, z=1.0415555555555556),
    up=dict(x=0.1304524092482481, y=-0.5041742384815869, z=0.8536922783842193),
)
PANEL_LETTER_SIZE = 20


# ════════════════════════════════════════════════════════════════════
# Loading
# ════════════════════════════════════════════════════════════════════
def load_run(tag=None, out_dir=None, splits=('train', 'test')):
    """Load prep + embeddings from data/cebra into a dict of split dicts."""
    run = {'tag': tag}
    for s in splits:
        p = np.load(prep(s), allow_pickle=True)
        e = np.load(embeddings(s))['embedding']
        run[s] = dict(
            emb=e,
            y=p['predictions'].astype(int),
            cpc=p['cpc_scores'].astype(float),
            cpc_b=p['cpc_binary'].astype(int),
            pid=p['patient_ids'],
            times=p['times'].astype(float),
            acts=p['activations'],
            dur=p['bin_durations'].astype(float),
        )
    return run


def find_patient(run, pid):
    """Return (split_name, per-patient dict sorted by time). Raises if absent."""
    for s in ('train', 'test'):
        d = run.get(s)
        if d is None or pid not in d['pid']:
            continue
        m = d['pid'] == pid
        order = np.argsort(d['times'][m])
        t = d['times'][m][order]
        return s, dict(
            pid=pid, split=s,
            emb=d['emb'][m][order],
            y=d['y'][m][order],
            cpc=float(d['cpc'][m][0]),
            times=t,
            hours=(t - t[0]) / 3600.0,
        )
    raise KeyError(f'{pid} not found in {list(run.keys())}')


def list_patients(run, split='test', outcome=None, min_bins=0):
    """Patient IDs, optionally filtered by outcome ('good'/'poor') and length."""
    d = run[split]
    out = []
    for pid in np.unique(d['pid']):
        m = d['pid'] == pid
        cpc = float(d['cpc'][m][0])
        oc = 'good' if cpc <= 2 else 'poor' if cpc <= 5 else 'unknown'
        if outcome and oc != outcome:
            continue
        if int(m.sum()) < min_bins:
            continue
        out.append(pid)
    return out


# ════════════════════════════════════════════════════════════════════
# Geometry
# ════════════════════════════════════════════════════════════════════
def class_centroids(run, split='train', normalize=True):
    d = run[split]
    C = np.zeros((N_CLASSES, d['emb'].shape[1]))
    for c in range(N_CLASSES):
        m = d['y'] == c
        if m.any():
            C[c] = d['emb'][m].mean(axis=0)
            if normalize:
                C[c] /= np.linalg.norm(C[c]) + 1e-12
    return C


def small_circle(points, n_pts=160, percentile=68, cap=np.pi * 0.35,
                 label_radius=1.34):
    """Small circle on the sphere enclosing `percentile`% of a class cloud."""
    c = points.mean(axis=0)
    c_dir = c / (np.linalg.norm(c) + 1e-12)
    r = float(np.linalg.norm(points, axis=1).mean())
    pu = points / (np.linalg.norm(points, axis=1, keepdims=True) + 1e-12)
    ang = np.arccos(np.clip(pu @ c_dir, -1, 1))
    alpha = float(min(np.percentile(ang, percentile), cap))
    t = np.array([1.0, 0, 0]) if abs(c_dir[0]) < 0.9 else np.array([0, 1.0, 0])
    x = t - (t @ c_dir) * c_dir
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(c_dir, x)
    th = np.linspace(0, 2 * np.pi, n_pts)
    circ = ((np.cos(th)[:, None] * x + np.sin(th)[:, None] * y) * np.sin(alpha)
            + np.cos(alpha) * c_dir) * r
    return circ, c_dir * r * label_radius


def slerp(p0, p1, t):
    n0, n1 = np.linalg.norm(p0), np.linalg.norm(p1)
    if n0 < 1e-12 or n1 < 1e-12:
        return p0 * (1 - t) + p1 * t
    u0, u1 = p0 / n0, p1 / n1
    dot = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
    om = np.arccos(dot)
    if om < 1e-6:
        return p0 * (1 - t) + p1 * t
    s = np.sin(om)
    return ((np.sin((1 - t) * om) / s) * u0
            + (np.sin(t * om) / s) * u1) * (n0 * (1 - t) + n1 * t)


def _sphere_normalize(P):
    return P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-12)


def _catmull_rom(P, n_per_seg=12):
    """
    Centripetal Catmull-Rom through P. C1-continuous, so no kink at the knots
    (piecewise SLERP is only C0, which is what made the old path look jagged).
    """
    P = np.asarray(P, float)
    if len(P) < 3:
        return P.copy()
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    t = np.linspace(0, 1, n_per_seg, endpoint=False)[:, None]
    t2, t3 = t * t, t * t * t
    out = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        out.append(0.5 * ((2 * p1)
                          + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1][None, :])
    return np.vstack(out)


SMOOTH_SIGMA_BINS = 9      # 45 min at 5-min resolution


def smooth_trajectory(emb, y, window=24, slerp_n=10, method='gaussian',
                      sigma_bins=None, n_render=900, waypoints=True):
    """
    A smooth path along a patient's course through the embedding.

    method='gaussian' (default)
        Gaussian-smooth the raw per-segment sequence, project back onto the
        sphere, then resample with a centripetal Catmull-Rom spline.  The line
        follows every segment rather than a sparse polyline, and is C1, so it
        reads as one continuous arc.
    method='slerp'
        The original modal-state waypoints joined by piecewise SLERP. Kept so
        older figures reproduce; visibly kinked.

    Waypoint markers (state-coloured) are placed ON the smoothed path at window
    centres, so they annotate the curve instead of defining it.
    """
    emb = np.asarray(emb, float)
    N = len(emb)
    if N < 2:
        return None

    if method == 'slerp':
        wp, wt, ws = [], [], []
        for s0 in range(0, N, window):
            s1 = min(s0 + window, N)
            seg_y, seg_e = y[s0:s1], emb[s0:s1]
            cls, cnt = np.unique(seg_y, return_counts=True)
            win = int(cls[np.argmax(cnt)])
            last = np.where(seg_y == win)[0][-1]
            wp.append(seg_e[last]); wt.append((s0 + last) / max(N - 1, 1))
            ws.append(win)
        wp, wt = np.array(wp), np.array(wt)
        pts, ts = [], []
        for i in range(len(wp) - 1):
            for k in range(slerp_n):
                f = k / slerp_n
                pts.append(slerp(wp[i], wp[i + 1], f))
                ts.append(wt[i] + f * (wt[i + 1] - wt[i]))
        pts.append(wp[-1]); ts.append(wt[-1])
        return dict(waypoints=wp, waypoint_t=wt, waypoint_state=np.array(ws),
                    path=np.array(pts), path_t=np.array(ts))

    # ── gaussian: smooth the raw sequence, stay on the sphere ──────────
    from scipy.ndimage import gaussian_filter1d
    # 9 bins (45 min) chosen by sweep: it maximises the fraction of windows
    # whose smoothed point still sits nearest its own dominant state
    # (0.80/0.78 on two 8-state patients, vs 0.74/0.75 at sigma=12) while the
    # drawn line is already smooth (mean turn ~5 deg). Larger sigma buys a
    # little smoothness by erasing the state structure the figure is about.
    sigma = float(SMOOTH_SIGMA_BINS) if sigma_bins is None else float(sigma_bins)
    sigma = max(sigma, 0.5)
    sm = gaussian_filter1d(emb, sigma=sigma, axis=0, mode='nearest')
    sm = _sphere_normalize(sm)
    t_raw = np.linspace(0.0, 1.0, N)

    # thin before splining so the spline smooths rather than re-tracing noise
    n_knots = int(np.clip(N // max(int(window // 2), 1), 4, 400))
    idx = np.unique(np.linspace(0, N - 1, n_knots).astype(int))
    knots, knot_t = sm[idx], t_raw[idx]

    n_per_seg = max(int(np.ceil(n_render / max(len(knots) - 1, 1))), 2)
    path = _sphere_normalize(_catmull_rom(knots, n_per_seg))
    path_t = _catmull_rom(knot_t[:, None], n_per_seg)[:, 0]
    path_t = np.clip(path_t, 0.0, 1.0)

    wp = wt = ws = None
    if waypoints:
        wp, wt, ws = [], [], []
        for s0 in range(0, N, window):
            s1 = min(s0 + window, N)
            seg_y = y[s0:s1]
            cls, cnt = np.unique(seg_y, return_counts=True)
            ws.append(int(cls[np.argmax(cnt)]))
            mid = (s0 + s1 - 1) // 2          # window centre, on the smooth path
            wp.append(sm[mid]); wt.append(t_raw[mid])
        wp, wt, ws = np.array(wp), np.array(wt), np.array(ws)

    return dict(waypoints=wp, waypoint_t=wt, waypoint_state=ws,
                path=path, path_t=path_t, smoothed=sm)


def trajectory_arc(patient, centroids, edge=0.25):
    """
    Where a patient's trajectory starts and ends, in state terms.
      drift          cosine distance between the early and late trajectory mean
      to_continuous  gain in cosine similarity to the Continuous centroid
                     (positive = background recovers toward continuous)
      from_suppressed  loss of similarity to the Burst-Suppression centroid
    These summarise the recovery axis the manuscript argues from (S2.10).
    """
    e = patient['emb']
    n = len(e)
    k = max(int(n * edge), 1)
    a = e[:k].mean(axis=0)
    b = e[-k:].mean(axis=0)
    au = a / (np.linalg.norm(a) + 1e-12)
    bu = b / (np.linalg.norm(b) + 1e-12)
    i_bs, i_cont = CLASS_NAMES.index('BurstSupp'), CLASS_NAMES.index('Continuous')
    cont = centroids[i_cont] / (np.linalg.norm(centroids[i_cont]) + 1e-12)
    bs = centroids[i_bs] / (np.linalg.norm(centroids[i_bs]) + 1e-12)
    return dict(
        drift=float(1.0 - np.dot(au, bu)),
        to_continuous=float(np.dot(bu, cont) - np.dot(au, cont)),
        from_suppressed=float(np.dot(au, bs) - np.dot(bu, bs)),
    )


def rank_by_arc(run, split='test', outcome='good', min_hours=40,
                centroids=None, top=8):
    """
    Rank patients by how clearly they show the recovery axis, not by how
    much they thrash.  Good-outcome patients are ranked by movement toward
    the Continuous region; poor-outcome patients by the absence of it.
    Returns [(pid, score, info), ...] best first.
    """
    C = centroids if centroids is not None else class_centroids(run)
    d = run[split]
    rows = []
    for pid in np.unique(d['pid']):
        m = d['pid'] == pid
        cpc = float(d['cpc'][m][0])
        oc = 'good' if cpc <= 2 else 'poor' if cpc <= 5 else 'unknown'
        if oc != outcome:
            continue
        order = np.argsort(d['times'][m])
        y = d['y'][m][order]
        hours = len(y) * 5 / 60.0
        if hours < min_hours:
            continue
        pat = dict(emb=d['emb'][m][order], y=y)
        arc = trajectory_arc(pat, C)
        n_states = len(np.unique(y))
        # legibility: reward a clear net displacement, mildly reward richness,
        # and do NOT reward raw transition count (that just makes scribble).
        base = arc['to_continuous'] if outcome == 'good' else -arc['to_continuous']
        score = base + 0.25 * arc['drift'] + 0.05 * n_states
        rows.append((pid, float(score),
                     dict(cpc=cpc, hours=hours, n_states=n_states,
                          transitions=int((y[1:] != y[:-1]).sum()), **arc)))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:top]


def prototype_landmarks(run, split='train', top_k=50):
    """Project each ProtoPNet prototype into the embedding (mean of top-k)."""
    d = run[split]
    acts = d['acts']
    n_proto = acts.shape[1]
    pos = np.zeros((n_proto, d['emb'].shape[1]))
    state = np.zeros(n_proto, dtype=int)
    for p in range(n_proto):
        top = np.argpartition(acts[:, p], -top_k)[-top_k:]
        pos[p] = d['emb'][top].mean(axis=0)
        state[p] = Counter(d['y'][top].tolist()).most_common(1)[0][0]
    return pos, state


def axis_range(*point_arrays, pad=1.10):
    pts = np.concatenate([np.atleast_2d(a) for a in point_arrays if a is not None
                          and len(a)], axis=0)
    half = float(np.abs(pts).max() * pad)
    return (-half, half)


def clinical_camera(centroids, elevation=0.40, distance=2.0):
    """
    Orient the camera so the burst-suppression -> continuous axis (the recovery
    axis the manuscript argues from, S2.10) lies exactly in the image plane,
    where motion along it is maximally visible.

    Build an orthonormal frame (a, w, eye) with a = the recovery axis and
    w = the part of world-up perpendicular to a.  Tilting the eye toward w
    keeps a . eye == 0, so the axis is never foreshortened.
    """
    bs = centroids[CLASS_NAMES.index('BurstSupp')]
    cont = centroids[CLASS_NAMES.index('Continuous')]
    a = cont - bs
    n = np.linalg.norm(a)
    if n < 1e-9:
        return dict(eye=dict(x=1.6, y=1.6, z=1.1), up=dict(x=0, y=0, z=1))
    a = a / n

    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(a, up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    w = up - np.dot(up, a) * a          # world-up, orthogonalised against a
    w /= np.linalg.norm(w) + 1e-12

    n_hat = np.cross(a, w)
    # face the camera toward the side the two anchor regions sit on, so the
    # recovery axis runs across the FRONT of the globe rather than behind it
    mid = 0.5 * (bs + cont)
    if np.dot(mid, n_hat) < 0:
        n_hat = -n_hat
    eye = n_hat + elevation * w            # both terms are perpendicular to a
    eye = eye / (np.linalg.norm(eye) + 1e-12) * distance
    return dict(eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])),
                up=dict(x=float(w[0]), y=float(w[1]), z=float(w[2])))


# ════════════════════════════════════════════════════════════════════
# Panel builders  (each appends traces to `fig` at row/col)
# ════════════════════════════════════════════════════════════════════
def _add_state_cloud(fig, emb, y, row, col, size=1.5, opacity=0.34,
                     legend=False, subsample=None, seed=0, legend_id='legend'):
    idx = np.arange(len(emb))
    if subsample and len(idx) > subsample:
        idx = np.random.RandomState(seed).choice(idx, subsample, replace=False)
    for c in DISPLAY_ORDER:
        m = y[idx] == c
        if not m.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=emb[idx][m, 0], y=emb[idx][m, 1], z=emb[idx][m, 2],
            mode='markers',
            marker=dict(size=size, color=CLASS_COLORS[c], opacity=opacity),
            name=CLASS_NAMES[c], legendgroup=CLASS_NAMES[c],
            showlegend=False, legend=legend_id,
            hoverinfo='skip'), row=row, col=col)
        if legend:          # opaque proxy: swatches at cloud opacity read washed out
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode='markers',
                marker=dict(size=7, color=CLASS_COLORS[c]),
                name=CLASS_NAMES[c], legendgroup=CLASS_NAMES[c],
                showlegend=True, legend=legend_id,
                hoverinfo='skip'), row=row, col=col)


NEUTRAL_CLOUD = '#b9bec6'


def _add_outcome_cloud(fig, emb, cpc_b, row, col, size=1.5, opacity=0.17,
                       legend=False, subsample=None, seed=0,
                       legend_id='legend2', balance=True, neutral=False):
    """
    Outcome cloud. Two corrections matter here:

    * the cohort is 441 poor vs 254 good, and poor patients record longer
      (228k vs 142k segments), so an unbalanced draw buries the good points;
      balance=True takes the same number from each outcome.
    * drawing the outcomes as two traces means the second is painted entirely
      on top of the first.  Both are merged into one shuffled trace with a
      per-point colour, so neither outcome occludes the other.

    Legend entries come from opaque zero-point proxies, since markers this
    small and transparent are invisible as swatches.
    """
    rs = np.random.RandomState(seed)
    if neutral:
        # one grey cloud: in the twin figure the cohort is context, and leaving
        # blue/orange unused there lets the analogues own those hues outright
        groups = [(0, NEUTRAL_CLOUD, 'Training cohort'),
                  (1, NEUTRAL_CLOUD, 'Training cohort')]
    else:
        groups = [(0, OUTCOME_GOOD_RGB, 'Good outcome (CPC 1-2)'),
                  (1, OUTCOME_BAD_RGB,  'Poor outcome (CPC 3-5)')]

    picks, colors = [], []
    for lab, color, _ in groups:
        idx = np.where(cpc_b == lab)[0]
        if not len(idx):
            continue
        if subsample:
            n = (subsample // 2 if balance
                 else int(subsample * len(idx) / len(cpc_b)))
            if len(idx) > n:
                idx = rs.choice(idx, n, replace=False)
        picks.append(idx)
        colors.append(np.full(len(idx), color, dtype=object))

    if picks:
        idx = np.concatenate(picks)
        col_arr = np.concatenate(colors)
        order = rs.permutation(len(idx))          # interleave the two outcomes
        idx, col_arr = idx[order], col_arr[order]
        e = emb[idx]
        fig.add_trace(go.Scatter3d(
            x=e[:, 0], y=e[:, 1], z=e[:, 2], mode='markers',
            marker=dict(size=size, color=list(col_arr), opacity=opacity),
            showlegend=False, legend=legend_id,
            hoverinfo='skip'), row=row, col=col)

    if legend:
        seen_names = set()
        for _, color, name in groups:
            if name in seen_names:
                continue
            seen_names.add(name)
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode='markers',
                marker=dict(size=7, color=color),
                name=name, legendgroup=name, showlegend=True,
                legend=legend_id, hoverinfo='skip'), row=row, col=col)


def _fibonacci_directions(n):
    """n roughly uniform unit vectors on the sphere."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


def score_camera(eye, centroids, paths, min_sep_target=0.42):
    """
    How well one viewing direction serves the figure.

      states     how many of the 8 region centroids face the viewer
      sep        smallest 2-D gap between any two region centroids
                 (regions that project on top of each other are unreadable)
      front      fraction of trajectory points not hidden behind the globe
      spread     2-D extent of the trajectories (a foreshortened path is
                 visible but useless -- start, middle and end must be legible)
    """
    eye = eye / (np.linalg.norm(eye) + 1e-12)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(eye, up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    up = up - np.dot(up, eye) * eye
    up /= np.linalg.norm(up) + 1e-12
    right = np.cross(up, eye)

    C = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    depth = C @ eye
    # graded, not binary: the cloud is translucent, so a region just past the
    # rim still reads. All eight can never face the viewer at once -- the
    # centroids span 171 deg -- so reward getting as many forward as possible.
    states = float(np.mean(np.clip((depth + 0.35) / 0.70, 0.0, 1.0)))
    P2 = np.stack([C @ right, C @ up], axis=1)
    d = np.linalg.norm(P2[:, None, :] - P2[None, :, :], axis=2)
    d[np.arange(len(d)), np.arange(len(d))] = np.inf
    sep = float(min(d.min() / min_sep_target, 1.0))

    fronts, spreads, ends = [], [], []
    for path in paths:
        u = path / (np.linalg.norm(path, axis=1, keepdims=True) + 1e-12)
        dp = u @ eye
        fronts.append(float(np.mean(dp > -0.10)))
        # start and end markers must both be on the visible face; a path whose
        # endpoint is behind the globe fails "fully visible" no matter how much
        # of its middle shows
        ends.append(float(min(dp[0], dp[-1]) > -0.05))
        q = np.stack([u @ right, u @ up], axis=1)
        step = np.linalg.norm(np.diff(q, axis=0), axis=1).sum()
        box = (q[:, 0].ptp() * q[:, 1].ptp()) ** 0.5
        spreads.append(0.5 * min(step / 2.5, 1.0) + 0.5 * min(box / 1.1, 1.0))
    # worst case, not average: averaging lets one hidden trajectory hide behind
    # several well-exposed ones
    front = float(np.min(fronts)) if fronts else 1.0
    spread = float(np.min(spreads)) if spreads else 1.0
    endpoints = float(np.min(ends)) if ends else 1.0

    total = (2.6 * states + 1.2 * sep + 1.6 * front + 1.0 * spread
             + 1.5 * endpoints)
    return total, dict(states=states, sep=sep, front=front, spread=spread,
                       endpoints=endpoints)


def optimal_camera(run, paths=(), split='train', n_candidates=20000,
                   distance=2.0, front_min=0.90, sep_min=0.85, verbose=False):
    """
    One shared viewing direction for every globe in the paper.

    Constrained, not a plain maximum: the class centroids span 171 deg, so no
    view can put all eight regions in front. Trajectory exposure is the binding
    requirement (a path hidden behind the globe is unreadable), so `front` and
    `sep` are hard floors and the number of forward-facing regions is maximised
    subject to them. Deterministic.
    """
    C = class_centroids(run, split)
    U = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    paths = [np.asarray(p, float) for p in paths if p is not None and len(p)]
    best, best_key, best_parts = None, None, None
    for eye in _fibonacci_directions(n_candidates):
        sc, parts = score_camera(eye, C, paths)
        if paths and (parts['front'] < front_min or parts['sep'] < sep_min
                      or parts['endpoints'] < 1.0):
            continue
        key = (int(((U @ eye) > 0).sum()), parts['states'], sc)
        if best_key is None or key > best_key:
            best, best_key, best_parts = eye, key, parts
    if best is None:                      # floors unreachable -> plain maximum
        for eye in _fibonacci_directions(n_candidates):
            sc, parts = score_camera(eye, C, paths)
            if best_key is None or sc > best_key[-1]:
                best, best_key, best_parts = eye, (0, parts['states'], sc), parts
    if verbose:
        print(f'  camera: {best_key[0]}/8 regions forward  ' +
              '  '.join(f'{k}={v:.2f}' for k, v in best_parts.items()))
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(best, up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    up = up - np.dot(up, best) * best
    up /= np.linalg.norm(up) + 1e-12
    e = best * distance
    return dict(eye=dict(x=float(e[0]), y=float(e[1]), z=float(e[2])),
                up=dict(x=float(up[0]), y=float(up[1]), z=float(up[2])))


def camera_frame(camera):
    """Orthonormal (right, up, eye) image-plane basis for a plotly camera."""
    e = camera.get('eye', {})
    eye = np.array([e.get('x', 0), e.get('y', 0), e.get('z', 0)], float)
    n = np.linalg.norm(eye)
    if n < 1e-9:
        return None
    eye = eye / n
    uu = camera.get('up', {'x': 0, 'y': 0, 'z': 1})
    up = np.array([uu.get('x', 0), uu.get('y', 0), uu.get('z', 1)], float)
    up = up - np.dot(up, eye) * eye
    up /= np.linalg.norm(up) + 1e-12
    return np.cross(up, eye), up, eye


def label_ring(run, camera, split='train', min_pts=20, ring=1.34,
               front=0.85, min_deg=30.0):
    """
    Lay the state names out on a ring around the globe.

    Each label keeps the compass bearing of its own cluster (so it still points
    at the right region), is pushed clear of the point cloud, and is placed at
    a front depth so it is never occluded.  Overlaps are resolved by sliding
    labels ALONG the ring, which cannot drag a label off its cluster the way
    free 2-D repulsion does.

    Returns {class_index: (label_xyz, anchor_xyz)} for drawing leader lines.
    """
    frame = camera_frame(camera)
    if frame is None:
        return {}
    right, up, eye = frame
    d = run[split]
    R = float(np.linalg.norm(d['emb'], axis=1).mean())

    items = []
    for c in DISPLAY_ORDER:
        m = d['y'] == c
        if m.sum() < min_pts:
            continue
        cen = d['emb'][m].mean(axis=0)
        cdir = cen / (np.linalg.norm(cen) + 1e-12)
        anchor = cdir * R
        th = float(np.arctan2(anchor @ up, anchor @ right))
        items.append([c, th, anchor])
    if not items:
        return {}

    # slide along the ring until every neighbour is >= min_deg apart
    items.sort(key=lambda it: it[1])
    step = np.deg2rad(min_deg)
    n = len(items)
    if n * step > 2 * np.pi:                      # cannot fit -> spread evenly
        base = items[0][1]
        for i, it in enumerate(items):
            it[1] = base + i * (2 * np.pi / n)
    else:
        for _ in range(400):
            moved = False
            for i in range(n):
                j = (i + 1) % n
                gap = items[j][1] - items[i][1]
                if j == 0:
                    gap += 2 * np.pi
                if gap < step:
                    push = (step - gap) / 2.0
                    items[i][1] -= push
                    items[j][1] += push
                    moved = True
            if not moved:
                break

    out = {}
    for c, th, anchor in items:
        pos = (ring * R) * (np.cos(th) * right + np.sin(th) * up) \
              + (front * R) * eye
        out[c] = (pos, anchor)
    return out


def _add_boundaries(fig, run, row, col, split='train', labels=True,
                    legend=False, width=5, min_pts=20, camera=None,
                    label_size=13, legend_id='legend', label_classes=None,
                    leaders=True, ring=1.34, min_deg=30.0):
    """State boundary circles, plus optional ring-laid state names."""
    d = run[split]
    for c in DISPLAY_ORDER:
        m = d['y'] == c
        if m.sum() < min_pts:
            continue
        circ, _ = small_circle(d['emb'][m])
        fig.add_trace(go.Scatter3d(
            x=circ[:, 0], y=circ[:, 1], z=circ[:, 2], mode='lines',
            line=dict(color=CLASS_COLORS[c], width=width),
            name=CLASS_NAMES[c], legendgroup=CLASS_NAMES[c],
            showlegend=legend, legend=legend_id,
            hoverinfo='skip'), row=row, col=col)

    if not labels or camera is None:
        return
    want = DISPLAY_ORDER if label_classes is None else list(label_classes)
    frame = camera_frame(camera)
    eye = frame[2] if frame is not None else None
    ring_pos = label_ring(run, camera, split, min_pts, ring, min_deg=min_deg)
    right_v = frame[0] if frame is not None else None
    span = max((abs(float(p @ right_v)) for p, _ in ring_pos.values()),
               default=1.0) if right_v is not None else 1.0
    for c, (pos, anchor) in ring_pos.items():
        if c not in want:
            continue
        # text is centred on its anchor and overflows it; near the frame edge
        # anchor it so the word grows inward instead of off the panel
        tpos = 'middle center'
        if right_v is not None and span > 1e-9:
            xr = float(pos @ right_v) / span
            if xr > 0.45:
                tpos = 'middle left'
            elif xr < -0.45:
                tpos = 'middle right'
        if leaders:
            # Start just outside the cloud, never at the centroid: a leader from
            # the centroid crosses the sphere and hides the points under it.
            # For a region on the FAR side, step out along the silhouette in the
            # same bearing instead, or the leader tunnels through the globe.
            outer = anchor * 1.07
            if eye is not None and float(anchor @ eye) < 0.0:
                rim = anchor - float(anchor @ eye) * eye
                n = np.linalg.norm(rim)
                if n > 1e-9:
                    outer = rim / n * np.linalg.norm(anchor) * 1.07
            fig.add_trace(go.Scatter3d(
                x=[outer[0], pos[0]], y=[outer[1], pos[1]],
                z=[outer[2], pos[2]], mode='lines',
                line=dict(color=CLASS_COLORS[c], width=1.6),
                opacity=0.6, showlegend=False,
                hoverinfo='skip'), row=row, col=col)
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]], mode='text',
            text=[f'<b>{CLASS_NAMES[c]}</b>'], textposition=tpos,
            textfont=dict(size=label_size, color=CLASS_COLORS[c], family=FONT),
            showlegend=False, hoverinfo='skip'), row=row, col=col)


def _add_trajectory(fig, traj, row, col, name, halo=True,
                    line_width=5, marker_size=3.2, legend=True,
                    colorbar=None, show_waypoints=True, legend_id='legend2'):
    p, t = traj['path'], traj['path_t']
    if halo:
        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='lines',
            line=dict(color='rgba(0,0,0,0.55)', width=line_width + 3),
            showlegend=False, hoverinfo='skip'), row=row, col=col)
    fig.add_trace(go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='lines',
        line=dict(color=t, colorscale='Plasma', width=line_width,
                  cmin=0, cmax=1,
                  showscale=colorbar is not None,
                  colorbar=colorbar or None),
        name=name, showlegend=legend, legend=legend_id,
        hoverinfo='skip'), row=row, col=col)
    wp, wst = traj['waypoints'], traj['waypoint_state']
    if show_waypoints:
        fig.add_trace(go.Scatter3d(
            x=wp[:, 0], y=wp[:, 1], z=wp[:, 2], mode='markers',
            marker=dict(size=marker_size,
                        color=[CLASS_COLORS[s] for s in wst],
                        line=dict(color='black', width=0.8)),
            text=[CLASS_NAMES[s] for s in wst],
            hovertemplate='%{text}<extra></extra>',
            showlegend=False), row=row, col=col)
    for pt, sym, col_, lbl in [(wp[0], 'diamond', '#00C853', 'Start'),
                               (wp[-1], 'diamond', '#D50000', 'End')]:
        fig.add_trace(go.Scatter3d(
            x=[pt[0]], y=[pt[1]], z=[pt[2]], mode='markers',
            marker=dict(size=6, color=col_, symbol=sym,
                        line=dict(color='black', width=1.2)),
            name=lbl, legendgroup=lbl, showlegend=legend,
            legend=legend_id, hoverinfo='skip'), row=row, col=col)


def _scene(rng, camera, show_axes=False):
    ax = dict(range=rng, showbackground=False, showticklabels=False,
              title='', showgrid=show_axes, zeroline=False,
              showline=False, visible=show_axes)
    return dict(bgcolor='white', aspectmode='cube',
                xaxis=ax, yaxis=ax, zaxis=ax, camera=camera)


# ════════════════════════════════════════════════════════════════════
# Figure: CEBRA trajectory globes
# ════════════════════════════════════════════════════════════════════
def fig_trajectory_globes(run, patients=None, good_pid=None, poor_pid=None,
                          split='train', atlas=True, atlas_subsample=70000,
                          cloud_subsample=55000, window=24, slerp_n=10,
                          camera=None, show_prototypes=False,
                          panel_width=470, height=575,
                          label_ring_r=1.20, pad=1.16, label_size=11.5):
    """
    A row of CEBRA globes on one shared camera and one shared axis range.

    patients : list of (pid, caption) — any length, any patients, any split.
               e.g. [('ICARE_0142', 'Recovery'),
                     ('ICARE_0279', 'Gray zone'),
                     ('ICARE_0277', 'Non-recovery')]
               A bare 'PID' string is accepted and captioned automatically.
    atlas    : prepend the state-atlas panel (manifold coloured by ACNS state).

    good_pid/poor_pid are kept as a two-panel shorthand for older callers.
    """
    if patients is None:
        if good_pid is None or poor_pid is None:
            raise ValueError('pass patients=[...] or both good_pid and poor_pid')
        patients = [(good_pid, 'Good outcome'), (poor_pid, 'Poor outcome')]
    if not patients and not atlas:
        raise ValueError('nothing to draw: pass patients=[...] or atlas=True')

    norm = []
    for item in patients:
        pid, cap = (item, None) if isinstance(item, str) else item
        _, pat = find_patient(run, pid)
        if cap is None:
            cap = 'Good outcome' if pat['cpc'] <= 2 else 'Poor outcome'
        norm.append((pid, cap, pat, smooth_trajectory(pat['emb'], pat['y'],
                                                      window, slerp_n)))

    d = run[split]
    camera = camera or PAPER_CAMERA

    label_pts = [p[None, :] for p, _ in
                 label_ring(run, camera, split, ring=label_ring_r).values()]
    rng = axis_range(d['emb'], *[t['path'] for _, _, _, t in norm],
                     *label_pts, pad=pad)

    letters = 'abcdefghij'
    titles, k = [], 0
    if atlas:
        titles.append('<b>a</b>   CEBRA state atlas'); k = 1
    for i, (pid, cap, pat, _) in enumerate(norm):
        cpc_txt = '' if 'CPC' in cap else f"CPC {pat['cpc']:.0f}, "
        titles.append(f"<b>{letters[i + k]}</b>   {cap} — {pid} "
                      f"({cpc_txt}{pat['hours'][-1]:.0f} h)")

    ncols = len(titles)
    fig = make_subplots(rows=1, cols=ncols,
                        specs=[[{'type': 'scatter3d'}] * ncols],
                        subplot_titles=titles, horizontal_spacing=0.008)

    col = 1
    if atlas:
        _add_state_cloud(fig, d['emb'], d['y'], 1, 1, opacity=0.35,
                         legend=True, subsample=atlas_subsample,
                         legend_id='legend')
        _add_boundaries(fig, run, 1, 1, split=split, labels=True,
                        legend=False, camera=camera, width=5,
                        label_size=label_size, ring=label_ring_r,
                        label_classes=ATLAS_LABEL_CLASSES)
        if show_prototypes:
            pos, st = prototype_landmarks(run, split)
            for c in DISPLAY_ORDER:
                m = st == c
                if m.any():
                    fig.add_trace(go.Scatter3d(
                        x=pos[m, 0], y=pos[m, 1], z=pos[m, 2], mode='markers',
                        marker=dict(size=5, color=CLASS_COLORS[c],
                                    symbol='cross',
                                    line=dict(color='white', width=1)),
                        showlegend=False, hoverinfo='skip'), row=1, col=1)
        col = 2

    cbar = dict(title=dict(text='Recording<br>progress', font=dict(size=11)),
                x=1.005, len=0.55, thickness=12,
                tickvals=[0, 0.5, 1], ticktext=['start', 'mid', 'end'],
                tickfont=dict(size=10))
    for i, (pid, cap, pat, traj) in enumerate(norm):
        first, last = (i == 0), (i == len(norm) - 1)
        _add_outcome_cloud(fig, d['emb'], d['cpc_b'], 1, col,
                           legend=first, subsample=cloud_subsample,
                           legend_id='legend2')
        _add_boundaries(fig, run, 1, col, split=split, labels=False,
                        legend=False, width=2.0)
        _add_trajectory(fig, traj, 1, col, name='Patient trajectory',
                        legend=first, legend_id='legend2',
                        colorbar=cbar if last else None)
        col += 1

    sc = _scene(rng, camera)
    layout = dict(paper_bgcolor='white',
                  font=dict(color='black', family=FONT, size=12),
                  legend=dict(title=dict(text='<b>EEG state</b>  ',
                                         font=dict(size=10), side='left'),
                              orientation='h', font=dict(size=10),
                              itemsizing='constant', x=0.5, y=-0.02,
                              xanchor='center', yanchor='top'),
                  legend2=dict(title=dict(text='<b>Cohort</b>  ',
                                          font=dict(size=10), side='left'),
                               orientation='h', font=dict(size=10),
                               itemsizing='constant', x=0.5, y=-0.10,
                               xanchor='center', yanchor='top'),
                  margin=dict(l=0, r=95, t=40, b=88),
                  width=panel_width * ncols, height=height)
    for j in range(ncols):
        layout['scene' if j == 0 else f'scene{j + 1}'] = sc
    fig.update_layout(**layout)
    for a in fig.layout.annotations:
        a.font.size = 12.5
        a.font.family = FONT
    return fig


# ════════════════════════════════════════════════════════════════════
# Export
# ════════════════════════════════════════════════════════════════════
_CAMERA_READOUT = """
var gd = document.getElementById('{plot_id}');
var NL = String.fromCharCode(10);
var scenes = Object.keys(gd.layout).filter(function (k) {
  return k.indexOf('scene') === 0;
});
var box = document.createElement('div');
box.style.cssText = 'position:fixed;bottom:10px;left:10px;max-width:46em;' +
  'background:rgba(255,255,255,.96);border:1px solid #bbb;border-radius:5px;' +
  'padding:7px 10px;font:11px/1.5 ui-monospace,Menlo,monospace;z-index:9999;' +
  'white-space:pre-wrap;color:#222;box-shadow:0 1px 4px rgba(0,0,0,.15)';
box.textContent = (scenes.length > 1)
  ? 'Rotate any globe - all ' + scenes.length + ' follow. Camera appears here.'
  : 'Rotate the globe - its camera appears here, ready to paste.';
document.body.appendChild(box);
function f(v) { return (Math.round(v * 1000) / 1000); }
var syncing = false;
gd.on('plotly_relayout', function (e) {
  if (syncing) return;
  var key = null;
  for (var k in e) { if (k.indexOf('.camera') !== -1) { key = k; break; } }
  if (!key) return;
  var src = key.split('.')[0];
  var cam = gd.layout[src] && gd.layout[src].camera;
  if (!cam || !cam.eye) return;
  var u = cam.up || {x: 0, y: 0, z: 1};
  box.textContent =
    'camera (' + src + ')  -  paste as camera=...' + NL +
    'dict(eye=dict(x=' + f(cam.eye.x) + ', y=' + f(cam.eye.y) +
    ', z=' + f(cam.eye.z) + '),' + NL +
    '     up=dict(x=' + f(u.x) + ', y=' + f(u.y) + ', z=' + f(u.z) + '))';
  if (scenes.length > 1) {
    var upd = {};
    scenes.forEach(function (s) { if (s !== src) { upd[s + '.camera'] = cam; } });
    syncing = true;
    Plotly.relayout(gd, upd).then(function () { syncing = false; },
                                 function () { syncing = false; });
  }
});
"""


def export(fig, stem, out_dir=None, formats=('html', 'png', 'pdf'),
           scale=EXPORT_SCALE, camera_readout=True, selfcontained=False):
    """
    Write interactive HTML plus publication-resolution static files.

    The HTML carries a small overlay that prints the live camera as a Python
    dict: rotate a globe to taste, copy the dict, pass it back as `camera=` and
    the static export will match exactly what you framed.
    """
    out_dir = out_dir or FIG_CEBRA
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for f in formats:
        path = os.path.join(out_dir, f'{stem}.{f}')
        if f == 'html':
            fig.write_html(path,
                           include_plotlyjs=True if selfcontained else 'cdn',
                           post_script=_CAMERA_READOUT if camera_readout else None)
        else:
            # static export needs a browser engine (kaleido >= 1 shells out to
            # Chrome). Missing Chrome must not destroy a run whose HTML is
            # already on disk -- warn and carry on.
            try:
                fig.write_image(path, scale=scale)
            except Exception as e:
                first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
                _log_written(f'SKIPPED {os.path.basename(path)} — static export '
                             f'unavailable ({first}); install with '
                             f'`plotly_get_chrome -y`')
                continue
        written.append(path)
        _log_written(f'wrote {os.path.basename(path)}', path)
    return written


def export_single_globes(run, panels, stem_prefix='Fig3_globe',
                         out_dir=None, atlas=True, **kw):
    """
    One standalone HTML per globe — easier to orient and screenshot than the
    combined row, where every scene rotates independently anyway.
    """
    written = []
    jobs = ([('a_atlas', [], True)] if atlas else [])
    letters = 'bcdefgh' if atlas else 'abcdefg'
    for i, (pid, caption) in enumerate(panels):
        slug = re.sub(r'[^a-z0-9]+', '', caption.lower())
        tag = f'{letters[i]}_{slug}_{pid}'
        jobs.append((tag, [(pid, caption)], False))
    for tag, pats, want_atlas in jobs:
        fig = fig_trajectory_globes(run, patients=pats, atlas=want_atlas,
                                    panel_width=900, height=900, **kw)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=90))
        written += export(fig, f'{stem_prefix}_{tag}', out_dir=out_dir,
                          formats=('html',))
    return written


# ════════════════════════════════════════════════════════════════════
# Figure: patient trajectory with ProtoPNet prototypes
# ════════════════════════════════════════════════════════════════════
def fig_patient_prototypes(run, pid, split='train', top_k=50,
                           cloud_subsample=45000, window=24, slerp_n=10,
                           camera=None, n_highlight=5, width=1480, height=720,
                           panel_b=True):
    """
    One patient against the prototype landmarks.

      (a) the CEBRA manifold with all ProtoPNet prototypes projected into it
          (each placed at the mean embedding of its top-k activating segments,
          coloured by the phenotype it codes for), the named state regions, and
          the patient's trajectory. Prototypes the patient activates most
          strongly are ringed.
      (b) the patient's prototype activations over time, prototypes grouped by
          phenotype -- the evidence trail behind the trajectory in (a).

    panel_b=False gives the globe alone, square and full-size -- easier to
    rotate and screenshot than a globe squeezed into half a figure.
    """
    d = run[split]
    camera = camera or PAPER_CAMERA
    pos, pstate = prototype_landmarks(run, split, top_k)

    _, p = find_patient(run, pid)
    traj = smooth_trajectory(p['emb'], p['y'], window, slerp_n)

    src = run['test'] if pid in run['test']['pid'] else run['train']
    m = src['pid'] == pid
    order = np.argsort(src['times'][m])
    acts = src['acts'][m][order]                     # (T, 45)
    hours = (src['times'][m][order] - src['times'][m][order][0]) / 3600.0

    # prototypes this patient leans on most
    mean_act = acts.mean(axis=0)
    top_idx = np.argsort(-mean_act)[:n_highlight]

    label_pts = [q[None, :] for q, _ in label_ring(run, camera, split).values()]
    rng = axis_range(d['emb'], traj['path'], pos, *label_pts, pad=1.04)

    if panel_b:
        fig = make_subplots(
            rows=1, cols=2, column_widths=[0.5, 0.5],
            specs=[[{'type': 'scatter3d'}, {'type': 'xy'}]],
            subplot_titles=(
                f'<b>a</b>   {pid} and the {len(pos)} ProtoPNet prototypes '
                f'in CEBRA space',
                f'<b>b</b>   {pid} prototype activations over time'),
            horizontal_spacing=0.08)
    else:
        fig = make_subplots(
            rows=1, cols=1, specs=[[{'type': 'scatter3d'}]],
            subplot_titles=(f'{pid} and the {len(pos)} ProtoPNet prototypes '
                            f'in CEBRA space',))

    _add_outcome_cloud(fig, d['emb'], d['cpc_b'], 1, 1, opacity=0.12, size=1.4,
                       legend=True, subsample=cloud_subsample,
                       legend_id='legend')
    _add_boundaries(fig, run, 1, 1, split=split, labels=True, legend=False,
                    camera=camera, width=2.5, label_size=11)

    for c in DISPLAY_ORDER:
        sel = pstate == c
        if not sel.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=pos[sel, 0], y=pos[sel, 1], z=pos[sel, 2], mode='markers',
            marker=dict(size=6, color=CLASS_COLORS[c], symbol='diamond',
                        line=dict(color='rgba(20,20,25,0.9)', width=1.4)),
            name=f'Prototype — {CLASS_NAMES[c]}', legendgroup='proto',
            showlegend=False, legend='legend',
            hovertemplate=f'{CLASS_NAMES[c]} prototype<extra></extra>'),
            row=1, col=1)
    fig.add_trace(go.Scatter3d(                       # one legend entry
        x=[None], y=[None], z=[None], mode='markers',
        marker=dict(size=8, color='#666', symbol='diamond',
                    line=dict(color='black', width=1.4)),
        name=f'ProtoPNet prototype (n={len(pos)})', legend='legend',
        showlegend=True, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=pos[top_idx, 0], y=pos[top_idx, 1], z=pos[top_idx, 2],
        mode='markers',
        marker=dict(size=13, color='rgba(0,0,0,0)', symbol='circle',
                    line=dict(color='#111', width=2.2)),
        name=f'Top-{n_highlight} for this patient', legend='legend',
        showlegend=True,
        text=[f'prototype {i} ({CLASS_NAMES[pstate[i]]})' for i in top_idx],
        hovertemplate='%{text}<extra></extra>'), row=1, col=1)

    _add_trajectory(fig, traj, 1, 1, name=f'{pid} trajectory',
                    legend=True, legend_id='legend', line_width=6,
                    marker_size=3.4)

    # ── (b) activation heatmap, prototypes grouped by phenotype ─────
    if not panel_b:
        fig.update_layout(
            scene=_scene(rng, camera),
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='black', family=FONT, size=12),
            legend=dict(title=dict(text='<b>CEBRA space</b>',
                                   font=dict(size=10)),
                        font=dict(size=10), itemsizing='constant',
                        x=0.005, y=0.99, xanchor='left', yanchor='top',
                        bgcolor='rgba(255,255,255,0.88)',
                        bordercolor='lightgray', borderwidth=1),
            margin=dict(l=0, r=0, t=44, b=0),
            width=950, height=950)
        for a in fig.layout.annotations[:1]:
            a.font.size = 13
            a.font.family = FONT
        return fig

    proto_order = [i for c in DISPLAY_ORDER
                   for i in np.where(pstate == c)[0]]
    fig.add_trace(go.Heatmap(
        z=acts[:, proto_order].T, x=hours,
        y=list(range(len(proto_order))),
        colorscale='Magma', zmin=0.0, zmax=float(acts.max()),
        colorbar=dict(title=dict(text='Prototype<br>activation',
                                 font=dict(size=10)),
                      x=1.005, len=0.62, thickness=12,
                      tickfont=dict(size=9)),
        hovertemplate='%{x:.1f} h · row %{y}: %{z:.3f}<extra></extra>'),
        row=1, col=2)
    # phenotype key down the left edge of the heatmap
    for row_i, pi in enumerate(proto_order):
        fig.add_trace(go.Scatter(
            x=[-1.8], y=[row_i], mode='markers',
            marker=dict(color=CLASS_COLORS[pstate[pi]], size=7,
                        symbol='square'),
            showlegend=False,
            hovertemplate=f'prototype {pi} — '
                          f'{CLASS_NAMES[pstate[pi]]}<extra></extra>'),
            row=1, col=2)

    fig.update_xaxes(title_text='Hours from recording start', row=1, col=2,
                     range=[-3.0, float(hours.max())])
    fig.update_yaxes(title_text='ProtoPNet prototype (grouped by phenotype)',
                     row=1, col=2, showticklabels=False,
                     autorange='reversed')

    fig.update_layout(
        scene=_scene(rng, camera),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(color='black', family=FONT, size=12),
        legend=dict(title=dict(text='<b>CEBRA space</b>', font=dict(size=10)),
                    font=dict(size=9.5), itemsizing='constant',
                    x=0.005, y=0.30, xanchor='left', yanchor='top',
                    bgcolor='rgba(255,255,255,0.88)',
                    bordercolor='lightgray', borderwidth=1),
        margin=dict(l=0, r=20, t=52, b=52),
        width=width, height=height)
    for a in fig.layout.annotations[:2]:
        a.font.size = 12.5
        a.font.family = FONT
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig
