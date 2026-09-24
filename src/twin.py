"""
twin.py — the digital twin rendered in CEBRA space (Figure 2).

Depends on the handoff files:
    twin/twin_step4_handoff.npz      per-block outcome, forecast, hidden state
    twin/twin_matching_handoff.npz   per-hour matching streams + roll_traj

Retrieval reproduces the deployed rule (manuscript S2.8):
    combined = alpha * trajectory + (1 - alpha) * feature
where trajectory is cosine similarity between transformer hidden states at the
query hour, and feature is the mean of per-stream standardised cosines over
CEBRA, ProtoPNet, prototype activations, label frequencies, qEEG and clinical.

Train patients with no EEG yet at the query hour carry NaN in the matching
streams and are excluded from the pool -- the same constraint a bedside system
would face.
"""
import pathlib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import figures as cf
from constants import CLASS_NAMES, CLASS_COLORS, DISPLAY_ORDER, N_CLASSES
from config import TWIN_HANDOFF_DIR

FEATURE_STREAMS = ['cebra', 'protopnet', 'proto_acts', 'labelfreq', 'qeeg']

# Analogue trajectory colours. Every hue on the globe is already spoken for by
# one of the eight state regions, so the analogues are separated by VALUE, not
# hue: they keep the outcome semantics of the cohort cloud (blue = good,
# orange = poor) but sit far darker than the faint cloud and the thin state
# rings, which puts them unambiguously in the foreground.
ANALOGUE_GOOD = '#0b3d91'    # deep blue  (cloud good is light blue)
ANALOGUE_POOR = '#9c3400'    # deep rust  (cloud poor is light orange)


def confidence_profile(twin, pid, split='test'):
    """
    What the twin actually believed about a patient over time.
      p_final      P(good) at the last observed block
      p_mean       mean P(good) over observed blocks
      uncertainty  mean(1 - 2*|p - 0.5|); 1.0 = sat on the fence throughout
      crossings    number of times P(good) crossed 0.5
    """
    t4 = twin['step4']
    i = int(np.where(t4[f'{split}_pids'] == pid)[0][0])
    m = t4[f'mask_{split}'][i].astype(bool)
    p = t4[f'outcome_prob_{split}'][i][m]
    if len(p) == 0:
        return None
    return dict(pid=pid, index=i, n_obs=int(m.sum()),
                good=bool(t4[f'y_{split}'][i] == 1),
                cpc=int(t4[f'cpc_{split}'][i]),
                p_final=float(p[-1]), p_mean=float(p.mean()),
                uncertainty=float(np.mean(1.0 - 2.0 * np.abs(p - 0.5))),
                crossings=int(np.sum(np.diff(np.sign(p - 0.5)) != 0)),
                p=p)


def rank_by_cpc_confidence(twin, cpc_value, split='test', min_obs=40, top=8):
    """
    Exemplars at one exact CPC grade that the twin was most confident about,
    and right about.

    Confidence is scored toward the true label: P(good) for CPC 1-2,
    1 - P(good) for CPC 3-5. Ties broken by low mean uncertainty, so the
    trace is a clean commitment rather than a lucky endpoint.
    """
    t4 = twin['step4']
    rows = []
    for pid in t4[f'{split}_pids']:
        c = confidence_profile(twin, str(pid), split)
        if c is None or c['n_obs'] < min_obs or c['cpc'] != cpc_value:
            continue
        toward = c['p_final'] if c['good'] else 1.0 - c['p_final']
        mean_t = c['p_mean'] if c['good'] else 1.0 - c['p_mean']
        rows.append((str(pid), float(toward + mean_t - c['uncertainty']), c))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:top]


def rank_by_confidence(twin, archetype, split='test', min_obs=40, top=8):
    """
    Pick exemplars by what the MODEL believed, not by how the EEG states moved.
      'recovery'     good outcome the twin was confidently right about
      'nonrecovery'  poor outcome the twin was confidently right about
      'gray'         the twin never left the fence (highest uncertainty)
    """
    t4 = twin['step4']
    rows = []
    for pid in t4[f'{split}_pids']:
        c = confidence_profile(twin, str(pid), split)
        if c is None or c['n_obs'] < min_obs:
            continue
        if archetype == 'recovery':
            if not c['good']:
                continue
            score = c['p_final'] + c['p_mean'] - c['uncertainty']
        elif archetype == 'nonrecovery':
            if c['good']:
                continue
            score = (1 - c['p_final']) + (1 - c['p_mean']) - c['uncertainty']
        elif archetype == 'gray':
            score = c['uncertainty'] + 0.10 * min(c['crossings'], 6)
        else:
            raise ValueError(archetype)
        rows.append((str(pid), float(score), c))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:top]


def _guide(fig, x, y, row, col, color='gray', dash=None, width=1):
    """A reference line as a trace (add_vline/add_hline break on mixed 2D/3D)."""
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                             line=dict(color=color, width=width, dash=dash),
                             showlegend=False, hoverinfo='skip'),
                  row=row, col=col)


def load_twin(twin_dir=None):
    """The twin handoffs. One location: config.TWIN_HANDOFF_DIR."""
    d = pathlib.Path(twin_dir) if twin_dir else TWIN_HANDOFF_DIR
    return dict(step4=np.load(d / 'twin_step4_handoff.npz', allow_pickle=True),
                match=np.load(d / 'twin_matching_handoff.npz', allow_pickle=True))


def _l2(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def retrieve_neighbors(twin, pid, hour=24, k=10, alpha=0.75):
    """Top-k training analogues for a test patient, as of `hour`."""
    t4, tm = twin['step4'], twin['match']
    cur = list(tm['current_hours'])
    if hour not in cur:
        raise ValueError(f'hour must be one of {cur}')
    hi = cur.index(hour)
    qi = int(np.where(tm['test_pids'] == pid)[0][0])

    # eligibility: train patients with observed EEG (no NaN) at this hour
    ok = ~np.isnan(tm['cebra_train'][:, hi, :]).any(axis=1)

    hq = t4['hidden_test'][qi, hour - 1]
    traj = _l2(t4['hidden_train'][:, hour - 1, :]) @ _l2(hq)

    feats = []
    for s in FEATURE_STREAMS:
        T = tm[f'{s}_train'][:, hi, :]
        q = tm[f'{s}_test'][qi, hi]
        mu = np.nanmean(T, axis=0)
        sd = np.nanstd(T, axis=0) + 1e-8
        feats.append(_l2(np.nan_to_num((T - mu) / sd)) @ _l2((q - mu) / sd))
    T, q = tm['clin_train'], tm['clin_test'][qi]
    mu, sd = T.mean(0), T.std(0) + 1e-8
    feats.append(_l2((T - mu) / sd) @ _l2((q - mu) / sd))
    feat = np.mean(feats, axis=0)

    comb = alpha * traj + (1 - alpha) * feat
    comb[~ok] = -np.inf
    top = np.argsort(-comb)[:k]
    return dict(
        query=pid, hour=hour, query_index=qi, alpha=alpha,
        eligible=int(ok.sum()),
        neighbors=[dict(pid=str(tm['train_pids'][j]), sim=float(comb[j]),
                        traj=float(traj[j]), feat=float(feat[j]),
                        cpc=int(t4['cpc_train'][j]), good=bool(t4['y_train'][j] == 1),
                        index=int(j)) for j in top],
    )


# Where the panel-b legend sits. The rolled-forward curves hug the top of the
# axis for a recovering patient and the bottom for a non-recovering one, so the
# legend has to move to the opposite corner or it covers them.
ROLL_LEGEND_POS = {
    'top-right':    dict(x=0.988, y=0.975, xanchor='right', yanchor='top'),
    'bottom-right': dict(x=0.988, y=0.545, xanchor='right', yanchor='bottom'),
    'top-left':     dict(x=0.600, y=0.975, xanchor='left',  yanchor='top'),
    'bottom-left':  dict(x=0.600, y=0.545, xanchor='left',  yanchor='bottom'),
}


def fig_twin_in_cebra_space(run, twin, pid, hour=24, k=10, alpha=0.75,
                            n_globe=3, split='train', cloud_subsample=35000,
                            roll_legend='bottom-right',
                            window=24, slerp_n=10, camera=None,
                            width=1450, height=760):
    """
    Figure 2 (CEBRA component) — the twin for one patient:
      (a) the query's observed trajectory on the manifold, with its retrieved
          analogues drawn in the same space and coloured by their outcome
      (b) the analogues' outcome trajectories rolled forward from the query
          hour, against the twin's own probability for the query
      (c) the query's EEG-state sequence, with the query hour marked
    """
    t4, tm = twin['step4'], twin['match']
    res = retrieve_neighbors(twin, pid, hour, k, alpha)
    qi = res['query_index']

    d = run[split]
    camera = camera or cf.PAPER_CAMERA

    _, q = cf.find_patient(run, pid)
    obs = q['hours'] <= hour
    q_traj = cf.smooth_trajectory(q['emb'][obs], q['y'][obs], window, slerp_n)

    # Ten trajectories on one globe is unreadable; S2.9 makes the top-1 the
    # twin, so only the closest n_globe are drawn here. All k stay in panel b.
    n_traj = []
    for nb in res['neighbors'][:n_globe]:
        try:
            _, npat = cf.find_patient(run, nb['pid'])
        except KeyError:
            continue
        m = npat['hours'] <= hour
        if m.sum() < 2:
            continue
        n_traj.append((nb, cf.smooth_trajectory(npat['emb'][m], npat['y'][m],
                                                window, slerp_n)))

    paths = [q_traj['path']] + [t['path'] for _, t in n_traj]
    label_pts = [p[None, :] for p, _ in
                 cf.label_ring(run, camera, split).values()]
    rng = cf.axis_range(d['emb'], *paths, *label_pts, pad=1.04)

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scatter3d', 'rowspan': 2}, {'type': 'xy'}],
               [None, {'type': 'xy'}]],
        column_widths=[0.52, 0.48], row_heights=[0.62, 0.38],
        subplot_titles=(
            f'<b>a</b>   {pid} and its {len(n_traj)} closest analogues '
            f'in CEBRA space (as of hour {hour})',
            f'<b>b</b>   Outcome trajectories rolled forward from hour {hour}',
            f'<b>c</b>   {pid} EEG-state sequence'),
        horizontal_spacing=0.07, vertical_spacing=0.11)

    # ── (a) globe ────────────────────────────────────────────────────
    # here the cohort cloud is context, not the subject: keep it faint enough
    # that the analogue trajectories (same blue) still read against it
    cf._add_outcome_cloud(fig, d['emb'], d['cpc_b'], 1, 1, opacity=0.13,
                          size=1.4, legend=True, subsample=cloud_subsample,
                          legend_id='legend')
    cf._add_boundaries(fig, run, 1, 1, split=split, labels=True, legend=False,
                       camera=camera, width=1.8, label_size=11,
                       label_classes=cf.ATLAS_LABEL_CLASSES)
    seen = set()
    for rank, (nb, t) in enumerate(n_traj, 1):
        col = ANALOGUE_GOOD if nb['good'] else ANALOGUE_POOR
        top1 = rank == 1
        nm = (f"Twin (rank 1) — {nb['pid']}" if top1 else
              'Other analogues (rank 2-%d)' % len(n_traj))
        p = t['path']
        fig.add_trace(go.Scatter3d(          # halo, so the line reads on the cloud
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='lines',
            line=dict(color='rgba(255,255,255,0.85)', width=9 if top1 else 6),
            showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='lines',
            line=dict(color=col, width=5.5 if top1 else 2.8),
            opacity=1.0 if top1 else 0.75,
            name=nm, legendgroup=nm, showlegend=nm not in seen,
            legend='legend',
            hovertemplate=f"{nb['pid']} rank {rank} (CPC {nb['cpc']})<extra></extra>"),
            row=1, col=1)
        seen.add(nm)
    cf._add_trajectory(fig, q_traj, 1, 1, name=f'{pid} (query)',
                       legend=True, legend_id='legend', line_width=7,
                       marker_size=3.4)

    # ── (b) rolled-forward outcome trajectories ──────────────────────
    cur = list(tm['current_hours'])
    fut = np.array(tm['future_hours'], float)
    hi = cur.index(hour)
    # show the same exemplars as the globe, not all k -- the rest are summarised
    # by the dotted aggregate below
    for rank, nb in enumerate(res['neighbors'][:n_globe], 1):
        col = ANALOGUE_GOOD if nb['good'] else ANALOGUE_POOR
        top1 = rank == 1
        fig.add_trace(go.Scatter(
            x=fut, y=tm['roll_traj_train'][nb['index'], hi, :],
            mode='lines',
            line=dict(color=col, width=3.0 if top1 else 1.6),
            opacity=1.0 if top1 else 0.7,
            name=(f"Twin (rank 1) — {nb['pid']}" if top1 else
                  f'Other analogues (rank 2-{n_globe})'),
            legendgroup=('twin1' if top1 else 'twinrest'),
            showlegend=(top1 or rank == 2), legend='legend2',
            hovertemplate=f"{nb['pid']} rank {rank}: %{{y:.2f}}<extra></extra>"),
            row=1, col=2)
    fwd = fut >= hour
    nb_mean = np.nanmean(
        [tm['roll_traj_train'][nb['index'], hi, :][fwd]
         for nb in res['neighbors']], axis=0)
    fig.add_trace(go.Scatter(x=fut[fwd], y=nb_mean, mode='lines',
                             line=dict(color='#444', width=2.2, dash='dot'),
                             name=f'Mean of all {len(res["neighbors"])} analogues',
                             legend='legend2'),
                  row=1, col=2)
    mk = t4['mask_test'][qi].astype(bool)
    hrs = np.arange(1, 85)
    fig.add_trace(go.Scatter(x=hrs[mk], y=t4['outcome_prob_test'][qi][mk],
                             mode='lines', line=dict(color='#111', width=3),
                             name=f'{pid} twin P(good)', legend='legend2'),
                  row=1, col=2)
    # guide lines drawn as traces: add_vline/add_hline choke on scatter3d
    # traces living in the same figure (they probe trace['xaxis'])
    _guide(fig, [hour, hour], [0, 1], 1, 2, dash='dash', color='gray')
    _guide(fig, [0, 85], [0.5, 0.5], 1, 2, color='lightgray')

    # ── (c) state sequence ───────────────────────────────────────────
    for c in DISPLAY_ORDER:
        m = q['y'] == c
        if not m.any():
            continue
        fig.add_trace(go.Scatter(
            x=q['hours'][m],
            y=np.full(m.sum(), DISPLAY_ORDER.index(c)),
            mode='markers',
            marker=dict(color=CLASS_COLORS[c], size=4, symbol='line-ns',
                        line=dict(color=CLASS_COLORS[c], width=3)),
            showlegend=False, name=CLASS_NAMES[c],
            hovertemplate=f'{CLASS_NAMES[c]} @ %{{x:.1f}} h<extra></extra>'),
            row=2, col=2)
    _guide(fig, [hour, hour], [-0.6, N_CLASSES - 0.4], 2, 2,
           dash='dash', color='gray')

    fig.update_xaxes(title_text='Hours from recording start', row=1, col=2,
                     range=[0, 85])
    fig.update_yaxes(title_text='P(good outcome)', row=1, col=2, range=[0, 1])
    fig.update_xaxes(title_text='Hours from recording start', row=2, col=2,
                     range=[0, 85])
    fig.update_yaxes(row=2, col=2, tickmode='array',
                     tickvals=list(range(N_CLASSES)),
                     ticktext=[CLASS_NAMES[c] for c in DISPLAY_ORDER],
                     autorange='reversed', tickfont=dict(size=9))

    gf = float(np.mean([n['good'] for n in res['neighbors']]))
    fig.update_layout(
        scene=cf._scene(rng, camera),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(color='black', family=cf.FONT, size=12),
        legend=dict(title=dict(text='<b>CEBRA space</b>', font=dict(size=10)),
                    font=dict(size=9.5), itemsizing='constant',
                    x=0.005, y=0.33, xanchor='left', yanchor='top',
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='lightgray', borderwidth=1),
        legend2=dict(font=dict(size=9.5), itemsizing='constant',
                     bgcolor='rgba(255,255,255,0.88)',
                     bordercolor='lightgray', borderwidth=1,
                     **ROLL_LEGEND_POS[roll_legend]),
        margin=dict(l=0, r=20, t=76, b=50),
        width=width, height=height,
        annotations=list(fig.layout.annotations) + [dict(
            text=(f"analogue good-fraction {gf:.0%}  ·  "
                  f"twin P(good) at h{hour} = "
                  f"{t4['outcome_prob_test'][qi, hour - 1]:.2f}  ·  "
                  f"true outcome CPC {t4['cpc_test'][qi]}"),
            xref='paper', yref='paper', x=0.76, y=1.075, showarrow=False,
            font=dict(size=10.5, color='#444'), xanchor='center')])
    for a in fig.layout.annotations[:3]:
        a.font.size = 12.5
        a.font.family = cf.FONT
    fig.update_xaxes(showgrid=True, gridcolor='#eee', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='#eee', zeroline=False)
    return fig
