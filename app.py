#!/usr/bin/env python3
"""Saturn E-Ring Composition Dashboard — Cassini CDA multi-panel explorer."""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from dash import Dash, html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
_data_dir = 'CS81 Cassini CDA Composition-selected'
_meta_cols = [
    'X2D', 'Y2D', 'Z_ring', 'R_sat', 'UTC', 'M3 Category',
    'Enceladus_Dist', 'Rhea_Dist', 'Dione_Dist', 'Tethys_Dist',
    'Titan_Dist', 'Mimas_Dist',
    'Latent1', 'Latent2',
    'Vrel_dust', 'V_sc', 'Inclination', 'Confidence', 'SNR',
    'PhiE_2008', 'PhiEncObs',
    'RF score', 'MSE',
]

_header = pd.read_csv(f'{_data_dir}/ConfDataSpicy.csv', nrows=0)
_available = set(_header.columns)
_meta_cols = [c for c in _meta_cols if c in _available]
_spec_cols = []
for c in _header.columns:
    try:
        float(c)
        _spec_cols.append(c)
    except ValueError:
        pass
_spec_cols.sort(key=float)

_use_cols = _meta_cols + _spec_cols
df = pd.read_csv(f'{_data_dir}/ConfDataSpicy.csv', usecols=_use_cols, dtype={c: np.float32 for c in _spec_cols})

_ext_path = f'{_data_dir}/ExtendedDataSpicy.csv'
if os.path.exists(_ext_path):
    df_ext = pd.read_csv(_ext_path, usecols=_use_cols, dtype={c: np.float32 for c in _spec_cols})
    df = pd.concat([df, df_ext], ignore_index=True)

SATURN_R_KM = 60268
df = df[(df['R_sat'] > 0) & (df['R_sat'] != -999)]

df['x'] = df['X2D'] / SATURN_R_KM
df['y'] = df['Y2D'] / SATURN_R_KM
df['z'] = df['Z_ring'].replace(-999, 0) / SATURN_R_KM

df['Vrel_dust'] = df['Vrel_dust'].replace(-999, np.nan)
df['V_sc'] = df['V_sc'].replace(-999, np.nan)
df['Inclination'] = df['Inclination'].replace(-999, np.nan)
df['SNR'] = df['SNR'].replace(-999, np.nan)
df['PhiE_2008'] = df['PhiE_2008'].replace(-999, np.nan)
df['PhiE_deg'] = np.degrees(df['PhiE_2008'])
for _opt_col in ['PhiEncObs', 'RF score', 'MSE']:
    if _opt_col in df.columns:
        df[_opt_col] = df[_opt_col].replace(-999, np.nan)
if 'PhiEncObs' in df.columns:
    df['PhiEncObs_deg'] = np.degrees(df['PhiEncObs'])

df['Year'] = pd.to_datetime(df['UTC'], format='%Y %b %d %H:%M:%S', errors='coerce').dt.year
df = df[df['Year'].notna()]
df['Year'] = df['Year'].astype(int)

def get_type(cat):
    cat = str(cat).replace('*', '')
    if cat in ['1L', '1H', '1M', '1S']:
        return 'Ice'
    elif cat == '2O':
        return 'Organic'
    elif cat in ['3M', '3L', '3W', '3C', '3K']:
        return 'Salt'
    return 'Other'

df['Type'] = df['M3 Category'].apply(get_type)

# ---------------------------------------------------------------------------
# Peak-chemistry classification
# ---------------------------------------------------------------------------
# Integer AMU -> nearest spectral column name (e.g. 18 -> '18.0')
_spec_vals = [float(c) for c in _spec_cols]
_amu_to_col = {}
for target in range(0, 201):
    best_idx = np.argmin([abs(v - target) for v in _spec_vals])
    if abs(_spec_vals[best_idx] - target) < 0.01:
        _amu_to_col[target] = _spec_cols[best_idx]

PEAK_CHEM_GROUPS = {
    'Water Ice':  [18, 19, 36, 37, 54, 55],
    'Silicate':   [28, 73, 91],
    'Iron':       [56],
    'Na Salt':    [23, 63],
    'K Salt':     [39],
    'Organic':    [12, 78],
    'Phosphorus': [31],
    'NH3/OH':     [17],
}

_group_sums = {}
for grp, amus in PEAK_CHEM_GROUPS.items():
    cols = [_amu_to_col[a] for a in amus if a in _amu_to_col]
    _group_sums[grp] = df[cols].clip(lower=0).sum(axis=1).values

_group_names = list(_group_sums.keys())
_group_matrix = np.column_stack([_group_sums[g] for g in _group_names])
df['PeakChem'] = np.array(_group_names)[_group_matrix.argmax(axis=1)]

spec_cols = _spec_cols
spec_values = [float(c) for c in spec_cols]

_moon_cols_all = {
    'Saturn': 'R_sat',
    'Enceladus': 'Enceladus_Dist',
    'Mimas': 'Mimas_Dist',
    'Tethys': 'Tethys_Dist',
    'Dione': 'Dione_Dist',
    'Rhea': 'Rhea_Dist',
    'Titan': 'Titan_Dist',
    'Enc. Azimuth': 'PhiE_deg',
    'Enc. Phase': 'PhiEncObs_deg',
    'Inclination': 'Inclination',
    'Impact Velocity': 'Vrel_dust',
}
moon_cols = {k: v for k, v in _moon_cols_all.items() if v in df.columns}

for col in moon_cols.values():
    df[col] = df[col].replace(-999, np.nan)

years = sorted(df['Year'].unique())
df = df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
FONT = "'Space Grotesk', 'Inter', system-ui, sans-serif"

C = {
    'bg': '#000000',
    'panel': '#050508',
    'card': '#0a0a0f',
    'border': 'rgba(255,255,255,0.08)',
    'grid': 'rgba(255,255,255,0.06)',
    'text': '#f0f0f4',
    'text2': '#8a8a98',
    'text3': '#55555f',
    'ice': '#0277BD',
    'salt': '#FFA726',
    'organic': '#66BB6A',
    'other': '#AB47BC',
    'saturn': '#d4a84b',
    'accent': '#7c3aed',
}

TYPE_COLORS = {
    'Ice': C['ice'], 'Other': C['other'], 'Salt': C['salt'], 'Organic': C['organic'],
}
PEAK_CHEM_COLORS = {
    'Water Ice':  '#0277BD',
    'Silicate':   '#FFD54F',
    'Iron':       '#A1887F',
    'Na Salt':    '#FF8A65',
    'K Salt':     '#CE93D8',
    'Organic':    '#81C784',
    'Phosphorus': '#F06292',
    'NH3/OH':     '#7986CB',
}
PEAK_CHEM_STYLES = {
    'Water Ice':  {'size_2d': 1.5, 'size_3d': 1.5, 'size_radial': 2.5, 'opacity': 1.0},
    'Silicate':   {'size_2d': 3.5, 'size_3d': 3.5, 'size_radial': 3.5, 'opacity': 1.0},
    'Iron':       {'size_2d': 3.5, 'size_3d': 3.5, 'size_radial': 3.5, 'opacity': 1.0},
    'Na Salt':    {'size_2d': 3,   'size_3d': 3,   'size_radial': 3,   'opacity': 1.0},
    'K Salt':     {'size_2d': 3,   'size_3d': 3,   'size_radial': 3,   'opacity': 1.0},
    'Organic':    {'size_2d': 4,   'size_3d': 4,   'size_radial': 3.5, 'opacity': 1.0},
    'Phosphorus': {'size_2d': 4,   'size_3d': 4,   'size_radial': 3.5, 'opacity': 1.0},
    'NH3/OH':     {'size_2d': 3,   'size_3d': 3,   'size_radial': 3,   'opacity': 1.0},
}


def get_color_config():
    """Return (type_col, colors, styles, type_order) for Peak Chemistry coloring."""
    return ('PeakChem', PEAK_CHEM_COLORS, PEAK_CHEM_STYLES,
            ['Water Ice', 'Silicate', 'Iron', 'Na Salt', 'K Salt', 'Organic', 'Phosphorus', 'NH3/OH'])


PEAK_LABELS = {
    1: 'H\u207a', 6: 'C\u207a', 7: 'N\u207a', 12: 'C\u207a', 16: 'O\u207a',
    17: 'OH\u207a/NH\u2083', 18: 'H\u2082O\u207a', 19: 'H\u2083O\u207a', 20: 'H\u2082O\u00b7H\u2082\u207a',
    23: 'Na\u207a', 24: 'Mg\u207a', 26: 'CN\u207a', 27: 'HCN\u207a',
    28: 'Si\u207a/CO\u207a', 29: 'CHO\u207a', 30: 'CH\u2082O\u207a', 31: 'P\u207a',
    36: '(H\u2082O)\u2082\u207a', 37: '(H\u2082O)\u2082H\u207a', 39: 'K\u207a', 40: 'Ca\u207a',
    44: 'SiO\u207a/CO\u2082\u207a', 48: 'Ti\u207a/SO\u207a',
    54: '(H\u2082O)\u2083\u207a', 55: '(H\u2082O)\u2083H\u207a', 56: 'Fe\u207a/CaO\u207a',
    63: 'Na(NaOH)\u207a', 73: 'SiO\u2083\u207b', 78: 'C\u2086H\u2086\u207a', 91: 'SiO\u2084H\u207b',
    103: 'Rh\u207a (target)',
}

# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------

def build_spatial_map(sample_df, highlight_idx=None):
    fig = go.Figure()
    type_col, colors, styles, type_order = get_color_config()
    local_counts = sample_df[type_col].value_counts()

    # E-ring shading
    theta_ering = np.linspace(0, 2*np.pi, 100)
    x_ering = np.concatenate([8*np.cos(theta_ering), 3*np.cos(theta_ering[::-1])])
    y_ering = np.concatenate([8*np.sin(theta_ering), 3*np.sin(theta_ering[::-1])])
    fig.add_trace(go.Scatter(
        x=x_ering, y=y_ering, fill='toself', fillcolor='rgba(74,158,255,0.03)',
        line=dict(width=0), hoverinfo='skip', showlegend=False,
    ))
    fig.add_annotation(x=5.5, y=-8.5, text='E-ring', showarrow=False,
                      font=dict(size=8, color='rgba(74,158,255,0.3)'), opacity=0.6)

    # Distance rings
    for r in [4, 6, 8, 10, 12, 14]:
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=r*np.cos(theta), y=r*np.sin(theta),
            mode='lines', line=dict(color=C['border'], width=1),
            hoverinfo='skip', showlegend=False,
        ))
        fig.add_annotation(x=0, y=-r-0.3, text=f'{r} R\u209b', showarrow=False,
                          font=dict(size=8, color=C['text3']), opacity=0.6)

    # Saturn body + rings
    theta = np.linspace(0, 2*np.pi, 80)
    fig.add_trace(go.Scatter(
        x=1.0*np.cos(theta), y=1.0*np.sin(theta),
        fill='toself', fillcolor=C['saturn'],
        line=dict(color='#b8944a', width=1.5), hoverinfo='skip', showlegend=False,
    ))
    for r_inner, r_outer, opacity in [(1.11,1.23,0.12),(1.23,1.52,0.20),(1.52,1.95,0.35),(2.02,2.27,0.28)]:
        theta_ring = np.linspace(0, 2*np.pi, 100)
        x_ring = np.concatenate([r_outer*np.cos(theta_ring), r_inner*np.cos(theta_ring[::-1])])
        y_ring = np.concatenate([r_outer*np.sin(theta_ring), r_inner*np.sin(theta_ring[::-1])])
        fig.add_trace(go.Scatter(
            x=x_ring, y=y_ring, fill='toself', fillcolor=f'rgba(180,160,120,{opacity})',
            line=dict(width=0), hoverinfo='skip', showlegend=False,
        ))

    # Moon labels
    fig.add_annotation(x=3.08, y=0.5, text='Mimas', showarrow=False, font=dict(size=8, color=C['text3']))
    fig.add_annotation(x=3.95, y=0.5, text='Enceladus', showarrow=False, font=dict(size=9, color=C['ice']))
    for name, r in [('Tethys', 4.89), ('Dione', 6.26), ('Rhea', 8.74)]:
        fig.add_annotation(x=r, y=0.5, text=name, showarrow=False, font=dict(size=9, color=C['text3']))

    # Grain scatter
    for t in type_order:
        s = sample_df[sample_df[type_col] == t]
        if len(s) == 0:
            continue
        style = styles[t]
        fig.add_trace(go.Scattergl(
            x=s['x'].values, y=s['y'].values, mode='markers',
            name=f'{t} ({local_counts.get(t, 0):,})',
            marker=dict(size=style['size_2d'], color=colors[t], opacity=style['opacity']),
            customdata=np.column_stack([s['_idx'].values, s['M3 Category'].values]),
            hovertemplate=f'<b>{t}</b> (%{{customdata[1]}})<br>R = %{{text:.1f}} R\u209b<extra></extra>',
            text=s['R_sat'].values, legendgroup=t,
        ))

    if highlight_idx is not None and highlight_idx in df.index:
        grain = df.loc[highlight_idx]
        fig.add_trace(go.Scattergl(
            x=[grain['x']], y=[grain['y']], mode='markers',
            marker=dict(size=16, color='rgba(0,0,0,0)', line=dict(width=2, color='white')),
            showlegend=False, hoverinfo='skip',
        ))

    fig.update_layout(
        plot_bgcolor=C['card'], paper_bgcolor=C['card'],
        font=dict(family=FONT, color=C['text']),
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text='<b>Spatial Distribution</b> <span style="font-size:9px;color:#55555f">\u2014 Saturn-centered equatorial plane (R\u209b)</span>',
                   font=dict(size=13), x=0.5, y=0.98),
        legend=dict(
            title=dict(text='<b>Peak Chemistry</b>', font=dict(size=10)),
            x=0.01, y=0.98, xanchor='left', yanchor='top',
            bgcolor='rgba(10,10,15,0.95)', bordercolor=C['border'], borderwidth=1,
            font=dict(size=9), itemclick='toggle', itemdoubleclick='toggleothers',
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False,
                   range=[-22,22], scaleanchor='y', constrain='domain'),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showline=False, range=[-22,22]),
        dragmode='pan',
        uirevision='spatial2d',
    )
    return fig


def build_spatial_map_3d(sample_df, highlight_idx=None):
    fig = go.Figure()
    type_col, colors, styles, type_order = get_color_config()
    local_counts = sample_df[type_col].value_counts()

    n_pts = 60
    theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    flat_light = dict(ambient=1, diffuse=0, specular=0, roughness=1, fresnel=0)

    # Saturn body mesh
    x_sat = np.concatenate([[0], np.cos(theta)]).tolist()
    y_sat = np.concatenate([[0], np.sin(theta)]).tolist()
    z_sat = [0] * (n_pts + 1)
    i_tri = [0] * n_pts
    j_tri = list(range(1, n_pts + 1))
    k_tri = list(range(2, n_pts + 1)) + [1]
    fig.add_trace(go.Mesh3d(
        x=x_sat, y=y_sat, z=z_sat, i=i_tri, j=j_tri, k=k_tri,
        color=C['saturn'], opacity=1.0, lighting=flat_light, hoverinfo='skip', showlegend=False,
    ))

    # Ring system
    ring_specs = [
        (1.11, 1.23, 'rgb(30,28,28)'), (1.23, 1.52, 'rgb(44,40,36)'),
        (1.52, 1.95, 'rgb(70,63,52)'), (2.02, 2.27, 'rgb(58,52,44)'),
    ]
    for r_inner, r_outer, ring_color in ring_specs:
        x_ring = np.concatenate([r_outer*np.cos(theta), r_inner*np.cos(theta)]).tolist()
        y_ring = np.concatenate([r_outer*np.sin(theta), r_inner*np.sin(theta)]).tolist()
        z_ring = [0] * (2 * n_pts)
        i_r, j_r, k_r = [], [], []
        for idx in range(n_pts):
            ni = (idx + 1) % n_pts
            i_r.extend([idx, idx])
            j_r.extend([ni, n_pts + ni])
            k_r.extend([n_pts + ni, n_pts + idx])
        fig.add_trace(go.Mesh3d(
            x=x_ring, y=y_ring, z=z_ring, i=i_r, j=j_r, k=k_r,
            color=ring_color, opacity=1.0, lighting=flat_light, hoverinfo='skip', showlegend=False,
        ))

    for r in [4, 8, 14]:
        fig.add_trace(go.Scatter3d(
            x=(r*np.cos(theta)).tolist(), y=(r*np.sin(theta)).tolist(), z=[0]*n_pts,
            mode='lines', line=dict(color=C['border'], width=1), hoverinfo='skip', showlegend=False,
        ))

    for t in type_order:
        s = sample_df[sample_df[type_col] == t]
        if len(s) == 0:
            continue
        style = styles[t]
        fig.add_trace(go.Scatter3d(
            x=s['x'].values.tolist(), y=s['y'].values.tolist(), z=s['z'].values.tolist(),
            mode='markers', name=f'{t} ({local_counts.get(t, 0):,})',
            marker=dict(size=style['size_3d'], color=colors[t], opacity=style['opacity'], line=dict(width=0)),
            customdata=np.column_stack([s['_idx'].values, s['M3 Category'].values, s[type_col].values]),
            hovertemplate='<b>%{customdata[2]}</b> (%{customdata[1]})<br>R = %{text:.1f} R\u209b<extra></extra>',
            text=s['R_sat'].values.tolist(), legendgroup=t,
        ))

    if highlight_idx is not None and highlight_idx in df.index:
        grain = df.loc[highlight_idx]
        fig.add_trace(go.Scatter3d(
            x=[grain['x']], y=[grain['y']], z=[grain['z']], mode='markers',
            marker=dict(size=8, color='rgba(0,0,0,0)', line=dict(width=2, color='white')),
            showlegend=False, hoverinfo='skip',
        ))

    fig.update_layout(
        plot_bgcolor=C['card'], paper_bgcolor=C['card'],
        font=dict(family=FONT, color=C['text']),
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text='<b>Spatial Distribution (3D)</b>', font=dict(size=13), x=0.5, y=0.98),
        legend=dict(
            title=dict(text='<b>Peak Chemistry</b>', font=dict(size=10)),
            x=0.01, y=0.98, xanchor='left', yanchor='top',
            bgcolor='rgba(10,10,15,0.95)', bordercolor=C['border'], borderwidth=1,
            font=dict(size=9), itemclick='toggle', itemdoubleclick='toggleothers',
        ),
        scene=dict(
            xaxis=dict(visible=False, range=[-160,160]),
            yaxis=dict(visible=False, range=[-80,80]),
            zaxis=dict(visible=False, range=[-1,1]),
            bgcolor=C['card'], aspectmode='manual', aspectratio=dict(x=2, y=1, z=0.01),
            camera=dict(eye=dict(x=0, y=0, z=0.25), up=dict(x=1, y=0, z=0)),
            dragmode='orbit',
        ),
        uirevision='spatial3d',
    )
    return fig


def build_spectrum(grain_idx=None, x_range=None):
    fig = go.Figure()

    if grain_idx is not None and grain_idx in df.index:
        grain = df.loc[grain_idx]
        spectrum = grain[spec_cols].values.astype(float)
        type_col, colors, _, _ = get_color_config()
        display_label = grain[type_col]
        color = colors.get(display_label, C['text2'])

        fig.add_trace(go.Scatter(
            x=spec_values, y=spectrum, mode='lines', line=dict(color=color, width=1.5),
            fill='tozeroy',
            fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)',
            hovertemplate='%{x:.1f} AMU: %{y:.2f}<extra></extra>',
        ))

        # Top peaks (deduplicated within 1.5 AMU)
        peak_threshold = max(np.nanmax(spectrum) * 0.05, 0.01)
        sorted_peaks = np.argsort(spectrum)[::-1]
        labeled_amus = []
        for pk in sorted_peaks:
            if len(labeled_amus) >= 6:
                break
            if spectrum[pk] <= peak_threshold:
                break
            amu_f = spec_values[pk]
            if any(abs(amu_f - prev) < 1.5 for prev in labeled_amus):
                continue
            labeled_amus.append(amu_f)
            amu_int = round(amu_f)
            label = PEAK_LABELS.get(amu_int, f'{amu_int}')
            fig.add_annotation(
                x=amu_f, y=spectrum[pk], text=label, showarrow=True,
                arrowhead=0, arrowcolor=C['text3'], font=dict(size=9, color=C['text']), ay=-25,
            )

        title_text = f'<b>Mass Spectrum</b> \u2014 {display_label} grain (R = {grain["R_sat"]:.1f} R\u209b, {int(grain["Year"])})'
    else:
        title_text = '<b>Mass Spectrum</b>'
        fig.add_annotation(
            x=0.5, y=0.5, xref='paper', yref='paper', text='Click a grain on any plot to view its spectrum',
            font=dict(size=12, color=C['text3'], family=FONT), showarrow=False,
        )

    fig.update_layout(
        plot_bgcolor=C['card'], paper_bgcolor=C['card'],
        font=dict(family=FONT, color=C['text']),
        margin=dict(l=40, r=10, t=28, b=32),
        title=dict(text=title_text, font=dict(size=11), x=0.5, y=0.97),
        xaxis=dict(
            title=dict(text='Mass (AMU)', font=dict(size=9, color=C['text3'])),
            tickfont=dict(size=8, color=C['text2']),
            showgrid=False, showline=True, linecolor=C['border'],
            range=x_range if x_range else [0, 200],
        ),
        yaxis=dict(
            title=dict(text='Intensity', font=dict(size=9, color=C['text3'])),
            tickfont=dict(size=8, color=C['text2']),
            showgrid=True, gridcolor=C['grid'], showline=True, linecolor=C['border'],
        ),
        showlegend=False,
    )
    return fig


bins = [3, 5, 7, 9, 11, 13, 15]
bin_labels = ['3-5', '5-7', '7-9', '9-11', '11-13', '13-15']

def build_bars(source_df, dist_col='R_sat', dist_label='Saturn'):
    fig = go.Figure()

    if dist_col == 'R_sat':
        b, bl = bins, bin_labels
    elif dist_col == 'PhiE_deg':
        b = [-180, -120, -60, 0, 60, 120, 180]
        bl = ['\u2212180\u00b0\u2013\u2212120\u00b0', '\u2212120\u00b0\u2013\u221260\u00b0', '\u221260\u00b0\u20130\u00b0', '0\u00b0\u201360\u00b0', '60\u00b0\u2013120\u00b0', '120\u00b0\u2013180\u00b0']
    else:
        valid = source_df[dist_col].dropna()
        if len(valid) < 50:
            b, bl = bins, bin_labels
        else:
            edges = [0] + [valid.quantile(q) for q in [0.15, 0.3, 0.5, 0.7, 0.85, 1.0]]
            b = sorted(set(round(e, 1) for e in edges))
            if len(b) < 3:
                b = list(np.linspace(valid.min(), valid.max(), 7))
            bl = [f'{b[i]:.0f}-{b[i+1]:.0f}' for i in range(len(b)-1)]

    x_bar = list(range(len(bl)))

    type_col, clrs, _, order = get_color_config()
    ice_key = 'Water Ice'
    show_cats = [(t, clrs[t]) for t in order]

    n_cats = len(show_cats)
    bar_width = min(0.8 / max(n_cats, 1), 0.35)

    bin_counts = []
    cat_ys = {cat: [] for cat, _ in show_cats}

    if len(source_df) > 0:
        working = source_df.dropna(subset=[dist_col]).copy()
        working['_bin'] = pd.cut(working[dist_col], bins=b, include_lowest=True)
        bin_sizes = working.groupby('_bin', observed=False).size()
        props = working.groupby('_bin', observed=False)[type_col].value_counts(normalize=True).unstack(fill_value=0) * 100

        for interval in sorted(bin_sizes.index):
            bin_counts.append(int(bin_sizes.get(interval, 0)))
            for cat, _ in show_cats:
                cat_ys[cat].append(float(props.loc[interval, cat]) if cat in props.columns else 0)
    else:
        for cat, _ in show_cats:
            cat_ys[cat] = [0] * len(bl)
        bin_counts = [0] * len(bl)

    # Point-biserial correlations
    corr_info = {}
    if len(source_df) > 10:
        valid_corr = source_df.dropna(subset=[dist_col])
        if len(valid_corr) > 10:
            for cat, _ in show_cats:
                try:
                    r_val, p_val = stats.pointbiserialr(valid_corr[type_col] == cat, valid_corr[dist_col])
                    if np.isfinite(r_val):
                        p_str = 'p<.001' if p_val < 0.001 else f'p={p_val:.3f}'
                        corr_info[cat] = f'r={r_val:.2f} ({p_str})'
                except Exception:
                    pass

    for i, (cat, color) in enumerate(show_cats):
        offset = (i - (n_cats - 1) / 2) * bar_width
        corr_str = f'  ({corr_info[cat]})' if cat in corr_info else ''
        fig.add_trace(go.Bar(
            x=x_bar, y=cat_ys[cat], marker=dict(color=color),
            width=bar_width, offset=offset, name=cat,
            hovertemplate=f'{cat}: %{{y:.1f}}%{corr_str}<extra></extra>',
            showlegend=True,
            visible='legendonly' if cat == ice_key else True,
        ))

    for i, n in enumerate(bin_counts):
        if n > 0:
            fig.add_annotation(
                x=i, y=-0.008, yref='paper', text=f'n={n:,}', showarrow=False,
                font=dict(size=8, color=C['text3'], family='JetBrains Mono, monospace'),
            )

    # Moon orbit markers
    if dist_col == 'R_sat':
        _moon_orbits = [('Enc', 3.95), ('Tet', 4.89), ('Dio', 6.26), ('Rhe', 8.74)]
        for name, r_s in _moon_orbits:
            # Rs -> bin-index x: x_pos = (Rs - bin_start) / bin_width - 0.5
            x_pos = (r_s - b[0]) / (b[1] - b[0]) - 0.5
            if -0.5 <= x_pos <= len(bl) - 0.5:
                fig.add_annotation(
                    x=x_pos, y=-0.06, yref='paper', text=f'\u25b2 {name}', showarrow=False,
                    font=dict(size=7, color=C['saturn'], family=FONT), textangle=0,
                )

    _units = {'R_sat': 'R\u209b', 'PhiE_deg': '\u00b0', 'PhiEncObs_deg': '\u00b0', 'Inclination': '\u00b0', 'Vrel_dust': 'km/s'}
    unit = _units.get(dist_col, 'R\u209b from moon')
    _labels = {'PhiE_deg': 'Azimuth from Enceladus', 'PhiEncObs_deg': 'Phase Angle (Enceladus)',
               'Inclination': 'Inclination', 'Vrel_dust': 'Impact Velocity'}
    x_label = _labels.get(dist_col, f'Distance ({unit})')
    fig.update_layout(
        plot_bgcolor=C['card'], paper_bgcolor=C['card'],
        font=dict(family=FONT, color=C['text']),
        margin=dict(l=40, r=70, t=28, b=38),
        title=dict(text=f'<b>Composition vs {dist_label}</b>', font=dict(size=11), x=0.5, y=0.98),
        barmode='group',
        bargap=0.3,
        legend=dict(
            orientation='v', x=1.02, xanchor='left', y=1.0, yanchor='top',
            font=dict(size=8, color=C['text2']),
            bgcolor='rgba(10,10,15,0.85)', borderwidth=0, bordercolor=C['border'],
            itemclick='toggle', itemdoubleclick='toggleothers',
            traceorder='normal', tracegroupgap=1,
        ),
        xaxis=dict(
            tickmode='array', tickvals=x_bar, ticktext=bl,
            tickfont=dict(size=9, color=C['text2']),
            title=dict(text=x_label, font=dict(size=9, color=C['text3'])),
            showgrid=False, showline=True, linecolor=C['border'],
        ),
        yaxis=dict(
            tickfont=dict(size=9, color=C['text2']),
            title=dict(text='Proportion (%)', font=dict(size=9, color=C['text3'])),
            showgrid=True, gridcolor=C['grid'], showline=True, linecolor=C['border'],
            rangemode='tozero',
        ),
    )
    return fig


def build_latent(sample_df, highlight_idx=None):
    fig = go.Figure()

    type_col, colors, styles, type_order = get_color_config()
    ice_key = 'Water Ice'
    _MAX_LATENT = 4000
    plot_cat = sample_df.sample(n=min(_MAX_LATENT, len(sample_df)), random_state=42) \
               if len(sample_df) > _MAX_LATENT else sample_df
    for t in type_order:
        sub = plot_cat[plot_cat[type_col] == t]
        if len(sub) == 0:
            continue
        is_ice = (t == ice_key)
        fig.add_trace(go.Scattergl(
            x=sub['Latent1'].values, y=sub['Latent2'].values, mode='markers',
            marker=dict(size=styles[t]['size_2d'],
                        color=colors[t],
                        opacity=0.12 if is_ice else 0.88,
                        line=dict(width=0)),
            customdata=sub['_idx'].values, name=t,
            hovertemplate=f'<b>{t}</b><br>L1=%{{x:.1f}}, L2=%{{y:.1f}}<extra></extra>',
            showlegend=False,
        ))
    subtitle = 'DustMAP autoencoder embedding (spectral similarity)'

    if highlight_idx is not None and highlight_idx in df.index:
        grain = df.loc[highlight_idx]
        if 'Latent1' in grain.index and pd.notna(grain['Latent1']):
            fig.add_trace(go.Scattergl(
                x=[grain['Latent1']], y=[grain['Latent2']], mode='markers',
                marker=dict(size=16, color='rgba(0,0,0,0)', line=dict(width=2, color='white')),
                showlegend=False, hoverinfo='skip',
            ))

    fig.update_layout(
        plot_bgcolor=C['card'], paper_bgcolor=C['card'],
        font=dict(family=FONT, color=C['text']),
        margin=dict(l=35, r=10, t=20, b=28),
        uirevision='latent',
        title=dict(text=f'<b>Latent Space</b> <span style="font-size:9px;color:#55555f">\u2014 {subtitle}</span>',
                   font=dict(size=11), x=0.5, y=0.97),
        xaxis=dict(title=dict(text='Latent 1', font=dict(size=9, color=C['text3'])),
                   tickfont=dict(size=8, color=C['text2']),
                   showgrid=True, gridcolor=C['grid'], showline=True, linecolor=C['border'], zeroline=False),
        yaxis=dict(title=dict(text='Latent 2', font=dict(size=9, color=C['text3'])),
                   tickfont=dict(size=8, color=C['text2']),
                   showgrid=True, gridcolor=C['grid'], showline=True, linecolor=C['border'], zeroline=False),
        showlegend=False, dragmode='pan',
    )
    return fig


def build_radial_strip(sample_df, highlight_idx=None, dist_col='R_sat', dist_label='Saturn'):
    fig = go.Figure()
    type_col, colors, styles, type_order = get_color_config()
    valid = sample_df.dropna(subset=[dist_col])

    # Deterministic jitter so highlight position is recoverable
    jitter_all = {}
    if len(valid) > 0:
        rng = np.random.RandomState(42)
        jitter_vals = rng.uniform(-0.35, 0.35, size=len(valid))
        jitter_all = dict(zip(valid.index, jitter_vals))

    ice_key = 'Water Ice'
    _hu = {'Vrel_dust': 'km/s', 'Inclination': '\u00b0', 'PhiE_deg': '\u00b0', 'PhiEncObs_deg': '\u00b0'}
    h_unit = _hu.get(dist_col, 'R\u209b')

    _MAX_STRIP_ICE = 800
    _MAX_STRIP_OTHER = 2000
    ice = valid[valid[type_col] == ice_key]
    if len(ice) > _MAX_STRIP_ICE:
        ice = ice.sample(n=_MAX_STRIP_ICE, random_state=42)
    if len(ice) > 0:
        fig.add_trace(go.Scattergl(
            x=ice[dist_col].values, y=[jitter_all[i] for i in ice.index], mode='markers',
            marker=dict(size=styles[ice_key]['size_radial'], color=colors[ice_key],
                        opacity=0.35, line=dict(width=0)),
            customdata=ice['_idx'].values,
            hovertemplate=f'<b>{ice_key}</b><br>%{{x:.1f}} {h_unit}<extra></extra>',
            showlegend=False,
        ))

    non_ice = valid[valid[type_col] != ice_key]
    if len(non_ice) > _MAX_STRIP_OTHER:
        non_ice = non_ice.sample(n=_MAX_STRIP_OTHER, random_state=42)
    if len(non_ice) > 0:
        ni_types = non_ice[type_col]
        _sz = {t: styles[t]['size_radial'] for t in type_order}
        fig.add_trace(go.Scattergl(
            x=non_ice[dist_col].values, y=[jitter_all[i] for i in non_ice.index], mode='markers',
            marker=dict(size=ni_types.map(_sz).astype(float).values,
                        color=ni_types.map(colors).values, opacity=1.0, line=dict(width=0)),
            customdata=non_ice['_idx'].values,
            text=ni_types.values,
            hovertemplate=f'<b>%{{text}}</b><br>%{{x:.1f}} {h_unit}<extra></extra>',
            showlegend=False,
        ))

    if dist_col == 'R_sat':
        for name, r in [('Mimas', 3.08), ('Enceladus', 3.95), ('Tethys', 4.89), ('Dione', 6.26), ('Rhea', 8.74)]:
            fig.add_vline(x=r, line=dict(color=C['text3'], width=1, dash='dot'), opacity=0.4)
            clr = C['ice'] if name == 'Enceladus' else C['text3']
            fig.add_annotation(x=r, y=0.48, text=name, showarrow=False,
                              font=dict(size=8, color=clr, family=FONT), yref='paper')

    if highlight_idx is not None and highlight_idx in df.index:
        dist_val = df.loc[highlight_idx].get(dist_col, np.nan)
        if pd.notna(dist_val):
            hy = jitter_all.get(highlight_idx, 0)
            fig.add_vline(x=dist_val, line=dict(color='white', width=1.5), opacity=0.4)
            fig.add_trace(go.Scattergl(
                x=[dist_val], y=[hy], mode='markers',
                marker=dict(size=10, color='rgba(0,0,0,0)', line=dict(width=2, color='white')),
                showlegend=False, hoverinfo='skip',
            ))

    _angle_cols = {'PhiE_deg', 'PhiEncObs_deg', 'Inclination'}
    if dist_col in _angle_cols:
        if dist_col == 'Inclination':
            x_hard_min, x_hard_max = -90.0, 90.0
        else:
            x_hard_min, x_hard_max = -180.0, 180.0
        if len(valid) > 0:
            x_min, x_max = valid[dist_col].quantile(0.01), valid[dist_col].quantile(0.99)
            pad = (x_max - x_min) * 0.05
            x_range = [max(x_hard_min, x_min - pad), min(x_hard_max, x_max + pad)]
        else:
            x_range = [x_hard_min, x_hard_max]
    elif dist_col == 'Vrel_dust':
        x_hard_min, x_hard_max = 0.0, 500.0
        if len(valid) > 0:
            x_range = [0, min(500, valid[dist_col].quantile(0.99) * 1.05)]
        else:
            x_range = [0, 200]
    else:
        x_hard_min, x_hard_max = 0.0, 40.0
        if dist_col == 'R_sat':
            x_range = [2.5, 16]
        elif len(valid) > 0:
            x_min, x_max = valid[dist_col].quantile(0.01), valid[dist_col].quantile(0.99)
            pad = (x_max - x_min) * 0.05
            x_range = [max(x_hard_min, x_min - pad), min(x_hard_max, x_max + pad)]
        else:
            x_range = [0, 20]

    _strip_labels = {
        'PhiE_deg': ('Azimuth from Enceladus (\u00b0)', '<b>Azimuthal Distribution \u2014 Enceladus</b>'),
        'PhiEncObs_deg': ('Phase Angle from Enceladus (\u00b0)', '<b>Phase Distribution \u2014 Enceladus</b>'),
        'Inclination': ('Inclination (\u00b0)', '<b>Distribution by Inclination</b>'),
        'Vrel_dust': ('Impact Velocity (km/s)', '<b>Distribution by Impact Velocity</b>'),
    }
    if dist_col in _strip_labels:
        x_title, strip_title = _strip_labels[dist_col]
    else:
        unit = 'R\u209b' if dist_col == 'R_sat' else 'R\u209b from moon'
        x_title = f'Distance ({unit})'
        strip_title = f'<b>Radial Distribution \u2014 from {dist_label}</b>'
    fig.update_layout(
        plot_bgcolor=C['card'], paper_bgcolor=C['card'],
        font=dict(family=FONT, color=C['text']),
        margin=dict(l=50, r=15, t=18, b=25),
        title=dict(text=strip_title, font=dict(size=11),
                   x=0.01, y=0.95, xanchor='left'),
        xaxis=dict(title=dict(text=x_title, font=dict(size=9, color=C['text3'])),
                   tickfont=dict(size=9, color=C['text2']),
                   showgrid=False, showline=True, linecolor=C['border'],
                   range=x_range, minallowed=x_hard_min, maxallowed=x_hard_max, fixedrange=False),
        yaxis=dict(visible=False, range=[-0.5, 0.5], fixedrange=True),
        showlegend=False, dragmode='zoom',
        uirevision='radial',
    )
    fig.add_annotation(x=0.99, y=0.95, xref='paper', yref='paper', xanchor='right', yanchor='top',
                      text='drag to zoom \u00b7 double-click to reset',
                      font=dict(size=8, color=C['text3']), showarrow=False, opacity=0.5)
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
_n_total = len(df)

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = 'Saturn E-Ring Composition'
app._favicon = 'favicon.svg'

_filter_opts = [{'label': 'All Grains', 'value': 'All'}]
for t in ['Water Ice', 'Silicate', 'Iron', 'Na Salt', 'K Salt', 'Organic', 'Phosphorus', 'NH3/OH']:
    _filter_opts.append({'label': t, 'value': f'PC:{t}'})

_toggle_btn = {'padding': '2px 8px', 'fontSize': '10px', 'fontFamily': FONT,
               'fontWeight': '600', 'cursor': 'pointer', 'lineHeight': '1.4'}
_active_btn = {**_toggle_btn, 'backgroundColor': C['accent'], 'color': '#fff', 'border': f'1px solid {C["accent"]}'}
_inactive_btn = {**_toggle_btn, 'backgroundColor': 'transparent', 'color': C['text2'], 'border': f'1px solid {C["border"]}'}

app.layout = dbc.Container([

    # Header
    html.Div([
        html.Div([
            html.Div([
                html.H4('Saturn E-Ring Composition',
                         style={'fontWeight': '700', 'color': C['text'], 'margin': '0',
                                'fontFamily': FONT, 'letterSpacing': '-0.5px'}),
                html.Span('Cassini CDA',
                           style={'fontSize': '12px', 'color': C['text3'], 'marginLeft': '10px', 'fontFamily': FONT}),
            ], style={'display': 'flex', 'alignItems': 'baseline', 'gap': '4px'}),
        ]),
        html.Span(f'{_n_total:,} grains  \u00b7  {years[0]}\u2013{years[-1]}',
                   style={'fontSize': '11px', 'color': C['text3'], 'fontFamily': 'JetBrains Mono, monospace'}),
    ], className='dashboard-header',
       style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
              'padding': '8px 12px', 'flexShrink': '0'}),

    # Controls
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Label('Year Range', style={'fontSize': '11px', 'color': C['text2'],
                                                 'fontFamily': FONT, 'fontWeight': '500'}),
                dcc.RangeSlider(
                    id='year-slider', min=int(years[0]), max=int(years[-1]), step=1,
                    value=[int(years[0]), int(years[-1])],
                    marks={int(y): {'label': str(y), 'style': {'color': '#e0e0e8'}} for y in years},
                    allowCross=False, className='mt-1',
                ),
            ], width=6),
            dbc.Col([
                html.Label('Plot By', style={'fontSize': '11px', 'color': C['text2'],
                                              'fontFamily': FONT, 'fontWeight': '500'}),
                dbc.Select(id='moon-dropdown',
                           options=[{'label': k, 'value': k} for k in moon_cols.keys()],
                           value='Saturn', size='sm', style={'fontSize': '12px', 'fontFamily': FONT}),
            ], width=3),
            dbc.Col([
                html.Label('Filter', style={'fontSize': '11px', 'color': C['text2'],
                                             'fontFamily': FONT, 'fontWeight': '500'}),
                dbc.Select(id='chem-filter', options=_filter_opts, value='All', size='sm',
                           style={'fontSize': '12px', 'fontFamily': FONT}),
            ], width=3),
        ]),
    ], className='controls-row', style={'flexShrink': '0'}),

    # Filters
    html.Div([
        html.Div([
            html.Span('Saturn Distance (R\u209b)', style={'fontSize': '10px', 'color': C['text2'], 'fontFamily': FONT, 'fontWeight': '500'}),
            dbc.Input(id='range-min', type='number', placeholder='min', size='sm', debounce=True,
                      style={'width': '65px', 'fontSize': '11px', 'fontFamily': 'JetBrains Mono, monospace',
                             'padding': '2px 6px', 'height': '26px'}),
            html.Span('\u2013', style={'color': C['text3'], 'fontSize': '13px', 'margin': '0 2px'}),
            dbc.Input(id='range-max', type='number', placeholder='max', size='sm', debounce=True,
                      style={'width': '65px', 'fontSize': '11px', 'fontFamily': 'JetBrains Mono, monospace',
                             'padding': '2px 6px', 'height': '26px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '4px', 'marginLeft': '16px'}),
        html.Button('Download Filtered CSV', id='btn-download', n_clicks=0,
                     className='download-btn',
                     style={**_toggle_btn, 'backgroundColor': C['accent'], 'color': '#fff',
                            'border': f'1px solid {C["accent"]}', 'borderRadius': '4px',
                            'padding': '3px 12px', 'fontSize': '10px', 'marginLeft': '12px'}),
        html.Span(id='filter-count', style={'fontSize': '11px', 'color': C['text3'],
                                             'fontFamily': 'JetBrains Mono, monospace', 'marginLeft': '10px'}),
    ], style={'padding': '4px 12px 2px 12px', 'display': 'flex', 'alignItems': 'center', 'flexShrink': '0'}),

    # Main panels
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.Button('2D', id='btn-2d', n_clicks=0,
                                style={**_active_btn, 'borderRadius': '4px 0 0 4px'}),
                    html.Button('3D', id='btn-3d', n_clicks=0,
                                style={**_inactive_btn, 'borderRadius': '0 4px 4px 0'}),
                ], style={'position': 'absolute', 'top': '4px', 'right': '50px', 'zIndex': '10', 'display': 'flex'}),
                html.Div(id='cam-help', children=[
                    html.Div('?', id='cam-help-btn', className='cam-help-btn'),
                    html.Div([
                        html.Div('3D Navigation', style={'fontWeight': '600', 'marginBottom': '4px', 'fontSize': '11px'}),
                        html.Div('W A S D / Arrows \u2014 Pan'),
                        html.Div('+ / \u2212 \u2014 Zoom'),
                        html.Div('Scroll \u2014 Zoom to center'),
                        html.Div('Right-drag \u2014 Pan'),
                        html.Div('Double-click \u2014 Re-center'),
                        html.Div('R \u2014 Reset camera'),
                    ], className='cam-help-tooltip'),
                ], style={'display': 'none'}),
                dcc.Graph(id='spatial-map', figure=go.Figure(),
                         config={'scrollZoom': True, 'displaylogo': False, 'doubleClick': False,
                                 'modeBarButtonsToRemove': ['lasso2d','select2d','autoScale2d'],
                                 'toImageButtonOptions': {'format':'png','filename':'spatial_distribution','scale':3}},
                         style={'height': '100%'}),
            ], className='chart-card mb-1', style={'position': 'relative', 'height': '47vh', 'flexShrink': '0'}),
            html.Div([
                dcc.Graph(id='latent-space', figure=go.Figure(),
                         config={'scrollZoom': True, 'displaylogo': False,
                                 'modeBarButtonsToRemove': ['lasso2d','select2d','autoScale2d'],
                                 'toImageButtonOptions': {'format':'png','filename':'latent_space','scale':3}},
                         style={'height': '100%'}),
            ], className='chart-card', style={'flex': '1', 'minHeight': '0'}),
        ], width=5, className='pe-1', style={'display': 'flex', 'flexDirection': 'column'}),

        dbc.Col([
            html.Div(id='grain-readout', className='grain-readout',
                     style={'height': '42px', 'overflow': 'hidden', 'flexShrink': '0'},
                     children=html.Div('Click a grain on any plot to inspect',
                                       style={'color': C['text3'], 'fontSize': '11px', 'padding': '10px 12px',
                                              'fontFamily': FONT, 'textAlign': 'center'})),
            html.Div([
                dcc.Graph(id='comp-bars', figure=go.Figure(),
                         config={'displaylogo': False,
                                 'toImageButtonOptions': {'format':'svg','filename':'composition_vs_distance','scale':3}},
                         style={'height': '100%'}),
            ], className='chart-card mb-1', style={'height': '24vh', 'flexShrink': '0'}),
            html.Div([
                html.Div([
                    html.Button('Full', id='zoom-full', n_clicks=0,
                                style={**_active_btn, 'borderRadius': '4px 0 0 4px'}),
                    html.Button('Na/NH\u2083', id='zoom-na', n_clicks=0,
                                style={**_inactive_btn, 'borderRadius': '0'}),
                    html.Button('Fe/Si', id='zoom-fe', n_clicks=0,
                                style={**_inactive_btn, 'borderRadius': '0'}),
                    html.Button('Water', id='zoom-water', n_clicks=0,
                                style={**_inactive_btn, 'borderRadius': '0 4px 4px 0'}),
                ], style={'position': 'absolute', 'top': '4px', 'right': '10px', 'zIndex': '10', 'display': 'flex'}),
                dcc.Graph(id='spectrum', figure=build_spectrum(),
                         config={'displaylogo': False,
                                 'toImageButtonOptions': {'format':'svg','filename':'grain_spectrum','scale':3}},
                         style={'height': '100%'}),
            ], className='chart-card', style={'position': 'relative', 'flex': '1', 'minHeight': '0'}),
        ], width=7, className='ps-1', style={'display': 'flex', 'flexDirection': 'column'}),
    ], className='mt-1', style={'flex': '1', 'minHeight': '0'}),

    # Radial strip
    html.Div([
        dcc.Graph(id='radial-strip', figure=go.Figure(),
                 config={'scrollZoom': True, 'displaylogo': False,
                         'modeBarButtonsToRemove': ['lasso2d','select2d','autoScale2d'],
                         'toImageButtonOptions': {'format':'png','filename':'radial_distribution','scale':3}},
                 style={'height': '14vh'}),
        html.Div(id='radial-stats', style={'display': 'none'}),
    ], className='chart-card mt-1', style={'position': 'relative', 'flexShrink': '0'}),

    dcc.Store(id='selected-grain', data=None),
    dcc.Store(id='spatial-view-mode', data='2D'),
    dcc.Store(id='spectrum-zoom', data=[0, 200]),
    dcc.Download(id='download-csv'),

], fluid=True, style={'backgroundColor': C['bg'], 'height': '100vh', 'padding': '4px 8px',
                      'overflow': 'hidden', 'display': 'flex', 'flexDirection': 'column'})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _filter_data(year_range, chem_filter='All', r_min=None, r_max=None):
    filtered = df
    if year_range and isinstance(year_range, (list, tuple)) and len(year_range) == 2:
        filtered = filtered[(filtered['Year'] >= year_range[0]) & (filtered['Year'] <= year_range[1])]
    if chem_filter and chem_filter != 'All':
        if chem_filter.startswith('M3:'):
            filtered = filtered[filtered['Type'] == chem_filter[3:]]
        elif chem_filter.startswith('PC:'):
            filtered = filtered[filtered['PeakChem'] == chem_filter[3:]]
    if r_min is not None:
        filtered = filtered[filtered['R_sat'] >= r_min]
    if r_max is not None:
        filtered = filtered[filtered['R_sat'] <= r_max]
    return filtered


@callback(
    Output('spatial-view-mode', 'data'),
    Output('btn-2d', 'style'), Output('btn-3d', 'style'),
    Output('cam-help', 'style'),
    Input('btn-2d', 'n_clicks'), Input('btn-3d', 'n_clicks'),
    prevent_initial_call=True,
)
def toggle_view(n2d, n3d):
    if ctx.triggered_id == 'btn-3d':
        return '3D', {**_inactive_btn, 'borderRadius': '4px 0 0 4px'}, {**_active_btn, 'borderRadius': '0 4px 4px 0'}, {'display': 'block'}
    return '2D', {**_active_btn, 'borderRadius': '4px 0 0 4px'}, {**_inactive_btn, 'borderRadius': '0 4px 4px 0'}, {'display': 'none'}


@callback(
    Output('spectrum-zoom', 'data'),
    Output('zoom-full', 'style'), Output('zoom-na', 'style'),
    Output('zoom-fe', 'style'), Output('zoom-water', 'style'),
    Input('zoom-full', 'n_clicks'), Input('zoom-na', 'n_clicks'),
    Input('zoom-fe', 'n_clicks'), Input('zoom-water', 'n_clicks'),
    prevent_initial_call=True,
)
def on_spectrum_zoom(*_):
    zoom_map = {'zoom-full': [0, 200], 'zoom-na': [14, 42], 'zoom-fe': [45, 105], 'zoom-water': [14, 60]}
    btn_ids = ['zoom-full', 'zoom-na', 'zoom-fe', 'zoom-water']
    radii = ['4px 0 0 4px', '0', '0', '0 4px 4px 0']
    styles = []
    for bid, rad in zip(btn_ids, radii):
        base = _active_btn if bid == ctx.triggered_id else _inactive_btn
        styles.append({**base, 'borderRadius': rad})
    return zoom_map.get(ctx.triggered_id, [0, 200]), *styles


@callback(
    Output('selected-grain', 'data'),
    Input('spatial-map', 'clickData'), Input('latent-space', 'clickData'), Input('radial-strip', 'clickData'),
    prevent_initial_call=True,
)
def on_click(spatial_click, latent_click, radial_click):
    click_data = {'spatial-map': spatial_click, 'latent-space': latent_click, 'radial-strip': radial_click}.get(ctx.triggered_id)
    if click_data and click_data['points']:
        pt = click_data['points'][0]
        if 'customdata' in pt and pt['customdata'] is not None:
            idx = int(pt['customdata']) if not isinstance(pt['customdata'], list) else int(pt['customdata'][0])
            return idx
    return no_update


@callback(
    Output('spectrum', 'figure'), Output('grain-readout', 'children'),
    Input('selected-grain', 'data'), Input('spectrum-zoom', 'data'),
)
def update_spectrum(grain_idx, zoom_range):
    x_range = zoom_range if zoom_range else [0, 200]
    fig = build_spectrum(grain_idx, x_range=x_range)

    if grain_idx is not None and grain_idx in df.index:
        grain = df.loc[grain_idx]
        m3_color = TYPE_COLORS.get(grain['Type'], C['text2'])
        peak_chem = grain.get('PeakChem', '?')
        peak_color = PEAK_CHEM_COLORS.get(peak_chem, C['text2'])

        pri_label = peak_chem
        pri_color = peak_color
        sec_label = grain['Type']
        sec_color = m3_color

        spectrum = grain[spec_cols].values.astype(float)
        peak_threshold = max(np.nanmax(spectrum) * 0.05, 0.01)
        sorted_peaks = np.argsort(spectrum)[::-1]
        deduped = []
        for pk in sorted_peaks:
            if len(deduped) >= 3:
                break
            if spectrum[pk] <= peak_threshold:
                break
            amu_f = spec_values[pk]
            if any(abs(amu_f - prev) < 1.5 for prev in deduped):
                continue
            deduped.append(amu_f)
        peaks_str = ', '.join(
            f'{round(a)} ({PEAK_LABELS.get(round(a), "?")})' for a in deduped
        )

        mono = {'fontFamily': 'JetBrains Mono, monospace', 'fontSize': '11px'}
        sm = {**mono, 'color': C['text2'], 'fontSize': '10px'}
        lbl = {'fontFamily': FONT, 'fontSize': '9px', 'color': C['text3'], 'textTransform': 'uppercase',
               'letterSpacing': '0.5px'}

        vrel = grain.get('Vrel_dust', np.nan)
        vrel_str = f'{vrel:.1f}' if pd.notna(vrel) else '\u2014'
        incl = grain.get('Inclination', np.nan)
        incl_str = f'{incl:.1f}\u00b0' if pd.notna(incl) else '\u2014'
        snr = grain.get('SNR', np.nan)
        snr_str = f'{snr:.0f}' if pd.notna(snr) and np.isfinite(snr) else '\u221e' if pd.notna(snr) else '\u2014'
        snr_color = '#00d9b1' if pd.notna(snr) and snr >= 30 else '#e6c83e' if pd.notna(snr) and snr >= 10 else '#ff6b35'
        phi_e = grain.get('PhiE_deg', np.nan)
        phi_str = f'{phi_e:.0f}\u00b0' if pd.notna(phi_e) else '\u2014'

        def _metric(label_text, value, color=C['text'], unit=''):
            return html.Div([
                html.Div(label_text, style=lbl),
                html.Div(f'{value}{unit}', style={**mono, 'color': color, 'fontSize': '11px'}),
            ], style={'display': 'inline-flex', 'flexDirection': 'column', 'alignItems': 'center',
                      'padding': '0 6px', 'minWidth': '40px'})

        divider = html.Div(style={'width': '1px', 'height': '24px', 'background': C['border'],
                                   'margin': '0 2px', 'flexShrink': '0'})

        readout = html.Div([
            html.Div([
                html.Span(f'#{grain_idx}', style={**mono, 'color': C['text3'], 'fontSize': '10px'}),
                html.Span(' \u25cf ', style={'color': pri_color, 'fontSize': '14px'}),
                html.Span(pri_label, style={**mono, 'color': pri_color, 'fontWeight': '600'}),
                html.Span(f' \u2022 {sec_label}', style={**sm, 'color': sec_color, 'fontSize': '9px', 'marginLeft': '4px'}),
                html.Span(f' \u2022 M3 {grain["M3 Category"]}', style={**sm, 'fontSize': '9px', 'marginLeft': '2px'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '10px', 'flexShrink': '0'}),
            divider,
            _metric('Dist', f'{grain["R_sat"]:.1f}', C['text'], ' R\u209b'),
            _metric('Year', str(int(grain['Year']))),
            _metric('Vrel', vrel_str, C['text2'], ' km/s'),
            _metric('Incl', incl_str, C['text2']),
            _metric('\u03a6_E', phi_str, C['text2']),
            _metric('SNR', snr_str, snr_color),
            divider,
            html.Div([
                html.Div('Peaks', style=lbl),
                html.Div(peaks_str if peaks_str else '\u2014', style={**sm, 'fontSize': '10px', 'whiteSpace': 'normal'}),
            ], style={'display': 'inline-flex', 'flexDirection': 'column', 'padding': '0 8px',
                      'minWidth': '80px', 'maxWidth': '300px'}),
        ], style={'padding': '4px 12px', 'display': 'flex', 'alignItems': 'center',
                  'flexWrap': 'wrap', 'gap': '2px', 'minHeight': '36px'})
    else:
        readout = html.Div('Click a grain on any plot to inspect',
                           style={'color': C['text3'], 'fontSize': '11px', 'padding': '10px 12px',
                                  'fontFamily': FONT, 'textAlign': 'center'})

    return fig, readout


@callback(
    Output('filter-count', 'children'),
    Input('year-slider', 'value'), Input('chem-filter', 'value'),
    Input('range-min', 'value'), Input('range-max', 'value'),
)
def update_filter_count(year_range, chem_filter, r_min, r_max):
    filtered = _filter_data(year_range, chem_filter, r_min, r_max)
    return f'{len(filtered):,} grains shown'


@callback(
    Output('latent-space', 'figure'),
    Input('year-slider', 'value'), Input('selected-grain', 'data'),
    Input('chem-filter', 'value'),
    Input('range-min', 'value'), Input('range-max', 'value'),
)
def update_latent(year_range, grain_idx, chem_filter, r_min, r_max):
    filtered = _filter_data(year_range, chem_filter, r_min, r_max).copy()
    filtered['_idx'] = filtered.index
    return build_latent(filtered, highlight_idx=grain_idx)


@callback(
    Output('spatial-map', 'figure'),
    Input('year-slider', 'value'), Input('spatial-view-mode', 'data'),
    Input('selected-grain', 'data'),
    Input('chem-filter', 'value'),
    Input('range-min', 'value'), Input('range-max', 'value'),
)
def update_spatial(year_range, view_mode, grain_idx, chem_filter, r_min, r_max):
    filtered = _filter_data(year_range, chem_filter, r_min, r_max).copy()
    filtered['_idx'] = filtered.index
    if view_mode == '3D':
        return build_spatial_map_3d(filtered, highlight_idx=grain_idx)
    return build_spatial_map(filtered, highlight_idx=grain_idx)


@callback(
    Output('comp-bars', 'figure'),
    Input('year-slider', 'value'), Input('moon-dropdown', 'value'),
    Input('chem-filter', 'value'),
    Input('range-min', 'value'), Input('range-max', 'value'),
)
def update_bars(year_range, moon, chem_filter, r_min, r_max):
    filtered = _filter_data(year_range, chem_filter, r_min, r_max)
    dist_col = moon_cols.get(moon, 'R_sat')
    return build_bars(filtered, dist_col=dist_col, dist_label=moon)


@callback(
    Output('radial-strip', 'figure'),
    Input('year-slider', 'value'), Input('selected-grain', 'data'),
    Input('moon-dropdown', 'value'),
    Input('chem-filter', 'value'),
    Input('range-min', 'value'), Input('range-max', 'value'),
)
def update_radial(year_range, grain_idx, moon, chem_filter, r_min, r_max):
    filtered = _filter_data(year_range, chem_filter, r_min, r_max).copy()
    filtered['_idx'] = filtered.index
    dist_col = moon_cols.get(moon, 'R_sat')
    return build_radial_strip(filtered, highlight_idx=grain_idx, dist_col=dist_col, dist_label=moon)


@callback(
    Output('radial-stats', 'children'), Output('radial-stats', 'style'),
    Input('radial-strip', 'relayoutData'),
    State('year-slider', 'value'), State('moon-dropdown', 'value'),
    State('chem-filter', 'value'),
    State('range-min', 'value'), State('range-max', 'value'),
)
def update_radial_stats(relayout_data, year_range, moon, chem_filter, r_min, r_max):
    base_style = {
        'position': 'absolute', 'bottom': '30px', 'left': '55px',
        'padding': '3px 10px', 'borderRadius': '6px',
        'background': 'rgba(10,10,15,0.92)', 'border': f'1px solid {C["border"]}',
        'fontSize': '10px', 'fontFamily': 'JetBrains Mono, monospace',
        'color': C['text'], 'zIndex': '10',
    }
    hidden = {**base_style, 'display': 'none'}

    if relayout_data is None:
        return '', hidden

    if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
        zoom_min = float(relayout_data['xaxis.range[0]'])
        zoom_max = float(relayout_data['xaxis.range[1]'])

        filtered = _filter_data(year_range, chem_filter, r_min, r_max)
        dist_col = moon_cols.get(moon, 'R_sat')
        in_range = filtered[(filtered[dist_col] >= zoom_min) & (filtered[dist_col] <= zoom_max)]

        if len(in_range) == 0:
            return '', hidden

        type_col, clrs, _, type_order = get_color_config()
        counts = in_range[type_col].value_counts()
        total = len(in_range)
        parts = []
        for t in type_order:
            pct = counts.get(t, 0) / total * 100
            if pct > 0:
                parts.append(html.Span([
                    html.Span(f'{t} ', style={'color': clrs[t], 'fontWeight': '600'}),
                    f'{pct:.0f}%'
                ], style={'marginRight': '8px'}))

        _su = {'R_sat': 'R\u209b', 'PhiE_deg': '\u00b0', 'PhiEncObs_deg': '\u00b0', 'Inclination': '\u00b0', 'Vrel_dust': 'km/s'}
        unit = _su.get(dist_col, 'R\u209b from moon')
        content = html.Div([
            html.Span(f'{zoom_min:.1f}\u2013{zoom_max:.1f} {unit}',
                       style={'fontWeight': '600', 'marginRight': '10px', 'color': C['text2']}),
            *parts,
            html.Span(f'  n={total:,}', style={'color': C['text3']}),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '2px'})
        return content, {**base_style, 'display': 'block'}

    if 'xaxis.autorange' in relayout_data:
        return '', hidden

    return no_update, no_update


@callback(
    Output('download-csv', 'data'),
    Input('btn-download', 'n_clicks'),
    State('year-slider', 'value'), State('chem-filter', 'value'),
    State('range-min', 'value'), State('range-max', 'value'),
    prevent_initial_call=True,
)
def on_download(n_clicks, year_range, chem_filter, r_min, r_max):
    filtered = _filter_data(year_range, chem_filter, r_min, r_max)
    export_cols = ['M3 Category', 'Type', 'PeakChem', 'Year', 'R_sat', 'UTC',
                   'Enceladus_Dist', 'Mimas_Dist', 'Tethys_Dist', 'Dione_Dist', 'Rhea_Dist', 'Titan_Dist',
                   'Latent1', 'Latent2', 'Vrel_dust', 'V_sc', 'Inclination', 'Confidence', 'SNR',
                   'RF score', 'MSE',
                   'PhiE_2008', 'PhiE_deg', 'PhiEncObs_deg']
    export = filtered[[c for c in export_cols if c in filtered.columns]]
    return dcc.send_data_frame(export.to_csv, 'cassini_filtered_grains.csv', index=False)


server = app.server

if __name__ == '__main__':
    app.run(debug=False, port=8050)