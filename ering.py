#!/usr/bin/env python3
"""
Saturn E-Ring Composition - with Time Slider
Shows: Chemistry doesn't change with distance, but DOES change over time
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('CS81 Cassini CDA Composition-selected/ConfDataSpicy.csv')

SATURN_R_KM = 60268
df = df[(df['R_sat'] > 0) & (df['R_sat'] != -999)]

df['x'] = df['X2D'] / SATURN_R_KM
df['y'] = df['Y2D'] / SATURN_R_KM

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

years = sorted(df['Year'].unique())
type_counts = df['Type'].value_counts()

FONT = "'Space Grotesk', 'Inter', system-ui, sans-serif"

C = {
    'bg': '#000000',
    'panel': '#050508',
    'card': '#0a0a0f',
    'border': 'rgba(255,255,255,0.06)',
    'grid': 'rgba(255,255,255,0.04)',
    'text': '#f0f0f4',
    'text2': '#8a8a98',
    'text3': '#55555f',
    'organic': '#ff6b35',
    'salt': '#00d9b1',
    'ice': '#4a9eff',
    'other': '#9d7cd8',
    'saturn': '#d4a84b',
}

bins = [3, 5, 7, 9, 11, 13, 15]
bin_labels = ['3-5', '5-7', '7-9', '9-11', '11-13', '13-15']

cumulative_data = {}
for year in years:
    cum_df = df[df['Year'] <= year]
    if len(cum_df) > 0:
        cumulative_data[year] = cum_df

all_sample = df

fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.55, 0.45],
    horizontal_spacing=0.06,
)

for r in [4, 6, 8, 10, 12, 14]:
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=r*np.cos(theta), y=r*np.sin(theta),
        mode='lines', line=dict(color=C['border'], width=1),
        hoverinfo='skip', showlegend=False
    ), row=1, col=1)

theta = np.linspace(0, 2*np.pi, 60)
fig.add_trace(go.Scatter(
    x=1.0*np.cos(theta), y=1.0*np.sin(theta),
    fill='toself', fillcolor=C['saturn'],
    line=dict(color='#b8944a', width=2),
    hoverinfo='skip', showlegend=False
), row=1, col=1)

rings = [
    (1.11, 1.23, 0.12),
    (1.23, 1.52, 0.20),
    (1.52, 1.95, 0.35),
    (2.02, 2.27, 0.28),
]
for r_inner, r_outer, opacity in rings:
    theta_ring = np.linspace(0, 2*np.pi, 100)
    x_ring = np.concatenate([r_outer*np.cos(theta_ring), r_inner*np.cos(theta_ring[::-1])])
    y_ring = np.concatenate([r_outer*np.sin(theta_ring), r_inner*np.sin(theta_ring[::-1])])
    fig.add_trace(go.Scatter(
        x=x_ring, y=y_ring,
        fill='toself', fillcolor=f'rgba(180,160,120,{opacity})',
        line=dict(width=0),
        hoverinfo='skip', showlegend=False
    ), row=1, col=1)

for name, r in [('Enceladus', 3.95), ('Tethys', 4.89), ('Dione', 6.26), ('Rhea', 8.74)]:
    fig.add_annotation(
        x=r, y=0.5, text=name, showarrow=False,
        font=dict(size=9, color=C['text3']),
        row=1, col=1
    )

for t, color, size, opacity in [
    ('Ice', C['ice'], 3, 0.5),
    ('Other', C['other'], 3.5, 0.6),
    ('Salt', C['salt'], 5, 0.85),
    ('Organic', C['organic'], 6, 0.9),
]:
    s = all_sample[all_sample['Type'] == t]
    fig.add_trace(go.Scattergl(
        x=s['x'].values, y=s['y'].values,
        mode='markers',
        name=f'{t} ({type_counts.get(t, 0):,})',
        marker=dict(size=size, color=color, opacity=opacity),
        hovertemplate=f'<b>{t}</b><br>R = %{{customdata[0]:.1f}} Rₛ<br>Year: %{{customdata[1]}}<extra></extra>',
        customdata=np.column_stack([s['R_sat'].values, s['Year'].values]),
        legendgroup=t,
    ), row=1, col=1)

props_all = df.groupby(pd.cut(df['R_sat'], bins=bins))['Type'].value_counts(normalize=True).unstack() * 100
x_bar = list(range(len(bin_labels)))
org_vals = [props_all.loc[props_all.index[i], 'Organic'] if 'Organic' in props_all.columns and i < len(props_all) else 0 for i in range(len(bin_labels))]
salt_vals = [props_all.loc[props_all.index[i], 'Salt'] if 'Salt' in props_all.columns and i < len(props_all) else 0 for i in range(len(bin_labels))]

fig.add_trace(go.Bar(
    x=x_bar, y=org_vals,
    marker=dict(color=C['organic']),
    width=0.35, offset=-0.18,
    hovertemplate='Organic: %{y:.1f}%<extra></extra>',
    showlegend=False,
), row=1, col=2)

fig.add_trace(go.Bar(
    x=x_bar, y=salt_vals,
    marker=dict(color=C['salt']),
    width=0.35, offset=0.18,
    hovertemplate='Salt: %{y:.1f}%<extra></extra>',
    showlegend=False,
), row=1, col=2)

SCATTER_IDX = [11, 12, 13, 14]
BAR_IDX = [15, 16]
ALL_IDX = SCATTER_IDX + BAR_IDX
types_order = ['Ice', 'Other', 'Salt', 'Organic']

def make_step(label, sample_df, bar_source_df):
    scatter_x = []
    scatter_y = []
    for t in types_order:
        s = sample_df[sample_df['Type'] == t]
        scatter_x.append(s['x'].values.tolist())
        scatter_y.append(s['y'].values.tolist())

    if len(bar_source_df) > 0:
        props = bar_source_df.groupby(pd.cut(bar_source_df['R_sat'], bins=bins), observed=True)['Type'].value_counts(normalize=True).unstack() * 100
        org_y = []
        salt_y = []
        for i in range(len(bin_labels)):
            bi = pd.Interval(bins[i], bins[i+1])
            if bi in props.index:
                org_y.append(float(props.loc[bi, 'Organic']) if 'Organic' in props.columns else 0)
                salt_y.append(float(props.loc[bi, 'Salt']) if 'Salt' in props.columns else 0)
            else:
                org_y.append(0)
                salt_y.append(0)
    else:
        org_y = [0] * len(bin_labels)
        salt_y = [0] * len(bin_labels)

    return {
        'args': [
            {'x': scatter_x + [x_bar, x_bar], 'y': scatter_y + [org_y, salt_y]},
            ALL_IDX
        ],
        'label': label,
        'method': 'restyle'
    }

steps = []

steps.append(make_step('All', all_sample, df))

for year in years:
    if year not in cumulative_data or len(cumulative_data[year]) < 10:
        continue
    cum_full = df[df['Year'] <= year]
    steps.append(make_step(str(year), cumulative_data[year], cum_full))

sliders = [{
    'active': 0,
    'currentvalue': {
        'prefix': 'Through: ',
        'font': {'size': 14, 'color': C['text']},
        'xanchor': 'center',
    },
    'pad': {'t': 30, 'b': 10},
    'len': 0.9,
    'x': 0.05,
    'xanchor': 'left',
    'y': 0,
    'yanchor': 'top',
    'bgcolor': C['panel'],
    'bordercolor': C['border'],
    'borderwidth': 1,
    'tickcolor': C['text2'],
    'font': {'color': C['text2'], 'size': 10},
    'steps': steps,
}]

r_org, _ = stats.pointbiserialr(df['Type'] == 'Organic', df['R_sat'])
r_salt, _ = stats.pointbiserialr(df['Type'] == 'Salt', df['R_sat'])

fig.update_layout(
    title=dict(
        text='<b>Saturn E-Ring Composition</b>',
        font=dict(size=22),
        x=0.5, y=0.96
    ),

    plot_bgcolor=C['card'],
    paper_bgcolor=C['bg'],
    font=dict(family=FONT, color=C['text']),

    width=1400,
    height=850,

    legend=dict(
        title=dict(text='<b>Grain Type</b>', font=dict(size=12)),
        x=0.01, y=0.98,
        xanchor='left', yanchor='top',
        bgcolor='rgba(10,10,15,0.95)',
        bordercolor=C['border'], borderwidth=1,
        font=dict(size=11),
        itemclick='toggle',
        itemdoubleclick='toggleothers',
    ),

    margin=dict(l=50, r=50, t=80, b=100),
    bargap=0.3,

    sliders=sliders,

    annotations=[
        dict(x=0.26, y=1.02, xref='paper', yref='paper',
             text='<b>Spatial Distribution</b>',
             font=dict(size=14), showarrow=False),

        dict(x=0.77, y=1.02, xref='paper', yref='paper',
             text='<b>Composition vs Distance</b>',
             font=dict(size=14), showarrow=False),

        dict(x=0.99, y=0.98, xref='paper', yref='paper',
             xanchor='right', yanchor='top',
             text=(
                 f'<span style="color:{C["organic"]}">●</span> Organic  r = {r_org:.2f}<br>'
                 f'<span style="color:{C["salt"]}">●</span> Salt  r = {r_salt:.2f}'
             ),
             font=dict(size=11), align='left',
             bgcolor=C['panel'], bordercolor=C['border'],
             borderwidth=1, borderpad=10, showarrow=False),
    ]
)

fig.update_xaxes(
    showgrid=False, zeroline=False, showticklabels=False, showline=False,
    range=[-22, 22], scaleanchor='y',
    row=1, col=1
)
fig.update_yaxes(
    showgrid=False, zeroline=False, showticklabels=False, showline=False,
    range=[-22, 22],
    row=1, col=1
)

fig.update_xaxes(
    tickmode='array', tickvals=x_bar, ticktext=bin_labels,
    tickfont=dict(size=11, color=C['text2']),
    title=dict(text='Distance from Saturn (Rₛ)', font=dict(size=11, color=C['text3'])),
    showgrid=False, showline=True, linecolor=C['border'],
    fixedrange=True,
    row=1, col=2
)
fig.update_yaxes(
    tickfont=dict(size=11, color=C['text2']),
    title=dict(text='Proportion (%)', font=dict(size=11, color=C['text3'])),
    showgrid=True, gridcolor=C['grid'],
    showline=True, linecolor=C['border'],
    range=[0, 14], dtick=2,
    fixedrange=True,
    row=1, col=2
)

fig.write_html(
    'ering.html',
    include_plotlyjs=True,
    full_html=True,
    config={
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d'],
        'displaylogo': False,
        'scrollZoom': True,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'saturn_ering_composition',
            'height': 850,
            'width': 1400,
            'scale': 2
        }
    }
)

try:
    fig.write_image('ering.png', width=1400, height=850, scale=2)
except Exception:
    pass
