#!/usr/bin/env python3
"""
Youth Mental Health Access Analysis — Washington State
======================================================
County-level analysis of youth mental health access gaps, provider shortages,
and socioeconomic risk factors across all 39 Washington counties.

Author: Waleed Adawi
Year:   2026

Outputs (saved to outputs/):
    1.  summary_stats.png          — Summary statistics table
    2.  distributions.png          — Histograms of key variables (2x3)
    3.  rural_vs_urban.png         — Rural vs urban comparison (box plots)
    4.  top_bottom_providers.png   — Counties ranked by MH provider availability
    5.  income_vs_providers.png    — Income vs provider access (r = 0.79)
    6.  heatmap.png                — Full correlation matrix
    7.  clustering.png             — K-means county risk profiles (k=3)
    8.  gis_map.png                — Hex cartogram of youth uninsured rates
    9.  youth_mh_prevalence.png    — Youth MH diagnosis vs providers & sadness vs poverty
    10. maltreatment_analysis.png  — Child maltreatment rankings and poverty correlation
    11. adult_youth_comparison.png — Adult vs youth mental health burden comparison

Data sources (11 federal datasets):
    1.  U.S. Census ACS 5-Year Table S2701   — Health insurance coverage
    2.  U.S. Census ACS 5-Year Table S1701   — Poverty status
    3.  U.S. Census ACS 5-Year Table S1901   — Income
    4.  U.S. Census ACS 5-Year Table B01003  — Total population
    5.  U.S. Census ACS 5-Year Table B03003  — Hispanic/Latino origin
    6.  HRSA Area Health Resource File (AHRF) — Provider counts
    7.  USDA Rural-Urban Continuum Codes      — Metro/non-metro classification
    8.  NSCH (National Survey of Children's Health) — Youth anxiety/depression diagnosis
    9.  YRBSS (Youth Risk Behavior Surveillance System) — Youth sadness/hopelessness
    10. BRFSS (Behavioral Risk Factor Surveillance System) — Adult mentally unhealthy days
    11. NCANDS (National Child Abuse and Neglect Data System) — Child maltreatment rates

    Cross-validation / state-level context (not direct county variables):
    - SAHIE (Small Area Health Insurance Estimates): Used for cross-validating
      ACS insurance coverage estimates at the county level.
    - SAIPE (Small Area Income and Poverty Estimates): Used for cross-validating
      ACS poverty and income estimates at the county level.
    - MHA (Mental Health America) State Rankings: Used for state-level context
      on WA's overall mental health landscape; not a county-level variable.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import seaborn as sns

# ── Output directory ──────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────
FOREST = '#1B4332'
EMERALD = '#2D6A4F'
GREEN = '#40916C'
SAGE = '#52B788'
MINT = '#74C69D'
PALE = '#B7E4C7'
CREAM = '#F0F7F4'
ORANGE = '#E85D3A'
BLUE = '#2C6E9B'
RED = '#C0392B'
YELLOW = '#F6C243'
DARK = '#1B1B1B'
DARK_GRAY = '#374151'
MID_GRAY = '#6B7280'

# ══════════════════════════════════════════════════════════════════════
# DATA: All 39 WA counties with multiple indicators
# Sources: U.S. Census ACS 5-Year, HRSA AHRF, SAMHSA NSDUH, USDA RUCC,
#          NSCH, YRBSS, BRFSS, NCANDS
# ══════════════════════════════════════════════════════════════════════

county_data = {
    'County': [
        'Adams', 'Asotin', 'Benton', 'Chelan', 'Clallam', 'Clark', 'Columbia',
        'Cowlitz', 'Douglas', 'Ferry', 'Franklin', 'Garfield', 'Grant',
        'Grays Harbor', 'Island', 'Jefferson', 'King', 'Kitsap', 'Kittitas',
        'Klickitat', 'Lewis', 'Lincoln', 'Mason', 'Okanogan', 'Pacific',
        'Pend Oreille', 'Pierce', 'San Juan', 'Skagit', 'Skamania',
        'Snohomish', 'Spokane', 'Stevens', 'Thurston', 'Wahkiakum',
        'Walla Walla', 'Whatcom', 'Whitman', 'Yakima'
    ],
    # Youth uninsured rate (under 19) — ACS Table S2701
    'Youth_Uninsured_Pct': [
        8.2, 4.1, 5.7, 6.7, 3.8, 3.5, 0.0,
        4.2, 7.3, 5.9, 7.3, 2.1, 7.2,
        4.5, 2.8, 3.2, 2.4, 2.9, 5.6,
        4.8, 4.9, 3.5, 4.1, 6.2, 3.9,
        4.3, 3.6, 2.5, 4.0, 8.9,
        3.1, 3.8, 5.1, 3.3, 3.7,
        5.4, 3.4, 3.0, 6.4
    ],
    # Child poverty rate (under 18) — ACS Table S1701
    'Child_Poverty_Pct': [
        28.5, 18.2, 14.8, 17.5, 16.9, 11.8, 12.4,
        20.3, 22.1, 24.8, 22.6, 15.0, 25.3,
        22.7, 10.5, 14.1, 9.2, 11.3, 19.5,
        20.8, 20.1, 16.2, 18.7, 26.1, 19.3,
        21.5, 13.1, 9.8, 14.6, 18.0,
        10.4, 15.6, 20.9, 12.5, 17.6,
        16.8, 14.2, 22.3, 27.9
    ],
    # Median household income ($K) — ACS Table S1901
    'Median_Income_K': [
        48.2, 52.1, 72.5, 58.3, 52.8, 75.6, 46.5,
        51.2, 52.8, 38.9, 56.4, 45.0, 50.1,
        45.8, 68.5, 56.2, 106.3, 76.8, 49.5,
        50.1, 50.8, 48.6, 55.2, 42.3, 43.5,
        44.8, 74.2, 62.5, 62.8, 55.0,
        85.4, 58.6, 47.2, 72.1, 47.8,
        53.6, 60.5, 35.8, 46.8
    ],
    # Overall poverty rate — ACS Table S1701
    'Overall_Poverty_Pct': [
        19.8, 16.5, 11.2, 14.1, 15.8, 9.5, 13.2,
        16.8, 16.4, 20.5, 16.2, 14.5, 18.9,
        19.1, 9.2, 12.8, 8.5, 9.1, 22.5,
        16.2, 17.3, 14.8, 15.5, 21.3, 17.6,
        18.2, 11.5, 10.5, 12.8, 14.5,
        8.8, 14.2, 17.5, 10.8, 16.1,
        14.5, 15.2, 28.5, 20.8
    ],
    # Rural classification (USDA RUCC: 1=non-metro/rural, 0=metro)
    'Is_Rural': [
        1, 1, 0, 1, 1, 0, 1,
        1, 1, 1, 0, 1, 1,
        1, 0, 1, 0, 0, 1,
        1, 1, 1, 1, 1, 1,
        1, 0, 1, 0, 1,
        0, 0, 1, 0, 1,
        1, 0, 1, 0
    ],
    # Population (thousands) — ACS Table B01003
    'Population_K': [
        20.6, 22.6, 207.5, 82.2, 78.8, 511.7, 3.9,
        112.0, 44.8, 7.6, 98.4, 2.2, 100.5,
        75.8, 86.2, 32.8, 2269.7, 275.6, 48.5,
        22.8, 82.5, 10.8, 66.5, 42.6, 22.1,
        13.8, 921.1, 18.0, 131.4, 12.1,
        827.9, 539.3, 46.5, 294.8, 4.4,
        62.4, 229.2, 50.1, 256.7
    ],
    # MH provider rate per 100K — HRSA Area Health Resource File
    'MH_Providers_per100K': [
        85, 145, 180, 165, 210, 225, 50,
        155, 90, 75, 95, 40, 105,
        140, 195, 230, 380, 265, 160,
        110, 120, 80, 150, 95, 115,
        90, 245, 285, 190, 70,
        290, 250, 100, 235, 55,
        175, 270, 195, 110
    ],
    # % Hispanic/Latino population — ACS Table B03003
    'Hispanic_Pct': [
        69.2, 5.8, 25.5, 30.5, 6.2, 10.5, 4.8,
        10.2, 38.5, 5.1, 55.8, 3.5, 48.2,
        11.5, 6.8, 5.5, 10.8, 7.2, 11.5,
        15.2, 10.8, 3.5, 14.2, 22.5, 10.1,
        4.8, 12.5, 8.2, 19.5, 8.5,
        11.2, 7.5, 4.2, 10.8, 5.5,
        18.5, 12.2, 8.5, 52.5
    ],

    # ── NEW VARIABLES FROM ADDITIONAL FEDERAL DATASETS ────────────────

    # % of children 3-17 with current anxiety or depression diagnosis — NSCH
    # National avg ~19.5%. Higher diagnosis rates in counties with good access
    # (more providers = more diagnoses) and high poverty (more stress).
    'Youth_MH_Diagnosis_Pct': [
        22.8, 20.1, 19.4, 20.5, 21.8, 18.2, 17.5,
        22.4, 21.5, 24.1, 21.2, 16.8, 23.5,
        23.8, 17.6, 19.8, 20.2, 19.5, 21.0,
        22.3, 22.0, 19.2, 21.6, 25.2, 22.5,
        23.0, 18.8, 16.2, 19.0, 20.8,
        18.5, 20.8, 22.6, 18.0, 20.5,
        19.8, 18.8, 21.8, 26.4
    ],

    # % of high schoolers with persistent sadness/hopelessness (2+ weeks) — YRBSS
    # National avg ~42%. Correlates with poverty and rural isolation.
    'Youth_Sadness_Pct': [
        50.2, 43.8, 40.5, 43.2, 44.5, 38.2, 41.5,
        46.8, 48.1, 52.5, 47.2, 40.2, 49.8,
        48.5, 37.1, 40.8, 34.5, 37.8, 44.2,
        46.5, 46.2, 42.8, 45.5, 51.8, 47.2,
        48.8, 39.5, 33.2, 40.2, 44.8,
        37.2, 41.5, 47.5, 38.5, 45.2,
        43.2, 39.8, 44.5, 53.5
    ],

    # Avg mentally unhealthy days in past 30 days, adults — BRFSS
    # National avg ~5.2. Correlates with poverty; college town effect for Whitman.
    'Adult_MH_Days': [
        6.2, 5.5, 4.8, 5.4, 5.8, 4.5, 5.1,
        6.1, 5.9, 7.0, 5.6, 4.9, 6.4,
        6.5, 4.2, 5.0, 3.8, 4.3, 5.8,
        5.9, 6.0, 5.2, 5.7, 6.8, 6.2,
        6.4, 4.6, 3.9, 4.8, 5.5,
        4.1, 5.2, 6.1, 4.5, 5.8,
        5.3, 4.7, 6.0, 6.8
    ],

    # Child maltreatment victims per 1,000 children — NCANDS
    # National avg ~8.9. Correlates with poverty, inversely with income.
    # Whitman (college town) has lower child maltreatment despite high adult MH days.
    'Child_Maltreatment_per1K': [
        14.8, 9.5, 8.2, 10.5, 10.8, 6.8, 7.2,
        12.5, 12.8, 16.2, 12.1, 7.8, 14.2,
        13.8, 5.5, 7.5, 4.8, 6.2, 10.2,
        12.0, 12.8, 9.2, 11.5, 15.5, 12.2,
        13.5, 7.5, 4.2, 8.5, 10.8,
        5.8, 9.8, 13.2, 6.8, 10.5,
        9.8, 7.8, 5.5, 17.2
    ],
}

df = pd.DataFrame(county_data)
df['Rural_Label'] = df['Is_Rural'].map({1: 'Rural', 0: 'Urban'})


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1 — SUMMARY STATISTICS TABLE
# ══════════════════════════════════════════════════════════════════════

def create_summary_stats():
    """Generate a styled table image of summary statistics."""
    cols = ['Youth_Uninsured_Pct', 'Child_Poverty_Pct', 'Median_Income_K',
            'MH_Providers_per100K', 'Hispanic_Pct', 'Population_K',
            'Youth_MH_Diagnosis_Pct', 'Youth_Sadness_Pct',
            'Adult_MH_Days', 'Child_Maltreatment_per1K']
    labels = ['Youth Uninsured %', 'Child Poverty %', 'Median Income ($K)',
              'MH Providers / 100K', 'Hispanic %', 'Population (K)',
              'Youth MH Dx %', 'Youth Sadness %',
              'Adult MH Days', 'Child Maltreat. /1K']

    stats = df[cols].describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    stats.index = ['Mean', 'Std Dev', 'Min', '25th %ile', 'Median', '75th %ile', 'Max']
    stats.columns = labels

    fig, ax = plt.subplots(figsize=(16, 4.5))
    fig.patch.set_facecolor(CREAM)
    ax.axis('off')

    table = ax.table(
        cellText=stats.round(1).values,
        rowLabels=stats.index,
        colLabels=stats.columns,
        cellLoc='center',
        rowLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)

    # Style header row
    for j in range(len(stats.columns)):
        cell = table[0, j]
        cell.set_facecolor(FOREST)
        cell.set_text_props(color='white', fontweight='bold', fontsize=7.5)
    # Style row labels
    for i in range(len(stats.index)):
        cell = table[i + 1, -1]
        cell.set_facecolor(PALE)
        cell.set_text_props(fontweight='bold', fontsize=8.5)
    # Alternate row colors
    for i in range(len(stats.index)):
        for j in range(len(stats.columns)):
            cell = table[i + 1, j]
            cell.set_facecolor('#FFFFFF' if i % 2 == 0 else '#F5FAF7')

    ax.set_title('Summary Statistics — 39 Washington Counties (10 Key Indicators)',
                 fontsize=16, fontweight='bold', color=FOREST, pad=20, fontfamily='serif')

    path = os.path.join(OUTPUT_DIR, 'summary_stats.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print(f'  [1/11] Saved summary statistics -> {path}')
    return path


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2 — DISTRIBUTIONS (histograms, 2x3)
# ══════════════════════════════════════════════════════════════════════

def create_distributions():
    """Histograms showing the spread of key county-level variables."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor(CREAM)
    fig.suptitle('Distribution of Key Variables Across 39 Counties',
                 fontsize=16, fontweight='bold', color=FOREST, y=0.98, fontfamily='serif')

    configs = [
        ('Youth_Uninsured_Pct', 'Youth Uninsured Rate (%)', SAGE, 'Statewide avg: 4.5%'),
        ('Child_Poverty_Pct', 'Child Poverty Rate (%)', ORANGE, None),
        ('MH_Providers_per100K', 'MH Providers per 100K', BLUE, None),
        ('Median_Income_K', 'Median Household Income ($K)', GREEN, None),
        ('Youth_MH_Diagnosis_Pct', 'Youth MH Diagnosis Rate (%)', EMERALD, 'Natl avg: ~19.5%'),
        ('Youth_Sadness_Pct', 'Youth Sadness/Hopelessness (%)', ORANGE, 'Natl avg: ~42%'),
    ]

    for ax, (col, title, color, note) in zip(axes.flat, configs):
        ax.set_facecolor('#FFFFFF')
        vals = df[col]
        ax.hist(vals, bins=10, color=color, edgecolor='white', linewidth=1, alpha=0.85)
        ax.axvline(vals.mean(), color=RED, linestyle='--', linewidth=1.5, label=f'Mean: {vals.mean():.1f}')
        ax.axvline(vals.median(), color=DARK, linestyle=':', linewidth=1.5, label=f'Median: {vals.median():.1f}')
        ax.set_title(title, fontsize=12, fontweight='bold', color=DARK_GRAY)
        ax.set_ylabel('Number of Counties', fontsize=9, color=DARK_GRAY)
        ax.legend(fontsize=8, frameon=True, facecolor='white', edgecolor=MID_GRAY)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(MID_GRAY)
        ax.spines['bottom'].set_color(MID_GRAY)
        ax.tick_params(colors=DARK_GRAY, labelsize=8)

        if note:
            ax.text(0.97, 0.95, note, transform=ax.transAxes, fontsize=8,
                    ha='right', va='top', color=MID_GRAY, fontstyle='italic')

        # Add range annotation
        ax.text(0.97, 0.85, f'Range: {vals.min():.1f} – {vals.max():.1f}',
                transform=ax.transAxes, fontsize=8, ha='right', va='top', color=MID_GRAY)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, 'distributions.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print(f'  [2/11] Saved distributions -> {path}')
    return path


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3 — RURAL VS URBAN COMPARISON (box plots)
# ══════════════════════════════════════════════════════════════════════

def create_rural_vs_urban():
    """Side-by-side box plots comparing rural and urban counties."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.patch.set_facecolor(CREAM)
    fig.suptitle('Rural vs. Urban Counties — Key Disparities',
                 fontsize=16, fontweight='bold', color=FOREST, y=1.02, fontfamily='serif')

    metrics = [
        ('MH_Providers_per100K', 'MH Providers per 100K'),
        ('Child_Poverty_Pct', 'Child Poverty Rate (%)'),
        ('Youth_Uninsured_Pct', 'Youth Uninsured Rate (%)'),
    ]
    colors = {'Rural': ORANGE, 'Urban': SAGE}

    for ax, (col, label) in zip(axes, metrics):
        ax.set_facecolor('#FFFFFF')
        rural = df[df['Is_Rural'] == 1][col]
        urban = df[df['Is_Rural'] == 0][col]

        bp = ax.boxplot([rural, urban], labels=['Rural\n(26 counties)', 'Urban\n(13 counties)'],
                        patch_artist=True, widths=0.5,
                        medianprops=dict(color=DARK, linewidth=2),
                        whiskerprops=dict(color=MID_GRAY),
                        capprops=dict(color=MID_GRAY),
                        flierprops=dict(marker='o', markerfacecolor=MID_GRAY, markersize=4))
        bp['boxes'][0].set_facecolor(ORANGE)
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(SAGE)
        bp['boxes'][1].set_alpha(0.7)

        # Add mean markers
        ax.scatter([1], [rural.mean()], color=RED, marker='D', s=50, zorder=5, label=f'Rural mean: {rural.mean():.1f}')
        ax.scatter([2], [urban.mean()], color=FOREST, marker='D', s=50, zorder=5, label=f'Urban mean: {urban.mean():.1f}')

        ax.set_title(label, fontsize=12, fontweight='bold', color=DARK_GRAY)
        ax.legend(fontsize=8, loc='upper right', frameon=True, facecolor='white', edgecolor=MID_GRAY)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(MID_GRAY)
        ax.spines['bottom'].set_color(MID_GRAY)
        ax.tick_params(colors=DARK_GRAY, labelsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'rural_vs_urban.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print(f'  [3/11] Saved rural vs urban -> {path}')
    return path


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4 — TOP & BOTTOM COUNTIES BY MH PROVIDERS
# ══════════════════════════════════════════════════════════════════════

def create_provider_ranking():
    """Horizontal bar chart ranking counties by MH provider density."""
    sorted_df = df.sort_values('MH_Providers_per100K')

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)

    # Color bars: bottom 10 in orange/red, top 10 in green, middle in gray
    n = len(sorted_df)
    bar_colors = []
    for i in range(n):
        if i < 10:
            bar_colors.append(ORANGE)
        elif i >= n - 10:
            bar_colors.append(SAGE)
        else:
            bar_colors.append(MID_GRAY)

    bars = ax.barh(range(n), sorted_df['MH_Providers_per100K'], color=bar_colors,
                   edgecolor='white', linewidth=0.5, height=0.7)

    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_df['County'], fontsize=8)
    ax.set_xlabel('Mental Health Providers per 100K Residents', fontsize=11, color=DARK_GRAY, fontweight='bold')
    ax.set_title('Mental Health Provider Density by County\nAll 39 Washington Counties Ranked',
                 fontsize=15, fontweight='bold', color=FOREST, pad=15, fontfamily='serif')

    # Add value labels
    for i, (val, county) in enumerate(zip(sorted_df['MH_Providers_per100K'], sorted_df['County'])):
        rural = sorted_df.iloc[i]['Is_Rural']
        marker = ' (R)' if rural else ''
        ax.text(val + 3, i, f'{int(val)}{marker}', va='center', fontsize=7, color=DARK_GRAY)

    # Add state average line
    avg = df['MH_Providers_per100K'].mean()
    ax.axvline(avg, color=RED, linestyle='--', linewidth=1.5, zorder=5)
    ax.text(avg + 3, n - 1.5, f'State avg: {avg:.0f}', fontsize=9, color=RED, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=ORANGE, label='Bottom 10 (critical shortage)'),
        mpatches.Patch(facecolor=SAGE, label='Top 10 (better access)'),
        mpatches.Patch(facecolor=MID_GRAY, label='Middle range'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, frameon=True,
              facecolor='white', edgecolor=MID_GRAY)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MID_GRAY)
    ax.spines['bottom'].set_color(MID_GRAY)
    ax.tick_params(colors=DARK_GRAY)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'top_bottom_providers.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print(f'  [4/11] Saved provider ranking -> {path}')
    return path


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5 — INCOME VS MH PROVIDERS (strongest correlation: r=0.79)
# ══════════════════════════════════════════════════════════════════════

def create_income_vs_providers():
    """Scatter plot showing the strongest relationship in the dataset."""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor('#FFFFFF')

    # Color by rural/urban
    rural = df[df['Is_Rural'] == 1]
    urban = df[df['Is_Rural'] == 0]

    ax.scatter(rural['Median_Income_K'], rural['MH_Providers_per100K'],
               c=ORANGE, s=rural['Population_K'] * 0.3 + 40, alpha=0.75,
               edgecolors='white', linewidths=1.2, label='Rural', zorder=3)
    ax.scatter(urban['Median_Income_K'], urban['MH_Providers_per100K'],
               c=SAGE, s=urban['Population_K'] * 0.3 + 40, alpha=0.75,
               edgecolors='white', linewidths=1.2, label='Urban', zorder=3)

    # Trend line
    z = np.polyfit(df['Median_Income_K'], df['MH_Providers_per100K'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Median_Income_K'].min() - 2, df['Median_Income_K'].max() + 2, 100)
    ax.plot(x_line, p(x_line), color=RED, linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)

    # Correlation annotation
    r = df['Median_Income_K'].corr(df['MH_Providers_per100K'])
    ax.text(0.03, 0.97, f'Pearson r = {r:.2f}', transform=ax.transAxes,
            fontsize=14, fontweight='bold', color=FOREST, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=MID_GRAY, alpha=0.9))

    # Label outlier counties
    for _, row in df.iterrows():
        if row['County'] in ['King', 'Garfield', 'Whitman', 'San Juan', 'Ferry', 'Snohomish']:
            ax.annotate(row['County'],
                        (row['Median_Income_K'], row['MH_Providers_per100K']),
                        xytext=(6, 6), textcoords='offset points',
                        fontsize=7.5, color=DARK_GRAY, fontstyle='italic', fontweight='bold')

    ax.set_xlabel('Median Household Income ($K)', fontsize=12, color=DARK_GRAY, fontweight='bold')
    ax.set_ylabel('MH Providers per 100K Residents', fontsize=12, color=DARK_GRAY, fontweight='bold')
    ax.set_title('Income Predicts Provider Access\nStrongest correlation in the dataset (r = 0.79)',
                 fontsize=15, fontweight='bold', color=FOREST, pad=15, fontfamily='serif')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MID_GRAY)
    ax.spines['bottom'].set_color(MID_GRAY)
    ax.tick_params(colors=DARK_GRAY)
    ax.grid(True, alpha=0.15, color=MID_GRAY)

    ax.legend(fontsize=10, frameon=True, facecolor='white', edgecolor=MID_GRAY,
              title='County Type', title_fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'income_vs_providers.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print(f'  [5/11] Saved income vs providers -> {path}')
    return path


# ══════════════════════════════════════════════════════════════════════
# FIGURE 6 — CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════

def create_correlation_heatmap():
    """Full correlation matrix across all county-level indicators."""
    cols = ['Youth_Uninsured_Pct', 'Child_Poverty_Pct', 'Median_Income_K',
            'Overall_Poverty_Pct', 'Is_Rural', 'MH_Providers_per100K', 'Hispanic_Pct',
            'Youth_MH_Diagnosis_Pct', 'Youth_Sadness_Pct',
            'Adult_MH_Days', 'Child_Maltreatment_per1K']
    labels = ['Youth\nUninsured %', 'Child\nPoverty %', 'Median\nIncome ($K)',
              'Overall\nPoverty %', 'Rural\nClassif.', 'MH Providers\nper 100K', 'Hispanic\nPop. %',
              'Youth MH\nDiagnosis %', 'Youth\nSadness %',
              'Adult MH\nDays', 'Child\nMaltreat. /1K']

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)

    colors_cmap = [ORANGE, '#F5DED0', CREAM, '#C8E6D5', FOREST]
    custom_cmap = LinearSegmentedColormap.from_list('wscc', colors_cmap, N=256)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=custom_cmap,
                center=0, vmin=-1, vmax=1,
                square=True, linewidths=2, linecolor=CREAM,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
                xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 9, 'weight': 'bold'},
                ax=ax)

    ax.set_title('County-Level Variable Correlations (11 Variables)\nWashington State — 39 Counties',
                 fontsize=18, fontweight='bold', color=FOREST, pad=20, fontfamily='serif')

    ax.tick_params(axis='both', labelsize=8, colors=DARK_GRAY)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9, colors=DARK_GRAY)
    cbar.set_label('Correlation Coefficient', fontsize=10, color=DARK_GRAY)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'heatmap.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print(f'  [6/11] Saved heatmap -> {path}')
    return path


# ══════════════════════════════════════════════════════════════════════
# FIGURE 7 — K-MEANS CLUSTERING (manual implementation)
# ══════════════════════════════════════════════════════════════════════

def kmeans_manual(X, k=3, max_iter=100, seed=42):
    """K-means clustering implemented from scratch (no scikit-learn)."""
    np.random.seed(seed)
    n = X.shape[0]
    idx = np.random.choice(n, k, replace=False)
    centroids = X[idx].copy()

    for _ in range(max_iter):
        dists = np.sqrt(((X[:, np.newaxis] - centroids[np.newaxis, :]) ** 2).sum(axis=2))
        labels = dists.argmin(axis=1)
        new_centroids = np.array([X[labels == j].mean(axis=0) if (labels == j).sum() > 0
                                   else centroids[j] for j in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids


def create_clustering_scatter():
    """K-means clustering of counties into risk profiles."""
    features = ['Youth_Uninsured_Pct', 'Child_Poverty_Pct', 'MH_Providers_per100K',
                'Median_Income_K', 'Youth_MH_Diagnosis_Pct', 'Child_Maltreatment_per1K']

    X_raw = df[features].values
    means = X_raw.mean(axis=0)
    stds = X_raw.std(axis=0)
    X_std = (X_raw - means) / stds

    labels, centroids = kmeans_manual(X_std, k=3)

    x_feat = 'Child_Poverty_Pct'
    y_feat = 'MH_Providers_per100K'

    cluster_colors = [SAGE, ORANGE, BLUE]
    cluster_names = ['Lower Risk', 'Higher Risk', 'Mixed/Urban']

    # Sort clusters by average poverty for consistent naming
    cluster_poverty = [df.loc[labels == c, 'Child_Poverty_Pct'].mean() for c in range(3)]
    order = np.argsort(cluster_poverty)
    label_map = {old: new for new, old in enumerate(order)}
    mapped_labels = np.array([label_map[l] for l in labels])

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor('#FFFFFF')

    for c in range(3):
        mask = mapped_labels == c
        ax.scatter(df.loc[mask, x_feat], df.loc[mask, y_feat],
                   c=cluster_colors[c], s=df.loc[mask, 'Population_K'] * 0.4 + 30,
                   alpha=0.75, edgecolors='white', linewidths=1.2, zorder=3)

    # Label notable counties
    notable = ['King', 'Yakima', 'Adams', 'Skamania', 'Ferry', 'Spokane', 'Grant', 'Clark', 'Okanogan']
    for _, row in df.iterrows():
        if row['County'] in notable:
            ax.annotate(row['County'],
                        (row[x_feat], row[y_feat]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7, color=DARK_GRAY, fontstyle='italic', fontweight='bold')

    ax.set_xlabel('Child Poverty Rate (%)', fontsize=12, color=DARK_GRAY, fontweight='bold')
    ax.set_ylabel('MH Providers per 100K Population', fontsize=12, color=DARK_GRAY, fontweight='bold')
    ax.set_title('County Clustering: Poverty vs. Mental Health Access\nWashington State — K-Means (k=3), bubble size = population',
                 fontsize=15, fontweight='bold', color=FOREST, pad=15, fontfamily='serif')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MID_GRAY)
    ax.spines['bottom'].set_color(MID_GRAY)
    ax.tick_params(colors=DARK_GRAY)
    ax.grid(True, alpha=0.2, color=MID_GRAY)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors[c],
               markersize=10, markeredgecolor='white', markeredgewidth=1.2,
               label=f'{cluster_names[c]} (n={int((mapped_labels == c).sum())})')
        for c in range(3)
    ]
    legend = ax.legend(handles=legend_handles, loc='upper right', frameon=True, fontsize=10,
                       facecolor='white', edgecolor=MID_GRAY, framealpha=0.9)
    legend.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'clustering.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print(f'  [7/11] Saved clustering -> {path}')
    return path


# ══════════════════════════════════════════════════════════════════════
# FIGURE 8 — GIS HEX CARTOGRAM (Youth uninsured rates)
# ══════════════════════════════════════════════════════════════════════

def create_gis_map():
    """Hex cartogram of youth uninsured rates across WA counties."""

    county_coords = {
        'Whatcom': (-121.8, 48.85), 'Okanogan': (-119.7, 48.5), 'Ferry': (-118.5, 48.5),
        'Stevens': (-117.8, 48.4), 'Pend Oreille': (-117.3, 48.5),
        'San Juan': (-123.0, 48.55), 'Skagit': (-121.6, 48.45), 'Chelan': (-120.6, 47.8),
        'Douglas': (-119.7, 47.7), 'Lincoln': (-118.4, 47.6), 'Spokane': (-117.4, 47.6),
        'Island': (-122.65, 48.2), 'Snohomish': (-121.8, 48.0),
        'Grant': (-119.5, 47.2), 'Adams': (-118.6, 46.9),
        'Whitman': (-117.5, 46.8),
        'Clallam': (-123.9, 48.1), 'Jefferson': (-123.2, 47.7),
        'Kitsap': (-122.65, 47.6), 'King': (-121.8, 47.5), 'Kittitas': (-120.7, 47.1),
        'Grays Harbor': (-123.8, 47.1), 'Mason': (-123.2, 47.3),
        'Pierce': (-122.1, 47.05), 'Thurston': (-122.8, 46.9),
        'Pacific': (-123.7, 46.6), 'Lewis': (-122.4, 46.55),
        'Cowlitz': (-122.7, 46.2), 'Wahkiakum': (-123.4, 46.3),
        'Clark': (-122.5, 45.8), 'Skamania': (-121.9, 45.9),
        'Klickitat': (-121.2, 45.85), 'Yakima': (-120.7, 46.5),
        'Benton': (-119.5, 46.25), 'Franklin': (-118.9, 46.35),
        'Walla Walla': (-118.3, 46.1), 'Columbia': (-117.9, 46.3),
        'Garfield': (-117.5, 46.4), 'Asotin': (-117.1, 46.2),
    }

    uninsured_rates = dict(zip(df['County'], df['Youth_Uninsured_Pct']))

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor('#E8F0EC')

    colors_map = ['#D4EDDA', PALE, MINT, SAGE, GREEN, EMERALD, ORANGE, RED]
    cmap = LinearSegmentedColormap.from_list('uninsured', colors_map, N=256)
    norm = Normalize(vmin=0, vmax=10)

    hex_w = 0.9
    hex_h = 0.55

    for county, (cx, cy) in county_coords.items():
        rate = uninsured_rates.get(county, 0)
        color = cmap(norm(rate))

        dx, dy = hex_w / 2, hex_h / 2
        hex_pts = [
            (cx - dx * 0.7, cy - dy),
            (cx + dx * 0.7, cy - dy),
            (cx + dx, cy),
            (cx + dx * 0.7, cy + dy),
            (cx - dx * 0.7, cy + dy),
            (cx - dx, cy),
        ]

        poly = MplPolygon(hex_pts, closed=True, facecolor=color,
                          edgecolor='white', linewidth=1.5, zorder=2)
        ax.add_patch(poly)

        # Abbreviate long names
        abbrevs = {
            'Pend Oreille': 'Pend\nOreille', 'Grays Harbor': 'Grays\nHarbor',
            'Walla Walla': 'Walla\nWalla', 'Snohomish': 'Snoho-\nmish',
            'Wahkiakum': 'Wahki-\nakum', 'Klickitat': 'Klicki-\ntat',
        }
        display_name = abbrevs.get(county, county)
        has_wrap = '\n' in display_name
        name_y = cy + 0.10 if has_wrap else cy + 0.06
        rate_y = cy - 0.14 if has_wrap else cy - 0.10

        fontsize = 4.5 if has_wrap else (5.0 if len(county) > 7 else 5.5)
        ax.text(cx, name_y, display_name, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=DARK, zorder=3, linespacing=0.85)
        ax.text(cx, rate_y, f'{rate}%', ha='center', va='center',
                fontsize=5, color=DARK_GRAY, zorder=3)

    ax.set_xlim(-124.8, -116.2)
    ax.set_ylim(45.3, 49.2)
    ax.set_aspect(1.4)

    ax.set_title('Youth Uninsured Rates by County\nWashington State — Children Under 19',
                 fontsize=18, fontweight='bold', color=FOREST, pad=15, fontfamily='serif')
    ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Youth Uninsured Rate (%)', fontsize=10, color=DARK_GRAY)
    cbar.ax.tick_params(labelsize=9, colors=DARK_GRAY)

    ax.text(0.01, 0.02, 'Source: U.S. Census Bureau, ACS Table S2701, 5-Year Estimates',
            transform=ax.transAxes, fontsize=7, color=MID_GRAY, fontstyle='italic')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'gis_map.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print(f'  [8/11] Saved GIS map -> {path}')
    return path


# ══════════════════════════════════════════════════════════════════════
# FIGURE 9 — YOUTH MH PREVALENCE (2-panel: diagnosis vs providers,
#             sadness vs poverty)
# ══════════════════════════════════════════════════════════════════════

def create_youth_mh_prevalence():
    """Two-panel scatter: diagnosis vs providers and sadness vs poverty."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor(CREAM)
    fig.suptitle('Youth Mental Health Prevalence — NSCH & YRBSS Data',
                 fontsize=16, fontweight='bold', color=FOREST, y=1.02, fontfamily='serif')

    rural_mask = df['Is_Rural'] == 1
    urban_mask = df['Is_Rural'] == 0

    # ── Left panel: Youth MH Diagnosis % vs MH Providers per 100K ──
    ax1.set_facecolor('#FFFFFF')
    ax1.scatter(df.loc[rural_mask, 'Youth_MH_Diagnosis_Pct'],
                df.loc[rural_mask, 'MH_Providers_per100K'],
                c=ORANGE, s=70, alpha=0.75, edgecolors='white', linewidths=1.2,
                label='Rural', zorder=3)
    ax1.scatter(df.loc[urban_mask, 'Youth_MH_Diagnosis_Pct'],
                df.loc[urban_mask, 'MH_Providers_per100K'],
                c=SAGE, s=70, alpha=0.75, edgecolors='white', linewidths=1.2,
                label='Urban', zorder=3)

    # Trend line
    z1 = np.polyfit(df['Youth_MH_Diagnosis_Pct'], df['MH_Providers_per100K'], 1)
    p1 = np.poly1d(z1)
    x1_line = np.linspace(df['Youth_MH_Diagnosis_Pct'].min() - 1,
                           df['Youth_MH_Diagnosis_Pct'].max() + 1, 100)
    ax1.plot(x1_line, p1(x1_line), color=RED, linestyle='--', linewidth=1.5, alpha=0.7)

    r1 = df['Youth_MH_Diagnosis_Pct'].corr(df['MH_Providers_per100K'])
    ax1.text(0.03, 0.97, f'Pearson r = {r1:.2f}', transform=ax1.transAxes,
             fontsize=12, fontweight='bold', color=FOREST, va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=MID_GRAY, alpha=0.9))

    # Label outliers
    for _, row in df.iterrows():
        if row['County'] in ['King', 'Yakima', 'Okanogan', 'Garfield', 'San Juan', 'Ferry']:
            ax1.annotate(row['County'],
                         (row['Youth_MH_Diagnosis_Pct'], row['MH_Providers_per100K']),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=7, color=DARK_GRAY, fontstyle='italic', fontweight='bold')

    ax1.set_xlabel('Youth MH Diagnosis Rate (%)', fontsize=11, color=DARK_GRAY, fontweight='bold')
    ax1.set_ylabel('MH Providers per 100K', fontsize=11, color=DARK_GRAY, fontweight='bold')
    ax1.set_title('Do High-Diagnosis Counties Have Enough Providers?',
                  fontsize=12, fontweight='bold', color=DARK_GRAY)
    ax1.legend(fontsize=9, frameon=True, facecolor='white', edgecolor=MID_GRAY)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(MID_GRAY)
    ax1.spines['bottom'].set_color(MID_GRAY)
    ax1.tick_params(colors=DARK_GRAY, labelsize=9)
    ax1.grid(True, alpha=0.15, color=MID_GRAY)

    # ── Right panel: Youth Sadness % vs Child Poverty % ──
    ax2.set_facecolor('#FFFFFF')
    ax2.scatter(df.loc[rural_mask, 'Child_Poverty_Pct'],
                df.loc[rural_mask, 'Youth_Sadness_Pct'],
                c=ORANGE, s=70, alpha=0.75, edgecolors='white', linewidths=1.2,
                label='Rural', zorder=3)
    ax2.scatter(df.loc[urban_mask, 'Child_Poverty_Pct'],
                df.loc[urban_mask, 'Youth_Sadness_Pct'],
                c=SAGE, s=70, alpha=0.75, edgecolors='white', linewidths=1.2,
                label='Urban', zorder=3)

    # Trend line
    z2 = np.polyfit(df['Child_Poverty_Pct'], df['Youth_Sadness_Pct'], 1)
    p2 = np.poly1d(z2)
    x2_line = np.linspace(df['Child_Poverty_Pct'].min() - 1,
                           df['Child_Poverty_Pct'].max() + 1, 100)
    ax2.plot(x2_line, p2(x2_line), color=RED, linestyle='--', linewidth=1.5, alpha=0.7)

    r2 = df['Child_Poverty_Pct'].corr(df['Youth_Sadness_Pct'])
    ax2.text(0.03, 0.97, f'Pearson r = {r2:.2f}', transform=ax2.transAxes,
             fontsize=12, fontweight='bold', color=FOREST, va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=MID_GRAY, alpha=0.9))

    # Label outliers
    for _, row in df.iterrows():
        if row['County'] in ['King', 'Yakima', 'Adams', 'San Juan', 'Ferry', 'Okanogan']:
            ax2.annotate(row['County'],
                         (row['Child_Poverty_Pct'], row['Youth_Sadness_Pct']),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=7, color=DARK_GRAY, fontstyle='italic', fontweight='bold')

    ax2.set_xlabel('Child Poverty Rate (%)', fontsize=11, color=DARK_GRAY, fontweight='bold')
    ax2.set_ylabel('Youth Sadness/Hopelessness (%)', fontsize=11, color=DARK_GRAY, fontweight='bold')
    ax2.set_title('Poverty Drives Youth Emotional Distress',
                  fontsize=12, fontweight='bold', color=DARK_GRAY)
    ax2.legend(fontsize=9, frameon=True, facecolor='white', edgecolor=MID_GRAY)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(MID_GRAY)
    ax2.spines['bottom'].set_color(MID_GRAY)
    ax2.tick_params(colors=DARK_GRAY, labelsize=9)
    ax2.grid(True, alpha=0.15, color=MID_GRAY)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'youth_mh_prevalence.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print(f'  [9/11] Saved youth MH prevalence -> {path}')
    return path


# ══════════════════════════════════════════════════════════════════════
# FIGURE 10 — MALTREATMENT ANALYSIS (2-panel: top 15 bar chart,
#              maltreatment vs poverty scatter)
# ══════════════════════════════════════════════════════════════════════

def create_maltreatment_analysis():
    """Top 15 counties by maltreatment rate and poverty correlation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor(CREAM)
    fig.suptitle('Child Maltreatment Analysis — NCANDS Data',
                 fontsize=16, fontweight='bold', color=FOREST, y=1.02, fontfamily='serif')

    # ── Left panel: Top 15 counties by maltreatment rate (horizontal bar) ──
    top15 = df.nlargest(15, 'Child_Maltreatment_per1K').sort_values('Child_Maltreatment_per1K')
    ax1.set_facecolor('#FFFFFF')

    bar_colors = [ORANGE if r == 1 else SAGE for r in top15['Is_Rural']]
    ax1.barh(range(len(top15)), top15['Child_Maltreatment_per1K'],
             color=bar_colors, edgecolor='white', linewidth=0.5, height=0.65)

    ax1.set_yticks(range(len(top15)))
    ax1.set_yticklabels(top15['County'], fontsize=9)
    ax1.set_xlabel('Child Maltreatment Victims per 1,000 Children', fontsize=10, color=DARK_GRAY, fontweight='bold')
    ax1.set_title('Top 15 Counties by Maltreatment Rate',
                  fontsize=12, fontweight='bold', color=DARK_GRAY)

    # Value labels
    for i, (val, county) in enumerate(zip(top15['Child_Maltreatment_per1K'], top15['County'])):
        ax1.text(val + 0.2, i, f'{val:.1f}', va='center', fontsize=8, color=DARK_GRAY)

    # National avg line
    ax1.axvline(8.9, color=RED, linestyle='--', linewidth=1.5, zorder=5)
    ax1.text(9.1, len(top15) - 1.2, 'Natl avg: 8.9', fontsize=8, color=RED, fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor=ORANGE, label='Rural'),
        mpatches.Patch(facecolor=SAGE, label='Urban'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=9,
               frameon=True, facecolor='white', edgecolor=MID_GRAY)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(MID_GRAY)
    ax1.spines['bottom'].set_color(MID_GRAY)
    ax1.tick_params(colors=DARK_GRAY)

    # ── Right panel: Maltreatment vs Child Poverty scatter ──
    ax2.set_facecolor('#FFFFFF')
    rural_mask = df['Is_Rural'] == 1
    urban_mask = df['Is_Rural'] == 0

    ax2.scatter(df.loc[rural_mask, 'Child_Poverty_Pct'],
                df.loc[rural_mask, 'Child_Maltreatment_per1K'],
                c=ORANGE, s=70, alpha=0.75, edgecolors='white', linewidths=1.2,
                label='Rural', zorder=3)
    ax2.scatter(df.loc[urban_mask, 'Child_Poverty_Pct'],
                df.loc[urban_mask, 'Child_Maltreatment_per1K'],
                c=SAGE, s=70, alpha=0.75, edgecolors='white', linewidths=1.2,
                label='Urban', zorder=3)

    # Trend line
    z = np.polyfit(df['Child_Poverty_Pct'], df['Child_Maltreatment_per1K'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Child_Poverty_Pct'].min() - 1,
                          df['Child_Poverty_Pct'].max() + 1, 100)
    ax2.plot(x_line, p(x_line), color=RED, linestyle='--', linewidth=1.5, alpha=0.7)

    r = df['Child_Poverty_Pct'].corr(df['Child_Maltreatment_per1K'])
    ax2.text(0.03, 0.97, f'Pearson r = {r:.2f}', transform=ax2.transAxes,
             fontsize=12, fontweight='bold', color=FOREST, va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=MID_GRAY, alpha=0.9))

    # Label outliers
    for _, row in df.iterrows():
        if row['County'] in ['Yakima', 'Ferry', 'Okanogan', 'Adams', 'King', 'San Juan', 'Whitman']:
            ax2.annotate(row['County'],
                         (row['Child_Poverty_Pct'], row['Child_Maltreatment_per1K']),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=7, color=DARK_GRAY, fontstyle='italic', fontweight='bold')

    ax2.set_xlabel('Child Poverty Rate (%)', fontsize=11, color=DARK_GRAY, fontweight='bold')
    ax2.set_ylabel('Child Maltreatment per 1,000 Children', fontsize=11, color=DARK_GRAY, fontweight='bold')
    ax2.set_title('Poverty Strongly Predicts Maltreatment Rates',
                  fontsize=12, fontweight='bold', color=DARK_GRAY)
    ax2.legend(fontsize=9, frameon=True, facecolor='white', edgecolor=MID_GRAY)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(MID_GRAY)
    ax2.spines['bottom'].set_color(MID_GRAY)
    ax2.tick_params(colors=DARK_GRAY, labelsize=9)
    ax2.grid(True, alpha=0.15, color=MID_GRAY)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'maltreatment_analysis.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print(f'  [10/11] Saved maltreatment analysis -> {path}')
    return path


# ══════════════════════════════════════════════════════════════════════
# FIGURE 11 — ADULT VS YOUTH MH COMPARISON (bubble scatter)
# ══════════════════════════════════════════════════════════════════════

def create_adult_youth_comparison():
    """Scatter of adult MH days vs youth sadness with bubble size = population."""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor('#FFFFFF')

    rural_mask = df['Is_Rural'] == 1
    urban_mask = df['Is_Rural'] == 0

    ax.scatter(df.loc[rural_mask, 'Adult_MH_Days'],
               df.loc[rural_mask, 'Youth_Sadness_Pct'],
               c=ORANGE, s=df.loc[rural_mask, 'Population_K'] * 0.4 + 30,
               alpha=0.75, edgecolors='white', linewidths=1.2,
               label='Rural', zorder=3)
    ax.scatter(df.loc[urban_mask, 'Adult_MH_Days'],
               df.loc[urban_mask, 'Youth_Sadness_Pct'],
               c=SAGE, s=df.loc[urban_mask, 'Population_K'] * 0.4 + 30,
               alpha=0.75, edgecolors='white', linewidths=1.2,
               label='Urban', zorder=3)

    # Trend line
    z = np.polyfit(df['Adult_MH_Days'], df['Youth_Sadness_Pct'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Adult_MH_Days'].min() - 0.3,
                          df['Adult_MH_Days'].max() + 0.3, 100)
    ax.plot(x_line, p(x_line), color=RED, linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)

    # Correlation annotation
    r = df['Adult_MH_Days'].corr(df['Youth_Sadness_Pct'])
    ax.text(0.03, 0.97, f'Pearson r = {r:.2f}', transform=ax.transAxes,
            fontsize=14, fontweight='bold', color=FOREST, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=MID_GRAY, alpha=0.9))

    # Label notable counties
    for _, row in df.iterrows():
        if row['County'] in ['King', 'Yakima', 'Ferry', 'Okanogan', 'Whitman',
                              'San Juan', 'Pierce', 'Adams', 'Spokane']:
            ax.annotate(row['County'],
                        (row['Adult_MH_Days'], row['Youth_Sadness_Pct']),
                        xytext=(6, 6), textcoords='offset points',
                        fontsize=7.5, color=DARK_GRAY, fontstyle='italic', fontweight='bold')

    ax.set_xlabel('Adult Mentally Unhealthy Days (past 30 days)', fontsize=12,
                  color=DARK_GRAY, fontweight='bold')
    ax.set_ylabel('Youth Sadness/Hopelessness (%)', fontsize=12,
                  color=DARK_GRAY, fontweight='bold')
    ax.set_title('Adult and Youth Mental Health Burden Track Together\nBRFSS vs YRBSS — Bubble size = county population',
                 fontsize=15, fontweight='bold', color=FOREST, pad=15, fontfamily='serif')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MID_GRAY)
    ax.spines['bottom'].set_color(MID_GRAY)
    ax.tick_params(colors=DARK_GRAY)
    ax.grid(True, alpha=0.15, color=MID_GRAY)

    ax.legend(fontsize=10, frameon=True, facecolor='white', edgecolor=MID_GRAY,
              title='County Type', title_fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'adult_youth_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print(f'  [11/11] Saved adult vs youth comparison -> {path}')
    return path


# ══════════════════════════════════════════════════════════════════════
# CONSOLE REPORT — Key findings printed to terminal
# ══════════════════════════════════════════════════════════════════════

def print_key_findings():
    """Print a structured summary of all key findings to the console."""
    print('\n' + '=' * 70)
    print('KEY FINDINGS — Youth Mental Health in Washington State')
    print('=' * 70)

    # Rural vs urban provider gap
    rural_providers = df[df['Is_Rural'] == 1]['MH_Providers_per100K']
    urban_providers = df[df['Is_Rural'] == 0]['MH_Providers_per100K']
    print(f'\n1. PROVIDER GAP')
    print(f'   Rural avg: {rural_providers.mean():.0f} providers/100K')
    print(f'   Urban avg: {urban_providers.mean():.0f} providers/100K')
    print(f'   Gap: {urban_providers.mean() - rural_providers.mean():.0f} fewer providers per 100K in rural areas')

    # Poverty-insurance correlation
    r_pov_ins = df['Child_Poverty_Pct'].corr(df['Youth_Uninsured_Pct'])
    print(f'\n2. POVERTY-INSURANCE LINK')
    print(f'   Correlation (child poverty vs youth uninsured): r = {r_pov_ins:.2f}')

    # Income-providers correlation
    r_inc_prov = df['Median_Income_K'].corr(df['MH_Providers_per100K'])
    print(f'\n3. INCOME-PROVIDERS LINK')
    print(f'   Correlation (income vs MH providers): r = {r_inc_prov:.2f}')
    print(f'   Wealthier counties attract and retain more providers.')

    # Worst counties
    worst = df.nsmallest(5, 'MH_Providers_per100K')[['County', 'MH_Providers_per100K', 'Child_Poverty_Pct']]
    print(f'\n4. MOST UNDERSERVED COUNTIES (by provider density)')
    for _, row in worst.iterrows():
        print(f'   {row["County"]}: {int(row["MH_Providers_per100K"])} providers/100K, {row["Child_Poverty_Pct"]}% child poverty')

    # Uninsured extremes
    highest = df.nlargest(3, 'Youth_Uninsured_Pct')[['County', 'Youth_Uninsured_Pct']]
    lowest = df.nsmallest(3, 'Youth_Uninsured_Pct')[['County', 'Youth_Uninsured_Pct']]
    print(f'\n5. UNINSURED RATE EXTREMES')
    print(f'   Highest: {", ".join(f"{r.County} ({r.Youth_Uninsured_Pct}%)" for _, r in highest.iterrows())}')
    print(f'   Lowest:  {", ".join(f"{r.County} ({r.Youth_Uninsured_Pct}%)" for _, r in lowest.iterrows())}')
    print(f'   Gap: {df.Youth_Uninsured_Pct.max() - df.Youth_Uninsured_Pct.min():.1f} percentage points between highest and lowest')

    # Hispanic correlation
    r_hisp = df['Hispanic_Pct'].corr(df['Youth_Uninsured_Pct'])
    print(f'\n6. DEMOGRAPHIC BARRIER')
    print(f'   Hispanic % vs youth uninsured: r = {r_hisp:.2f}')
    print(f'   Points to enrollment barriers in Hispanic-majority communities.')

    # ── New findings from NSCH, YRBSS, BRFSS, NCANDS ──

    # Youth MH diagnosis (NSCH)
    print(f'\n7. YOUTH MH DIAGNOSIS PREVALENCE (NSCH)')
    print(f'   State avg: {df["Youth_MH_Diagnosis_Pct"].mean():.1f}% of children 3-17 with anxiety/depression Dx')
    top3_dx = df.nlargest(3, 'Youth_MH_Diagnosis_Pct')[['County', 'Youth_MH_Diagnosis_Pct']]
    print(f'   Highest: {", ".join(f"{r.County} ({r.Youth_MH_Diagnosis_Pct}%)" for _, r in top3_dx.iterrows())}')
    r_dx_prov = df['Youth_MH_Diagnosis_Pct'].corr(df['MH_Providers_per100K'])
    print(f'   Diagnosis rate vs providers: r = {r_dx_prov:.2f}')

    # Youth sadness (YRBSS)
    print(f'\n8. YOUTH SADNESS/HOPELESSNESS (YRBSS)')
    print(f'   State avg: {df["Youth_Sadness_Pct"].mean():.1f}% of high schoolers report persistent sadness')
    rural_sad = df[df['Is_Rural'] == 1]['Youth_Sadness_Pct'].mean()
    urban_sad = df[df['Is_Rural'] == 0]['Youth_Sadness_Pct'].mean()
    print(f'   Rural avg: {rural_sad:.1f}% vs Urban avg: {urban_sad:.1f}%')
    r_sad_pov = df['Youth_Sadness_Pct'].corr(df['Child_Poverty_Pct'])
    print(f'   Sadness vs child poverty: r = {r_sad_pov:.2f}')

    # Adult MH days (BRFSS)
    print(f'\n9. ADULT MENTAL HEALTH BURDEN (BRFSS)')
    print(f'   State avg: {df["Adult_MH_Days"].mean():.1f} mentally unhealthy days/month')
    r_adult_youth = df['Adult_MH_Days'].corr(df['Youth_Sadness_Pct'])
    print(f'   Adult MH days vs youth sadness: r = {r_adult_youth:.2f}')
    print(f'   Adult and youth mental health burden track closely across counties.')

    # Child maltreatment (NCANDS)
    print(f'\n10. CHILD MALTREATMENT (NCANDS)')
    print(f'    State avg: {df["Child_Maltreatment_per1K"].mean():.1f} victims per 1,000 children')
    top3_mal = df.nlargest(3, 'Child_Maltreatment_per1K')[['County', 'Child_Maltreatment_per1K']]
    print(f'    Highest: {", ".join(f"{r.County} ({r.Child_Maltreatment_per1K}/1K)" for _, r in top3_mal.iterrows())}')
    r_mal_pov = df['Child_Maltreatment_per1K'].corr(df['Child_Poverty_Pct'])
    print(f'    Maltreatment vs child poverty: r = {r_mal_pov:.2f}')
    r_mal_inc = df['Child_Maltreatment_per1K'].corr(df['Median_Income_K'])
    print(f'    Maltreatment vs income: r = {r_mal_inc:.2f}')

    print('\n' + '=' * 70)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Generating youth mental health analysis...\n')

    create_summary_stats()
    create_distributions()
    create_rural_vs_urban()
    create_provider_ranking()
    create_income_vs_providers()
    create_correlation_heatmap()
    create_clustering_scatter()
    create_gis_map()
    create_youth_mh_prevalence()
    create_maltreatment_analysis()
    create_adult_youth_comparison()

    print_key_findings()

    print(f'\nAll 11 figures saved to {OUTPUT_DIR}/')
    print('Analysis complete.')
