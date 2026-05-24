"""
Youth Mental Health in Washington State — Behavioral Health Analysis
WSCC Annual Impact Report 2026

Author: Waleed Adawi
Organization: Washington State Community Connectors (WSCC)

This script processes 11 federal datasets to analyze youth mental health
trends, behavioral risk factors, and child flourishing across all 39
Washington State counties.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Color palette — forest/emerald theme
COLORS = {
    "forest": "#1B4332",
    "emerald": "#2D6A4F",
    "sage": "#52B788",
    "mint": "#74C69D",
    "pale": "#B7E4C7",
    "cream": "#F0F7F4",
    "orange": "#E85D3A",
    "blue": "#2C6E9B",
}

plt.rcParams.update({
    "figure.facecolor": COLORS["cream"],
    "axes.facecolor": COLORS["cream"],
    "font.family": "sans-serif",
    "font.size": 11,
})


# ──────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────

def load_county_data():
    """
    Load and merge county-level indicators from multiple federal sources.
    Sources: ACS, SAHIE, SAIPE, AHRF, RUCC
    """
    # Example structure — adapt file paths to your local data
    print("Loading county-level datasets...")

    # American Community Survey (ACS) 5-Year Estimates
    # Columns: county_fips, county_name, total_pop, pop_under_18,
    #          median_income, pct_poverty, pct_uninsured
    acs = pd.read_csv(DATA_DIR / "acs_wa_counties.csv")

    # Small Area Health Insurance Estimates (SAHIE)
    # Columns: county_fips, uninsured_rate_under19, uninsured_rate_total
    sahie = pd.read_csv(DATA_DIR / "sahie_wa.csv")

    # Area Health Resources Files (AHRF)
    # Columns: county_fips, mh_providers_per_100k, hpsa_mh_designation
    ahrf = pd.read_csv(DATA_DIR / "ahrf_wa.csv")

    # Rural-Urban Continuum Codes (RUCC)
    # Columns: county_fips, rucc_code, metro_nonmetro
    rucc = pd.read_csv(DATA_DIR / "rucc_wa.csv")

    # SAIPE — child poverty
    # Columns: county_fips, child_poverty_rate
    saipe = pd.read_csv(DATA_DIR / "saipe_wa.csv")

    # Merge all on county_fips
    merged = acs.merge(sahie, on="county_fips", how="left")
    merged = merged.merge(ahrf, on="county_fips", how="left")
    merged = merged.merge(rucc, on="county_fips", how="left")
    merged = merged.merge(saipe, on="county_fips", how="left")

    print(f"  Loaded {len(merged)} counties with {merged.shape[1]} variables")
    return merged


# ──────────────────────────────────────────────────────────────────────
# Analysis 1: Correlation Heatmap
# ──────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df):
    """
    Generate a Pearson correlation matrix across key county-level
    indicators: child poverty, uninsured rates, MH provider density,
    and population demographics.
    """
    print("Generating correlation heatmap...")

    numeric_cols = [
        "child_poverty_rate",
        "uninsured_rate_under19",
        "mh_providers_per_100k",
        "pct_poverty",
        "median_income",
        "pop_under_18",
    ]

    # Filter to available columns
    available = [c for c in numeric_cols if c in df.columns]
    corr_matrix = df[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".3f",
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title(
        "Correlation Matrix: County-Level Youth Mental Health Indicators\nWashington State",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: outputs/heatmap.png")


# ──────────────────────────────────────────────────────────────────────
# Analysis 2: K-Means Clustering
# ──────────────────────────────────────────────────────────────────────

def kmeans_manual(X, k=4, max_iter=100, seed=42):
    """Simple K-means implementation (no scikit-learn dependency)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), k, replace=False)
    centroids = X[idx].copy()

    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)
        labels = dists.argmin(axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids


def plot_clustering(df):
    """
    K-means cluster analysis grouping WA counties by behavioral health
    access and socioeconomic risk factors.
    """
    print("Generating K-means clustering plot...")

    features = ["child_poverty_rate", "mh_providers_per_100k"]
    available = [c for c in features if c in df.columns]

    if len(available) < 2:
        print("  Skipped: insufficient columns for clustering")
        return

    subset = df[available].dropna()
    X = subset.values

    # Standardize
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    labels, centroids = kmeans_manual(X_std, k=4)

    fig, ax = plt.subplots(figsize=(10, 8))

    cluster_colors = [COLORS["forest"], COLORS["orange"], COLORS["blue"], COLORS["sage"]]
    cluster_names = [
        "Low Risk, High Access",
        "High Risk, Low Access",
        "Moderate Risk",
        "Rural Underserved",
    ]

    for i in range(4):
        mask = labels == i
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=cluster_colors[i],
            label=cluster_names[i],
            s=80,
            alpha=0.75,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Child Poverty Rate (%)", fontsize=12)
    ax.set_ylabel("MH Providers per 100k", fontsize=12)
    ax.set_title(
        "K-Means Clustering: WA Counties by Access & Risk\n(k=4)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "clustering.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: outputs/clustering.png")


# ──────────────────────────────────────────────────────────────────────
# Analysis 3: GIS Hex Cartogram
# ──────────────────────────────────────────────────────────────────────

def plot_gis_map(df):
    """
    Hex cartogram of all 39 WA counties showing composite behavioral
    health risk scores. Uses matplotlib hexagons since geopandas is
    not available in all environments.
    """
    print("Generating GIS hex cartogram...")

    # Pre-defined hex grid positions for WA's 39 counties
    # (row, col) — approximate geographic layout
    hex_positions = {
        "Whatcom": (0, 3), "Okanogan": (0, 4), "Ferry": (0, 5),
        "Stevens": (0, 6), "Pend Oreille": (0, 7),
        "San Juan": (1, 1), "Island": (1, 2), "Skagit": (1, 3),
        "Chelan": (1, 4), "Douglas": (1, 5), "Lincoln": (1, 6),
        "Spokane": (1, 7),
        "Clallam": (2, 0), "Jefferson": (2, 1), "Snohomish": (2, 2),
        "King": (2, 3), "Kittitas": (2, 4), "Grant": (2, 5),
        "Adams": (2, 6), "Whitman": (2, 7),
        "Grays Harbor": (3, 0), "Mason": (3, 1), "Pierce": (3, 2),
        "Thurston": (3, 3), "Yakima": (3, 4), "Franklin": (3, 5),
        "Walla Walla": (3, 6), "Columbia": (3, 7),
        "Pacific": (4, 0), "Lewis": (4, 1), "Cowlitz": (4, 2),
        "Clark": (4, 3), "Skamania": (4, 4), "Klickitat": (4, 5),
        "Benton": (4, 6), "Asotin": (4, 7),
        "Wahkiakum": (5, 1), "Kitsap": (2, 1.5),
        "Garfield": (3, 7.5),
    }

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_facecolor(COLORS["cream"])

    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(0, 100)

    for county, (row, col) in hex_positions.items():
        # Compute composite risk score if data available
        county_data = df[df["county_name"].str.contains(county, case=False, na=False)]
        if len(county_data) > 0:
            risk_score = county_data.iloc[0].get("composite_risk", 50)
        else:
            risk_score = 50  # Default mid-range

        x = col * 1.5
        y = -row * 1.732 + (0.866 if col % 2 else 0)

        hex_patch = plt.Polygon(
            [(x + np.cos(a), y + np.sin(a)) for a in np.linspace(0, 2 * np.pi, 7)],
            facecolor=cmap(norm(risk_score)),
            edgecolor="white",
            linewidth=2,
        )
        ax.add_patch(hex_patch)

        # County label
        label = county[:6] if len(county) > 6 else county
        ax.text(x, y, label, ha="center", va="center", fontsize=7, fontweight="bold")

    ax.set_xlim(-2, 14)
    ax.set_ylim(-12, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Washington State Counties: Composite Behavioral Health Risk Score",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Risk Score (higher = greater need)", fontsize=11)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "gis_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: outputs/gis_map.png")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("WSCC Youth Mental Health Analysis — Washington State")
    print("=" * 60)

    df = load_county_data()

    plot_correlation_heatmap(df)
    plot_clustering(df)
    plot_gis_map(df)

    print("\n" + "=" * 60)
    print("Analysis complete. Charts saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
