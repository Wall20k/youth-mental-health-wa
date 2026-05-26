# Youth Mental Health Access Analysis — Washington State

County-level analysis of youth mental health access gaps, provider shortages, and socioeconomic risk factors across all 39 Washington counties.

**Author:** Waleed Adawi · **Year:** 2026  
**Stack:** Python 3 · pandas · NumPy · Matplotlib · Seaborn

---

## Overview

### The Problem

Rural and low-income communities in Washington State face disproportionate barriers to youth mental health care. Provider shortages, high uninsured rates among children, and poverty create compounding access gaps that vary dramatically from county to county.

### Why It Matters

Washington has 39 counties spanning dense urban centers like King County (2.3M residents, 380 MH providers per 100K) to remote rural areas like Garfield County (2,200 residents, 40 providers per 100K). Understanding where access breaks down — and what drives those gaps — is the first step toward equitable resource allocation.

### Objective

Quantify the relationships between socioeconomic indicators (income, poverty, insurance coverage) and mental health provider availability at the county level, identify the most underserved communities, and classify counties into risk profiles using unsupervised clustering.

---

## Methodology

### Approach

This analysis uses a single self-contained Python script with all county-level data embedded directly. No external CSV files or databases are required — the data dictionary is built into `Code.py` for full reproducibility.

The analytical pipeline includes descriptive statistics across all 39 counties, distribution analysis of key variables, rural vs. urban disparity comparisons, Pearson correlation analysis between all indicator pairs, a manually implemented K-means clustering algorithm (k=3, no scikit-learn) to classify counties into risk profiles, and a hex cartogram for geographic visualization.

### Tools

| Tool | Purpose |
|------|---------|
| pandas | Data manipulation and summary statistics |
| NumPy | Array operations and K-means implementation |
| Matplotlib | All figure generation (8 outputs) |
| Seaborn | Correlation heatmap styling |

No external machine learning libraries (scikit-learn, scipy, etc.) are used. The K-means algorithm is implemented from scratch using NumPy for educational transparency.

---

## Data Processing

### Data Sources

County-level indicators were compiled from four federal sources:

| Source | Tables / Dataset | Variables |
|--------|-----------------|-----------|
| U.S. Census ACS 5-Year (2019–2023) | S2701, S1701, S1901, B01003, B03003 | Youth uninsured rate, child poverty, median income, population, Hispanic % |
| HRSA Area Health Resource File (2023) | AHRF | Mental health provider rate per 100K |
| SAMHSA NSDUH (2022–2023) | State-level estimates | Contextual prevalence benchmarks |
| USDA Rural-Urban Continuum Codes (2023) | RUCC | Metro/non-metro county classification |

A full list of all 11 reference datasets is available in [`data/sources.md`](data/sources.md).

### Variables

| Variable | Description | Source Table |
|----------|-------------|-------------|
| `Youth_Uninsured_Pct` | % of residents under 19 without health insurance | ACS S2701 |
| `Child_Poverty_Pct` | % of residents under 18 below poverty line | ACS S1701 |
| `Median_Income_K` | Median household income in thousands ($K) | ACS S1901 |
| `Overall_Poverty_Pct` | % of all residents below poverty line | ACS S1701 |
| `Is_Rural` | Binary rural classification (USDA RUCC: 1 = non-metro) | USDA RUCC |
| `Population_K` | County population in thousands | ACS B01003 |
| `MH_Providers_per100K` | Licensed mental health providers per 100K residents | HRSA AHRF |
| `Hispanic_Pct` | % Hispanic/Latino population | ACS B03003 |

### Data Evaluation

All data comes from federally administered surveys with established methodologies. ACS estimates use 5-year pooling (2019–2023) for county-level reliability, which is standard practice for small-area estimation. RUCC codes provide a binary metro/non-metro split — a simplification that trades granularity for interpretability across just 39 observations.

### Cleaning

County data was entered directly from source tables and cross-verified. No imputation was needed — all 39 counties have complete records across all 8 variables. The `Rural_Label` column is derived from `Is_Rural` for visualization purposes.

---

## Exploratory Data Analysis

### Summary Statistics

![Summary Statistics](outputs/summary_stats.png)

The summary table shows considerable variation across Washington's 39 counties. Youth uninsured rates range from 0.0% (Garfield) to 8.9% (Skamania), while mental health provider density spans a 9.5x gap between the least-served county (Garfield, 40 per 100K) and the best-served (King, 380 per 100K). Median household income ranges from $35,800 (Whitman) to $106,300 (King).

### Distributions

![Distributions](outputs/distributions.png)

Youth uninsured rates are right-skewed, with most counties falling between 3–6% but a handful of agricultural counties pulling the tail above 7%. Child poverty shows a wider, more uniform spread. Provider density is bimodal — urban counties cluster above 200 per 100K while rural counties cluster below 150.

### Rural vs. Urban Disparities

![Rural vs Urban](outputs/rural_vs_urban.png)

Box plots reveal consistent disadvantage across rural counties (26 of 39). Rural counties average fewer mental health providers, higher child poverty, and higher youth uninsured rates compared to the 13 urban counties. The provider gap is the most pronounced disparity.

### Provider Ranking

![Provider Ranking](outputs/top_bottom_providers.png)

A full ranking of all 39 counties by mental health provider density. The five most underserved counties — Garfield (40), Columbia (50), Wahkiakum (55), Skamania (70), and Ferry (75) — are all rural with populations under 13,000. The top five — King (380), Snohomish (290), San Juan (285), Whatcom (270), and Kitsap (265) — are predominantly urban or high-income.

### Income vs. Provider Access

![Income vs Providers](outputs/income_vs_providers.png)

The strongest relationship in the dataset: median household income correlates with mental health provider density at r = 0.79. Bubble size represents county population. The scatter plot reveals that wealthier counties attract and retain more providers, while low-income rural counties face compounding disadvantages. King County is a clear outlier with both the highest income and highest provider rate.

### Correlation Heatmap

![Correlation Heatmap](outputs/heatmap.png)

The full correlation matrix confirms several expected relationships. Income and providers show the strongest positive link (r = 0.79). Child poverty and youth uninsured rates are positively correlated. Rural classification is negatively associated with both income and provider availability. Hispanic population percentage correlates with youth uninsured rates, suggesting enrollment barriers in Hispanic-majority communities.

### K-Means Clustering

![Clustering](outputs/clustering.png)

Counties are clustered into three risk profiles using a from-scratch K-means implementation (k=3) on standardized values of youth uninsured rate, child poverty, provider density, and median income. The three groups separate into lower-risk (low poverty, high providers), higher-risk (high poverty, low providers), and mixed/urban profiles. Bubble size represents population.

### Geographic Distribution

![GIS Map](outputs/gis_map.png)

A hex cartogram showing youth uninsured rates across all 39 counties. Eastern agricultural counties (Adams, Skamania, Grant, Franklin, Douglas) show the highest rates, while western urban counties (King, Island, San Juan) have the lowest. The geographic pattern closely mirrors the income and provider gradients seen in earlier figures.

---

## Key Findings

1. **Rural-urban provider gap.** Rural counties average significantly fewer mental health providers per 100K residents than urban counties — a structural shortage affecting 26 of 39 counties.

2. **Income predicts access.** Median household income and provider density correlate at r = 0.79, the strongest relationship in the dataset. Wealthier counties attract and retain more providers.

3. **Poverty compounds risk.** Child poverty is positively correlated with youth uninsured rates, meaning the counties where children most need coverage are least likely to have it.

4. **Five critical counties.** Garfield (40 providers/100K), Columbia (50), Wahkiakum (55), Skamania (70), and Ferry (75) represent the most acute access deserts — all rural, all with populations under 13,000.

5. **Uninsured rate extremes.** Skamania (8.9%), Adams (8.2%), and Franklin/Douglas (7.3%) have the highest youth uninsured rates, while Garfield (0.0%), King (2.4%), and San Juan (2.5%) have the lowest.

6. **Demographic barriers.** Hispanic population percentage correlates with youth uninsured rates, pointing to enrollment barriers in communities like Adams (69% Hispanic, 8.2% uninsured) and Franklin (56% Hispanic, 7.3% uninsured).

---

## Recommendations

1. **Target provider recruitment in the five critical counties** — Garfield, Columbia, Wahkiakum, Skamania, and Ferry lack the population base to sustain private-practice models. Telehealth subsidies or state-funded provider rotations could bridge the gap.

2. **Expand insurance outreach in high-Hispanic communities** — The correlation between Hispanic population share and youth uninsured rates suggests language and documentation barriers to enrollment, not lack of eligibility. Bilingual navigators and community-based enrollment drives should be prioritized in Adams, Franklin, Grant, and Yakima counties.

3. **Use clustering to prioritize funding** — The three risk profiles identified by K-means can guide tiered resource allocation: higher-risk counties need crisis-level intervention, mixed counties need targeted support, and lower-risk counties need maintenance funding.

4. **Invest in rural infrastructure** — The consistent rural disadvantage across providers, poverty, and insurance coverage reflects systemic underinvestment. Broadband expansion for telehealth, loan forgiveness for rural mental health professionals, and mobile crisis teams would address root causes.

---

## Repository Structure

```
youth-mental-health-wa/
├── Code.py                          # Full analysis script (all data embedded)
├── data/
│   └── sources.md                   # Dataset documentation and download links
├── outputs/
│   ├── summary_stats.png            # Fig 1: Summary statistics table
│   ├── distributions.png            # Fig 2: Variable distributions
│   ├── rural_vs_urban.png           # Fig 3: Rural vs urban box plots
│   ├── top_bottom_providers.png     # Fig 4: Provider density ranking
│   ├── income_vs_providers.png      # Fig 5: Income vs providers (r = 0.79)
│   ├── heatmap.png                  # Fig 6: Correlation matrix
│   ├── clustering.png               # Fig 7: K-means risk profiles
│   └── gis_map.png                  # Fig 8: Hex cartogram
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn
python Code.py
```

All 8 figures are saved to `outputs/`. No external data files are needed — the county data is embedded in the script.

---

## Copyright

© 2026 Waleed Adawi. All rights reserved.

This project and its contents are shared for portfolio and educational purposes. Data sourced from U.S. Census Bureau (ACS), HRSA, SAMHSA, and USDA — all publicly available federal datasets. See [`data/sources.md`](data/sources.md) for full citations.

Licensed under the [MIT License](LICENSE).
