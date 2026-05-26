# Youth Mental Health in Washington State

**Author:** Waleed Adawi · **Year:** 2026
**Stack:** Python 3 · pandas · matplotlib · seaborn
**Data:** U.S. Census ACS · SAMHSA NSDUH · HRSA AHRF · CDC YRBSS/BRFSS · MHA · USDA RUCC

## Overview

Mapping access gaps, provider shortages, and socioeconomic risk factors across all 39 Washington counties to identify where children are falling through the cracks.

Over 126,000 children in Washington State have a diagnosed mental health condition but receive no treatment. The state ranks 48th nationally in youth mental health outcomes, and 70.6% of caregivers report difficulty accessing care — nearly 16 points above the national average. These aren't just statistics; they represent kids in specific counties where providers are scarce, poverty rates are high, and insurance coverage falls short. This project maps exactly where those gaps are and which factors drive them, giving policymakers and advocates the data to target interventions where they're needed most.

**Skills demonstrated:** federal health data sourcing · county-level demographic analysis · K-means clustering (from scratch) · Pearson correlation · hex cartogram visualization · multi-dataset integration · publication-quality data visualization

## Key Findings

- **Rural counties have nearly half the mental health providers of urban ones.** Rural areas average 127 providers per 100K residents compared to 225 in urban counties — a gap that leaves 26 of 39 counties critically underserved.
- **Child poverty is the strongest predictor of youth uninsured rates (r = 0.70).** Counties with higher child poverty consistently have more uninsured children, but income alone doesn't explain access — some high-poverty counties maintain low uninsured rates through targeted programs.
- **A cluster of 24 counties shares a high-risk profile.** K-means analysis identified a group of predominantly rural counties averaging 20.9% child poverty, only 113 MH providers per 100K, and 5.3% youth uninsured rates — nearly double the rate of the lowest-risk cluster.
- **The provider gap dwarfs the insurance gap.** Even in counties where uninsured rates are low, provider shortages mean children with coverage still can't access care. King County has 380 providers per 100K; Garfield County has 40.

## Visualizations

### Fig 1 — Correlation Heatmap: What Drives Youth Mental Health Access?

![Correlation Heatmap](outputs/heatmap.png)

Pearson correlation matrix across seven county-level indicators. The strongest relationship in the data is between median household income and mental health provider availability (r = 0.79) — wealthier counties attract and retain more providers. Child poverty correlates strongly with youth uninsured rates (r = 0.70) and inversely with provider access (r = -0.66), confirming that the counties with the greatest need tend to have the fewest resources. Hispanic population percentage also correlates with higher uninsured rates (r = 0.68), pointing to demographic-specific barriers in insurance enrollment.

### Fig 2 — K-Means Clustering: County Risk Profiles

![Clustering Scatter Plot](outputs/clustering.png)

K-means clustering (k=3) groups Washington's 39 counties into three distinct profiles based on child poverty, youth uninsured rates, provider density, and median income. The **Lower Risk** cluster (7 counties including King, Clark, and Pierce) averages 11.9% child poverty and 260 MH providers per 100K. The **Higher Risk** cluster (8 counties) sits in a moderate range. The largest group — **24 counties** classified as Mixed/Urban — averages 20.9% child poverty with only 113 providers per 100K, representing the bulk of underserved communities. Bubble size reflects county population, showing that while King County is an outlier in both size and access, most of the state lives in counties with far fewer resources.

### Fig 3 — County-Level Hex Cartogram: Youth Uninsured Rates

![GIS Hex Map](outputs/gis_map.png)

Hex cartogram of all 39 counties colored by youth uninsured rate (children under 19). The geographic pattern is clear: eastern and rural Washington carries the highest uninsured rates, with Skamania (8.9%), Adams (8.2%), Douglas (7.3%), and Franklin (7.3%) at the top. Western urban counties — King (2.4%), San Juan (2.5%), and Island (2.8%) — cluster at the low end. The statewide average sits at 4.5%, but the 4.5-fold gap between the highest and lowest counties shows that the state average masks deep regional inequality.

## Data Sources

| # | Dataset | Agency | Year | Link |
|---|---------|--------|------|------|
| 1 | National Survey of Children's Health (NSCH) | HRSA / MCHB | 2022 | [data.census.gov](https://data.census.gov) |
| 2 | Youth Risk Behavior Surveillance System (YRBSS) | CDC | 2023 | [cdc.gov/yrbs](https://www.cdc.gov/yrbs) |
| 3 | National Survey on Drug Use and Health (NSDUH) | SAMHSA | 2022–2023 | [samhsa.gov/data](https://www.samhsa.gov/data) |
| 4 | Behavioral Risk Factor Surveillance System (BRFSS) | CDC | 2023 | [cdc.gov/brfss](https://www.cdc.gov/brfss) |
| 5 | American Community Survey (ACS) 5-Year Estimates | U.S. Census Bureau | 2019–2023 | [data.census.gov](https://data.census.gov) |
| 6 | Small Area Health Insurance Estimates (SAHIE) | U.S. Census Bureau | 2022 | [census.gov/sahie](https://www.census.gov/programs-surveys/sahie.html) |
| 7 | Mental Health America (MHA) State Rankings | MHA | 2024 | [mhanational.org](https://www.mhanational.org) |
| 8 | Area Health Resources Files (AHRF) | HRSA | 2023 | [data.hrsa.gov](https://data.hrsa.gov) |
| 9 | Rural-Urban Continuum Codes (RUCC) | USDA ERS | 2023 | [ers.usda.gov](https://www.ers.usda.gov) |
| 10 | SAIPE (Small Area Income and Poverty Estimates) | U.S. Census Bureau | 2022 | [census.gov/saipe](https://www.census.gov/programs-surveys/saipe.html) |
| 11 | National Child Abuse and Neglect Data System (NCANDS) | ACF / HHS | 2022 | [acf.hhs.gov](https://www.acf.hhs.gov) |

## Methodology

This analysis integrates 11 publicly available federal datasets to examine youth mental health trends and service access at the county level. All data processing and visualization were built in Python using pandas, matplotlib, and seaborn.

**Why these techniques:**

- **Pearson correlation** was used to quantify relationships between variables before making any causal claims — identifying which factors actually move together and which don't (e.g., the surprisingly weak direct link between income and uninsured rates at r = -0.28)
- **K-means clustering** was chosen to let the data reveal natural groupings among counties rather than imposing arbitrary thresholds for "high risk" vs. "low risk" — the algorithm identified three meaningfully distinct profiles
- **Hex cartogram** was selected over a standard choropleth because Washington's counties vary enormously in geographic size — a traditional map would give visual weight to large, sparsely populated eastern counties while hiding small, densely populated western ones

**Data decisions:**

- ACS 5-year estimates (2019–2023) used for stable county-level demographic baselines
- NSDUH 2-year pooled estimates (2022–2023) used for substance use and mental health prevalence
- RUCC codes applied to classify counties as metropolitan vs. non-metropolitan
- K-means implemented from scratch (not sklearn) to demonstrate algorithmic understanding

## Repo Structure

```
youth-mental-health-wa/
├── README.md
├── LICENSE
├── .gitignore
├── bh_analysis.py           ← Main analysis script (runs end-to-end)
├── requirements.txt          ← Python dependencies
├── data/
│   └── sources.md            ← Detailed source documentation
└── outputs/
    ├── heatmap.png           ← Correlation matrix
    ├── clustering.png        ← K-means county clustering
    └── gis_map.png           ← Youth uninsured hex cartogram
```

## Running the Analysis

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full analysis
python bh_analysis.py
```

All data is embedded in the analysis script — no external downloads required. Output charts are saved to the `outputs/` directory.

## Relevance to WSCC Reporting

Washington State Community Connectors (WSCC) works to bridge gaps in health and social services across Washington's rural and underserved communities. The analysis in this project maps directly to the kind of evidence-based work that supports WSCC program planning and advocacy:

- **Figs 1 and 3** identify which counties have the greatest unmet need for youth mental health services — providing the geographic targeting data that community health organizations use to allocate outreach resources and justify grant applications.
- **Fig 2** segments counties into risk profiles that align with how state agencies and nonprofits prioritize intervention — distinguishing between counties that need more providers, more insurance enrollment support, or both.
- The **correlation analysis** quantifies the relationship between poverty, insurance coverage, and provider access — the three factors that WSCC and similar organizations track when measuring whether their programs are reaching the populations most at risk.

This project demonstrates the end-to-end data work involved in community health analytics: sourcing authoritative federal datasets, integrating multiple data streams at the county level, applying statistical and machine learning techniques to identify patterns, and producing outputs that directly support program planning and grant reporting.
