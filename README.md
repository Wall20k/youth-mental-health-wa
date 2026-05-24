# Youth Mental Health in Washington State
### WSCC Annual Impact Report — 2026

**Tools:** Python · pandas · matplotlib · seaborn  
**Data:** 11 federal datasets (SAMHSA, HRSA, U.S. Census Bureau, USDA, CDC, MHA)  
**Author:** Waleed Adawi · Data Analysis Intern, WSCC · 2026

---

## Research Question

What is the current experience for parents and caregivers navigating Washington's youth mental health services?

---

## Key Findings

- **70.6%** of WA caregivers report difficulty accessing youth mental health care — 15.6 percentage points above the national average (55.0%)
- **~126,370** WA children have a diagnosed mental health condition but are receiving no treatment
- **47.8%** treatment gap — nearly half of children with a diagnosed condition receive no care
- Washington ranks **48th out of 51** states on youth mental health (MHA 2024)
- **3× gap** in uninsured rates between highest and lowest racial groups
- Correlation between child poverty and uninsured rates: r = 0.050 — challenging income-only policy assumptions

---

## Data Sources

| # | Dataset | Agency | Year | Link |
|---|---------|--------|------|------|
| 1 | National Survey of Children's Health (NSCH) | HRSA / MCHB | 2022 | [data.census.gov](https://data.census.gov) |
| 2 | Youth Risk Behavior Surveillance System (YRBSS) | CDC | 2023 | [cdc.gov/yrbs](https://www.cdc.gov/yrbs/) |
| 3 | National Survey on Drug Use and Health (NSDUH) | SAMHSA | 2022–2023 | [samhsa.gov/data](https://www.samhsa.gov/data/) |
| 4 | Behavioral Risk Factor Surveillance System (BRFSS) | CDC | 2023 | [cdc.gov/brfss](https://www.cdc.gov/brfss/) |
| 5 | American Community Survey (ACS) 5-Year Estimates | U.S. Census Bureau | 2019–2023 | [data.census.gov](https://data.census.gov) |
| 6 | Small Area Health Insurance Estimates (SAHIE) | U.S. Census Bureau | 2022 | [census.gov/sahie](https://www.census.gov/programs-surveys/sahie.html) |
| 7 | Mental Health America (MHA) State Rankings | MHA | 2024 | [mhanational.org](https://www.mhanational.org/) |
| 8 | Area Health Resources Files (AHRF) | HRSA | 2023 | [data.hrsa.gov](https://data.hrsa.gov/) |
| 9 | Rural-Urban Continuum Codes (RUCC) | USDA ERS | 2023 | [ers.usda.gov](https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/) |
| 10 | SAIPE (Small Area Income and Poverty Estimates) | U.S. Census Bureau | 2022 | [census.gov/saipe](https://www.census.gov/programs-surveys/saipe.html) |
| 11 | National Child Abuse and Neglect Data System (NCANDS) | ACF / HHS | 2022 | [acf.hhs.gov](https://www.acf.hhs.gov/cb/data-research/child-maltreatment) |

---

## Analysis Themes

1. **Youth Mental Health Prevalence** — depression, anxiety, and behavioral health indicators across WA counties
2. **Access to Mental Health Services** — provider shortages, HPSA designations, and geographic barriers
3. **Insurance & Financial Barriers** — uninsured rates by county, age group, and race/ethnicity
4. **Regional Differences (Rural vs. Urban)** — RUCC-based comparisons of access and outcomes
5. **Lived Experience & Caregiver Perspectives** — NSCH-derived measures of family navigation burden
6. **Equity Gaps** — racial, ethnic, and socioeconomic disparities in treatment access

---

## Visualizations

### Correlation Heatmap
![Correlation Heatmap](outputs/heatmap.png)  
*Pearson correlation matrix across county-level indicators — child poverty, uninsured rates, provider access, and mental health prevalence.*

### K-Means Clustering
![K-Means Clustering](outputs/clustering.png)  
*K-means cluster analysis grouping WA counties by behavioral health access and socioeconomic risk factors.*

### County-Level GIS Map
![GIS Map](outputs/gis_map.png)  
*Hex cartogram of all 39 WA counties showing composite behavioral health risk scores.*

---

## Methodology

This analysis integrates 11 publicly available federal datasets to examine youth mental health trends and service access across Washington State's 39 counties. Data processing and visualization were performed in Python using pandas, matplotlib, and seaborn.

**Key methodological decisions:**
- ACS 5-year estimates (2019–2023) used for stable county-level demographic baselines
- NSDUH 2-year pooled estimates (2022–2023) used for substance use and mental health prevalence
- RUCC codes applied to classify counties as metropolitan vs. non-metropolitan
- NSCH child flourishing measured using 3 criteria: curiosity about learning, resilience when faced with challenges, and emotional regulation
- K-means clustering (k=4) applied to identify county typologies based on access and risk indicators
- Pearson correlation coefficients computed across all numeric county-level variables

---

## Repository Structure

```
youth-mental-health-wa/
├── README.md
├── bh_analysis.py          ← Main Python analysis script
├── requirements.txt        ← Python dependencies
├── data/                   ← Source dataset links (public aggregate data)
│   └── sources.md
├── outputs/                ← All chart PNGs from the analysis
│   ├── heatmap.png
│   ├── clustering.png
│   └── gis_map.png
└── methodology.md          ← Detailed methodology notes
```

---

## Note

> The full WSCC Annual Impact Report PDF is available upon request. The data used in this project is entirely from public, aggregate federal sources — no personally identifiable information is included.

---

## License

This project was produced during an internship with the Washington State Community Connectors (WSCC). Code is shared for portfolio and educational purposes.
