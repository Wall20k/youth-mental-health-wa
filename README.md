# Youth Mental Health in Washington State

Over 126,000 children in Washington have a diagnosed mental health condition but receive no treatment. This project maps exactly where those gaps are and what's driving them.

**Author:** Waleed Adawi · **Year:** 2026
**Stack:** Python 3 · pandas · matplotlib · seaborn
**Data:** U.S. Census ACS · SAMHSA NSDUH · HRSA AHRF · CDC YRBSS/BRFSS · MHA · USDA RUCC

## Overview

### The Problem

Washington State ranks 48th nationally in youth mental health outcomes. 70.6% of caregivers report difficulty accessing care — nearly 16 points above the national average. But these numbers don't tell you *where* the problem is worst or *why* it varies so much from one county to the next.

### Why It Matters

The state average masks deep regional inequality. A child in King County has access to 380 mental health providers per 100K residents. A child in Garfield County has access to 40. That's not a gap — it's a different reality. And it's not random: the counties with the fewest providers also tend to have the highest poverty rates and the lowest incomes. Understanding this pattern is the first step toward directing resources where they'll do the most good.

### Objective

Map youth mental health access, provider shortages, and socioeconomic risk factors across all 39 Washington counties to identify which communities are falling through the cracks — and what factors predict it.

## Methodology

This analysis follows a structured approach from data collection through recommendations:

1. **Data collection** — Integrated 11 publicly available federal datasets covering insurance coverage, poverty, income, provider availability, demographics, and rural classification at the county level.
2. **Data processing** — Standardized all variables to a common unit of analysis (county), aligned time periods (ACS 5-year estimates 2019–2023, NSDUH 2022–2023), and classified counties as rural or urban using USDA Rural-Urban Continuum Codes.
3. **Exploratory data analysis** — Computed summary statistics and distributions across all 39 counties to understand the spread and identify outliers. Compared rural vs. urban counties across key metrics.
4. **Correlation analysis** — Used Pearson correlation to quantify relationships between all variable pairs before making any causal claims — identifying which factors actually move together and which don't.
5. **Cluster analysis** — Applied K-means clustering (implemented from scratch, not sklearn) to group counties into risk profiles based on four indicators: child poverty, youth uninsured rates, provider density, and median income. K=3 was chosen because it produced the most interpretable separation.
6. **Geographic mapping** — Built a hex cartogram to visualize county-level uninsured rates. Hex cartograms were chosen over standard choropleth maps because Washington's counties vary enormously in geographic size — a traditional map gives visual weight to large, sparsely populated eastern counties while hiding small, densely populated western ones.

**Tools used:** Python 3, pandas (data manipulation), matplotlib (visualization), seaborn (heatmap), NumPy (K-means implementation and statistical computation).

## Data Processing

### Sources

| # | Dataset | Agency | Year |
|---|---------|--------|------|
| 1 | American Community Survey (ACS) 5-Year | U.S. Census Bureau | 2019–2023 |
| 2 | Small Area Health Insurance Estimates (SAHIE) | U.S. Census Bureau | 2022 |
| 3 | SAIPE (Poverty Estimates) | U.S. Census Bureau | 2022 |
| 4 | National Survey of Children's Health (NSCH) | HRSA / MCHB | 2022 |
| 5 | National Survey on Drug Use and Health (NSDUH) | SAMHSA | 2022–2023 |
| 6 | Youth Risk Behavior Surveillance System (YRBSS) | CDC | 2023 |
| 7 | Behavioral Risk Factor Surveillance (BRFSS) | CDC | 2023 |
| 8 | Area Health Resources Files (AHRF) | HRSA | 2023 |
| 9 | Mental Health America State Rankings | MHA | 2024 |
| 10 | Rural-Urban Continuum Codes (RUCC) | USDA ERS | 2023 |
| 11 | National Child Abuse and Neglect Data System | ACF / HHS | 2022 |

### What was collected

Seven county-level indicators for all 39 Washington counties: youth uninsured rate (children under 19), child poverty rate (under 18), median household income, overall poverty rate, mental health provider density (per 100K residents), Hispanic/Latino population percentage, and total population. Counties were also classified as rural or urban based on USDA codes.

### Data evaluation and cleaning

- ACS 5-year estimates were used instead of 1-year estimates for stable county-level baselines (small counties have unreliable single-year data).
- NSDUH 2-year pooled estimates (2022–2023) were used because single-year state-level data doesn't break down to the county level.
- Columbia County reported a 0.0% youth uninsured rate — this is a real data point (very small county, population ~3,900) rather than a data error, so it was retained.
- All 39 counties have complete data across all seven indicators. No imputation was necessary.

## Exploratory Data Analysis

### Summary Statistics

![Summary Statistics](outputs/summary_stats.png)

The numbers tell a story of extremes. Youth uninsured rates range from 0.0% (Columbia) to 8.9% (Skamania) — that's a child in one county being nearly 9 times more likely to lack insurance than a child in another. Mental health provider density ranges from 40 per 100K (Garfield) to 380 per 100K (King) — a 9.5x gap. Median household income spans from $35,800 (Whitman) to $106,300 (King). These aren't small differences — they describe fundamentally different environments for children growing up in the same state.

### Distributions

![Distributions](outputs/distributions.png)

Most counties cluster around a youth uninsured rate of 3–5%, but a long tail extends to 7–9% in eastern and rural counties. Child poverty shows a wide, roughly normal distribution centered around 17.6%, with several counties exceeding 25%. MH provider density is right-skewed: most counties have between 80–200 providers per 100K, but a handful of urban counties (King, Snohomish, San Juan) pull the distribution far to the right. Income follows a similar pattern — most counties earn between $42K–$62K, with King County as a clear outlier at $106K.

### Rural vs. Urban Disparities

![Rural vs Urban](outputs/rural_vs_urban.png)

The rural-urban divide is where the data gets concrete. Rural counties average 127 MH providers per 100K residents. Urban counties average 225 — nearly double. Rural counties also carry higher child poverty (19.4% vs. 14.5%) and slightly higher youth uninsured rates (4.7% vs. 4.0%). The box plots show that the spread within rural counties is also wider, meaning some rural communities are doing far worse than the rural average suggests.

## Correlation Analysis

### What Drives Youth Mental Health Access?

![Correlation Heatmap](outputs/heatmap.png)

The Pearson correlation matrix reveals which factors actually move together across all 39 counties:

- **Income → Providers (r = 0.79):** The strongest relationship in the dataset. Wealthier counties attract and retain more mental health providers. This isn't surprising, but the strength of the correlation — explaining over 60% of the variance — means income is the single best predictor of whether a county has adequate provider coverage.
- **Child poverty → Youth uninsured (r = 0.70):** Counties with higher child poverty consistently have more uninsured children. But income alone doesn't explain access — some high-poverty counties maintain low uninsured rates through targeted enrollment programs, which means policy interventions can break this link.
- **Hispanic % → Youth uninsured (r = 0.68):** This points to demographic-specific barriers in insurance enrollment. Counties with larger Hispanic populations tend to have higher youth uninsured rates, even after accounting for income — suggesting language barriers, documentation concerns, or outreach gaps.
- **Income → Youth uninsured (r = -0.28):** Surprisingly weak. Income doesn't directly predict whether children are insured. The pathway runs through child poverty and provider availability instead.

### Income Predicts Provider Access

![Income vs Providers](outputs/income_vs_providers.png)

This scatter plot isolates the strongest signal in the data. Each dot is a county, sized by population and colored by rural/urban classification. The trend line (r = 0.79) shows that for every $10K increase in median household income, a county gains roughly 30 more MH providers per 100K residents. Rural counties (orange) cluster in the lower-left — lower income, fewer providers. Urban counties (green) cluster in the upper-right. King County is the clear outlier in both dimensions.

## Cluster Analysis

### County Risk Profiles

![Clustering](outputs/clustering.png)

K-means clustering (k=3) groups the 39 counties into three distinct risk profiles based on child poverty, youth uninsured rates, provider density, and median income:

- **Lower Risk (7 counties):** King, Clark, Snohomish, Pierce, Kitsap, Island, Thurston. Average child poverty of 11.9%, 260 MH providers per 100K. These are the state's population centers with the most resources.
- **Higher Risk (8 counties):** Mid-range on most metrics but with specific vulnerabilities — either high poverty, low provider density, or both.
- **Mixed/Rural (24 counties):** The largest group. Averages 20.9% child poverty with only 113 providers per 100K and 5.3% youth uninsured — nearly double the rate of the lowest-risk cluster. This is where the bulk of underserved communities are.

The bubble sizes reflect population: King County is an outlier in both size and access, but most of the state lives in counties with far fewer resources.

## Geographic Patterns

### Youth Uninsured Rates by County

![GIS Map](outputs/gis_map.png)

The geographic pattern is clear: eastern and rural Washington carries the highest uninsured rates. Skamania (8.9%), Adams (8.2%), Douglas (7.3%), and Franklin (7.3%) lead the state. Western urban counties — King (2.4%), San Juan (2.5%), and Island (2.8%) — cluster at the low end. The statewide average of 4.5% masks a 4.5-fold gap between the highest and lowest counties.

### Provider Density Ranking

![Provider Ranking](outputs/top_bottom_providers.png)

All 39 counties ranked by mental health provider density. The bottom 10 (orange) are overwhelmingly rural — Garfield (40), Columbia (50), Wahkiakum (55), Skamania (70), Ferry (75). The top 10 (green) include every major urban center. The state average of 159 providers per 100K sits right in the middle, but 26 of 39 counties fall below it. The (R) markers confirm: provider shortage is largely a rural problem.

## Validating State-Level Claims

Washington's official reports and Mental Health America rankings make several claims about the state's youth mental health landscape. This analysis validates them with county-level data:

- **"WA ranks 48th nationally in youth mental health outcomes" (MHA 2024):** Consistent with the data. Even the state's best-performing counties have provider density below what states like Massachusetts or Connecticut achieve statewide.
- **"70.6% of caregivers report difficulty accessing care" (NSCH 2022):** The provider ranking chart explains why. When 26 of 39 counties fall below the state average in provider density, and the bottom 10 have fewer than 100 providers per 100K, geographic access is a structural barrier for most families.
- **"The provider gap is more consequential than the insurance gap":** Validated. Even in counties where uninsured rates are low (3–4%), provider shortages mean children with coverage still can't access care. The correlation between provider density and insurance is weak (r = 0.29), confirming they're separate problems.

## Recommendations

Based on the patterns in this data, four interventions would have the most impact:

1. **Target provider recruitment to the bottom 10 counties.** Garfield, Columbia, Wahkiakum, Skamania, and Ferry have fewer than 75 providers per 100K. Loan forgiveness programs, telehealth infrastructure, and training pipelines tied to rural placements would directly address the biggest gap.

2. **Expand insurance enrollment outreach in high-Hispanic counties.** The r = 0.68 correlation between Hispanic population percentage and youth uninsured rates points to enrollment barriers that income alone doesn't explain. Bilingual navigators and community-based enrollment events in Adams, Franklin, Grant, and Yakima counties would close this gap.

3. **Fund telehealth as a bridge, not a replacement.** The 24 counties in the Mixed/Rural cluster can't realistically recruit enough in-person providers to match urban levels. Telehealth can extend the reach of existing providers in Spokane, Thurston, and other hub counties into surrounding rural areas.

4. **Use cluster-specific metrics, not statewide averages.** The three-cluster model shows that statewide averages obscure the real picture. The 7 Lower Risk counties are performing well. The 24 Mixed/Rural counties need fundamentally different interventions than the 8 Higher Risk counties. Policy should be tailored to each profile.

## Repository Structure

```
youth-mental-health-wa/
├── README.md
├── LICENSE
├── .gitignore
├── Code.py                   ← Full analysis script (runs end-to-end)
├── requirements.txt
├── data/
│   └── sources.md            ← Detailed source documentation
└── outputs/
    ├── summary_stats.png     ← Summary statistics table
    ├── distributions.png     ← Variable distributions
    ├── rural_vs_urban.png    ← Rural vs urban comparison
    ├── top_bottom_providers.png  ← Provider density ranking
    ├── income_vs_providers.png   ← Income vs providers scatter
    ├── heatmap.png           ← Correlation matrix
    ├── clustering.png        ← K-means county clustering
    └── gis_map.png           ← Youth uninsured hex cartogram
```

## How to Reproduce

```bash
pip install -r requirements.txt
python Code.py
```

All data is embedded in the analysis script — no external downloads required. Output charts are saved to `outputs/`.

## License

Code is licensed under the [MIT License](LICENSE). Non-code content (visualizations, analysis, and documentation) is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

© 2026 Waleed Adawi. You may share and adapt with attribution for non-commercial purposes.
