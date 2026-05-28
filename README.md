# Behavioral Health Treatment Gap Analysis — Washington State

A state-level analysis of mental health and substance use disorder treatment gaps using
SAMHSA NSDUH 2021–2022 survey estimates and HRSA mental health shortage area data.
Built with Python 3, SQLite, NumPy, and Matplotlib — no scikit-learn, no pandas.

**Author:** Waleed Adawi &nbsp;|&nbsp; **Year:** 2026

---

## Key Findings

| Metric | Value |
|--------|-------|
| WA adult AMI prevalence | **27.14%** — 4th highest nationally |
| WA mental health treatment gap | **+3.26 pp** (rank #17 of 51) |
| National average MH gap | +2.20 pp |
| WA unmet AMI need | **51.2%** — rank #31 nationally |
| WA counties with HPSA score ≥ 16 | **Seven** (federal high-priority threshold) |
| National SUD prevalence average | **18.64%** |

Washington's AMI prevalence (27.14%) is among the highest in the country, yet the state's
treatment gap (+3.26 percentage points) falls close to the national average (+2.20 pp),
suggesting moderate treatment system capacity relative to need — though still leaving
roughly one-quarter of adults with any mental illness without care.

---

## Charts

### Fig 1 — Mental Health Treatment Gap: All 51 States Ranked

![MH Treatment Gap Ranking](outputs/fig1_treatment_gap_ranking.png)

Nevada leads the country with the largest MH treatment gap (+8.66 pp), while
Massachusetts, Connecticut, and New Jersey have negative gaps — meaning estimated
treatment rates exceed prevalence estimates, likely reflecting measurement uncertainty
at the tails of the distribution. Washington ranks **#17** with a gap of +3.26 pp.

---

### Fig 2 — Washington vs. National Average

![WA vs National](outputs/fig2_wa_vs_national.png)

Washington's AMI prevalence (27.14%) sits 4.16 points above the national average (22.98%),
but its treatment rate (23.88%) is only 3.10 points above the national average (20.78%).
The result is a gap that is slightly larger than the national mean but not an outlier.

---

### Fig 3 — HRSA HPSA Score Ranking: Washington Counties

![Yakima HPSA Rank](outputs/fig3_yakima_hpsa_rank.png)

Of Washington's 30 HPSA-designated counties, **seven** score at or above the federal
high-priority threshold of 16: Ferry (20), Yakima (19), Pend Oreille (18), Stevens (17),
Garfield (17), Lincoln (16), and Adams (16). Yakima County — the state's second-largest
by HPSA score — serves a population of roughly 256,000 with severely limited access.

---

### Fig 4 — AMI Prevalence vs. Treatment Rate (All 51 States)

![Prevalence vs Treatment](outputs/fig4_prevalence_vs_treatment.png)

Higher prevalence states tend to have higher treatment rates, but the relationship is
imperfect. Several high-prevalence states (Nevada, Texas, Idaho) show disproportionately
low treatment rates, while DC and Massachusetts show high treatment rates relative
to prevalence.

---

### Fig 5 — Treatment Gap Distribution

![Gap Distribution](outputs/fig5_gap_distribution.png)

The distribution of MH treatment gaps across all 51 states is right-skewed, with a mean
of +2.20 pp and a maximum of +8.66 pp (Nevada). A handful of states have negative gaps;
these likely reflect limitations in small-area estimation methodology rather than genuine
over-treatment.

---

### Fig 6 — MH Gap vs. SUD Gap by State

![MH vs SUD Gaps](outputs/fig6_mh_vs_sud_gaps.png)

Substance use disorder treatment gaps are consistently **2–5× larger** than mental health
treatment gaps across all states. The national SUD prevalence averages 18.64% while
treatment rates are far lower, producing gaps that range from 6.08 pp (Alabama) to
20.21 pp (Oregon). Washington's SUD gap (15.55 pp) is close to the national median.
Washington's MH gap is marked at **rank #17**.

---

### Fig 7 — Unmet AMI Need: All 51 States Ranked

![Unmet Need Ranking](outputs/fig7_unmet_need_ranking.png)

Unmet need is calculated as the share of adults *with* AMI who received *no* mental health
treatment in the past year. The national average is 52.0%. Washington's unmet need is
51.2%, placing it at **rank #31** — just below the midpoint, meaning most states have
a higher share of untreated AMI than Washington. Wyoming (61.8%) and Nevada (61.7%)
have the highest unmet need nationally.

---

### Fig 8 — HPSA Bubble Chart: Score × Population × Geographic Context

![HPSA Bubble](outputs/fig8_hpsa_bubble.png)

Each bubble represents a HPSA-designated WA county, sized by population and colored by
HPSA score tier. Yakima County stands out as the highest-burden county combining a very
high shortage score (19) with a large affected population (~256,000). Ferry County has
the state's highest raw score (20) but a much smaller population (~8,500).

---

## Methodology

### Data Sources

| Dataset | Description | Rows |
|---------|-------------|------|
| SAMHSA NSDUH 2021–2022 | State-level AMI/SMI/SUD prevalence and treatment estimates | 51 |
| HRSA HPSA Designations | Mental health shortage area scores for WA counties | 30 |

**NSDUH estimates** are produced using Small Area Estimation (SAE) methodology and
represent model-based statistical estimates, not direct survey counts. They carry
confidence intervals that are not reflected in point-estimate comparisons. State-level
estimates are available for all 50 states plus DC.

**HRSA HPSA scores** are composite indices (0–25) incorporating provider-to-population
ratios, poverty rates, and travel distance to care. A score of ≥ 16 triggers federal
high-priority designation for workforce development programs.

### Database Schema

```
behavioral_health.db
├── nsduh_state  (51 rows)
│   ├── state
│   ├── year
│   ├── ami_prevalence_pct
│   ├── ami_received_treatment_pct
│   ├── ami_unmet_need_pct
│   ├── smi_prevalence_pct
│   ├── sud_prevalence_pct
│   └── sud_received_treatment_pct
│
└── hrsa_shortage  (30 rows)
    ├── county
    ├── hpsa_score
    └── population_of_designation
```

**Treatment gap** is computed as `ami_prevalence_pct − ami_received_treatment_pct`.
**Unmet need** (`ami_unmet_need_pct`) uses a different denominator — it is the share
of adults *with AMI* who received no treatment — and is stored directly from NSDUH
rather than derived.

### Implementation Notes

- All data is loaded into SQLite via named-column SQL queries (`SELECT col1, col2 …`) to
  avoid ordering bugs from `SELECT *`.
- Charts are generated with Matplotlib only (no seaborn, no scikit-learn).
- K-means or clustering was not used; all groupings are threshold-based (e.g., HPSA ≥ 16).

---

## Repository Structure

```
behavioral-health-access-wa/
├── Code.py                          # All analysis and chart generation
├── behavioral_health.db             # SQLite database
├── README.md
├── LICENSE
│
├── data/
│   ├── nsduh_state_estimates.csv    # SAMHSA NSDUH 2021-2022, all 51 states
│   └── hrsa_hpsa_wa.csv             # HRSA HPSA designations, WA counties
│
└── outputs/
    ├── fig1_treatment_gap_ranking.png
    ├── fig2_wa_vs_national.png
    ├── fig3_yakima_hpsa_rank.png
    ├── fig4_prevalence_vs_treatment.png
    ├── fig5_gap_distribution.png
    ├── fig6_mh_vs_sud_gaps.png
    ├── fig7_unmet_need_ranking.png
    ├── fig8_hpsa_bubble.png
    ├── treatment_gap_by_state.csv
    ├── wa_vs_national.csv
    └── yakima_shortage_rank.csv
```

---

## Limitations

1. **NSDUH estimates are modeled, not measured.** State-level figures come from Small
   Area Estimation, which uses survey microdata combined with demographic covariates.
   Point estimates have confidence intervals that are not shown in the charts; small
   state-to-state differences (< 1 pp) are unlikely to be statistically meaningful.

2. **Treatment gap ≠ unmet need.** The gap metric (`prevalence − treatment rate`) is
   computed from two separately estimated quantities, each with its own error. A
   "negative gap" (e.g., DC: −11.07 pp) does not mean more people are treated than
   have AMI; it reflects estimation uncertainty and should not be interpreted literally.

3. **HPSA scores reflect designated areas, not all counties.** Only counties with an
   active HPSA designation appear in the HRSA dataset. Counties without a designation
   may still have provider shortages that fall below the federal threshold for formal
   recognition.

4. **Single survey year.** The analysis uses 2021–2022 NSDUH data. Behavioral health
   capacity and prevalence estimates shift year to year; these findings describe a
   snapshot, not a stable trend.

5. **No causal claims.** Correlations between HPSA scores, prevalence, and treatment
   rates in this analysis are descriptive. They do not establish that shortage area
   status causes lower treatment rates; unmeasured confounders (income, rurality,
   insurance coverage) likely contribute.

6. **Population-level estimates only.** NSDUH does not provide county-level data. The
   HRSA shortage data is county-level but limited to Washington State, so no
   county-to-county national comparison is possible.

---

## Running the Analysis

```bash
# Install dependencies
pip install numpy matplotlib

# Run (generates all 8 charts + exports CSVs)
python Code.py
```

All outputs are saved to `outputs/`. The SQLite database is created automatically on
first run and populated from the data in `data/`.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Code and analysis by Waleed Adawi, 2026.
Underlying data is sourced from U.S. federal agencies (SAMHSA, HRSA) and is public domain.

---

*Data sources: SAMHSA NSDUH 2021–2022 State Estimates; HRSA Shortage Area Locator.*  
*NSDUH estimates use SAE methodology and should be interpreted with appropriate uncertainty.*
