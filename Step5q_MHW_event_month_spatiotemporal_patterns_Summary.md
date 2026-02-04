# Step 5q Analysis Summary: MHW Event Month Spatiotemporal Patterns

## Analysis Overview

### 1.1 Analysis Objective
Systematically characterize the **spatiotemporal heterogeneity** of Marine Heatwave (MHW) impacts on mangrove Gross Primary Productivity (GPP) during MHW event months. The analysis quantifies the **direction (facilitation vs. suppression), magnitude, and spatial-seasonal patterns** of ΔGPP responses when MHWs are present.

### 1.2 Core Conceptual Framework
The analysis uses the **model-internal effect** definition:

```
ΔGPP = GPP_cf - GPP_factual
```

Where:
- `GPP_factual`: Predicted GPP under actual MHW conditions
- `GPP_cf`: Predicted GPP under counterfactual (no-MHW) conditions
- **Interpretation**:
  - `ΔGPP < 0` → **Facilitation**: MHW presence increases contemporaneous GPP
  - `ΔGPP > 0` → **Suppression**: MHW presence decreases contemporaneous GPP
  - `ΔGPP = 0` → Neutral (theoretically possible but rare)

### 1.3 Analysis Scope
- **Time scale**: Month-level analysis (only months with MHW occurrence)
- **Spatial scale**: 142 mangrove grid cells across tropical/subtropical coasts
- **Temporal coverage**: 2007-2020 (full analysis period)
- **Event definition**: Months with `isMHW = 1` (at least one MHW occurrence)
- **Statistical approach**: Group-wise descriptive statistics with directional scoring

---

## 2. Data and Methods

### 2.1 Data Inputs
| Data File | Purpose | Key Columns |
|-----------|---------|-------------|
| `factual_rolling_predictions.csv` | Factual GPP predictions | `grid_id`, `year`, `month`, `gpp_pred` |
| `counterfactual_rolling_predictions.csv` | Counterfactual GPP predictions | `grid_id`, `year`, `month`, `gpp_pred` |
| `data.csv` | Original observational data | `grid_id`, `year`, `month`, `isMHW`, `lat_c`, `lon_c` |

### 2.2 Processing Pipeline
1. **Data Alignment**: Inner join factual and counterfactual predictions on `(grid_id, year, month)`
2. **MHW Filtering**: Select only months with `isMHW == 1` (MHW event months)
3. **ΔGPP Calculation**: Compute `delta_gpp = gpp_cf - gpp_factual` for each event month
4. **Direction Classification**:
   - `facilitation`: `delta_gpp < 0`
   - `suppression`: `delta_gpp > 0`
   - `neutral`: `delta_gpp == 0`
5. **Spatial Binning**:
   - Latitude bands: 10° increments (e.g., -20°~-10°, -10°~0°, ...)
   - Longitude bands: 20° increments (e.g., -80°~-60°, -60°~-40°, ...)
   - Combined lat-lon bands: Cartesian product of above
6. **Hemisphere Assignment**: Northern Hemisphere (NH, lat ≥ 0°), Southern Hemisphere (SH, lat < 0°)

### 2.3 Statistical Metrics
For each grouping (month, latitude band, etc.):
- **Counts**: `n_facilitation`, `n_suppression`, `n_neutral`
- **Directional Fractions**:
  - `frac_facilitation_nonzero = n_facilitation / (n_facilitation + n_suppression)`
  - `frac_suppression_nonzero = n_suppression / (n_facilitation + n_suppression)`
- **Directional Score**: `score_fac_minus_sup = frac_facilitation_nonzero - frac_suppression_nonzero`
  - Positive: Facilitation-dominant group
  - Negative: Suppression-dominant group
- **Distribution Statistics**: Mean, median, 25th/75th percentiles of ΔGPP

### 2.4 Script Implementation
**Primary Script**: `code/analysis/analysis_step5q_event_month_spatiotemporal_patterns.py`

**Key Functions**:
- `summarize_group()`: Core grouping and statistics computation
- `month_hemisphere_summary()`: Hemisphere-specific monthly analysis
- `plot_month_bar_two_panels()`: Visualization of NH/SH seasonal patterns
- `top_bottom_k()`: Identification of extreme facilitation/suppression groups

**Supporting Scripts**:
- `analysis_step5q_event_month_hemisphere_barplot.py`: Enhanced hemisphere-month visualization
- `plot_step5q_gating_heatmaps.py`: Heatmap generation from gating data
- `analysis_step5q_event_month_spatial_heatmaps.py`: Spatial pattern visualization

**Execution Command**:
```bash
python -m code.analysis.analysis_step5q_event_month_spatiotemporal_patterns \
  --factual results/predictions/factual_rolling_predictions.csv \
  --counterfactual results/predictions/counterfactual_rolling_predictions.csv \
  --data data/data.csv \
  --outdir results/step5q \
  --lat_step 10.0 \
  --lon_step 20.0 \
  --min_n 50
```

---

## 3. Key Results

### 3.1 Dataset Summary
- **Total MHW event months**: 8,746 (months with at least one MHW occurrence)
- **Minimum samples per group**: 50 (statistical reliability threshold)
- **Zero neutral cases**: All event months showed non-zero ΔGPP (no exact zeros)

### 3.2 Seasonal Patterns (Monthly Ranking)

**Most Facilitation-Dominant Months** (ranked by `score_fac_minus_sup`):
| Month | n | Facilitation % | Suppression % | Score | Mean ΔGPP (×10⁸) |
|-------|---|----------------|---------------|-------|------------------|
| 8 (Aug) | 749 | **65.0%** | 35.0% | +0.300 | -6.05 |
| 10 (Oct) | 851 | **61.3%** | 38.7% | +0.227 | -3.09 |
| 5 (May) | 759 | **58.6%** | 41.4% | +0.173 | -0.42 |
| 9 (Sep) | 715 | **57.6%** | 42.4% | +0.152 | -3.16 |
| 7 (Jul) | 714 | **55.9%** | 44.1% | +0.118 | -2.12 |

**Most Suppression-Dominant Months**:
| Month | n | Facilitation % | Suppression % | Score | Mean ΔGPP (×10⁸) |
|-------|---|----------------|---------------|-------|------------------|
| 3 (Mar) | 677 | 42.8% | **57.2%** | -0.143 | +0.84 |
| 2 (Feb) | 651 | 44.2% | **55.8%** | -0.115 | +1.88 |
| 12 (Dec) | 760 | 47.9% | **52.1%** | -0.042 | +1.76 |
| 11 (Nov) | 685 | 49.3% | **50.7%** | -0.013 | +1.06 |
| 6 (Jun) | 709 | 49.6% | **50.4%** | -0.007 | +0.22 |

**Interpretation**: Late summer/early autumn (Aug-Oct) shows strongest facilitation, while late winter/early spring (Feb-Mar) shows strongest suppression.

### 3.3 Spatial Patterns (Latitude Bands)

**Most Facilitation-Dominant Latitude Bands**:
| Latitude Band | n | Facilitation % | Suppression % | Score |
|---------------|---|----------------|---------------|-------|
| -20°~-10° | 242 | **61.6%** | 38.4% | +0.231 |
| -10°~0° | 2,864 | **58.8%** | 41.2% | +0.176 |
| 10°~20° | 1,675 | **54.5%** | 45.5% | +0.090 |

**Most Suppression-Dominant Latitude Bands**:
| Latitude Band | n | Facilitation % | Suppression % | Score |
|---------------|---|----------------|---------------|-------|
| -30°~-20° | 58 | 44.8% | **55.2%** | -0.103 |
| 20°~30° | 1,106 | 48.0% | **52.0%** | -0.040 |
| 0°~10° | 2,801 | 49.8% | **50.2%** | -0.004 |

**Interpretation**: Southern tropics (-20°~0°) show strongest facilitation, while higher latitudes (both hemispheres) show suppression dominance.

### 3.4 Combined Spatial Patterns (Latitude × Longitude)

**Strongest Facilitation Regions**:
| Lat × Lon Band | n | Facilitation % | Suppression % | Score |
|----------------|---|----------------|---------------|-------|
| -20°~-10° | 120°~140° | 146 | **71.9%** | 28.1% | +0.438 |
| -10°~0° | -80°~-60° | 127 | **64.6%** | 35.3% | +0.291 |
| -10°~0° | 140°~160° | 164 | **64.0%** | 36.0% | +0.280 |

**Strongest Suppression Regions**:
| Lat × Lon Band | n | Facilitation % | Suppression % | Score |
|----------------|---|----------------|---------------|-------|
| 20°~30° | -80°~-60° | 80 | **18.8%** | 81.3% | -0.625 |
| 20°~30° | -120°~-100° | 64 | **37.5%** | 62.5% | -0.250 |
| 0°~10° | -60°~-40° | 250 | **38.4%** | 61.6% | -0.232 |

**Interpretation**: Western Pacific warm pool regions show strongest facilitation, while Caribbean/Western Atlantic regions show strongest suppression.

### 3.5 Hemisphere-Specific Seasonal Patterns

**Northern Hemisphere (NH)**:
- **Strongest Facilitation**: August (61.0% facilitation, score +0.220)
- **Strongest Suppression**: March (41.1% facilitation, score -0.178)
- **Seasonal Pattern**: Summer facilitation, winter/spring suppression

**Southern Hemisphere (SH)**:
- **Strongest Facilitation**: August (72.8% facilitation, score +0.457)
- **Strongest Suppression**: March (45.9% facilitation, score -0.082)
- **Seasonal Pattern**: More pronounced facilitation overall, especially in local winter months

**Key Contrast**: SH shows stronger facilitation signals than NH across most months, particularly in August (SH: 72.8% vs NH: 61.0% facilitation).

### 3.6 Statistical Reliability
- **Group validity threshold**: n ≥ 50 samples
- **Invalid groups**: Only 1 latitude band (-30°~-20°) had n < 100 but still ≥ 50
- **All monthly groups**: n ≥ 651 samples, well above threshold
- **Directional consistency**: Clear statistical separation between top and bottom groups

---

## 4. Interpretation and Discussion

### 4.1 Ecological Implications
1. **Seasonal Asymmetry**: MHW impacts are not uniform across seasons. Late summer/autumn MHWs tend to **enhance** mangrove productivity, while winter/spring MHWs tend to **suppress** it.
2. **Latitudinal Gradient**: Tropical regions (-20°~0°) show net facilitation, while subtropical regions (20°~30°) show net suppression.
3. **Hemispheric Contrast**: Southern Hemisphere mangroves show stronger facilitation responses, possibly due to different species composition, tidal regimes, or MHW characteristics.
4. **Regional Hotspots**: Western Pacific warm pool regions exhibit exceptionally strong facilitation, suggesting these ecosystems may be adapted to warm water anomalies.

### 4.2 Methodological Contributions
1. **Event-Month Focus**: By analyzing only MHW months, this approach isolates the **contemporaneous** effects of MHWs, separate from legacy or lagged effects.
2. **Directional Scoring**: The `score_fac_minus_sup` metric provides an intuitive measure of net directional tendency.
3. **Multi-Scale Analysis**: Simultaneous examination of seasonal, latitudinal, longitudinal, and combined patterns reveals complex spatiotemporal heterogeneity.
4. **Statistical Rigor**: Minimum sample thresholds ensure reliable group-level estimates.

### 4.3 Limitations and Caveats
1. **Temporal Aggregation**: Month-scale analysis may mask within-month dynamics.
2. **Spatial Resolution**: 10°×20° bins may obscure finer-scale patterns.
3. **Causal Attribution**: ΔGPP represents model-predicted differences, not directly observed responses.
4. **Neutral Cases**: Absence of exact zero ΔGPP values suggests continuous response spectrum rather than binary facilitation/suppression.

### 4.4 Future Research Directions
1. **Mechanistic Investigation**: Link spatiotemporal patterns to environmental drivers (SST anomalies, precipitation, light availability).
2. **Species-Specific Analysis**: Disaggregate responses by mangrove species composition.
3. **Event Characteristics**: Incorporate MHW intensity, duration, and rate of onset as covariates.
4. **Lag Effects**: Extend analysis to post-MHW months to examine legacy effects.

---

## 5. Output Files

### 5.1 Core Data Files (`results/step5q/`)
| File | Description | Key Columns |
|------|-------------|-------------|
| `step5q_event_rows.csv` | Raw event-month data | `grid_id`, `year`, `month`, `delta_gpp`, `effect`, `lat_c`, `lon_c`, `hemisphere`, `lat_band`, `lon_band`, `latlon_band` |
| `step5q_by_month.csv` | Monthly statistics | `month`, `n`, `n_facilitation`, `n_suppression`, `frac_facilitation_nonzero`, `frac_suppression_nonzero`, `score_fac_minus_sup`, distribution statistics |
| `step5q_by_latband.csv` | Latitude band statistics | `lat_band`, statistics as above |
| `step5q_by_lonband.csv` | Longitude band statistics | `lon_band`, statistics as above |
| `step5q_by_latlonband.csv` | Combined lat-lon statistics | `latlon_band`, statistics as above |
| `step5q_by_month_hemisphere.csv` | Hemisphere-month statistics | `month`, `hemisphere`, statistics + `is_valid_n` flag |
| `step5q_gating_month_lat.csv` | Month-latitude gating data | For heatmap visualization |

### 5.2 Summary Files
| File | Description |
|------|-------------|
| `step5q_highlights.txt` | Key findings in plain text format |
| `step5q_month_hemisphere_bar.png` | Two-panel bar chart of NH/SH monthly patterns |

### 5.3 Visualization Files (`results/step5q/figs/`)
- Heatmaps of month-latitude patterns
- Spatial distribution maps
- Directional dominance plots

---

## 6. References to Code and Data

### 6.1 Primary Script Location
`code/analysis/analysis_step5q_event_month_spatiotemporal_patterns.py`

### 6.2 Supporting Scripts
- `code/analysis/analysis_step5q_event_month_hemisphere_barplot.py`
- `code/analysis/plot_step5q_gating_heatmaps.py`
- `code/analysis/analysis_step5q_event_month_spatial_heatmaps.py`
- `code/analysis/analysis_step5q_event_season_spatial_heatmaps.py`
- `code/analysis/analysis_step5q_event_month_hemisphere_directional_dominance.py`

### 6.3 Data Dependencies
- **Factual predictions**: `results/predictions/factual_rolling_predictions.csv`
- **Counterfactual predictions**: `results/predictions/counterfactual_rolling_predictions.csv`
- **Original data**: `data/data.csv`

### 6.4 Configuration Parameters
- Latitude bin size: 10°
- Longitude bin size: 20°
- Minimum group size: 50 samples
- Hemisphere boundary: 0° latitude

---

## 7. Conclusion

The Step 5q analysis reveals **significant spatiotemporal heterogeneity** in MHW impacts on mangrove GPP during event months. Key conclusions:

1. **Directional Dominance**: MHW months show clear facilitation (ΔGPP < 0) in 55.3% of cases overall, but this masks strong seasonal and spatial variation.
2. **Seasonal Pattern**: Late summer/autumn MHWs (Aug-Oct) are predominantly facilitative, while winter/spring MHWs (Feb-Mar) are predominantly suppressive.
3. **Spatial Pattern**: Tropical regions (-20°~0°) show net facilitation, particularly in the Western Pacific, while subtropical regions (20°~30°) show net suppression, especially in the Caribbean.
4. **Hemispheric Asymmetry**: Southern Hemisphere mangroves exhibit stronger facilitation responses than Northern Hemisphere mangroves.
5. **Methodological Value**: The event-month focused approach successfully isolates contemporaneous MHW effects and provides a framework for spatiotemporal impact characterization.

These findings advance our understanding of MHW impacts on coastal blue carbon ecosystems and provide empirical evidence for climate-ecosystem interactions at regional to global scales.

---

*Document generated from analysis of Step 5q results, February 2026*
*Analysis framework: Temporal Fusion Transformer (TFT) model for mangrove GPP prediction*
*Data period: 2007-2020, 142 mangrove grid cells*