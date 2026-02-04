# Step 5q: MHW Event Month Spatiotemporal Patterns - Key Results for Paper

## Overview
This analysis quantifies the spatiotemporal heterogeneity of Marine Heatwave (MHW) impacts on mangrove Gross Primary Productivity (GPP) during MHW event months (months with `isMHW = 1`). Using the model-internal effect definition (`ΔGPP = GPP_cf - GPP_factual`), we characterize the direction (facilitation vs. suppression), magnitude, and spatial-seasonal patterns of contemporaneous MHW responses across 8,746 MHW event months (2007-2020, 142 mangrove grid cells).

## Key Findings

### 1. Seasonal Patterns (Month-Level)
MHW impacts show strong seasonal asymmetry in directionality:

| Month | n | Facilitation % | Suppression % | Dominant Response | Mean ΔGPP (×10⁸) |
|-------|---|----------------|---------------|-------------------|------------------|
| **August** | 749 | **65.0%** | 35.0% | **Facilitation** | -6.05 |
| **October** | 851 | **61.3%** | 38.7% | **Facilitation** | -3.09 |
| **May** | 759 | **58.6%** | 41.4% | **Facilitation** | -0.42 |
| **March** | 677 | 42.8% | **57.2%** | **Suppression** | +0.84 |
| **February** | 651 | 44.2% | **55.8%** | **Suppression** | +1.88 |

**Interpretation**: Late summer/autumn MHWs (Aug-Oct) are predominantly facilitative, while late winter/early spring MHWs (Feb-Mar) are predominantly suppressive.

### 2. Spatial Patterns (Latitude Bands)
Clear latitudinal gradient in response directionality:

| Latitude Band | n | Facilitation % | Suppression % | Dominant Response | Score |
|---------------|---|----------------|---------------|-------------------|-------|
| **-20° to -10°** | 242 | **61.6%** | 38.4% | **Facilitation** | +0.23 |
| **-10° to 0°** | 2,864 | **58.8%** | 41.2% | **Facilitation** | +0.18 |
| **20° to 30°** | 1,106 | 48.0% | **52.0%** | **Suppression** | -0.04 |
| **-30° to -20°** | 58 | 44.8% | **55.2%** | **Suppression** | -0.10 |

**Interpretation**: Tropical regions (-20° to 0°) show net facilitation, while subtropical regions (20°-30° and -30° to -20°) show net suppression.

### 3. Combined Spatial Patterns (Latitude × Longitude)
Extreme facilitation and suppression hotspots:

| Region (Lat × Lon) | n | Facilitation % | Suppression % | Dominant Response | Score |
|--------------------|---|----------------|---------------|-------------------|-------|
| **-20°~-10° | 120°~140°** (W. Pacific) | 146 | **71.9%** | 28.1% | **Facilitation** | **+0.44** |
| **20°~30° | -80°~-60°** (Caribbean) | 80 | 18.8% | **81.3%** | **Suppression** | **-0.63** |

**Interpretation**: Western Pacific warm pool regions show strongest facilitation, while Caribbean/Western Atlantic regions show strongest suppression.

### 4. Hemisphere-Specific Seasonal Patterns
**Northern Hemisphere (NH)**:
- Strongest facilitation in August (61.0% facilitation, score +0.22)
- Strongest suppression in March (41.1% facilitation, score -0.18)

**Southern Hemisphere (SH)**:
- Strongest facilitation in August (72.8% facilitation, score +0.46)
- Strongest suppression in March (45.9% facilitation, score -0.08)

**Key contrast**: SH mangroves show stronger facilitation signals than NH mangroves, particularly in August (SH: 72.8% vs NH: 61.0% facilitation).

### 5. Environmental Correlates (from Gating Analysis)
For the strongest facilitation case (August, 20°-30°N, 81.1% facilitation):
- **NDVI**: Facilitated sites have higher NDVI (+1182 median difference)
- **Solar radiation**: Facilitated sites have lower radiation (-481 MJ/m²/day)
- **VPD**: Facilitated sites have higher VPD (+9.08 Pa)
- **Temperature**: Facilitated sites are warmer (+0.85°C)

**Interpretation**: Mature, high-NDVI mangroves in warm, high-VPD conditions may be particularly responsive to MHW facilitation.

## Statistical Summary
- **Total MHW event months**: 8,746 (months with at least one MHW occurrence)
- **Overall directionality**: 55.3% facilitation, 44.7% suppression
- **Minimum group size**: n ≥ 50 for reliable statistics
- **Zero neutral cases**: All event months showed non-zero ΔGPP

## Main Conclusions
1. **Spatiotemporal heterogeneity**: MHW impacts on mangrove GPP during event months show significant variation across seasons and space.
2. **Seasonal asymmetry**: Late summer/autumn MHWs tend to enhance productivity, while winter/spring MHWs tend to suppress it.
3. **Latitudinal gradient**: Tropical mangroves show net facilitation, subtropical mangroves show net suppression.
4. **Regional hotspots**: Western Pacific mangroves are particularly resilient (facilitative response), while Caribbean mangroves are particularly vulnerable (suppressive response).
5. **Hemispheric asymmetry**: Southern Hemisphere mangroves exhibit stronger facilitation responses than Northern Hemisphere mangroves.

## Methods Summary
- **Data**: Factual and counterfactual GPP predictions from TFT model (2007-2020)
- **Analysis**: Months with `isMHW = 1` only (contemporaneous effects)
- **Metric**: `ΔGPP = GPP_cf - GPP_factual`; ΔGPP < 0 = facilitation, ΔGPP > 0 = suppression
- **Directional score**: `score_fac_minus_sup = P(ΔGPP<0) - P(ΔGPP>0)` (positive = facilitation-dominant)
- **Grouping**: Month, latitude bands (10°), longitude bands (20°), combined lat-lon bands, hemisphere

---
*Results generated from analysis of Step 5q scripts, February 2026*
*For detailed methodology and full results, see: `Step5q_MHW_event_month_spatiotemporal_patterns_Summary.md`*