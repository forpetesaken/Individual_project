# CGM EDA Summary

**Rows:** 87663  
**Glucose non-missing:** 86500 (missing: 1163)  
**Time coverage:** 2025-04-25 00:01:50 → 2026-02-15 05:48:08 (span: 296 days 05:46:18)  
**Inferred sampling cadence:** 5.00 min per reading if detected, else N/A

## Glucose Stats
- mean: 139.944
- median: 123.000
- variance: 4043.607
- std dev: 63.589
- min/max: 40.000 / 400.000
- skew: 1.413
- kurtosis: 2.086
- quartiles (Q1/median/Q3): 97.000 / 123.000 / 166.000
- IQR: 69.000
- p01 / p05 / p95 / p99: 52.000 / 68.000 / 274.000 / 357.000

## IQR Outliers
- count: 4662
- proportion: 5.318%

## Label Distribution
- label=0: 86770
- label=2: 782
- label=1: 111

## Figures
- Time series: `timeseries.png`
- Rolling mean: `rolling_mean.png` (if cadence inferred)
- Histogram: `histogram.png`
- Boxplot: `boxplot.png`
- Autocorrelation: `autocorrelation.png`
