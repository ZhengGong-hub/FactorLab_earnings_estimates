# Financial Data Preprocessing Methodology

## Data Cleaning and Variable Construction

### Financial Variables Construction
1. Standardized Unexpected Earnings (SUE)
   - EPS SUE: Calculated when EPS count ≥ 3
     * Formula: EPS_surprise / EPS_std
   - Normalized EPS SUE: Calculated when EPSNormalized count ≥ 3
     * Formula: EPSNormalized_surprise / EPSNormalized_std
   - Revenue SUE: Calculated when revenue count ≥ 3
     * Formula: revenue_surprise / revenue_std

### Key Financial Variables Preserved
1. Earnings Per Share (EPS) Metrics:
   - EPS_actual, EPSDiff, EPS_surprise
   - EPS_count, EPS_std
   - EPS_guidance_high, EPS_guidance_low
   
2. Normalized EPS Metrics:
   - EPSNormalized_actual, EPSNormalized_diff
   - EPSNormalized_surprise, EPSNormalized_count
   - EPSNormalized_std
   - EPSNormalized_guidance_high, EPSNormalized_guidance_low
   
3. Revenue Metrics:
   - revenue_actual, revenueDiff, revenue_surprise
   - revenue_count, revenue_std
   - revenue_guidance_high, revenue_guidance_low

## Variable Selection Criteria

### Missing Value Treatment
- Variables with > 20% missing values are removed

### Correlation Analysis
- Variables with correlation > 0.8 are considered highly correlated
- One variable from each highly correlated pair is removed

### Low Information Content Removal
1. Zero Variance
   - Removes constant features
   
2. Coefficient of Variation (CV)
   - Formula: CV = std/mean
   - Threshold: CV < 0.01
   - Special handling for near-zero means (|mean| < 1e-10)
   
3. Same Value Dominance
   - Variables where > 95% of values are identical

### Categorical Variable Treatment
- Binary encoding for categorical variable "calendar year" and "calendarquarter"
- Original categorical columns preserved

### Variable Classification
- Dependent Variables (y_): 
  * All SUE metrics
  * EPS metrics (actual, diff, surprise)
  * Revenue metrics (actual, diff, surprise)
- Independent Variables (x_):
  * All remaining numeric variables after cleaning
  * Must pass variance and correlation criteria

### Data Standardization
- Method: Z-score standardization (x - mean) / std
- Applied to: All numeric variables except:
  * Financial metrics (EPS, Revenue)
  * SUE variables
  * Categorical variables
  * Guidance variables

## Configuration Parameters
- Missing values threshold: 20%
- Correlation threshold: 0.8
- Coefficient of variation threshold: 0.01
- Same value proportion threshold: 95%


