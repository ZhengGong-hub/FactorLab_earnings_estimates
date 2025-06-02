from typing import Set

# Variables to be excluded from processing or kept in specific operations
EXCLUDE_VARIABLES: Set[str] = {
    # Identifier and date columns
    'companyid',
    'calendardate',
    
    # EPS related variables
    'EPS_actual',
    'EPSDiff',
    'EPS_surprise',
    'EPS_count',
    'EPS_std',
    'EPS_guidance_high',
    'EPS_guidance_low',
    
    # Normalized EPS variables
    'EPSNormalized_actual',
    'EPSNormalized_diff',
    'EPSNormalized_surprise',
    'EPSNormalized_count',
    'EPSNormalized_std',
    'EPSNormalized_guidance_high',
    'EPSNormalized_guidance_low',
    
    # Revenue related variables
    'revenue_actual',
    'revenueDiff',
    'revenue_surprise',
    'revenue_count',
    'revenue_std',
    'revenue_guidance_high',
    'revenue_guidance_low',
} 