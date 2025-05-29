import pandas as pd
import numpy as np
import re
from collections import defaultdict
from typing import List, Dict, Optional, Pattern

DEFAULT_SUFFIX_PATTERNS = [
    r"(.+)_\d+$",  # matches columns with numeric suffixes (e.g., column_1, column_2)
    r"(.+)_\w+$"   # matches columns with word suffixes (e.g., column_normalized)
]

def extract_near_duplicate_groups(
    columns: List[str],
    suffix_patterns: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Identifies and groups "near-duplicate" columns by detecting base names with known suffixes.
    
    This function helps identify related columns in a dataset that follow a common naming pattern,
    such as variations of the same measurement or different versions of a feature.

    Parameters
    ----------
    columns : List[str]
        List of column names to analyze for near-duplicates.
        Example: ['revenue', 'revenue_2', 'revenue_normalized', 'profit', 'profit_adjusted']
    
    suffix_patterns : Optional[List[str]]
        List of regex patterns, each containing exactly one capture group for the base name.
        If None, defaults to matching:
        - Numeric suffixes (e.g., '_1', '_2')
        - Word suffixes (e.g., '_normalized', '_adjusted')
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping base column names to their variant columns.
        Example:
        {
            'revenue': ['revenue_2', 'revenue_normalized'],
            'profit': ['profit_adjusted']
        }
        
    Raises
    ------
    ValueError
        If columns is empty or contains non-string elements
        If suffix_patterns contains invalid regex patterns
    """
    # Input validation
    if not columns:
        raise ValueError("columns list cannot be empty")
    if not all(isinstance(col, str) for col in columns):
        raise ValueError("all column names must be strings")
    
    # Use default patterns if none provided
    patterns = suffix_patterns if suffix_patterns is not None else DEFAULT_SUFFIX_PATTERNS
    
    # Validate regex patterns
    try:
        compiled_patterns = [re.compile(pattern) for pattern in patterns]
    except re.error as e:
        raise ValueError(f"Invalid regex pattern provided: {str(e)}")
    
    # Create set for O(1) lookups
    columns_set = set(columns)
    duplicate_groups: Dict[str, List[str]] = defaultdict(list)
    
    # Find near-duplicates
    for column in columns:
        for pattern in compiled_patterns:
            match = pattern.match(column)
            if not match:
                continue
                
            base_name = match.group(1)
            if base_name in columns_set:
                duplicate_groups[base_name].append(column)
    
    # Convert to regular dict and sort variants for consistency
    return {
        base: sorted(variants) 
        for base, variants in duplicate_groups.items()
        if variants  # Only include groups that have variants
    }
