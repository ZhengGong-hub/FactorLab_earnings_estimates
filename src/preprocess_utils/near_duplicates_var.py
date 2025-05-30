"""
Identify and group near-duplicate columns based on naming patterns.
"""

import re
from collections import defaultdict
from typing import List, Dict

# internal imports
from logger import setup_logger

# setup logger
logger = setup_logger(__name__)

# Common suffix patterns for identifying related columns
SUFFIX_PATTERNS = [
    r"(.+)_\d+$",  # numeric suffixes (e.g., column_1)
    r"(.+)_\w+$"   # word suffixes (e.g., column_normalized)
]

def find_duplicates(columns: List[str], patterns: List[str] = None) -> Dict[str, List[str]]:
    """
    Group columns that share the same base name but have different suffixes.
    
    Example:
        Input: ['revenue', 'revenue_2', 'revenue_normalized', 'profit', 'profit_adjusted']
        Output: {
            'revenue': ['revenue_2', 'revenue_normalized'],
            'profit': ['profit_adjusted']
        }
    """
    if not columns:
        logger.warning("Empty column list provided")
        return {}
        
    # Use default patterns if none provided
    patterns = patterns or SUFFIX_PATTERNS
    compiled_patterns = [re.compile(p) for p in patterns]
    
    # Track base names and their variants
    base_names = set(columns)
    variants = defaultdict(list)
    
    # Find variants for each column
    for col in columns:
        for pattern in compiled_patterns:
            if match := pattern.match(col):
                base = match.group(1)
                if base in base_names:
                    variants[base].append(col)
    
    # Return only groups that have variants
    return {
        base: sorted(vars) 
        for base, vars in variants.items() 
        if vars
    }
