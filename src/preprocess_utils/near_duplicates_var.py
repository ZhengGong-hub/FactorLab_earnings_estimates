import pandas as pd
import numpy as np
import re
from collections import defaultdict
from typing import List, Dict, Tuple

def extract_near_duplicate_groups(
    columns: List[str],
    suffix_patterns: List[str] = None
) -> Dict[str, List[str]]:
    """
    Find "near-duplicate" columns by looking for a base name plus known suffixes.

    Parameters
    ----------
    columns : List[str]
        List of all column names in the DataFrame.
    suffix_patterns : List[str], optional
        List of regex patterns with one capture group,  
        e.g. [r'(.+)_2$', r'(.+)_normalized$'].  
        Defaults to matching any underscore + word‐chars suffix.

    Returns
    -------
    groups : Dict[str, List[str]]
        A dict mapping the base column name to a list of its variants,
        e.g. { 'A': ['A_2', 'A_normalized'], … }
    """
    if suffix_patterns is None:
        # Default: any '_<word>' or '_<digits>' suffix
        suffix_patterns = [r"(.+)_\d+$", r"(.+)_\w+$"]
    
    cols_set = set(columns)
    groups: Dict[str, List[str]] = defaultdict(list)

    for col in columns:
        for pat in suffix_patterns:
            m = re.match(pat, col)
            if not m:
                continue
            base = m.group(1)
            # only consider it near-duplicate if the base actually exists
            if base in cols_set:
                groups[base].append(col)

    # Convert to regular dict, and sort for consistency
    return {base: sorted(variants) for base, variants in groups.items()}
