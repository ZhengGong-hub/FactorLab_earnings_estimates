# FactorLab Earnings Estimates

This project processes and analyzes earnings estimates data using various preprocessing utilities and machine learning techniques.

## Project Structure

```
.
├── src/                          # Source code directory
│   ├── main.py                  # Main entry point of the application
│   ├── preprocess.py            # Main preprocessing pipeline
│   ├── ml_run.py               # Machine learning execution script
│   └── preprocess_utils/        # Preprocessing utility functions
│       ├── near_duplicates_var.py    # Handle near-duplicate variables
│       ├── drop_fuzzy_var.py         # Drop fuzzy matching variables
│       └── missing_var_treat.py      # Handle missing values
```

## Code Structure Details

### Main Components

1. **main.py**
   - Entry point of the application
   - Orchestrates the overall data processing pipeline
   - Size: 674B (31 lines)

2. **preprocess.py**
   - Implements the main preprocessing pipeline
   - Coordinates various preprocessing steps
   - Size: 654B (32 lines)

3. **ml_run.py**
   - Handles machine learning model execution
   - (Currently empty, prepared for future implementation)

### Preprocessing Utilities (`preprocess_utils/`)

1. **near_duplicates_var.py**
   - Purpose: Identifies and groups "near-duplicate" columns in datasets
   - Key Features:
     - Detects columns with similar base names but different suffixes
     - Supports customizable suffix patterns
     - Handles both numeric and word-based suffixes
   - Example patterns:
     - Numeric suffixes (e.g., column_1, column_2)
     - Word suffixes (e.g., column_normalized, column_adjusted)
   - Size: 2.9KB (86 lines)

2. **drop_fuzzy_var.py**
   - Purpose: Handles fuzzy matching for variable removal
   - Size: 562B (21 lines)

3. **missing_var_treat.py**
   - Purpose: Implements missing value treatment strategies
   - Size: 1.1KB (39 lines)

## Usage

To run the main application:

```bash
uv run src/main.py
```

## Dependencies

The project uses various Python libraries including:
- pandas
- numpy
- re (Python standard library)
- collections (Python standard library)

## Development

### Code Quality Standards
- Type hints are used throughout the codebase
- Comprehensive docstrings with NumPy style
- Input validation and error handling
- Consistent code formatting
- Performance optimizations where applicable

### Best Practices
- Modular code structure
- Clear separation of concerns
- Efficient data structures
- Comprehensive error handling
- Well-documented functions and modules

## Contributing

When contributing to this project, please:
1. Follow the existing code style
2. Add appropriate type hints
3. Include comprehensive docstrings
4. Add proper error handling
5. Update tests if applicable
6. Update this README when adding new functionality 