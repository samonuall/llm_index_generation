# Testing Suite

## Overview

Comprehensive test suite ensuring data reliability and code correctness throughout the analysis pipeline.

# 1. Data Layer Tests

## Test Structure

```
tests/
├── conftest.py                                    # Shared pytest fixtures and configuration
├── TESTS_README.md                                # Test documentation
│
├── data/                                          # Data Layer Tests
│   ├── __init__.py                                # Package initializer
│   ├── test_data_loader.py                        # Test data loading from various sources
│   ├── test_data_storage.py                       # Test data persistence and saving
│   ├── test_data_transformation.py                # Test data processing and transformations
│   ├── test_data_cache.py                         # Test caching mechanisms
│   └── test_data_pipeline.py                      # Test end-to-end data workflows
│
├── evaluation/                                    # Evaluation and validation tests
│   ├── test_base_preprocessor.py                  # Test base preprocessing functionality
│   ├── test_data_quality_validation.py            # Test data quality checks (7 tests)
│   ├── test_get_data.py                           # Test data retrieval functions
│   ├── test_pipeline_integration.py               # Test pipeline integration (2 tests, needs update)
│   └── test_schema.py                             # Test schema validation (3 tests)
│
└── agents/analysis_code_agent/                    # Agent-specific tests
    ├── test_bm25_client.py                        # Test BM25 client communication
    └── test_eval_utils.py                         # Test evaluation utilities
```

### Description

The data layer test suite ensures end-to-end data reliability from initial loading through processing to final output. Tests are distributed across `test_data_layer.py`, which covers core data access operations and database interactions, and the `evaluation/` directory, which focuses on data quality and integrity.

The `test_get_data.py` file validates data loading mechanisms, cache management, and file system operations, ensuring data can be reliably retrieved from various sources. Schema validation is handled by `test_schema.py`, which enforces data type consistency and validates schema evolution over time.

The most comprehensive data integrity testing occurs in `test_data_quality_validation.py`, which monitors data health at each iteration. This includes detecting missing values, identifying duplicates, flagging statistical outliers, and tracking quality degradation over time. The tests also validate that critical data properties (row counts, column names, ID ranges) remain invariant across processing steps and detect distribution shifts that might indicate data drift or corruption. Together, these tests ensure that invalid or corrupted data never progresses through the pipeline to produce misleading results.


## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/agents/ -v                              # All agent tests
pytest tests/evaluation/ -v                          # All evaluation tests

# Run specific test file
pytest tests/evaluation/test_data_quality_validation.py -v

# Run tests by pattern
pytest tests/ -k "agent" -v                          # All tests with "agent" in name
pytest tests/ -k "quality" -v                        # All quality-related tests

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Show detailed output (including print statements)
pytest tests/ -v -s

# Stop on first failure (fast feedback)
pytest tests/ -x

# Run specific test method
pytest tests/evaluation/test_data_quality_validation.py::TestDataQualityAcrossIterations::test_detects_missing_values -v
```
