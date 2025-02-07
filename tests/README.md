# Automated Test Suite Documentation  

This project includes a comprehensive suite of tests to validate the functionality, robustness, and reliability of key components. This document provides a categorized overview of the tests, including their purpose and expected outcomes.

---

## Table of Contents  

- [Automated Test Suite Documentation](#automated-test-suite-documentation)
  - [Table of Contents](#table-of-contents)
  - [Test Categories and Coverage](#test-categories-and-coverage)
    - [1. REST API Tests](#1-rest-api-tests)
      - [Test Cases:](#test-cases)
    - [2. Data Ingestion Tests](#2-data-ingestion-tests)
      - [Test Cases:](#test-cases-1)
    - [3. Data Transformation Tests](#3-data-transformation-tests)
      - [Test Cases:](#test-cases-2)
    - [4. Model Selection Tests](#4-model-selection-tests)
      - [Test Cases:](#test-cases-3)
    - [5. Command-Line Argument Tests](#5-command-line-argument-tests)
      - [Test Cases:](#test-cases-4)
  - [Execution Instructions](#execution-instructions)
  - [Running Specific Tests](#running-specific-tests)
  - [Generating a Test Coverage Report](#generating-a-test-coverage-report)

---

## Test Categories and Coverage  

### 1. REST API Tests  
**Located in**: `tests/test_rest_api.py`  

#### Test Cases:  
| **Test Name**                   | **Purpose**                                                                                           | **Expected Outcome**                                                                                          |
|----------------------------------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `test_predict_success`           | Tests the `/predict` endpoint with valid input data.                                                  | Returns a `200 OK` response with a valid prediction.                                                         |
| `test_predict_validation_error`  | Validates that the `/predict` endpoint handles missing or invalid input fields correctly.             | Returns a `400 Bad Request` response with detailed validation errors.                                         |
| `test_predict_missing_payload`   | Ensures the `/predict` endpoint handles empty or missing JSON payloads.                               | Returns a `422 Unprocessable Entity` response with an appropriate error message.                              |
| `test_root_endpoint`             | Verifies the root (`/`) endpoint functionality.                                                       | Returns a `200 OK` response with a health check message.                                                      |

---

### 2. Data Ingestion Tests  
**Located in**: `tests/test_services/test_data_ingestion.py`  

#### Test Cases:  
| **Test Name**                       | **Purpose**                                                                                     | **Expected Outcome**                                                                                          |
|-------------------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `test_data_ingestion_with_missing_file` | Ensures the data ingestion process handles missing input files gracefully.                      | Raises a `CustomException` caused by a `FileNotFoundError`, with a message referencing the missing file.       |
| `test_data_ingestion_creates_files`  | Validates that data ingestion correctly creates train and test files from the input dataset.     | Generated train and test files exist in the specified paths and contain valid data.                           |

---

### 3. Data Transformation Tests  
**Located in**: `tests/test_services/test_data_transformation.py`  

#### Test Cases:  
| **Test Name**                         | **Purpose**                                                                                     | **Expected Outcome**                                                                                          |
|---------------------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `test_data_transformation`            | Ensures the data transformation process works as expected with valid inputs.                    | Returns transformed train and test arrays with expected shapes and saves the preprocessor object.             |
| `test_data_transformation_missing_columns` | Validates that the transformation process handles missing required columns appropriately.       | Raises a `CustomException` indicating the missing columns.                                                    |

---

### 4. Model Selection Tests  
**Located in**: `tests/test_services/test_model_selection.py`  

#### Test Cases:  
| **Test Name**                       | **Purpose**                                                                                     | **Expected Outcome**                                                                                          |
|-------------------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `test_initiate_model_trainer_success` | Validates the selection and evaluation of the best model from the given data.                   | Ensures a valid model report is returned, the best model is identified, and its score meets expectations.      |
| `test_no_model_meets_threshold`     | Ensures an exception is raised when no model meets the minimum performance threshold.            | Raises a `CustomException` with a message indicating no best model was found.                                 |
| `test_invalid_data_handling`        | Tests the behavior of model selection when invalid training and testing data are provided.       | Returns a model report with `None` scores for all but Linear Regression, which should score 1.0.              |

---

### 5. Command-Line Argument Tests  
**Located in**: `tests/test_command_line.py`  

#### Test Cases:  
| **Test Name**                         | **Purpose**                                                                                     | **Expected Outcome**                                                                                          |
|---------------------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `test_parse_arguments_ingest_monkey`  | Tests argument parsing for the `ingest` command using `monkeypatch`.                            | Ensures `ingest` is recognized and `config` is properly set.                                                 |
| `test_parse_arguments_train_monkey`   | Tests argument parsing for the `train` command with `monkeypatch`.                             | Ensures `train` command is parsed and `--debug` is correctly set to `True`.                                  |
| `test_parse_arguments_ingest_mock`    | Tests `ingest` command parsing using `patch`.                                                 | Ensures correct argument assignment for `ingest`.                                                            |
| `test_parse_arguments_train_mock`     | Tests `train` command parsing using `patch`.                                                  | Ensures correct argument assignment for `train`.                                                             |
| `test_parse_arguments_ingest_no_config` | Ensures `ingest` works without a specified config file.                                        | Confirms default config behavior when none is provided.                                                      |
| `test_parse_arguments_train_no_debug` | Ensures `train` command behavior when `--debug` is not passed.                                | Confirms `debug` defaults to `False` and correct model config is used.                                       |
| `test_parse_arguments_ingest_with_debug` | Ensures `ingest` command accepts the `--debug` flag.                                         | Verifies `debug` is correctly set to `True`.                                                                |
| `test_parse_arguments_no_command`     | Tests behavior when no command is provided.                                                  | Ensures the application exits with an error.                                                                |

---

## Execution Instructions  

1. Ensure all dependencies are installed using:  
   ```bash
   pip install -r requirements.txt
   ```

2. Run the tests using `pytest`:  
   ```bash
   PYTHONPATH=$(pwd) pytest -v
   ```

---

## Running Specific Tests  

To run a specific test module, use:  
```bash
PYTHONPATH=$(pwd) pytest -v tests/<test_module_name>.py
```

For example, to run only the command-line argument tests:  
```bash
PYTHONPATH=$(pwd) pytest -v tests/test_command_line.py
```

---

## Generating a Test Coverage Report  

To generate a test coverage report:  
```bash
PYTHONPATH=$(pwd) pytest --cov=src --cov-report=html
```
