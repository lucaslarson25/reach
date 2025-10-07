# Tests Directory

This directory contains unit tests and integration tests for the REACH project.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_environment.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Test Structure

- `test_environment.py` - Tests for simulation environments
- `test_agents.py` - Tests for RL agents
- `test_vision.py` - Tests for YOLO detector and camera
- `test_utils.py` - Tests for utility functions
- `test_integration.py` - End-to-end integration tests

## Writing Tests

Each test file should:
- Test a specific module or component
- Include both positive and negative test cases
- Use fixtures for common setup
- Be independent (no reliance on test execution order)

