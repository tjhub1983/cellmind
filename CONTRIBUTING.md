# Contributing to CellMind

Thank you for your interest in contributing to CellMind.

## How to Contribute

### Reporting Bugs

Open an issue at https://github.com/cellmind-team/cellmind with:
- Clear title describing the bug
- Steps to reproduce
- Expected vs actual behavior
- Python version and CellMind version

### Suggesting Features

Open an issue with tag `enhancement`. Describe:
- The problem you're solving
- How your proposed feature addresses it
- Any relevant context or references

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests for your change
4. Ensure all tests pass: `pytest tests/ -v`
5. Commit with clear message
6. Open a PR against `main`

### Code Style

- Python 3.8+
- Follow PEP 8
- Add docstrings for all public methods
- Keep lines under 100 characters

### Testing

All new features must include tests. Run the test suite:
```bash
pytest tests/ -v --tb=short
```

Coverage target: 90%+ for new code.

## Good First Issues

Look for issues tagged `good first issue`:
- Fix typos in documentation
- Add examples to README
- Write additional tests for untested modules
- Translate docs to other languages

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.