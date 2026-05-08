# Contributing to CellMind

Thank you for your interest in contributing to CellMind.

## What is CellMind?

CellMind is a biological cell-based AI memory architecture with:
- Hebbian learning mechanisms
- Emotional state systems
- REM-like memory consolidation
- Temporal coherence memory loops

## Prerequisites

- **Python 3.8+**
- **pip** (comes with Python)
- **Git** (for cloning and branching)

Optional for development:
- **pytest** (for running tests)
- **flake8** or **ruff** (for code linting)

## Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/cellmind.git
cd cellmind
```

### 2. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scriptsctivate   # Windows

# Install in editable mode with dev dependencies
pip install -e .
pip install pytest
```

### 3. Verify Installation

```bash
python -c "import cellmind; print('CellMind installed successfully')"
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=. --cov-report=term-missing
pytest tests/test_cmind.py -v
```

## Code Style

- Follow **PEP 8** guidelines
- Maximum line length: **100 characters**
- Add **docstrings** for all public methods
- Use **type hints** where appropriate

## Project Structure

```
cellmind/
├── cellmind/          # Main package
│   └── cell_core/     # Core memory engine
├── tests/             # Test suite
├── examples/          # Usage examples
└── docs/              # Documentation
```

## Submitting Changes

### 1. Create a Branch

```bash
git checkout -b feature/my-feature
git checkout -b fix/my-bug-fix
```

### 2. Make Your Changes

- Write code following the style guidelines
- Add or update tests
- Update documentation if needed

### 3. Commit with DCO Sign-off

CellMind uses the **Developer Certificate of Origin (DCO)**.

```bash
git add .
git commit -s -m "Add: brief description of changes"
```

The `-s` flag adds a Signed-off-by line automatically.

### 4. Push and Open PR

```bash
git push origin feature/my-feature
```

Then open a Pull Request on GitHub.

## Pull Request Checklist

- [ ] Code follows PEP 8 style guidelines
- [ ] All tests pass: `pytest tests/ -v`
- [ ] New code includes tests
- [ ] Documentation updated if needed
- [ ] Commit message includes DCO sign-off
- [ ] PR description explains the change

## Good First Issues

Looking for a place to start? Check these:

- `good first issue` tagged issues — beginner-friendly tasks
- Fix typos in documentation
- Add examples to README
- Write additional tests for untested modules
- Improve error messages

## License

By contributing to CellMind, you agree that your contributions will be licensed under the **Apache License 2.0**.
