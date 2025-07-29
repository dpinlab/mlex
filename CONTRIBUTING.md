# Contributing to MLEX

Thank you for your interest in contributing to the Money Laundering Expert System (MLEX)! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a virtual environment**:
   ```bash
   python -m venv mlex-venv
   source mlex-venv/bin/activate  # On Windows: mlex-venv\Scripts\activate
   ```
4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Setup

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run the formatters:
```bash
black mlex/
isort mlex/
```

Run the linters:
```bash
flake8 mlex/
mypy mlex/
```

### Pre-commit Hooks

Install pre-commit hooks to automatically format and check your code:
```bash
pre-commit install
```

### Running Tests

Run the test suite:
```bash
pytest tests/
```

Run tests with coverage:
```bash
pytest tests/ --cov=mlex --cov-report=html
```

## Making Changes

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards

3. **Add tests** for new functionality

4. **Update documentation** if needed

5. **Commit your changes** with a descriptive message:
   ```bash
   git commit -m "Add new feature: description of what was added"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## Pull Request Guidelines

- **Title**: Clear and descriptive
- **Description**: Explain what the PR does and why
- **Tests**: Ensure all tests pass
- **Documentation**: Update docs if needed
- **Type hints**: Add type hints for new functions
- **Docstrings**: Add docstrings for new classes and functions

## Release Process

1. **Update version** in `pyproject.toml` and `mlex/__init__.py`
2. **Update CHANGELOG.md** with new features/fixes
3. **Create a release** on GitHub
4. **GitHub Actions** will automatically publish to PyPI

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Questions?

If you have questions about contributing, please:
- Open an issue on GitHub
- Join our discussions
- Contact the maintainers

Thank you for contributing to MLEX! 