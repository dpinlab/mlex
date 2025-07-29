# MLEX PyPI Deployment Guide

This guide walks you through the process of publishing the MLEX framework to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account on [PyPI](https://pypi.org/account/register/)
2. **TestPyPI Account**: Create an account on [TestPyPI](https://test.pypi.org/account/register/)
3. **API Tokens**: Generate API tokens for both PyPI and TestPyPI
4. **GitHub Repository**: Ensure your code is in a GitHub repository

## Step 1: Setup API Tokens

### PyPI API Token
1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Click "Add API token"
3. Give it a name (e.g., "MLEX Upload Token")
4. Select "Entire account (all projects)"
5. Copy the token (you'll only see it once!)

### TestPyPI API Token
1. Go to [TestPyPI Account Settings](https://test.pypi.org/manage/account/)
2. Follow the same steps as above
3. Copy the token

## Step 2: Configure GitHub Secrets

Add these secrets to your GitHub repository:

1. Go to your repository → Settings → Secrets and variables → Actions
2. Add the following secrets:
   - `PYPI_API_TOKEN`: Your PyPI API token
   - `TEST_PYPI_API_TOKEN`: Your TestPyPI API token

## Step 3: Update Package Information

Before publishing, update these files:

### 1. Update `pyproject.toml`
- Change `your-email@example.com` to your actual email
- Update the GitHub URLs to match your repository
- Update version number if needed

### 2. Update `mlex/__init__.py`
- Update the email address
- Update version number to match `pyproject.toml`

### 3. Update `README.md`
- Update GitHub URLs
- Update email address
- Verify installation instructions

## Step 4: Test Build Locally

```bash
# Activate your virtual environment
source mlex-venv/bin/activate

# Install build tools
pip install build twine

# Build the package
python -m build

# Check the package
twine check dist/*

# Test upload to TestPyPI
twine upload --repository testpypi dist/*
```

## Step 5: Test Installation

Test the package installation from TestPyPI:

```bash
# Create a new virtual environment for testing
python -m venv test-env
source test-env/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ mlex

# Test the installation
python -c "import mlex; print(mlex.__version__)"
```

## Step 6: Publish to PyPI

### Option A: Manual Upload
```bash
# Upload to PyPI
twine upload dist/*
```

### Option B: GitHub Actions (Recommended)
1. Create a new release on GitHub
2. Tag it with the version (e.g., `v0.1.0`)
3. The GitHub Actions workflow will automatically publish to PyPI

## Step 7: Verify Publication

1. Check your package on PyPI: https://pypi.org/project/mlex/
2. Test installation: `pip install mlex`
3. Verify functionality: `python -c "import mlex; print(mlex.__version__)"`

## Step 8: Update Documentation

1. Update your GitHub repository README
2. Create documentation on Read the Docs (optional)
3. Update any external references to your package

## Troubleshooting

### Common Issues

1. **Package name already exists**: Choose a different name or contact the owner
2. **Build errors**: Check your `pyproject.toml` syntax
3. **Import errors**: Verify your package structure and `__init__.py` files
4. **Authentication errors**: Check your API tokens

### Version Management

- Use semantic versioning (MAJOR.MINOR.PATCH)
- Update version in both `pyproject.toml` and `mlex/__init__.py`
- Create a new release for each version

### Security Best Practices

- Never commit API tokens to version control
- Use GitHub Secrets for sensitive information
- Regularly rotate your API tokens
- Use TestPyPI for testing before publishing to PyPI

## Maintenance

### Updating the Package

1. Make your changes
2. Update version numbers
3. Update CHANGELOG.md
4. Test locally
5. Create a new release on GitHub
6. GitHub Actions will automatically publish

### Deprecating the Package

If you need to deprecate the package:

1. Add a deprecation notice to the README
2. Update the package description
3. Consider transferring ownership if appropriate

## Support

For issues with PyPI publishing:
- [PyPI Help](https://pypi.org/help/)
- [Python Packaging User Guide](https://packaging.python.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions) 