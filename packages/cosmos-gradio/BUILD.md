# Building the cosmos-gradio Package

This document provides instructions for building the cosmos-gradio package using two different approaches.

## Prerequisites

- Python 3.10 or higher
- Git (for cloning the repository)
- just: apt install just

## Option 1: pip + build (Recommended for CI/CD and Publishing)

The `pip + build` approach uses the standard Python build tools and is widely supported across different environments.

### Installation

```bash
# Install the build tool
pip install build

# If you're in an externally managed environment (like some Docker containers)
pip install build --break-system-packages
```

### Building the Package

```bash
# Navigate to the project directory
cd cosmos-gradio

# Build both wheel and source distributions
python -m build
```

This creates:
- `dist/cosmos_gradio-0.1.0-py3-none-any.whl` (wheel distribution)
- `dist/cosmos_gradio-0.1.0.tar.gz` (source distribution)



## Option 2: uvx + hatchling (Recommended for Local Development)

The `uvx + hatchling` approach uses modern Python tooling for faster, cleaner builds.

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using brew
brew install uv
```

### Building the Package

```bash
# Navigate to the project directory
cd cosmos-gradio

# Build using uvx with hatchling directly
uvx hatchling build
```

This creates:
- `dist/cosmos_gradio-0.1.0-py3-none-any.whl` (wheel distribution)

### Alternative uvx Commands

```bash
# Using build tool via uvx
uvx --from build pyproject-build

# Using uv directly (also works)
uv build
```

## Publishing

After building, you can publish to PyPI:

```bash
# Using the project's just command (if available)
just publish <pypi_token>

# Or using twine directly
pip install twine
twine upload dist/*
```

---

For more information about the cosmos-gradio package, see the main [README.md](README.md).
