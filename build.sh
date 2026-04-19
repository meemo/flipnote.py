#!/bin/bash
set -e

VENV_DIR=".venv"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install build tools
pip install -U build twine

# Build package
python3 -m build

# Upload package to PyPi.org
python3 -m twine upload --repository pypi dist/*

# Move uploaded package files to the archive folder
mkdir -p dist_archive
mv dist/* dist_archive/

echo "Done."
