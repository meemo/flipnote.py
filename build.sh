#!/bin/bash

# Install latest build dependencies
python3 -m pip install -U -r requirements.txt

# Build package
python3 -m build

# Upload package to PyPi.org
python3 -m twine upload --repository pypi dist/*

# Move uploaded package files to the archive folder
mkdir -p dist_archive
mv dist/* dist_archive/
