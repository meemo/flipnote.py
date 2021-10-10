#!/bin/bash

# Build package
python3 -m build

# Upload package to PyPi.org
python3 -m twine upload --repository pypi dist/*

# Move uploaded package files to the archive folder
mv dist/* dist_archive/
