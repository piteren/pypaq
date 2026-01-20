#!/bin/bash

# Check if a tag parameter is provided and if it follows the specific pattern
if [ "$#" -ne 1 ] || ! [[ "$1" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Usage: $0 <tag> where <tag> is like 'v1.2.3', example: $0 v1.16.1"
    exit 1
fi

TAG=$1
SETUP_PY_FILE="setup.py"

# Update the version in setup.py
sed -i "8s/version=.*',/version=            '$TAG',/" "$SETUP_PY_FILE"

# Check if sed command succeeded
if [ $? -ne 0 ]; then
    echo "Error updating version in $SETUP_PY_FILE"
    exit 1
fi

echo "Version updated to $TAG in $SETUP_PY_FILE"

# Commit the change
git add "$SETUP_PY_FILE"
git commit -m "Update version to $TAG"

# Check if git commit succeeded
if [ $? -ne 0 ]; then
    echo "Error committing changes"
    exit 1
fi

# Tag the commit
git tag -a "$TAG" -m "Version $TAG"

# Check if git tag succeeded
if [ $? -ne 0 ]; then
    echo "Error tagging commit"
    exit 1
fi

# Push the commit and tag to remote
git push origin master --tags

# Check if git push succeeded
if [ $? -ne 0 ]; then
    echo "Error pushing changes"
    exit 1
fi

echo "Version $TAG committed, tagged, and pushed successfully."

python setup.py sdist bdist_wheel
python -m twine upload -r pypi dist/*
rm -rf build
rm -rf dist
rm -rf *.egg-info

echo "Version $TAG published to pypi."