#!/bin/bash
# Render build script to ensure proper setuptools installation

set -e

echo "=== Installing build tools first ==="
pip install --upgrade pip setuptools wheel

echo "=== Installing requirements ==="
pip install -r requirements.txt

echo "=== Build completed successfully ==="
