#!/bin/bash
set -e

echo "Current directory: $(pwd)"
echo "Listing files:"
ls -la
echo "Python version:"
python3 --version
echo "Starting universal_mind.py..."
exec python3 universal_mind.py
