#!/bin/bash
# Run pourOver unit tests (pure-logic helpers — no coffea file or Flask needed).
set -e

echo "=== pourOver unit tests ==="
python -m unittest discover -s coffea4bees/plots/tests -p pourOver_test.py -v
