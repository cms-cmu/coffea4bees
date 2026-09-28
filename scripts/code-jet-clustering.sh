#!/bin/bash
set -e

echo "############### Running jet clustering test"
python coffea4bees/jet_clustering/tests/test_clustering.py 
python coffea4bees/jet_clustering/tests/test_splitting_library.py
