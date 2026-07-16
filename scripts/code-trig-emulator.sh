#!/bin/bash

echo "############### Running trigger emulator test"
python -m unittest coffea4bees.analysis.tests.test_trigger_emulator

python -m unittest coffea4bees.analysis.tests.test_vectorized_trigger_sf

