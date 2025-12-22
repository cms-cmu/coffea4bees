#!/bin/bash

echo "############### Running hemisphere mixing test"
#python coffea4bees/hemisphere_mixing/tests/test_mixing.py 

#python -m unittest coffea4bees.hemisphere_mixing.tests.test_mixing.mixingTestCase.test_reading_hemisphere_library
#python -m unittest coffea4bees.hemisphere_mixing.tests.test_mixing.mixingTestCase.test_hemi_making
#python -m unittest coffea4bees.hemisphere_mixing.tests.test_mixing.mixingTestCase.test_reading_all_hemisphere_libraries
python -m unittest coffea4bees.hemisphere_mixing.tests.test_mixing.mixingTestCase.test_hemi_mixing
#python -m unittest coffea4bees.hemisphere_mixing.tests.test_mixing.mixingTestCase.test_chatGPT
