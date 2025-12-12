# Coffea4bees skimmer

To run the skimmer, remember first to set the coffea environment and your grid certificate. If you followed the instructions in the [README.md](../../README.md), the `set_shell.sh` file must be located right after the package, and then:
```
voms-proxy-init -rfc -voms cms --valid 168:00
cd coffea4bees/ ## Or go to this directory
source set_shell.sh
```

## In this folder

Here you find the code to create `picoAODs` (skims from nanoAOD)

Each folder contains:
 - [metadata](./metadata/): yml files to run the processors
 - [processors](./processors/): python files with the processors for each skimms.

Then, the run-all script is called `runner.py` and it is one directory below (in [coffea4bees/](../../coffea4bees/)). This script will run local or condor depending on the flag used. To learn all the options of the script, just run:
```
# (inside /coffea4bees/coffea4bees/)
python runner.py --help
```

## Run Analysis

### Example to run the analysis

For example, to run a processor you can do:
```
#  (inside /coffea4bees/coffea4bees/)
python runner.py -s -p skimmer/processor/skimmer_4b.py -c skimmer/metadata/HH4b.yml -y UL18 -d TTTo2L2Nu -t
```

The output file of this process will be located under `coffea4bees/skimmer/test/`.
