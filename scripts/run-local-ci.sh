#!/bin/bash

# Check if the job name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <job_name>"
fi

JOB_NAME=$1
SNAKEFILE="coffea4bees/workflows/Snakefile_testCI.smk"

# Check if JOB_NAME contains '-'
if [[ "$JOB_NAME" == *-* ]]; then
  # Replace '-' with '_'
  JOB_NAME=${JOB_NAME//-/_}
fi

# Check if the folder named 'output' exists
if [ -d "CI_output" ]; then
  echo "The folder 'CI_output' exists. Remember that snakemake will not run a step if the output files already exist."
else
  echo "Output files will be created in the 'CI_output' folder."
fi

# Check if the file ~/x509up* exists
if ls ~/x509up* 1> /dev/null 2>&1; then
  echo "Copying ~/x509up* to /proxy/x509_proxy."
  /bin/cp ~/x509up* proxy/x509_proxy
else
  echo "File ~/x509up* does not exist. Run voms-proxy-init -voms cms before running this script."
  return 0
fi

./run_container snakemake --snakefile $SNAKEFILE --use-apptainer --apptainer-args "--bind $PWD:/srv --pwd /srv" --cores 1 $JOB_NAME
