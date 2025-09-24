export REANA_SERVER_URL=https://reana.cern.ch
export REANA_ACCESS_TOKEN="${REANA_TOKEN}"
export workflow_name="${1:-test}"
reana-client ping
echo """
##########################################################
#### THIS JOB WILL FAILED IF YOU DONT HAVE A REANA ACCOUNT
#### AND THE REANA TOKEN AS CI SECRETS VARIABLE
#### BECAUSE AT THE MOMENT REANA DOES NOT HAVE GROUP ACCOUNTS
#### BUT IT HAS TO RUN IN THE CMU CENTRAL REPO AND IT IS
#### ALLOWED TO FAILED FOR MERGE REQUEST
##########################################################
"""

if [[ "$workflow_name" == coffea4bees_* ]]; then
    hash="${workflow_name#coffea4bees_}"
else
    hash="$(git rev-parse --short HEAD)"
fi

# Check if we're in the right directory, if not try to navigate to barista_framework
if [ ! -f "coffea4bees/workflows/inputs_reana.yaml" ]; then
    echo "Warning: coffea4bees/workflows/inputs_reana.yaml not found in current directory"
    echo "Attempting to change to barista_framework directory..."
    cd barista_framework
    if [ ! -f "coffea4bees/workflows/inputs_reana.yaml" ]; then
        echo "Error: coffea4bees/workflows/inputs_reana.yaml not found in barista_framework either"
        echo "Current directory: $(pwd)"
        echo "Available files:"
        ls -la
        exit 1
    fi
    echo "Successfully found inputs_reana.yaml in barista_framework"
fi

sed -e 's|--githash.*|--githash '${hash}'"|' -i coffea4bees/workflows/inputs_reana.yaml
git diff HEAD > gitdiff.txt
cat coffea4bees/workflows/inputs_reana.yaml
reana-client run -f coffea4bees/workflows/reana.yaml -w ${workflow_name}
