# change these vars #
export LPCUSER="jda102"
export CERNUSER="j/johnda"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/HH4b_Run3_v2"
export MODEL="${BASE}/classifier/MvD/"
export MvD="${BASE}/friend/MvD/"
export PLOT="root://eosuser.cern.ch//eos/user/${CERNUSER}/www/HH4b/classifier/HH4b_Run3_v2/"
#####################

export WFS="coffea4bees/classifier/config/workflows/HH4b_Run3/MvD"

# the first argument can be a port
if [ -z "$1" ]; then
    port=10200
else
    port=$1
fi

# train with train.yml and common.yml configs
./src/pyml.py \
    template "model: ${MODEL}" $WFS/train.yml \
    -from $WFS/../common.yml \
    -setting Monitor "address: :${port}" \
    -flag debug

# plot the AUC and ROC
./src/pyml.py analyze \
    --results ${MODEL}/result.json \
    -analysis HCR.LossROC \
    -setting IO "output: ${PLOT}" \
    -setting IO "report: MvD" \
    -setting Monitor "address: :${port}"

# evaluate with evaluate.yml and common.yml configs
./src/pyml.py \
    template "{model: ${MODEL}, MvD: ${MvD}}" $WFS/evaluate.yml \
    -from $WFS/../common.yml \
    -setting Monitor "address: :${port}"
