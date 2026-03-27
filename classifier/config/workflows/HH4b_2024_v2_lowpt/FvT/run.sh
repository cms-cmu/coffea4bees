# change these vars #
export LPCUSER="algomez"
export CERNUSER="a/algomez"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/XX4b/2024_v2/"
export MODEL="${BASE}/classifier/FvT_lowpt/"
export FvT="${BASE}/friend/FvT_lowpt/"
export PLOT="root://eosuser.cern.ch//eos/user/${CERNUSER}/www/HH4b/Plots/$(date +%Y%m%d)_classifier_lowpt/"
#####################

export CLASSIFIER_CONFIG_PATHS="coffea4bees"
export WFS="coffea4bees/classifier/config/workflows/HH4b_2024_v2_lowpt/FvT"

# Generate a random port between 10000 and 60000 so jobs don't crash
export PORT=$(shuf -i 10000-60000 -n 1)

train with train.yml and common.yml configs
./src/pyml.py \
    template "model: ${MODEL}" $WFS/train.yml \
    -from $WFS/../common.yml \
    -setting Monitor "address: '127.0.0.1:${PORT}'" \
    -flag debug # use debug flag

# plot the AUC and ROC
./src/pyml.py analyze \
    --results ${MODEL}/result.json \
    -analysis HCR.LossROC \
    -setting IO "output: ${PLOT}" \
    -setting IO "report: FvT" \
    -setting Monitor "address: '127.0.0.1:${PORT}'" 

# evaluate with evaluate.yml and common.yml configs
./src/pyml.py \
    template "{model: ${MODEL}, FvT: ${FvT}}" $WFS/evaluate.yml \
    -from $WFS/../common.yml \
    -setting Monitor "address: '127.0.0.1:${PORT}'" \
    -flag debug
