# change these vars #
export LPCUSER="algomez"
export CERNUSER="a/algomez"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/XX4b/2024_v2/"
export MODEL="${BASE}/classifier/SvB_lowpt/"
export SvB="${BASE}/friend/SvB_lowpt/"
export FvT="${BASE}/friend/FvT_lowpt/"
export PLOT="root://eosuser.cern.ch//eos/user/${CERNUSER}/www/HH4b/$(date +%Y%m%d)_classifier_SvB_lowpt/"
#####################

export CLASSIFIER_CONFIG_PATHS="coffea4bees"
export WFS="coffea4bees/classifier/config/workflows/HH4b_2024_v2_lowpt/SvB"
export PYTHONUNBUFFERED=1

# Generate a random port between 10000 and 60000 so jobs don't crash
export PORT=$(shuf -i 10000-60000 -n 1)

# train
echo "[$(date)] Starting SvB training on port ${PORT}..."
./src/pyml.py \
    template "{model: ${MODEL}, FvT: ${FvT}}" $WFS/train.yml \
    -from $WFS/../common.yml \
    -setting Monitor "address: '127.0.0.1:${PORT}'" \
    -flag debug
echo "[$(date)] Training done."

# # plot the AUC and ROC
# echo "[$(date)] Starting AUC/ROC plot..."
# ./src/pyml.py analyze \
#     --results ${MODEL}/result.json \
#     -analysis HCR.LossROC \
#     -setting IO "output: ${PLOT}" \
#     -setting IO "report: SvB" \
#     -setting Monitor "enable: false"
# echo "[$(date)] Plotting done."

# # evaluate
# ./src/pyml.py \
#     template "{model: ${MODEL}, SvB: ${SvB}}" $WFS/evaluate.yml \
#     -from $WFS/../common.yml \
#     -setting Monitor "enable: false"
