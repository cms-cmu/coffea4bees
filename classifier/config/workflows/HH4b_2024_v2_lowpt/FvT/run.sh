# change these vars #
export LPCUSER="algomez"
export CERNUSER="a/algomez"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/XX4b/2024_v2/"
export MODEL="${BASE}/classifier/FvT/"
export FvT="${BASE}/friend/FvT/"
export PLOT="root://eosuser.cern.ch//eos/user/${CERNUSER}/www/HH4b/$(date +%Y%m%d)_classifier_lowpt/"
#####################

export CLASSIFIER_CONFIG_PATHS="coffea4bees"
export WFS="coffea4bees/classifier/config/workflows/HH4b_2024_v2_lowpt/FvT"

# train with train.yml and common.yml configs
# ./src/pyml.py \
#     template "model: ${MODEL}" $WFS/train.yml \
#     -from $WFS/../common.yml \
#     -setting Monitor "enable: false" \
#     -flag debug # use debug flag

# # plot the AUC and ROC
# ./src/pyml.py analyze \
#     --results ${MODEL}/result.json \
#     -analysis HCR.LossROC \
#     -setting IO "output: ${PLOT}" \
#     -setting IO "report: FvT" \
#     -setting Monitor "enable: false"

# # evaluate with evaluate.yml and common.yml configs
./src/pyml.py \
    template "{model: ${MODEL}, FvT: ${FvT}}" $WFS/evaluate.yml \
    -from $WFS/../common.yml \
    -setting Monitor "enable: false"
