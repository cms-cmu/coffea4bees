# change these vars #
export LPCUSER="algomez"
export CERNUSER="a/algomez"
export BASE="root://cmseos.fnal.gov//store/user/${LPCUSER}/XX4b/2024_v2/nominal_sel/"
export MODEL="${BASE}/classifier/SvB_nominal/"
export SvB="${BASE}/friend/SvB_nominal/"
export FvT="${BASE}/friend/FvT_nominal/"
export PLOT="root://eosuser.cern.ch//eos/user/${CERNUSER}/www/HH4b/classifier/HH4b_2024_v2/"
#####################

export WFS="coffea4bees/classifier/config/workflows/HH4b_2024_v2/SvB"

# check port
if [ -z "$1" ]; then
    port=10200
else
    port=$1
fi


# train
./src/pyml.py \
    template "{model: ${MODEL}, FvT: ${FvT}}" $WFS/train.yml \
    -from $WFS/../common.yml \
    -setting Monitor "address: :${port}" -flag debug

# plot the AUC and ROC
# ./src/pyml.py analyze \
#     --results ${MODEL}/result.json \
#     -analysis HCR.LossROC \
#     -setting IO "output: ${PLOT}" \
#     -setting IO "report: SvB" \
#     -setting Monitor "address: :${port}"

# # evaluate
# ./src/pyml.py \
#     template "{model: ${MODEL}, SvB: ${SvB}}" $WFS/evaluate.yml \
#     -from $WFS/../common.yml \
#     -setting Monitor "address: :${port}"
