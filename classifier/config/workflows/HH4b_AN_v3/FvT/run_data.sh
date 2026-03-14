export LPCUSER="chuyuanl"
export WFS="coffea4bees/classifier/config/workflows/HH4b_AN_v3/FvT"
export GMAIL=~/gmail.yml

# check port
if [ -z "$1" ]; then
    port=10200
else
    port=$1
fi

# train
./src/pyml.py from $WFS/train.yml \
    -template "user: ${LPCUSER}" $WFS/train_data.yml \
    -setting Monitor "address: :${port}" -flag debug

# evaluate
./src/pyml.py template "user: ${LPCUSER}" $WFS/evaluate_data.yml

# merge
./src/pyml.py template "user: ${LPCUSER}" $WFS/merge_data.yml

if [ -e "$GMAIL" ]; then
    ./src/pyml.py analyze \
        -analysis notify.Gmail \
        --title "FvT done" \
        --body "All jobs done at $(date)" \
        --labels HH4b AN_v4 \
        -from $GMAIL \
        -setting Monitor "address: :${port}"
fi