#!/bin/bash
# usage: bash run_dp.sh ROUNDS MAX_TRAIN CLIP NOISE
ROUNDS=${1:-3}; MAXTRAIN=${2:-12}; CLIP=${3:-0}; NOISE=${4:-0}
pkill -f src.federated 2>/dev/null; sleep 2
mkdir -p logs results
OUT=results/dp_clip${CLIP}_noise${NOISE}.json
python -m src.federated.server --rounds $ROUNDS --tls --out $OUT > logs/server_dp.log 2>&1 &
sleep 5
for i in 0 1 2; do
  python -m src.federated.client --id $i --max-train $MAXTRAIN --tls --dp-clip $CLIP --dp-noise $NOISE > logs/client_dp$i.log 2>&1 &
done
wait
echo "All finished -> $OUT"
grep -h "DP:" logs/client_dp*.log | head -3
grep "global Dice" logs/server_dp.log
