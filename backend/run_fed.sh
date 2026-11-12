#!/bin/bash
# usage: bash run_fed.sh ROUNDS MAX_TRAIN LOCAL_EPOCHS [--tls]
ROUNDS=${1:-5}; MAXTRAIN=${2:-60}; EPOCHS=${3:-1}; TLS=${4:-}
pkill -f src.federated 2>/dev/null; sleep 2
mkdir -p logs results
python -m src.federated.server --rounds $ROUNDS $TLS > logs/server.log 2>&1 &
sleep 5
for i in 0 1 2; do
  python -m src.federated.client --id $i --max-train $MAXTRAIN --local-epochs $EPOCHS $TLS > logs/client$i.log 2>&1 &
done
wait
echo "All finished"
grep -E "SSL|global Dice" logs/server.log
