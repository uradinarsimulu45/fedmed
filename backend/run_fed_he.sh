#!/bin/bash
# usage: bash run_fed_he.sh ROUNDS MAX_TRAIN LOCAL_EPOCHS [--tls]
ROUNDS=${1:-2}; MAXTRAIN=${2:-12}; EPOCHS=${3:-1}; TLS=${4:-}
pkill -f src.federated 2>/dev/null; sleep 2
mkdir -p logs results
python -m src.federated.server_he --rounds $ROUNDS $TLS > logs/server_he.log 2>&1 &
sleep 5
for i in 0 1 2; do
  python -m src.federated.client_he --id $i --max-train $MAXTRAIN --local-epochs $EPOCHS $TLS > logs/client_he$i.log 2>&1 &
done
wait
echo "All finished"
grep -E "secret key|WITHOUT|global Dice|Run finished" logs/server_he.log
grep -h "sent" logs/client_he*.log
