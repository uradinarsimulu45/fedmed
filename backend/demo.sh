#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p logs results

echo "=== 1. Federated vs centralized (audit table) ==="
python -m src.audit_table | tail -2

echo; echo "=== 2. Payload audit: what leaves a hospital ==="
python -m src.federated.audit_payload 2>/dev/null

echo; echo "=== 3. TLS: a plain client must be rejected ==="
pkill -f src.federated 2>/dev/null; sleep 2
python -m src.federated.server --rounds 1 --tls > logs/demo_server.log 2>&1 &
sleep 5
timeout 40 python -m src.federated.client --id 0 --max-train 6 2>&1 | grep -E "UNAVAILABLE|Socket closed" | head -2
pkill -f src.federated 2>/dev/null; sleep 2

echo; echo "=== 4. Crash survival: hospital 2 dies in round 2 ==="
python -m src.federated.server --rounds 3 --tls --timeout 300 > logs/server.log 2>&1 &
sleep 5
python -m src.federated.client --id 0 --max-train 12 --tls > logs/client0.log 2>&1 &
python -m src.federated.client --id 1 --max-train 12 --tls > logs/client1.log 2>&1 &
python -m src.federated.client --id 2 --max-train 12 --tls --crash-at-round 2 > logs/client2.log 2>&1 &
wait
grep -E "aggregate_fit|Run finished" logs/server.log
grep "simulated crash" logs/client2.log
echo; echo "Demo complete."
