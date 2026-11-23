"""WebSocket endpoint: streams federated metrics to the dashboard as rounds finish."""
import asyncio, json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FILES = {"plain": "results/federated_metrics.json",
         "encrypted": "results/federated_he_metrics.json",
         "baseline": "results/baseline_metrics.json"}


def read_all():
    out = {}
    for name, path in FILES.items():
        try:
            out[name] = json.load(open(path))["history"]
        except Exception:
            out[name] = []
    return out


@app.get("/metrics")
def metrics():
    return read_all()


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    last = None
    try:
        while True:
            data = read_all()
            if data != last:
                await sock.send_text(json.dumps(data))
                last = data
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
