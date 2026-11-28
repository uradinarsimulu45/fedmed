import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

function useJson(path) {
  const [data, setData] = useState(null);
  useEffect(() => {
    fetch(path).then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setData).catch(() => setData(null));
  }, [path]);
  return data;
}

function Chart({ data, xKey, lines }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xKey} />
        <YAxis domain={[0, 0.5]} />
        <Tooltip />
        <Legend />
        {lines.map((l) => (
          <Line key={l.key} type="monotone" dataKey={l.key} stroke={l.color} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
function LiveFederated() {
  const [data, setData] = useState({ plain: [], encrypted: [] });
  const [status, setStatus] = useState("connecting...");
  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");
    ws.onopen = () => setStatus("live");
    ws.onclose = () => setStatus("disconnected (start the API)");
    ws.onmessage = (e) => setData(JSON.parse(e.data));
    return () => ws.close();
  }, []);
  return (
    <>
      <h2>Live encrypted federated training: {status}</h2>
      <Chart data={data.encrypted} xKey="round"
        lines={[{ key: "dice_mean", color: "#54a24b" }, { key: "dice_wt", color: "#f58518" }]} />
    </>
  );
}
function EncryptedVsPlain() {
  const plain = useJson("/federated_audit_15rounds.json");
  const enc = useJson("/federated_he_audit_15rounds.json");
  if (!plain || !enc) return <p>Encrypted comparison data not found in frontend/public.</p>;
  const n = Math.min(plain.history.length, enc.history.length);
  const data = Array.from({ length: n }, (_, i) => ({
    round: i + 1,
    plain: plain.history[i].dice_mean,
    encrypted: enc.history[i].dice_mean,
  }));
  return (
    <>
      <h2>Encrypted vs plain FedAvg (first {n} rounds)</h2>
      <p>
        Upload per hospital per round: plain 4.8 MB | encrypted{" "}
        {enc.history[0].upload_mb_per_hospital} MB. Server holds no secret key.
      </p>
      <Chart data={data} xKey="round"
        lines={[{ key: "plain", color: "#4c78a8" }, { key: "encrypted", color: "#e45756" }]} />
    </>
  );
}

export default function App() {
  const base = useJson("/baseline_audit_15epochs.json");
  const fed = useJson("/federated_audit_15rounds.json");

  const merged = base && fed
    ? Array.from({ length: Math.min(base.history.length, fed.history.length) }, (_, i) => ({
        step: i + 1,
        baseline: base.history[i].dice_mean,
        federated: fed.history[i].dice_mean,
      }))
    : [];

  const bestB = base ? Math.max(...base.history.map((r) => r.dice_mean)) : null;
  const bestF = fed ? Math.max(...fed.history.map((r) => r.dice_mean)) : null;

  return (
    <div style={{ maxWidth: 900, margin: "2rem auto", fontFamily: "sans-serif", padding: "0 1rem" }}>
      <h1>FedMed Training Dashboard</h1>

      <h2>Hospital nodes</h2>
      <div style={{ display: "flex", gap: 12 }}>
        {[0, 1, 2].map((i) => (
          <div key={i} style={{ flex: 1, border: "1px solid #ccc", borderRadius: 8, padding: 12 }}>
            <strong>Hospital {i}</strong>
            <div>20 private cases</div>
            <div>TLS: enabled</div>
          </div>
        ))}
      </div>

      <h2>Federated vs centralized (mean Dice)</h2>
      {bestB !== null && bestF !== null && (
        <p>
          Best baseline: {bestB.toFixed(4)} | Best federated: {bestF.toFixed(4)} |
          Ratio: {Math.round((bestF / bestB) * 100)}%
        </p>
      )}
      {merged.length > 0 ? (
        <Chart data={merged} xKey="step"
          lines={[{ key: "baseline", color: "#4c78a8" }, { key: "federated", color: "#54a24b" }]} />
      ) : (
        <p>Audit files not found in frontend/public.</p>
      )}

      {fed && (
        <>
          <h2>Federated Dice by tumor region</h2>
          <Chart data={fed.history} xKey="round"
            lines={[
              { key: "dice_tc", color: "#4c78a8" },
              { key: "dice_wt", color: "#f58518" },
              { key: "dice_et", color: "#e45756" },
              { key: "dice_mean", color: "#54a24b" },
            ]} />
        </>
      )}
            <h2>Differential privacy trade-off</h2>
      <img src="/dp_tradeoff.png" alt="DP trade-off" style={{ maxWidth: "100%" }} />
      <p>Clip 1.0, 5 rounds. Epsilon is a loose upper bound.</p>
      <h2>Tumor segmentation (centralized baseline model)</h2>
<p>Left to right: MRI, true tumor, predicted tumor. Baseline Dice is about 0.3, so masks are rough.</p>
{[0, 1, 2].map((i) => (
  <img key={i} src={`/masks/case${i}.png`} alt={`case ${i}`} style={{ maxWidth: "100%" }} />
))}
      
    </div>
  );
}