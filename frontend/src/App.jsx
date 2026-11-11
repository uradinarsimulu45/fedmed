import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const hospitals = [
  { name: "Hospital 0", cases: 20, status: "Idle" },
  { name: "Hospital 1", cases: 20, status: "Idle" },
  { name: "Hospital 2", cases: 20, status: "Idle" },
];

function useJson(path) {
  const [data, setData] = useState(null);
  useEffect(() => {
    fetch(path)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setData)
      .catch(() => setData(null));
  }, [path]);
  return data;
}

function Chart({ data, xKey, lines }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xKey} />
        <YAxis />
        <Tooltip />
        <Legend />
        {lines.map((l) => (
          <Line key={l.key} type="monotone" dataKey={l.key} stroke={l.color} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export default function App() {
  const baseline = useJson("/baseline_metrics.json");
  const fed = useJson("/federated_metrics.json");

  return (
    <div style={{ maxWidth: 900, margin: "2rem auto", fontFamily: "sans-serif", padding: "0 1rem" }}>
      <h1>FedMed Training Dashboard</h1>

      <h2>Hospital nodes</h2>
      <div style={{ display: "flex", gap: 12 }}>
        {hospitals.map((h) => (
          <div key={h.name} style={{ flex: 1, border: "1px solid #ccc", borderRadius: 8, padding: 12 }}>
            <strong>{h.name}</strong>
            <div>{h.cases} private cases</div>
            <div>Status: {h.status}</div>
          </div>
        ))}
      </div>

      <h2>Centralized baseline</h2>
      {baseline ? (
        <>
          <p>Best mean Dice: {baseline.best_dice_mean.toFixed(4)}</p>
          <Chart
            data={baseline.history}
            xKey="epoch"
            lines={[
              { key: "train_loss", color: "#e45756" },
              { key: "dice_mean", color: "#4c78a8" },
            ]}
          />
        </>
      ) : (
        <p>No baseline results found.</p>
      )}

      <h2>Federated training (FedAvg)</h2>
      {fed ? (
        <Chart
          data={fed.history}
          xKey="round"
          lines={[
            { key: "loss", color: "#e45756" },
            { key: "dice_mean", color: "#54a24b" },
          ]}
        />
      ) : (
        <p>No federated results yet. These appear after Day 5.</p>
      )}
    </div>
  );
}