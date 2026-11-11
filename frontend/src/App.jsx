import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

export default function App() {
  const [data, setData] = useState([]);
  const [best, setBest] = useState(null);

  useEffect(() => {
    fetch("/baseline_metrics.json")
      .then((r) => r.json())
      .then((d) => { setData(d.history); setBest(d.best_dice_mean); })
      .catch(() => {});
  }, []);

  return (
    <div style={{ maxWidth: 900, margin: "2rem auto", fontFamily: "sans-serif" }}>
      <h1>FedMed Training Dashboard</h1>
      <p>Centralized baseline {best !== null && `| Best mean Dice: ${best.toFixed(4)}`}</p>
      {data.length === 0 && <p>No results loaded yet.</p>}
      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="epoch" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="train_loss" stroke="#e45756" />
          <Line type="monotone" dataKey="dice_mean" stroke="#4c78a8" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}