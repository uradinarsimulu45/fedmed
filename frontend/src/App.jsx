import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

// Mock data for now; Day 13+ replaces this with a live WebSocket feed
const mock = Array.from({ length: 20 }, (_, i) => ({
  round: i + 1,
  loss: +(1.0 * Math.exp(-i / 6) + 0.05).toFixed(3),
  dice: +(0.85 * (1 - Math.exp(-i / 5))).toFixed(3),
}));

export default function App() {
  return (
    <div style={{ maxWidth: 900, margin: "2rem auto", fontFamily: "sans-serif" }}>
      <h1>FedMed Training Dashboard</h1>
      <p>Global model convergence across federated rounds (mock data)</p>
      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={mock}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="round" label={{ value: "Round", position: "insideBottom", offset: -5 }} />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="loss" stroke="#e45756" dot={false} />
          <Line type="monotone" dataKey="dice" stroke="#4c78a8" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}