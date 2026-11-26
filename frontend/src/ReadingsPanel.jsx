import { useState } from "react";
import { api } from "./api.js";
import { CATEGORIES, SEGMENTS, formatDate } from "./demoData.js";

const today = () => new Date().toISOString().slice(0, 10);
const num = (v) => (v === "" || v == null ? null : Number(v));

// Manual entry of body-composition readings (typed in from a Tanita printout). Stored as typed; nothing is
// computed from the scan. A category is stored only if chosen — no standard ranges are applied here.
export default function ReadingsPanel({ data, onChange }) {
  const [f, setF] = useState({ date: today(), weight_kg: "", body_fat_pct: "", muscle_kg: "", note: "" });
  const [withSeg, setWithSeg] = useState(false);
  const [seg, setSeg] = useState({});
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [tg, setTg] = useState({ weight_kg: "", body_fat_pct: "", muscle_kg: "" });

  const run = async (fn) => {
    setErr(""); setBusy(true);
    try { await fn(); await onChange(); } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };
  const submit = (e) => {
    e.preventDefault();
    const body = { date: f.date, weight_kg: num(f.weight_kg), body_fat_pct: num(f.body_fat_pct),
      muscle_kg: num(f.muscle_kg), note: f.note, segments: null };
    if (withSeg) body.segments = Object.fromEntries(SEGMENTS.map((s) => [s.key,
      { fat_pct: num(seg[s.key]?.fat_pct), category: seg[s.key]?.category || null }]));
    run(async () => { await api.addReading(body); setF({ ...f, weight_kg: "", body_fat_pct: "", muscle_kg: "", note: "" }); setSeg({}); });
  };
  const field = (key, label, step = "0.1") => (
    <label>{label}<input type="number" step={step} value={f[key]} required
      onChange={(e) => setF({ ...f, [key]: e.target.value })} /></label>
  );

  return (
    <section className="card readings">
      <h3>My readings</h3>
      <p className="muted">Type values from your Tanita printout. They are stored on this computer exactly as entered
        and are not checked against the scan. Nothing is classified for you: pick a category only if the printout shows one.</p>

      <form onSubmit={submit} className="rform">
        <label>Date<input type="date" value={f.date} required max={today()} onChange={(e) => setF({ ...f, date: e.target.value })} /></label>
        {field("weight_kg", "Weight (kg)")}
        {field("body_fat_pct", "Body fat (%)")}
        {field("muscle_kg", "Muscle (kg)")}
        <label className="check"><input type="checkbox" checked={withSeg} onChange={(e) => setWithSeg(e.target.checked)} /> Add segment body fat</label>
        {withSeg && (
          <div className="segs">
            {SEGMENTS.map((s) => (
              <div key={s.key}>
                <small>{s.label}</small>
                <input type="number" step="0.1" placeholder="%" required value={seg[s.key]?.fat_pct ?? ""}
                  onChange={(e) => setSeg({ ...seg, [s.key]: { ...seg[s.key], fat_pct: e.target.value } })} />
                <select value={seg[s.key]?.category ?? ""} onChange={(e) => setSeg({ ...seg, [s.key]: { ...seg[s.key], category: e.target.value } })}>
                  <option value="">no category</option>
                  {CATEGORIES.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
                </select>
              </div>
            ))}
          </div>
        )}
        <button type="submit" className="primary" disabled={busy}>Add reading</button>
      </form>
      {err && <p className="bad">{err}</p>}

      {data.readings.length > 0 && (
        <div className="table-wrap">
        <table className="rtable">
          <thead><tr><th>Date</th><th>Weight</th><th>Body fat</th><th>Muscle</th><th>Segments</th><th /></tr></thead>
          <tbody>
            {data.readings.map((r) => (
              <tr key={r.id}>
                <td>{formatDate(r.date)}</td><td>{r.weight_kg} kg</td><td>{r.body_fat_pct} %</td><td>{r.muscle_kg} kg</td>
                <td>{r.segments ? "yes" : "–"}</td>
                <td><button className="linkish" disabled={busy} onClick={() => run(() => api.deleteReading(r.id))}>Delete</button></td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      )}

      <details>
        <summary>Profile and target (optional)</summary>
        <div className="rform">
          <label>Sex<select value={data.profile?.sex ?? ""} onChange={(e) => run(() => api.putProfile({ ...data.profile, sex: e.target.value || null }))}>
            <option value="">–</option><option>Male</option><option>Female</option></select></label>
          <label>Height (cm)<input type="number" defaultValue={data.profile?.height_cm ?? ""}
            onBlur={(e) => run(() => api.putProfile({ ...data.profile, height_cm: num(e.target.value) }))} /></label>
          {["weight_kg", "body_fat_pct", "muscle_kg"].map((k, i) => (
            <label key={k}>Target {["weight (kg)", "body fat (%)", "muscle (kg)"][i]}
              <input type="number" step="0.1" value={tg[k] || (data.target?.[k] ?? "")} onChange={(e) => setTg({ ...tg, [k]: e.target.value })} /></label>
          ))}
          <button disabled={busy} onClick={() => run(() => api.putTarget({ weight_kg: num(tg.weight_kg || data.target?.weight_kg),
            body_fat_pct: num(tg.body_fat_pct || data.target?.body_fat_pct), muscle_kg: num(tg.muscle_kg || data.target?.muscle_kg) }))}>Save target</button>
          {data.target && <button className="linkish" onClick={() => run(() => api.deleteTarget())}>Remove target</button>}
        </div>
      </details>

      <p className="muted">Tanita file import will be added once a sample export is available.</p>
    </section>
  );
}
