import { useEffect, useState } from "react";
import { api } from "./api.js";

const fmt = (v, d = 1) => (v == null ? "–" : Number(v).toFixed(d));
const PILL = { measured: "ok", estimated: "warn", unreliable: "bad", unavailable: "" };

// Mesh vs tape for one finished depth job. Values come from backend validation/compare.py; nothing is computed here.
export default function Comparison({ job }) {
  const [res, setRes] = useState(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    setRes(null); setErr("");
    api.comparison(job.id).then(setRes).catch((e) => setErr(e.message));
  }, [job.id]);

  if (err) return <p className="muted">Tape comparison unavailable: {err}</p>;
  if (!res) return null;
  return (
    <div className="comparison">
      <h3>Mesh vs tape</h3>
      {res.rows.length === 0 ? (
        <p className="muted">No tape measurements were entered for this upload. Enter them on the upload form (three
          readings each; for circumferences also the height above the floor where the tape was held) to compare.</p>
      ) : (
        <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Dimension</th><th className="num">Tape (mean ± SD)</th><th className="num">Mesh</th><th className="num">Difference</th><th>Status</th><th>Why</th></tr>
          </thead>
          <tbody>
            {res.rows.map((r) => (
              <tr key={r.name}>
                <td>{r.name}{r.level_mm ? ` @ ${fmt(r.level_mm, 0)} mm` : ""}</td>
                <td className="num">{fmt(r.tape.mean_mm)}{r.tape.sd_mm != null ? ` ± ${fmt(r.tape.sd_mm)}` : ""} mm</td>
                <td className="num">{r.mesh_mm == null ? "–" : `${fmt(r.mesh_mm)} mm`}</td>
                <td className="num">{r.error_mm == null ? "–" : `${r.error_mm > 0 ? "+" : ""}${fmt(r.error_mm)} mm (${fmt(r.error_pct, 1)} %)`}</td>
                <td><span className={`pill ${PILL[r.status] || ""}`}>{r.status}</span></td>
                <td className="muted">{r.why}</td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      )}
      <p className="muted">
        {res.note} Run status: {res.run_status}. Only "measured" values passed every check; "unreliable" values are
        shown for diagnosis only. Definition {res.compare_version}.
      </p>
    </div>
  );
}
