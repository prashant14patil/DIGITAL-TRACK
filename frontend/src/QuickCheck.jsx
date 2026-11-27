import { useEffect, useState } from "react";
import { api } from "./api.js";

const ZIP = /\.zip$/i;

// Quick recording check for a LiDAR (Stray Scanner .zip) upload: ~15 s, no mesh. Run it before leaving the room.
export default function QuickCheck({ upload }) {
  const zips = upload.files.map((f) => f.path).filter((p) => ZIP.test(p));
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [bg, setBg] = useState("");
  const [others, setOthers] = useState([]);
  useEffect(() => {
    api.list().then((l) => setOthers(l.filter((u) => u.id !== upload.id && u.status === "complete"))).catch(() => {});
  }, [upload.id]);
  if (upload.status !== "complete" || zips.length !== 1) return null;

  const run = async () => {
    setBusy(true); setErr(""); setRes(null);
    try { setRes(await api.quickcheck(upload.id, bg)); } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };

  return (
    <div className="quickcheck">
      <label className="field">
        <span className="lbl">Empty-scene recording for the check<small> · needed for distance, legs-apart and arm checks</small></span>
        <select value={bg} onChange={(e) => setBg(e.target.value)} disabled={busy}>
          <option value="">none</option>
          {others.map((u) => <option key={u.id} value={u.id}>{u.label} · {u.created_utc.replace("T", " ").replace("Z", "")}</option>)}
        </select>
      </label>
      <div className="action-row">
        <button onClick={run} disabled={busy}>{busy ? "Checking… (about 15 s)" : "Check recording"}</button>
        <span className="muted">Fast check (no mesh): pauses, iPad still, floor visible, full turn completed. Run it
          before leaving the room so you can re-record straight away.</span>
      </div>
      {err && <div className="banner bad">{err}</div>}
      {res && (
        <>
          <div className={`banner ${res.verdict === "ok" ? "ok" : res.verdict === "warn" ? "warn" : "bad"}`}>
            <strong>{res.verdict_text}</strong>
          </div>
          <ul className="reasons">
            {res.checks.map((c) => (
              <li key={c.id}>
                <span className={`pill ${c.status === "ok" ? "ok" : c.status === "warn" ? "warn" : "bad"}`}>{c.status}</span>{" "}
                {c.message}
              </li>
            ))}
          </ul>
          <p className="muted">Limits are proposed values from one real recording; "ok" means no problem was detected,
            not that the scan is accurate.</p>
        </>
      )}
    </div>
  );
}
