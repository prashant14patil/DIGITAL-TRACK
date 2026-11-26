import { useState } from "react";
import { api } from "./api.js";
import MeshViewer from "./MeshViewer.jsx";
import Comparison from "./Comparison.jsx";

const IMAGES = [
  ["mesh_render.png", "Mesh from several sides"],
  ["coverage.png", "Which parts each view saw"],
  ["views_front.png", "All views overlaid, front (colour ghosting = misalignment)"],
  ["views_side.png", "All views overlaid, side"],
  ["views_top.png", "All views overlaid, top"],
  ["poses.png", "Estimated turn per view"],
  ["slices.png", "Horizontal slices"],
  // RGB-D tracked pipeline (rgbd/tracked.py)
  ["landmarks.png", "Landmarks and measurement levels (proposed definitions)"],
  ["turn_angle.png", "Estimated body turn over time"],
  ["view_consistency.png", "How well each view agrees with the fused surface"],
  ["cross_sections.png", "Cross-sections: full mesh vs two independent halves of the views"],
  ["circumference_profile.png", "Outline length per height (diagnostic, not a body measurement)"],
  ["alignment_rgb_depth.png", "RGB / depth alignment (red = depth outline)"],
];

// "Show on Body avatar": prepares a display copy with body regions (backend experimental/depth_avatar.py), then
// opens the avatar page on this scan. Display only: nothing is measured from it.
function SendToAvatar({ job }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const go = async () => {
    setBusy(true); setErr("");
    try { await api.prepareAvatar(job.id); window.location.hash = `#avatar/${job.id}`; }
    catch (e) { setErr(e.message); } finally { setBusy(false); }
  };
  return (
    <div className="action-row">
      <button className="primary" onClick={go} disabled={busy}>{busy ? "Preparing…" : "Show on Body avatar"}</button>
      <span className="muted">Display only{job.result?.status && job.result.status !== "pass" ? ` · this scan's status is ${job.result.status}, its shape may be distorted` : ""}.</span>
      {err && <span className="bad">{err}</span>}
    </div>
  );
}

// Body measurements from backend measurement/body.py. Every number is shown WITH its status and reasons; nothing
// is computed here. Status stays "unreliable" until the definitions are validated against tape.
function BodyMeasurements({ body }) {
  if (!body?.measurements?.length) return null;
  const label = (n) => n.replace(/_/g, " ");
  return (
    <>
      <h3>Body measurements · {body.definition_version} · not validated</h3>
      <p className="muted">
        Proposed definitions. Torso values use only the torso part of each slice (arms excluded where separated).
        Every value is <strong>unreliable</strong> until compared with a tape measure on this setup.
      </p>
      {body.flags?.length > 0 && <ul className="reasons">{body.flags.map((f) => <li key={f}>{f}</li>)}</ul>}
      <table className="measure-table">
        <thead><tr><th>Measurement</th><th>Value</th><th>Level</th><th>Status</th><th>Why</th></tr></thead>
        <tbody>
          {body.measurements.map((m) => (
            <tr key={m.name}>
              <td>{label(m.name)}</td>
              <td>{m.value != null ? `${(m.value * 100).toFixed(1)} cm` : "–"}</td>
              <td>{m.level_z != null ? `${m.level_z.toFixed(2)} m` : ""}</td>
              <td><span className={m.status === "unavailable" ? "bad" : "warn-text"}>{m.status}</span></td>
              <td className="muted">{(m.reasons || []).join("; ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

// Result of a LiDAR depth job (Stray Scanner zip). Failed runs still show their diagnostic images.
export default function DepthResult({ job }) {
  const file = (name) => api.jobFileUrl(job.id, name);
  const r = job.result || {};
  const s = r.stray || {};
  const has = (n) => r.files?.includes(n);
  return (
    <>
      {job.status === "complete" && (
        <div className={`banner ${r.status === "pass" ? "ok" : "warn"}`}>
          <strong>LiDAR depth mesh · status {r.status} · not validated.</strong> {r.status_note} Units are metres
          from the iPad depth sensor, but nothing has been checked against a tape yet, so no body measurements are
          shown.
        </div>
      )}
      {has("mesh.glb") && <MeshViewer url={file("mesh.glb")} />}
      {job.status === "complete" && has("mesh.ply") && <SendToAvatar job={job} />}
      {job.status === "complete" && <Comparison job={job} />}
      <div className="links">
        {has("mesh.glb") && <a className="link" href={file("mesh.glb")} download={`${job.id}_mesh.glb`}>Download mesh (.glb)</a>}
        {has("mesh.ply") && <a className="link" href={file("mesh.ply")} download={`${job.id}_mesh.ply`}>Download mesh (.ply)</a>}
        {has("mesh.obj") && <a className="link" href={file("mesh.obj")} download={`${job.id}_mesh.obj`}>Download mesh (.obj)</a>}
        {has("report.html") && <a className="link" href={file("report.html")} target="_blank" rel="noreferrer">Full report</a>}
        {has("quality_report.json") && <a className="link" href={file("quality_report.json")} target="_blank" rel="noreferrer">quality_report.json</a>}
        {has("run_log.json") && <a className="link" href={file("run_log.json")} target="_blank" rel="noreferrer">run_log.json</a>}
      </div>
      <dl>
        <dt>How views were picked</dt><dd>{r.mode || "–"} · {r.n_views ?? "–"} views used</dd>
        <dt>Recording</dt>
        <dd>{s.duration_s ?? "–"} s · {s.depth_files ?? "–"} depth frames ({s.depth_size?.join("×")}) · confidence {s.has_confidence ? "yes" : "no"}</dd>
        <dt>iPad movement</dt>
        <dd>{s.ipad_moved_m != null ? `${(s.ipad_moved_m * 100).toFixed(1)} cm` : "–"}{s.ipad_moved_m > 0.02 && <span className="bad"> · the iPad moved; it should stay fixed</span>}</dd>
        {r.mesh?.height_extent_m != null && (<><dt>Mesh height extent</dt><dd>{r.mesh.height_extent_m} m (visual check only, not a measurement)</dd></>)}
        {r.mesh?.triangles != null && (<><dt>Mesh</dt><dd>{r.mesh.triangles.toLocaleString()} triangles</dd></>)}
      </dl>
      <BodyMeasurements body={r.body_measurements} />
      {r.warnings?.length > 0 && (
        <>
          <h3>Warnings</h3>
          <ul className="reasons">{r.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
        </>
      )}
      {IMAGES.filter(([n]) => has(n)).map(([n, title]) => (
        <div key={n}>
          <h3>{title}</h3>
          <img className="preview" src={file(n)} alt={title} />
        </div>
      ))}
    </>
  );
}
