import { useCallback, useEffect, useState } from "react";
import { api, formatUtc } from "./api.js";
import { Chevron, SectionHead } from "./ui.jsx";
import MeshViewer from "./MeshViewer.jsx";
import ScanReview from "./ScanReview.jsx";
import DepthResult from "./DepthResult.jsx";

const VIDEO = /\.(mov|mp4|m4v|zip)$/i;
const ZIP = /\.zip$/i;

export const videoFiles = (upload) => (upload?.files || []).map((f) => f.path).filter((p) => VIDEO.test(p));

// Button shown in an upload's details: start the experimental mesh job for one of its videos.
export function ProcessButton({ upload, onStarted }) {
  const videos = videoFiles(upload);
  const [file, setFile] = useState(videos[0] || "");
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [bg, setBg] = useState("");
  const [others, setOthers] = useState([]);
  const isZip = ZIP.test(file);
  useEffect(() => {
    if (isZip) api.list().then((l) => setOthers(l.filter((u) => u.id !== upload.id && u.status === "complete"))).catch(() => {});
  }, [isZip, upload.id]);
  if (upload.status !== "complete" || !videos.length) return null;
  async function start() {
    setBusy(true);
    setErr(null);
    try {
      onStarted(await api.process(upload.id, file, isZip ? bg : ""));
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="process">
      {videos.length > 1 && (
        <select value={file} onChange={(e) => setFile(e.target.value)}>
          {videos.map((v) => <option key={v} value={v}>{v}</option>)}
        </select>
      )}
      {isZip && (
        <label className="field">
          <span className="lbl">Empty-scene recording<small> · recommended: separates the person from walls and furniture</small></span>
          <select value={bg} onChange={(e) => setBg(e.target.value)} disabled={busy}>
            <option value="">none</option>
            {others.map((u) => <option key={u.id} value={u.id}>{u.label} · {u.created_utc.replace("T", " ").replace("Z", "")}</option>)}
          </select>
        </label>
      )}
      <button className="primary" disabled={busy} onClick={start}>
        {ZIP.test(file) ? "Create 3D mesh from LiDAR depth" : "Create 3D mesh (experimental)"}
      </button>
      <small className="muted">
        {ZIP.test(file) ? "Stray Scanner zip: depth fusion in metres, not yet validated. Takes a few minutes."
          : "RGB-only demo: non-metric, poses assumed. Takes a few minutes."}
      </small>
      {err && <div className="errors">{err}</div>}
    </div>
  );
}

// Section listing processing jobs; polls while any job is queued or running; shows the selected result.
export default function MeshJobs({ refreshKey, focusJobId }) {
  const [jobs, setJobs] = useState([]);
  const [openId, setOpenId] = useState(null);
  const load = useCallback(() => api.jobs().then(setJobs).catch(() => {}), []);

  useEffect(() => { load(); }, [load, refreshKey]);
  useEffect(() => { if (focusJobId) setOpenId(focusJobId); }, [focusJobId]);
  const active = jobs.some((j) => j.status === "queued" || j.status === "running");
  useEffect(() => {
    if (!active) return undefined;
    const t = setInterval(load, 1500);
    return () => clearInterval(t);
  }, [active, load]);

  const open = jobs.find((j) => j.id === openId) || null;
  return (
    <section className="card">
      <SectionHead title="3D meshes" desc="Processing jobs. Running jobs update automatically."
        actions={<button className="link" onClick={load}>Refresh</button>} />
      {jobs.length === 0 ? (
        <p className="empty">No mesh yet. Open a finished upload above and choose "Create 3D mesh".</p>
      ) : (
        <div className="table-wrap">
        <table className="uploads">
          <thead>
            <tr><th>Started (UTC)</th><th>Upload</th><th>File</th><th>Status</th><th>Progress</th><th /></tr>
          </thead>
          <tbody>
            {jobs.map((j) => (
              <tr key={j.id} className={openId === j.id ? "sel" : ""} onClick={() => setOpenId(j.id)}>
                <td className="ts">{formatUtc(j.created_utc)}</td>
                <td>{j.label || j.upload_id}</td>
                <td className="path">{j.file}</td>
                <td><span className={`pill ${j.status === "complete" ? "ok" : j.status === "failed" ? "bad" : "warn"}`}>{j.status}</span></td>
                <td>
                  {j.status === "running" || j.status === "queued" ? (
                    <div className="mini"><div className="bar"><div style={{ width: `${Math.round(100 * j.progress)}%` }} /></div>
                      <small>{j.stage} · {Math.round(100 * j.progress)}%</small></div>
                  ) : j.status === "failed" ? <small className="muted">–</small> : <small className="muted">done</small>}
                </td>
                <td className="go"><Chevron /></td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      )}
      {open && <JobResult job={open} onClose={() => setOpenId(null)}
        onDeleted={() => { setOpenId(null); load(); }} />}
    </section>
  );
}

function JobResult({ job, onClose, onDeleted }) {
  const file = (name) => api.jobFileUrl(job.id, name);
  const r = job.result;
  const [delErr, setDelErr] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const busy = job.status === "queued" || job.status === "running";
  async function remove() {
    if (!window.confirm("Remove this mesh and its results? The uploaded recording is kept, so you can create the mesh again.")) return;
    setDeleting(true);
    setDelErr(null);
    try {
      await api.deleteJob(job.id);
      onDeleted();
    } catch (e) {
      setDelErr(e.message);
      setDeleting(false);
    }
  }
  return (
    <div className="details">
      <div className="dhead">
        <strong>{job.label || job.upload_id} · {job.file}</strong>
        <span className="mono">{job.id}</span>
        <button className="link right" onClick={onClose}>Close</button>
        <button className="link danger" disabled={busy || deleting} onClick={remove}
          title={busy ? "Wait until processing finishes" : "Delete this mesh (the upload is kept)"}>
          {deleting ? "Removing…" : "Remove mesh"}
        </button>
      </div>
      {delErr && <div className="errors">{delErr}</div>}
      {job.status === "failed" && <div className="banner bad">Processing failed: {job.error}</div>}
      {(job.status === "queued" || job.status === "running") && (
        <p className="muted">Working: {job.stage} ({Math.round(100 * job.progress)}%). This page updates automatically.</p>
      )}
      {job.kind === "rgbd_depth" && job.status !== "queued" && job.status !== "running" && <DepthResult job={job} />}
      {job.kind !== "rgbd_depth" && job.status === "complete" && r && (
        <>
          <div className="banner warn">
            <strong>Experimental · not measured · poses assumed.</strong> This mesh is height-normalised (not in
            metres), made without camera calibration or a scale reference, from estimated turn angles. Do not read
            body dimensions from it. Status: <span className="mono">{r.status}</span>
          </div>
          <MeshViewer url={file("mesh.glb")} />
          <ScanReview job={job} />
          <div className="links">
            <a className="link" href={file("mesh.glb")} download={`${job.id}_mesh.glb`}>Download mesh (.glb)</a>
            <a className="link" href={file("report.json")} target="_blank" rel="noreferrer">report.json</a>
          </div>
          <dl>
            <dt>Mesh</dt><dd>{r.mesh?.triangles?.toLocaleString()} triangles · units: {r.mesh?.units}</dd>
            <dt>Frames used</dt><dd>{r.frames_used} (every {r.frame_step}th video frame)</dd>
            <dt>Turn angles</dt>
            <dd>estimated, not measured: {(r.angle_anchors || []).map((a) => `${a.angle_deg}° @ ${a.time_s.toFixed(1)} s`).join(" · ")}</dd>
            <dt>Measurements</dt><dd>{r.measurements}</dd>
            <dt>Runtime</dt><dd>{r.runtime_s} s</dd>
          </dl>
          <h3>Why this is unreliable</h3>
          <ul className="reasons">{(r.status_reasons || []).map((s) => <li key={s}>{s}</li>)}</ul>
          {r.files?.includes("preview_silhouettes.png") && (
            <>
              <h3>Person segmentation (sample frames)</h3>
              <img className="preview" src={file("preview_silhouettes.png")} alt="Person segmentation on sample frames" />
            </>
          )}
          {r.files?.includes("preview_angle_estimation.png") && (
            <>
              <h3>Turn-angle estimation</h3>
              <img className="preview" src={file("preview_angle_estimation.png")} alt="Turn-angle estimation" />
            </>
          )}
        </>
      )}
    </div>
  );
}
