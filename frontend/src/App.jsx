import RecordingGuide from "./RecordingGuide.jsx";
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, filesFromDrop, formatBytes, formatUtc } from "./api.js";
import { BrandMark, Chevron, SectionHead, ThemeSwitch, UploadIcon } from "./ui.jsx";
import MeshJobs, { ProcessButton } from "./MeshJobs.jsx";
import ZipContents from "./ZipContents.jsx";
import QuickCheck from "./QuickCheck.jsx";
import AvatarView from "./AvatarView.jsx";

const RECORDING_TYPES = [
  { id: "R0", title: "Empty scene", hint: "About 10 s, nothing in view. Background, floor plane, camera stability." },
  { id: "R1", title: "Flat wall or floor", hint: "Static. Depth noise and bias." },
  { id: "R2", title: "Box, static", hint: "One view, no turning. Sensor-only dimension error." },
  { id: "R3", title: "Box on floor, turned in steps", hint: "Turn by hand in small steps, pause at each stop, full 360°." },
  { id: "R4", title: "Box on marker disc, turned in steps", hint: "Same as R3, with the box on the printed-marker disc." },
  { id: "R5", title: "Cylinder, turned in steps", hint: "Plain or textured cylinder, same stepping protocol." },
  { id: "other", title: "Other", hint: "Describe it in the notes." },
];

const DEFAULT_MEASUREMENTS = {
  box: ["length", "width", "height"],
  cylinder: ["circumference", "height"],
  plane: [],
  none: [],
  other: [],
};

const emptyMeasurement = (name = "") => ({ name, readings: ["", "", ""], instrument: "tape", level: "" });

const INITIAL_FORM = {
  mode: "person", // "person" (default, short form) | "test" (R0-R5 sensor test recordings)
  recording_type: "R2",
  label: "",
  contains_person: "",
  consent_confirmed: false,
  subject_id: "",
  device_model: "",
  os_version: "",
  capture_app: "Camera",
  capture_app_version: "",
  capture_settings: "",
  object_kind: "box",
  object_textured: "",
  on_marker_disc: "",
  camera_distance_m: "",
  camera_height_m: "",
  step_deg: "",
  n_stops: "",
  pause_s: "",
  ipad_touched_during_recording: "",
  notes: "",
};

const STEPPED = new Set(["R3", "R4", "R5"]);

function toNumber(v) {
  if (v === "" || v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

function triState(v) {
  return v === "" ? null : v === "yes";
}

function buildMetadata(form, measurements, files) {
  const { mode, ...fields } = form;
  if (mode === "person") {
    return {
      recording_type: "other",
      label: form.label.trim() || `Person scan ${new Date().toISOString().slice(0, 10)}`,
      contains_person: true,
      consent_confirmed: form.consent_confirmed,
      subject_id: form.subject_id,
      device_model: form.device_model,
      os_version: form.os_version,
      capture_app: form.capture_app === "Camera" && files.some((f) => /\.zip$/i.test(f.path)) ? "Stray Scanner" : form.capture_app,
      capture_app_version: form.capture_app_version,
      notes: form.notes,
      object_kind: "none",
      measurements: [],
    };
  }
  const meta = {
    ...fields,
    object_textured: triState(form.object_textured),
    on_marker_disc: triState(form.on_marker_disc),
    ipad_touched_during_recording: triState(form.ipad_touched_during_recording),
    contains_person: form.contains_person === "yes",
    camera_distance_m: toNumber(form.camera_distance_m),
    camera_height_m: toNumber(form.camera_height_m),
    step_deg: toNumber(form.step_deg),
    n_stops: toNumber(form.n_stops),
    pause_s: toNumber(form.pause_s),
    measurements: measurements
      .filter((m) => m.name.trim() && m.readings.some((r) => r !== ""))
      .map((m) => ({
        name: m.name.trim(),
        instrument: m.instrument.trim(),
        readings_mm: m.readings.filter((r) => r !== "").map(Number),
        level_mm: m.level === "" || m.level == null ? null : Number(m.level),
      })),
  };
  return meta;
}

function validate(form, measurements, files) {
  const errs = [];
  if (form.subject_id && !/^[A-Za-z0-9_-]+$/.test(form.subject_id))
    errs.push("Subject ID must be a pseudonymous code (letters, digits, - or _), never a name.");
  if (form.mode === "person") {
    if (!form.consent_confirmed) errs.push("Tick the consent box before uploading a person recording.");
    if (!files.some((f) => /\.(mov|mp4|m4v|zip)$/i.test(f.path))) errs.push("Choose a Stray Scanner .zip or a video (.mov, .mp4).");
    return errs;
  }
  if (!form.label.trim()) errs.push("Add a short label (no personal names).");
  if (form.contains_person === "") errs.push("Say whether a person is visible in the recording.");
  if (form.contains_person === "yes" && !form.consent_confirmed)
    errs.push("Confirm consent was obtained under your team's process before uploading a person recording.");
  for (const k of ["camera_distance_m", "camera_height_m", "step_deg", "n_stops", "pause_s"]) {
    const n = toNumber(form[k]);
    if (Number.isNaN(n) || (n !== null && n <= 0)) errs.push(`${k.replaceAll("_", " ")} must be a positive number.`);
  }
  for (const m of measurements) {
    for (const r of m.readings)
      if (r !== "" && !(Number(r) > 0)) errs.push(`Measurement "${m.name || "?"}": readings must be positive millimetres.`);
    if (m.level && !(Number(m.level) > 0 && Number(m.level) < 2500))
      errs.push(`Measurement "${m.name || "?"}": level must be millimetres above the floor (0 to 2500).`);
  }
  if (!files.length) errs.push("Choose the export folder (or files) to upload.");
  return [...new Set(errs)];
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [form, setForm] = useState(INITIAL_FORM);
  const [measurements, setMeasurements] = useState(DEFAULT_MEASUREMENTS.box.map(emptyMeasurement));
  const [files, setFiles] = useState([]); // {path, file}
  const [run, setRun] = useState(null); // upload progress state
  const [uploads, setUploads] = useState([]);
  const [selected, setSelected] = useState(null);
  const [errors, setErrors] = useState([]);
  const [page, setPageState] = useState(window.location.hash.startsWith("#avatar") ? "avatar" : "capture");
  useEffect(() => {  // "#avatar/<job id>" (from "Show on Body avatar") opens the avatar page on that scan
    const onHash = () => setPageState(window.location.hash.startsWith("#avatar") ? "avatar" : "capture");
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const setPage = (p) => { setPageState(p); window.history.replaceState(null, "", p === "avatar" ? "#avatar" : "#"); };
  const [jobsKey, setJobsKey] = useState(0);
  const [focusJob, setFocusJob] = useState(null);
  const onJobStarted = (job) => { setFocusJob(job.id); setJobsKey((k) => k + 1); };

  const refresh = useCallback(() => api.list().then(setUploads).catch(() => {}), []);
  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(false));
    refresh();
  }, [refresh]);

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.type === "checkbox" ? e.target.checked : e.target.value }));

  const setMode = (mode) => {
    setForm((f) => ({ ...f, mode, capture_app: mode === "person" ? "Camera" : "Stray Scanner", consent_confirmed: false }));
    setErrors([]);
  };

  const setObjectKind = (e) => {
    const kind = e.target.value;
    setForm((f) => ({ ...f, object_kind: kind }));
    setMeasurements(DEFAULT_MEASUREMENTS[kind].map(emptyMeasurement));
  };

  const addFiles = (list) =>
    setFiles((prev) => {
      const map = new Map(prev.map((f) => [f.path, f]));
      for (const f of list) map.set(f.path, f);
      return [...map.values()].sort((a, b) => a.path.localeCompare(b.path));
    });

  const totalBytes = useMemo(() => files.reduce((s, f) => s + f.file.size, 0), [files]);
  const topFolders = useMemo(() => [...new Set(files.map((f) => f.path.split("/")[0]))], [files]);
  const busy = run && run.phase === "uploading";

  async function startUpload() {
    const errs = validate(form, measurements, files);
    setErrors(errs);
    if (errs.length) return;
    const state = { phase: "uploading", id: null, done: new Set(), failed: new Map(), sent: 0, total: totalBytes, current: "" };
    setRun({ ...state });
    try {
      state.id = (await api.create(buildMetadata(form, measurements, files))).id;
      await uploadFiles(state);
    } catch (e) {
      setRun({ ...state, phase: "error", message: e.message });
    }
  }

  async function uploadFiles(state) {
    const queue = files.filter((f) => !state.done.has(f.path));
    state.failed = new Map();
    const inFlight = new Map(); // path -> bytes sent for the file currently uploading
    const doneBytes = () => files.filter((f) => state.done.has(f.path)).reduce((s, f) => s + f.file.size, 0);
    const update = () =>
      setRun({ ...state, sent: doneBytes() + [...inFlight.values()].reduce((a, b) => a + b, 0) });

    const worker = async () => {
      while (queue.length) {
        const item = queue.shift();
        inFlight.set(item.path, 0);
        state.current = item.path;
        update();
        try {
          const res = await api.putFile(state.id, item.path, item.file, (n) => (inFlight.set(item.path, n), update()));
          if (res.size !== item.file.size) throw new Error(`size mismatch (${res.size} vs ${item.file.size})`);
          state.done.add(item.path);
        } catch (e) {
          state.failed.set(item.path, e.message);
        }
        inFlight.delete(item.path);
        update();
      }
    };
    await Promise.all(Array.from({ length: 4 }, worker)); // 4 parallel transfers; fine on localhost

    if (state.failed.size) {
      setRun({ ...state, phase: "partial", sent: doneBytes() });
      return;
    }
    setRun({ ...state, phase: "finalizing", sent: doneBytes() });
    try {
      const manifest = await api.finalize(state.id, files.map((f) => ({ path: f.path, size: f.file.size })));
      setRun({ ...state, phase: "complete", manifest, sent: doneBytes() });
      refresh();
    } catch (e) {
      setRun({ ...state, phase: "error", message: `Verification failed: ${e.message}` });
    }
  }

  async function retry() {
    const state = { ...run, phase: "uploading", done: new Set(run.done) };
    setRun(state);
    await uploadFiles(state);
  }

  async function discardRun() {
    if (run?.id) await api.discard(run.id).catch(() => {});
    setRun(null);
    refresh();
  }

  function resetForNext() {
    setRun(null);
    setFiles([]);
    setForm((f) => ({ ...f, label: "", notes: "", contains_person: "", consent_confirmed: false, subject_id: "" }));
    setMeasurements(DEFAULT_MEASUREMENTS[form.object_kind].map(emptyMeasurement));
  }

  return (
    <div className="app">
      <header className="appbar">
        <div className="appbar-inner">
          <div className="brand"><BrandMark />DIGITAL TRACK</div>
          <nav className="tabs" aria-label="Pages">
            <button className={page === "capture" ? "on" : ""} onClick={() => setPage("capture")}>Capture &amp; meshes</button>
            <button className={page === "avatar" ? "on" : ""} onClick={() => setPage("avatar")}>Body avatar</button>
          </nav>
          <div className={`conn ${health ? "up" : health === false ? "down" : ""}`} title="Runs on this computer only">
            <i aria-hidden="true" />
            {health ? `Local API v${health.version}` : health === false ? "API offline" : "Connecting…"}
          </div>
          <ThemeSwitch />
        </div>
      </header>

      <div className="page">
      {health === false && (
        <div className="banner bad">
          <strong>The local API is not running.</strong> Start it with <code>backend\.venv\Scripts\dt.exe serve</code>, then reload this page.
        </div>
      )}

      {page === "avatar" && (
        <>
          <div className="page-head">
            <h1>Body avatar</h1>
            <p>Your own scan shown with body-composition readings. Display only: nothing is measured from the shape.</p>
          </div>
          <AvatarView />
        </>
      )}
      {page === "capture" && (
      <main className="stack">
        <div className="page-head">
          <h1>Capture &amp; meshes</h1>
          <p>Upload a recording, check it, then build a 3D mesh. Files stay on this computer.</p>
        </div>
        <RecordingGuide />
        <section className="card composer">
          <div className="pane">
          <SectionHead step="Step 1" title="Recording details" />
          <div className="seg kind">
            <button className={form.mode === "person" ? "on" : ""} disabled={busy} onClick={() => setMode("person")}>Person scan</button>
            <button className={form.mode === "test" ? "on" : ""} disabled={busy} onClick={() => setMode("test")}>Test recording</button>
          </div>

          {form.mode === "person" && (
            <>
              <div className="row">
                <Field label="Label" hint="optional, no personal names">
                  <input value={form.label} onChange={set("label")} maxLength={80} placeholder="e.g. front pose, take 1" disabled={busy} />
                </Field>
              </div>
              <div className="row">
                <Field label="Subject ID" hint="optional code, never a name">
                  <input value={form.subject_id} onChange={set("subject_id")} maxLength={40} placeholder="e.g. S001" disabled={busy} />
                </Field>
              </div>
              <details className="more">
                <summary>More details (optional)</summary>
                <div className="row three">
                  <Field label="iPad model"><input value={form.device_model} onChange={set("device_model")} disabled={busy} /></Field>
                  <Field label="iPadOS version"><input value={form.os_version} onChange={set("os_version")} disabled={busy} /></Field>
                  <Field label="Capture app"><input value={form.capture_app} onChange={set("capture_app")} disabled={busy} /></Field>
                </div>
                <Field label="Notes"><textarea rows={2} value={form.notes} onChange={set("notes")} disabled={busy} /></Field>
              </details>
              <label className="confirm">
                <input type="checkbox" checked={form.consent_confirmed} onChange={set("consent_confirmed")} disabled={busy} />
                <span>Consent for this recording was obtained.</span>
              </label>
            </>
          )}

          {form.mode === "test" && (
          <>
          <fieldset className="types">
            <legend>Recording type</legend>
            {RECORDING_TYPES.map((t) => (
              <label key={t.id} className={`type ${form.recording_type === t.id ? "on" : ""}`}>
                <input type="radio" name="rt" value={t.id} checked={form.recording_type === t.id}
                  onChange={set("recording_type")} disabled={busy} />
                <span className="tid">{t.id}</span>
                <span><strong>{t.title}</strong><small>{t.hint}</small></span>
              </label>
            ))}
          </fieldset>

          <div className="row">
            <Field label="Label" hint="Short description, no personal names">
              <input value={form.label} onChange={set("label")} maxLength={80} placeholder="e.g. box A, 15° steps, take 1" disabled={busy} />
            </Field>
          </div>

          <div className="row three">
            <Field label="iPad model"><input value={form.device_model} onChange={set("device_model")} placeholder="Settings → General → About" disabled={busy} /></Field>
            <Field label="iPadOS version"><input value={form.os_version} onChange={set("os_version")} disabled={busy} /></Field>
            <Field label="Capture app / version">
              <div className="pair">
                <input value={form.capture_app} onChange={set("capture_app")} disabled={busy} />
                <input value={form.capture_app_version} onChange={set("capture_app_version")} placeholder="version" disabled={busy} />
              </div>
            </Field>
          </div>
          <Field label="Capture settings" hint="Resolution, frame rate, depth/confidence options exactly as set in the app">
            <textarea rows={2} value={form.capture_settings} onChange={set("capture_settings")} disabled={busy} />
          </Field>

          <h3>Object and reference measurements</h3>
          <div className="row three">
            <Field label="Object">
              <select value={form.object_kind} onChange={setObjectKind} disabled={busy}>
                <option value="none">none (empty scene)</option>
                <option value="plane">flat wall / floor</option>
                <option value="box">box</option>
                <option value="cylinder">cylinder</option>
                <option value="other">other</option>
              </select>
            </Field>
            <Field label="Textured surface?"><YesNo value={form.object_textured} onChange={set("object_textured")} disabled={busy} /></Field>
            <Field label="On marker disc?"><YesNo value={form.on_marker_disc} onChange={set("on_marker_disc")} disabled={busy} /></Field>
          </div>

          {measurements.length > 0 && (
            <div className="table-wrap">
            <table className="meas">
              <thead>
                <tr><th>Dimension</th><th colSpan={3}>Readings (mm), measure 3×</th><th>Instrument</th>
                  <th title="Circumferences: height above the floor where the tape was held">Level (mm above floor)</th><th></th></tr>
              </thead>
              <tbody>
                {measurements.map((m, i) => (
                  <tr key={i}>
                    <td><input value={m.name} disabled={busy}
                      onChange={(e) => setMeasurements((ms) => ms.map((x, j) => (j === i ? { ...x, name: e.target.value } : x)))} /></td>
                    {m.readings.map((r, k) => (
                      <td key={k}>
                        <input inputMode="decimal" value={r} disabled={busy} className="num"
                          onChange={(e) => setMeasurements((ms) => ms.map((x, j) => (j === i
                            ? { ...x, readings: x.readings.map((y, l) => (l === k ? e.target.value : y)) } : x)))} />
                      </td>
                    ))}
                    <td><input value={m.instrument} disabled={busy}
                      onChange={(e) => setMeasurements((ms) => ms.map((x, j) => (j === i ? { ...x, instrument: e.target.value } : x)))} /></td>
                    <td><input inputMode="decimal" className="num" value={m.level ?? ""} disabled={busy} placeholder="circumf. only"
                      onChange={(e) => setMeasurements((ms) => ms.map((x, j) => (j === i ? { ...x, level: e.target.value } : x)))} /></td>
                    <td><button className="link quiet" disabled={busy} onClick={() => setMeasurements((ms) => ms.filter((_, j) => j !== i))}>Remove</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          )}
          <button className="link add" disabled={busy} onClick={() => setMeasurements((ms) => [...ms, emptyMeasurement()])}>Add dimension</button>

          <h3>Setup</h3>
          <div className="row three">
            <Field label="Camera → object distance (m)"><input inputMode="decimal" value={form.camera_distance_m} onChange={set("camera_distance_m")} disabled={busy} /></Field>
            <Field label="Camera height (m)"><input inputMode="decimal" value={form.camera_height_m} onChange={set("camera_height_m")} disabled={busy} /></Field>
            <Field label="iPad touched during recording?"><YesNo value={form.ipad_touched_during_recording} onChange={set("ipad_touched_during_recording")} disabled={busy} /></Field>
          </div>
          {STEPPED.has(form.recording_type) && (
            <div className="row three">
              <Field label="Step angle (°)"><input inputMode="decimal" value={form.step_deg} onChange={set("step_deg")} placeholder="15" disabled={busy} /></Field>
              <Field label="Number of stops"><input inputMode="numeric" value={form.n_stops} onChange={set("n_stops")} placeholder="24" disabled={busy} /></Field>
              <Field label="Pause per stop (s)"><input inputMode="decimal" value={form.pause_s} onChange={set("pause_s")} placeholder="2–3" disabled={busy} /></Field>
            </div>
          )}
          <Field label="Notes" hint="Lighting, material, anything unusual, start/stop time">
            <textarea rows={3} value={form.notes} onChange={set("notes")} disabled={busy} />
          </Field>

          <h3>People and consent</h3>
          <div className="row three">
            <Field label="Person visible in the recording?">
              <select value={form.contains_person} onChange={set("contains_person")} disabled={busy}>
                <option value="">choose…</option>
                <option value="no">no (object / empty scene)</option>
                <option value="yes">yes</option>
              </select>
            </Field>
            {form.contains_person === "yes" && (
              <Field label="Subject ID" hint="pseudonymous code, never a name">
                <input value={form.subject_id} onChange={set("subject_id")} maxLength={40} placeholder="e.g. S001" disabled={busy} />
              </Field>
            )}
          </div>
          {form.contains_person === "yes" && (
            <label className="confirm">
              <input type="checkbox" checked={form.consent_confirmed} onChange={set("consent_confirmed")} disabled={busy} />
              <span>Consent for this recording was obtained under our team's process. The files stay on this computer and are never committed to git.</span>
            </label>
          )}
          </>
          )}
          </div>

          <div className="pane">
          <SectionHead step="Step 2" title={form.mode === "person" ? "Scan zip or video" : "Export files"}
            desc="Uploaded byte for byte, checksummed, then made read-only." />
          <DropZone onFiles={addFiles} disabled={busy} person={form.mode === "person"} />
          {files.length > 0 && (
            <>
              <div className="summary">
                <span><strong>{files.length}</strong> {files.length === 1 ? "file" : "files"}</span>
                <span><strong>{formatBytes(totalBytes)}</strong></span>
                <span className="path">{topFolders.slice(0, 3).join(", ")}{topFolders.length > 3 ? "…" : ""}</span>
                <button className="link" disabled={busy} onClick={() => setFiles([])}>Clear</button>
              </div>
              <div className="filelist">
                {files.slice(0, 400).map((f) => (
                  <div key={f.path} className={`file ${run?.done?.has(f.path) ? "done" : ""} ${run?.failed?.has(f.path) ? "fail" : ""}`}>
                    <span className="path">{f.path}</span>
                    <span className="size">{formatBytes(f.file.size)}</span>
                  </div>
                ))}
                {files.length > 400 && <div className="file more">… and {files.length - 400} more</div>}
              </div>
            </>
          )}

          {errors.length > 0 && (
            <ul className="errors">{errors.map((e) => <li key={e}>{e}</li>)}</ul>
          )}

          {!run && (
            <div className="actions-bar">
              <button className="primary" onClick={startUpload} disabled={health === false}>Upload</button>
            </div>
          )}
          {run && <Progress run={run} onRetry={retry} onDiscard={discardRun} onNext={resetForNext} />}
          </div>
        </section>

        <section className="card">
          <SectionHead title="Uploads" desc="Select a row to see its files, check the recording or create a mesh."
            actions={<button className="link" onClick={refresh}>Refresh</button>} />
          {uploads.length === 0 ? (
            <p className="empty">Nothing uploaded yet.</p>
          ) : (
            <div className="table-wrap">
            <table className="uploads">
              <thead>
                <tr><th>Created (UTC)</th><th>Test</th><th>Label</th><th className="num">Files</th><th className="num">Size</th><th>Status</th><th /></tr>
              </thead>
              <tbody>
                {uploads.map((u) => (
                  <tr key={u.id} className={selected?.id === u.id ? "sel" : ""}
                    onClick={() => api.get(u.id).then(setSelected)}>
                    <td className="ts">{formatUtc(u.created_utc)}</td>
                    <td className="mono">{u.recording_type === "other" ? "–" : u.recording_type}</td>
                    <td>{u.label}</td>
                    <td className="num">{u.file_count}</td>
                    <td className="num">{formatBytes(u.total_bytes)}</td>
                    <td><span className={`pill ${u.status === "complete" ? "ok" : "warn"}`}>{u.status}</span></td>
                    <td className="go"><Chevron /></td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          )}
          {selected && <Details upload={selected} onClose={() => setSelected(null)} onJobStarted={onJobStarted} />}
        </section>

        <MeshJobs refreshKey={jobsKey} focusJobId={focusJob} />
      </main>
      )}
      </div>
    </div>
  );
}

function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span className="lbl">{label}{hint && <small> · {hint}</small>}</span>
      {children}
    </label>
  );
}

function YesNo({ value, onChange, disabled }) {
  return (
    <select value={value} onChange={onChange} disabled={disabled}>
      <option value="">not stated</option>
      <option value="yes">yes</option>
      <option value="no">no</option>
    </select>
  );
}

function DropZone({ onFiles, disabled, person }) {
  const [over, setOver] = useState(false);
  const folderRef = useRef(null);
  const filesRef = useRef(null);
  const fromInput = (e) => {
    onFiles([...e.target.files].map((f) => ({ path: f.webkitRelativePath || f.name, file: f })));
    e.target.value = "";
  };
  return (
    <div
      className={`drop ${over ? "over" : ""} ${disabled ? "off" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={async (e) => {
        e.preventDefault();
        setOver(false);
        if (!disabled) onFiles(await filesFromDrop(e.dataTransfer));
      }}
    >
      <span className="drop-icon"><UploadIcon /></span>
      <p className="drop-title">{person ? "Drop a scan .zip or video" : "Drop the export folder"}</p>
      <p className="drop-sub">{person ? "Stray Scanner .zip, .mov or .mp4, uploaded unchanged" : "The whole folder, unchanged"}</p>
      <div className="btns">
        {!person && <button disabled={disabled} onClick={() => folderRef.current.click()}>Choose folder…</button>}
        <button disabled={disabled} onClick={() => filesRef.current.click()}>{person ? "Browse files…" : "Choose files…"}</button>
      </div>
      <input ref={folderRef} type="file" webkitdirectory="" directory="" multiple hidden onChange={fromInput} />
      <input ref={filesRef} type="file" multiple hidden onChange={fromInput} />
    </div>
  );
}

function Progress({ run, onRetry, onDiscard, onNext }) {
  const pct = run.total ? Math.min(100, (100 * run.sent) / run.total) : 0;
  return (
    <div className={`progress ${run.phase}`}>
      <div className="bar"><div style={{ width: `${pct}%` }} /></div>
      <div className="ptext">
        {run.phase === "uploading" && <>Uploading {formatBytes(run.sent)} / {formatBytes(run.total)} · <span className="path">{run.current}</span></>}
        {run.phase === "finalizing" && <>Verifying sizes and checksums…</>}
        {run.phase === "complete" && (
          <>Upload complete and verified: {run.manifest.file_count} files, {formatBytes(run.manifest.total_bytes)}. Files are now read-only. <span className="mono">{run.id}</span></>
        )}
        {run.phase === "partial" && <>{run.failed.size} file(s) failed. Already uploaded files are kept; retry sends only the rest.</>}
        {run.phase === "error" && <>Upload stopped: {run.message}</>}
      </div>
      {run.phase === "partial" && (
        <ul className="errors">
          {[...run.failed.entries()].slice(0, 20).map(([p, m]) => <li key={p}><span className="path">{p}</span>: {m}</li>)}
        </ul>
      )}
      <div className="btns">
        {run.phase === "partial" && <button className="primary" onClick={onRetry}>Retry failed files</button>}
        {(run.phase === "partial" || run.phase === "error") && run.id && <button onClick={onDiscard}>Discard this upload</button>}
        {run.phase === "error" && !run.id && <button onClick={onDiscard}>Back</button>}
        {run.phase === "complete" && <button className="primary" onClick={onNext}>Upload another recording</button>}
      </div>
    </div>
  );
}

function Details({ upload, onClose, onJobStarted }) {
  const m = upload.metadata;
  return (
    <div className="details">
      <div className="dhead">
        <strong>{m.label}</strong>
        <span className="mono">{upload.id}</span>
        <button className="link right" onClick={onClose}>Close</button>
      </div>
      <dl>
        <dt>Device</dt><dd>{[m.device_model, m.os_version].filter(Boolean).join(", ") || "not stated"}</dd>
        <dt>App</dt><dd>{[m.capture_app, m.capture_app_version].filter(Boolean).join(" ") || "not stated"}</dd>
        {m.object_kind !== "none" && <><dt>Object</dt><dd>{m.object_kind}</dd></>}
        {m.measurements.map((x, i) => (
          <Fragment key={i}><dt>{x.name}</dt><dd>{x.readings_mm.join(" / ")} mm {x.instrument && `(${x.instrument})`}{x.level_mm ? ` at ${x.level_mm} mm above floor` : ""}</dd></Fragment>
        ))}
        {m.notes && <><dt>Notes</dt><dd>{m.notes}</dd></>}
      </dl>
      <ProcessButton key={`p-${upload.id}`} upload={upload} onStarted={onJobStarted} />
      <QuickCheck key={`q-${upload.id}`} upload={upload} />
      <ZipContents key={upload.id} upload={upload} />
      <div className="filelist small">
        {upload.files.map((f) => (
          <div key={f.path} className="file">
            <span className="path">{f.path}</span>
            <span className="size">{formatBytes(f.size)}</span>
            <span className="mono hash" title={f.sha256}>{f.sha256.slice(0, 12)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
