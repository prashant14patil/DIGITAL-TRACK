import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { api } from "./api.js";
import AvatarViewer from "./AvatarViewer.jsx";
import ReadingsPanel from "./ReadingsPanel.jsx";
import { CATEGORIES, CATEGORY, DEMO_PROFILE, DEMO_READINGS, DEMO_TARGET, SEGMENTS, formatDate, readingAt } from "./demoData.js";

const fmt = (x, d = 1) => (x == null ? "–" : x.toFixed(d));

function blendColor(seg) {
  if (!seg) return null;
  let { from, to, f } = seg.blend || { from: seg.category, to: seg.category, f: 0 };
  if (!from && !to) return null; // no category entered: stay neutral, never guess one
  if (!from) from = to;
  if (!to) to = from;
  return new THREE.Color(CATEGORY[from].color).lerp(new THREE.Color(CATEGORY[to].color), f).getStyle();
}

function Chip({ label, from, to, unit, goodWhen }) {
  if (from == null) return <span className="chip">{label} <strong>{fmt(to)} {unit}</strong></span>;
  const d = to - from;
  const good = goodWhen === "down" ? d < 0 : goodWhen === "up" ? d > 0 : null;
  return (
    <span className="chip">
      {label} {fmt(from)} {unit} → <strong>{fmt(to)} {unit}</strong>{" "}
      <b className={good == null ? "" : good ? "good" : "bad"}>{d > 0 ? "+" : ""}{fmt(d)}</b>
    </span>
  );
}

export default function AvatarView() {
  const [jobs, setJobs] = useState(null);
  const [jobId, setJobId] = useState("");
  const [regions, setRegions] = useState(null);
  const [mode, setMode] = useState("since"); // today | since | target
  const [view, setView] = useState("front");
  const [t, setT] = useState(1e9); // clamped to the last reading
  const [playing, setPlaying] = useState(false);
  const [keyOpen, setKeyOpen] = useState(true);
  const [source, setSource] = useState(null); // "demo" | "mine"; null until readings have loaded
  const [mine, setMine] = useState({ profile: {}, target: null, readings: [] });
  const loadMine = () => api.readings().then((d) => { setMine(d); return d; });
  useEffect(() => {
    loadMine().then((d) => setSource((s) => s ?? (d.readings.length ? "mine" : "demo"))).catch(() => setSource("demo"));
  }, []);
  const raf = useRef(0);

  useEffect(() => {
    // RGB demo meshes (regions written by the job) + LiDAR depth meshes sent here with "Show on Body avatar"
    Promise.all([api.jobs().catch(() => []), api.avatarScans().catch(() => [])]).then(([all, depth]) => {
      const rgb = all.filter((j) => j.status === "complete" && j.result?.files?.includes("mesh_regions.glb"));
      const ok = [...depth.map((d) => ({ ...d, _depth: true })), ...rgb];
      setJobs(ok);
      const wanted = window.location.hash.match(/^#avatar\/(job_[A-Za-z0-9_]+)$/)?.[1];
      setJobId(ok.find((j) => j.id === wanted)?.id || ok[0]?.id || "");
    });
  }, []);
  const fileUrl = (j, name) => (j?._depth ? api.avatarFileUrl(j.id, name) : api.jobFileUrl(j.id, name));
  useEffect(() => {
    setRegions(null);
    const j = jobs?.find((x) => x.id === jobId);
    if (j) fetch(fileUrl(j, "regions.json")).then((r) => r.json()).then(setRegions).catch(() => {});
  }, [jobId, jobs]);

  const demo = source !== "mine";
  const readings = demo ? DEMO_READINGS : mine.readings;
  const target = demo ? DEMO_TARGET : mine.target && { ...mine.target, date: "target", segments: null };
  const profile = demo ? DEMO_PROFILE : mine.profile;
  const have = readings.length > 0;
  const last = Math.max(0, readings.length - 1);
  const first = readings[0];
  const latest = readings[last];
  const canSince = readings.length > 1;
  const effMode = mode === "since" && !canSince ? "today" : mode === "target" && !target ? "today" : mode;
  const emptySegs = (r) => r.segments ?? null;
  const current = !have ? null : effMode === "target" ? { ...target, segments: target.segments &&
    Object.fromEntries(Object.entries(target.segments).map(([k, v]) => [k, { ...v, blend: null }])) }
    : effMode === "today" ? readingAt(readings, last) : readingAt(readings, Math.min(t, last));
  const base = effMode === "since" ? first : effMode === "target" ? latest : null;

  const regionColors = useMemo(() => Object.fromEntries(SEGMENTS.map((s) =>
    [s.key, blendColor(current?.segments?.[s.key])])), [current]);
  const callouts = SEGMENTS.map((s) => ({ key: s.key, label: s.label.toUpperCase(), side: s.side, slot: s.slot,
    value: fmt(current?.segments?.[s.key]?.fat_pct), unit: current?.segments ? "%" : "", demo }));

  // replay: animate t from the first to the last reading
  useEffect(() => {
    if (!playing) return undefined;
    const start = performance.now();
    const dur = last * 1400;
    const step = (now) => {
      const f = Math.min(1, (now - start) / dur);
      setT(f * last);
      if (f < 1) raf.current = requestAnimationFrame(step);
      else setPlaying(false);
    };
    setT(0);
    raf.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, last]);

  const job = jobs?.find((j) => j.id === jobId);
  const sinceLabel = `Since ${new Date(`${first.date}T00:00:00`).toLocaleDateString("en-GB", { month: "long" })}`;

  return (
    <main className="avatar-page">
      <div className="seg source">
        <button className={demo ? "on" : ""} onClick={() => setSource("demo")}>Demo data</button>
        <button className={!demo ? "on" : ""} onClick={() => setSource("mine")}>My readings{mine.readings.length ? ` (${mine.readings.length})` : ""}</button>
      </div>

      {demo ? (
        <div className="banner warn">
          <strong>DEMO values</strong>, <b>not measurements of this person</b>. The body shape is an experimental scan
          and the body-part split is approximate.
        </div>
      ) : (
        <div className="banner">
          Values are shown <b>as you typed them</b>, not from the scan. The body shape is an experimental scan and the
          body-part split is approximate.
        </div>
      )}

      {!demo && <ReadingsPanel data={mine} onChange={loadMine} />}

      {jobs && jobs.length === 0 && (
        <p className="muted">No mesh with body regions yet. Create a 3D mesh on the "Capture &amp; meshes" page first.</p>
      )}

      {job && (
        <section className="card avatar-card">
          <div className="avatar-top">
            <div className="seg">
              <button className={effMode === "today" ? "on" : ""} onClick={() => setMode("today")}>Today</button>
              <button className={effMode === "since" ? "on" : ""} disabled={!canSince} onClick={() => setMode("since")}
                title={canSince ? "" : "needs at least two readings"}>{canSince ? sinceLabel : "Since …"}</button>
              <button className={effMode === "target" ? "on" : ""} disabled={!target} onClick={() => setMode("target")}
                title={target ? "" : "no target set"}>Target</button>
            </div>
            {jobs.length > 1 && (
              <select value={jobId} onChange={(e) => setJobId(e.target.value)} title="scan">
                {jobs.map((j) => <option key={j.id} value={j.id}>{j._depth ? "LiDAR · " : "RGB demo · "}{j.label || j.upload_id} · {j.created_utc.slice(0, 10)}{j._depth ? ` · ${j.run_status}` : ""}</option>)}
              </select>
            )}
            <div className="seg">
              {["front", "side", "back"].map((v) => (
                <button key={v} className={view === v ? "on" : ""} onClick={() => setView(v)}>{v[0].toUpperCase() + v.slice(1)}</button>
              ))}
            </div>
          </div>

          {job._depth && (
            <div className={`banner ${job.run_status === "pass" ? "" : "warn"}`}>
              <strong>LiDAR scan · status {job.run_status} · display only, not measured.</strong>{" "}
              {job.run_status !== "pass" && "The scan did not pass every check, so its shape may be distorted. "}
              The body-part split is approximate and no size is read from it.
            </div>
          )}
          {regions ? (
            <AvatarViewer meshUrl={fileUrl(job, "mesh_regions.glb")} regions={regions}
              regionColors={regionColors} view={view} callouts={callouts} />
          ) : <div className="avatar-stage" style={{ height: 560 }}><div className="viewer-msg">Loading…</div></div>}
          <div className="avatar-meta">
            <span>{demo ? "DEMO profile · " : ""}{[profile?.sex, profile?.height_cm && `${profile.height_cm} cm`].filter(Boolean).join(" · ") || (demo ? "" : "no profile entered")}</span>
            <span>{!have ? "no readings yet" : effMode === "target" ? (demo ? "DEMO target" : "Target") : formatDate(current.date)}</span>
          </div>

          {effMode === "since" && (
            <div className="timeline">
              <span>{formatDate(first.date)}</span>
              <input type="range" min={0} max={last} step={0.01} value={Math.min(t, last)}
                onChange={(e) => { setPlaying(false); setT(Number(e.target.value)); }} />
              <span>{formatDate(latest.date)}</span>
              <button className="replay" onClick={() => setPlaying(true)} disabled={playing}>▷ Replay</button>
            </div>
          )}

          {current && (
            <div className="chips">
              <Chip label="Weight" from={base?.weight_kg} to={current.weight_kg} unit="kg" />
              <Chip label="Body fat" from={base?.body_fat_pct} to={current.body_fat_pct} unit="%" goodWhen="down" />
              <Chip label="Muscle" from={base?.muscle_kg} to={current.muscle_kg} unit="kg" goodWhen="up" />
            </div>
          )}
          {!have && !demo && <p className="muted">Add your first reading above to fill in the avatar.</p>}

          <div className="key">
            <button className="keyhead" onClick={() => setKeyOpen((o) => !o)}>
              <strong>Colour key</strong><span>{keyOpen ? "▴" : "▾"}</span>
            </button>
            {keyOpen && (
              <div className="keybody">
                {CATEGORIES.map((c) => <span key={c.key}><i style={{ background: c.color }} />{c.label}</span>)}
                <small>{demo ? "Categories are part of the sample data." : "Only categories you entered are coloured; the rest stay grey."}</small>
              </div>
            )}
          </div>
        </section>
      )}
    </main>
  );
}
