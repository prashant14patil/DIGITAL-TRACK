import { useEffect, useState } from "react";
import { api } from "./api.js";

const load = (url) => fetch(url).then((r) => (r.ok ? r.json() : null)).catch(() => null);

// Recording warnings and rough slice readings of one finished job (backend experimental/scan_review.py).
export default function ScanReview({ job }) {
  const files = job.result?.files || [];
  const [quality, setQuality] = useState(null);
  const [slices, setSlices] = useState(null);

  useEffect(() => {
    setQuality(null);
    setSlices(null);
    if (files.includes("quality.json")) load(api.jobFileUrl(job.id, "quality.json")).then(setQuality);
    if (files.includes("slices.json")) load(api.jobFileUrl(job.id, "slices.json")).then(setSlices);
  }, [job.id, files.join()]);

  return (
    <>
      {quality && (
        <>
          <h3>Recording check: {quality.warnings ? `${quality.warnings} warning(s)` : "no problem detected"}</h3>
          <ul className="reasons">
            {quality.checks.map((c) => (
              <li key={c.id}>
                <span className={`pill ${c.status === "ok" ? "ok" : "warn"}`}>{c.status}</span> {c.message}
              </li>
            ))}
          </ul>
          <p className="muted">{quality.note}</p>
        </>
      )}
      {slices && (
        <>
          <h3>Rough slice readings (diagnosis only)</h3>
          <div className="table-wrap">
          <table className="rtable">
            <thead><tr><th>Level</th><th>Outline length (% of body height)</th><th>Status</th></tr></thead>
            <tbody>
              {slices.items.map((i) => (
                <tr key={i.name}>
                  <td>{i.name} · {Math.round(100 * i.height_fraction)}% up</td>
                  <td>{i.value ?? `– (${i.reason})`}</td>
                  <td><span className="pill warn">{i.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
          <p className="muted">{slices.note}</p>
        </>
      )}
    </>
  );
}
