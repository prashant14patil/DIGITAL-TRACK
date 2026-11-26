import { useState } from "react";
import { api, formatBytes } from "./api.js";

export default function ZipContents({ upload }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  if (upload.status !== "complete" || !upload.files.some((f) => /\.zip$/i.test(f.path))) return null;

  async function open() {
    setBusy(true);
    setErr(null);
    try {
      setData(await api.inspect(upload.id));
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="process">
      <button disabled={busy} onClick={open}>{busy ? "Unpacking…" : "Open zip contents"}</button>
      <small className="muted">Lists what is inside. The zip stays unchanged; a copy is unpacked.</small>
      {err && <div className="errors">{err}</div>}
      {data && (
        <div className="filelist small">
          <div className="file"><span className="path">{data.file_count} files, {formatBytes(data.total_bytes)}</span></div>
          {data.groups.map((g) => (
            <div key={g.folder + g.extension} className="file">
              <span className="path">{g.folder} · {g.extension}</span>
              <span className="size">{g.count} files, {formatBytes(g.total_bytes)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
