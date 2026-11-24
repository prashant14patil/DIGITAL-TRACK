// Thin client for the FastAPI backend (/api). Uploads stream each File as the raw request body.

async function json(res) {
  if (res.ok) return res.status === 204 ? null : res.json();
  let detail = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (typeof body.detail === "string") detail = body.detail;
    else if (Array.isArray(body.detail))
      detail = body.detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
  } catch {
    /* keep status text */
  }
  throw new Error(detail);
}

const send = (method, url, body) =>
  fetch(url, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined }).then(json);

export const api = {
  health: () => fetch("/api/health").then(json),
  list: () => fetch("/api/uploads").then(json),
  get: (id) => fetch(`/api/uploads/${id}`).then(json),
  create: (meta) =>
    fetch("/api/uploads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(meta),
    }).then(json),
  finalize: (id, files) =>
    fetch(`/api/uploads/${id}/finalize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ files }),
    }).then(json),
  discard: (id) => fetch(`/api/uploads/${id}`, { method: "DELETE" }).then(json),

  // processing jobs (experimental mesh)
  process: (id, file) =>
    fetch(`/api/uploads/${id}/process`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(file ? { file } : {}),
    }).then(json),
  jobs: () => fetch("/api/jobs").then(json),
  readings: () => fetch("/api/readings").then(json),
  addReading: (r) => send("POST", "/api/readings", r),
  deleteReading: (id) => send("DELETE", `/api/readings/${id}`),
  putProfile: (p) => send("PUT", "/api/readings/profile", p),
  putTarget: (t) => send("PUT", "/api/readings/target", t),
  deleteTarget: () => send("DELETE", "/api/readings/target"),
  job: (id) => fetch(`/api/jobs/${id}`).then(json),
  jobFileUrl: (id, name) => `/api/jobs/${id}/files/${name}`,

  // XHR (not fetch) so upload progress is observable.
  putFile(id, path, file, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("PUT", `/api/uploads/${id}/files?path=${encodeURIComponent(path)}`);
      xhr.setRequestHeader("Content-Type", "application/octet-stream");
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded);
      xhr.onload = () => {
        let body = null;
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          /* ignore */
        }
        if (xhr.status === 200) resolve(body);
        else reject(new Error((body && body.detail) || `${xhr.status} ${xhr.statusText}`));
      };
      xhr.onerror = () => reject(new Error("network error (is `dt serve` running?)"));
      xhr.send(file);
    });
  },
};

// Recursively collect files from a drag-and-drop (folders included), keeping relative paths.
export async function filesFromDrop(dataTransfer) {
  const entries = [...dataTransfer.items].map((i) => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
  if (!entries.length) return [...dataTransfer.files].map((f) => ({ path: f.name, file: f }));
  const out = [];
  const readAll = (reader) =>
    new Promise((resolve, reject) => {
      const all = [];
      const next = () =>
        reader.readEntries((batch) => (batch.length ? (all.push(...batch), next()) : resolve(all)), reject);
      next();
    });
  async function walk(entry) {
    if (entry.isFile) {
      const file = await new Promise((res, rej) => entry.file(res, rej));
      out.push({ path: entry.fullPath.replace(/^\/+/, ""), file });
    } else if (entry.isDirectory) {
      for (const child of await readAll(entry.createReader())) await walk(child);
    }
  }
  for (const e of entries) await walk(e);
  return out;
}

export function formatBytes(n) {
  if (n == null) return "";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) (n /= 1024), i++;
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
}
