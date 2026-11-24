# frontend: capture upload page

Local web page for uploading capture exports (for example a Stray Scanner folder) to the backend. Upload only:
nothing is processed or converted. Files are stored byte-for-byte with SHA-256, verified on finalize, then made
read-only, under `data/real/uploads/<upload_id>/` (git-ignored).

## Run (Windows, PowerShell, from the repository root)

One-time setup:

```powershell
backend\.venv\Scripts\python.exe -m pip install -e "backend[dev]"   # adds fastapi + uvicorn
cd frontend; npm install; npm run build; cd ..
```

Start, then open http://127.0.0.1:8000 :

```powershell
backend\.venv\Scripts\dt.exe serve
```

Frontend development with hot reload: keep `dt serve` running, then `cd frontend; npm run dev` and open
http://127.0.0.1:5173 (Vite forwards `/api` to port 8000).

## Rules built in

- Local only (binds 127.0.0.1; a warning is printed for any other host). No authentication.
- Recordings with a person require a ticked consent confirmation (consent is handled by the team's own
  process) and a pseudonymous subject ID (letters, digits, `-`, `_`; never a name).
- Existing files are never overwritten; finalized uploads cannot be changed or deleted from the page.
- Unsafe paths (`..`, absolute paths, drive letters, Windows reserved names) are rejected by the backend.
