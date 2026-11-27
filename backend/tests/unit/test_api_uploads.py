"""Capture-upload API: raw files stored byte-for-byte, hashed, never overwritten, frozen on finalize."""

from __future__ import annotations

import hashlib
import json
import os

import pytest
from fastapi.testclient import TestClient

from digitaltrack.api.app import create_app
from digitaltrack.api.storage import UploadError, safe_relpath

META = {"recording_type": "R2", "label": "box static", "contains_person": False, "object_kind": "box",
        "measurements": [{"name": "length", "readings_mm": [400.0, 401.0, 400.5], "instrument": "tape"}]}

FILES = {
    "71de12f9/rgb.mp4": os.urandom(300_000),
    "71de12f9/camera_matrix.csv": b"1,0,0\n0,1,0\n0,0,1\n",
    "71de12f9/depth/000000.npy": os.urandom(5000),
}


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(tmp_path / "uploads", serve_frontend=False))


def _upload(client, files=FILES, meta=META):
    uid = client.post("/api/uploads", json=meta).json()["id"]
    for p, data in files.items():
        r = client.put(f"/api/uploads/{uid}/files", params={"path": p}, content=data)
        assert r.status_code == 200, r.text
        assert r.json() == {"path": p, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return uid


def test_upload_roundtrip_is_byte_exact_and_frozen(client, tmp_path):
    uid = _upload(client)
    r = client.post(f"/api/uploads/{uid}/finalize", json={"files": [{"path": p, "size": len(d)} for p, d in FILES.items()]})
    assert r.status_code == 200, r.text
    man = r.json()
    assert man["file_count"] == 3 and man["total_bytes"] == sum(map(len, FILES.values()))

    root = tmp_path / "uploads" / uid
    for p, data in FILES.items():
        f = root / "export" / p
        assert f.read_bytes() == data
        assert not os.access(f, os.W_OK)  # read-only after finalize
    disk_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert {f["path"]: f["sha256"] for f in disk_manifest["files"]} == \
        {p: hashlib.sha256(d).hexdigest() for p, d in FILES.items()}
    assert json.loads((root / "metadata.json").read_text(encoding="utf-8"))["label"] == "box static"

    got = client.get(f"/api/uploads/{uid}").json()
    assert got["status"] == "complete" and got["file_count"] == 3
    assert client.get("/api/uploads").json()[0]["id"] == uid
    assert not list(root.rglob("*.part-*"))


def test_finalized_upload_is_immutable(client):
    uid = _upload(client)
    client.post(f"/api/uploads/{uid}/finalize", json={"files": [{"path": p, "size": len(d)} for p, d in FILES.items()]})
    assert client.put(f"/api/uploads/{uid}/files", params={"path": "new.txt"}, content=b"x").status_code == 409
    assert client.delete(f"/api/uploads/{uid}").status_code == 409
    assert client.post(f"/api/uploads/{uid}/finalize", json={"files": [{"path": "a", "size": 1}]}).status_code == 409


def test_no_overwrite_of_raw_files(client):
    uid = _upload(client)
    r = client.put(f"/api/uploads/{uid}/files", params={"path": "71de12f9/rgb.mp4"}, content=b"other")
    assert r.status_code == 409 and "never overwritten" in r.json()["detail"]


def test_finalize_detects_missing_extra_and_size_mismatch(client):
    uid = _upload(client)
    exp = [{"path": p, "size": len(d)} for p, d in FILES.items()]
    r = client.post(f"/api/uploads/{uid}/finalize", json={"files": exp + [{"path": "71de12f9/odometry.csv", "size": 10}]})
    assert r.status_code == 409 and "missing: 71de12f9/odometry.csv" in r.json()["detail"]
    r = client.post(f"/api/uploads/{uid}/finalize", json={"files": exp[:2]})
    assert "not expected" in r.json()["detail"]
    exp[0]["size"] += 1
    r = client.post(f"/api/uploads/{uid}/finalize", json={"files": exp})
    assert "size differs" in r.json()["detail"]
    assert client.get(f"/api/uploads/{uid}").json()["status"] == "receiving"  # nothing frozen on failure


def test_incomplete_upload_can_be_discarded(client, tmp_path):
    uid = _upload(client)
    assert client.delete(f"/api/uploads/{uid}").status_code == 204
    assert client.get(f"/api/uploads/{uid}").status_code == 404
    assert not (tmp_path / "uploads" / uid).exists()


def test_finalized_upload_needs_explicit_permanent_delete(client, tmp_path):
    uid = _upload(client)
    client.post(f"/api/uploads/{uid}/finalize", json={"files": [{"path": p, "size": len(d)} for p, d in FILES.items()]})
    r = client.delete(f"/api/uploads/{uid}")  # plain discard never removes raw data
    assert r.status_code == 409 and (tmp_path / "uploads" / uid / "export").is_dir()
    assert client.delete(f"/api/uploads/{uid}", params={"permanently": "true"}).status_code == 204  # read-only files too
    assert not (tmp_path / "uploads" / uid).exists()
    assert client.get(f"/api/uploads/{uid}").status_code == 404
    assert all(u["id"] != uid for u in client.get("/api/uploads").json())
    assert client.delete(f"/api/uploads/{uid}", params={"permanently": "true"}).status_code == 404


@pytest.mark.parametrize("bad", ["../x", "a/../../x", "/etc/passwd", "C:/x", "C:x", "a//b", "a/./b", "", "a\x00b",
                                 "CON", "nul.txt", "a/b.", "a/b ", 'a"b', "a?b"])
def test_unsafe_paths_rejected(bad):
    with pytest.raises(UploadError):
        safe_relpath(bad)


def test_backslash_paths_normalized():
    assert safe_relpath("71de12f9\\depth\\000000.npy") == "71de12f9/depth/000000.npy"


def test_unsafe_path_rejected_over_http(client, tmp_path):
    uid = client.post("/api/uploads", json=META).json()["id"]
    r = client.put(f"/api/uploads/{uid}/files", params={"path": "../../escape.txt"}, content=b"x")
    assert r.status_code == 409
    assert not (tmp_path / "escape.txt").exists() and not (tmp_path / "uploads" / "escape.txt").exists()


def test_person_recordings_need_consent(client):
    r = client.post("/api/uploads", json={**META, "contains_person": True})
    assert r.status_code == 422 and "consent" in r.text
    r = client.post("/api/uploads", json={**META, "contains_person": True, "consent_confirmed": True,
                                          "subject_id": "S001"})
    assert r.status_code == 201


def test_subject_id_is_pseudonymous_token_only(client):
    r = client.post("/api/uploads", json={**META, "contains_person": True, "consent_confirmed": True,
                                          "subject_id": "John Smith"})
    assert r.status_code == 422


@pytest.mark.parametrize("patch", [{"recording_type": "R9"}, {"label": ""}, {"camera_distance_m": -1},
                                   {"measurements": [{"name": "x", "readings_mm": [0]}]}])
def test_metadata_validation(client, patch):
    assert client.post("/api/uploads", json={**META, **patch}).status_code == 422


def test_unknown_upload_ids(client):
    assert client.get("/api/uploads/up_20260101T000000Z_abcdef").status_code == 404
    assert client.get("/api/uploads/..%2F..%2Fetc").status_code == 404
