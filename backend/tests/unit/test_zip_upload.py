import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from digitaltrack.api.app import create_app


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def _upload(client, payload: bytes) -> str:
    uid = client.post("/api/uploads", json={"recording_type": "other", "label": "zip", "contains_person": False}).json()["id"]
    assert client.put(f"/api/uploads/{uid}/files", params={"path": "scan.zip"}, content=payload).status_code == 200
    assert client.post(f"/api/uploads/{uid}/finalize", json={"files": [{"path": "scan.zip", "size": len(payload)}]}).status_code == 200
    return uid


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(tmp_path / "uploads", serve_frontend=False)), tmp_path / "uploads"


def test_inspect_unpacks_a_copy_and_keeps_the_zip(client):
    c, data = client
    payload = _zip({"rec/depth/000000.png": b"x", "rec/depth/000001.png": b"y", "rec/odometry.csv": b"a,b\n1,2\n",
                    "__MACOSX/junk": b"z"})
    uid = _upload(c, payload)
    for _ in range(2):  # repeatable
        r = c.post(f"/api/uploads/{uid}/inspect")
        assert r.status_code == 200, r.text
        assert r.json()["file_count"] == 3
    assert (data / uid / "export" / "scan.zip").read_bytes() == payload
    assert (data / uid / "extracted" / "scan" / "rec" / "odometry.csv").is_file()


@pytest.mark.parametrize("payload", [_zip({"../evil.txt": b"x"}), b"not a zip"])
def test_inspect_rejects_unsafe_or_broken_zip(client, payload):
    c, data = client
    uid = _upload(c, payload)
    assert c.post(f"/api/uploads/{uid}/inspect").status_code == 409
    assert not list(data.rglob("evil.txt"))
    assert not (data / uid / "extracted" / "scan").exists()
