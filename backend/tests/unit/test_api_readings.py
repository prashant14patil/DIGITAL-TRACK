from fastapi.testclient import TestClient

from digitaltrack.api.app import create_app

SEG = {k: {"fat_pct": 20.0} for k in ("left_arm", "right_arm", "trunk", "left_leg", "right_leg")}
R = {"date": "2026-03-12", "weight_kg": 67.4, "body_fat_pct": 21.6, "muscle_kg": 50.6}


def _c(tmp_path):
    return TestClient(create_app(tmp_path / "uploads", serve_frontend=False, jobs_inline=True))


def test_add_list_delete_persist(tmp_path):
    c = _c(tmp_path)
    assert c.get("/api/readings").json()["readings"] == []
    a = c.post("/api/readings", json={**R, "date": "2026-05-02", "segments": SEG}).json()
    c.post("/api/readings", json=R)
    doc = _c(tmp_path).get("/api/readings").json()  # new app instance reads the same file
    assert [r["date"] for r in doc["readings"]] == ["2026-03-12", "2026-05-02"]
    assert doc["readings"][0]["source"] == "manual" and doc["readings"][0]["segments"] is None
    assert c.delete(f"/api/readings/{a['id']}").status_code == 200
    assert c.delete(f"/api/readings/{a['id']}").status_code == 404
    assert len(c.get("/api/readings").json()["readings"]) == 1


def test_validation_rejects_bad_input(tmp_path):
    c = _c(tmp_path)
    for bad in ({**R, "date": "12/03/2026"}, {**R, "date": "2026-13-40"}, {**R, "body_fat_pct": 0},
                {**R, "weight_kg": -1}, {**R, "segments": {"trunk": {"fat_pct": 20}}},
                {**R, "segments": {**SEG, "trunk": {"fat_pct": 20, "category": "great"}}}):
        assert c.post("/api/readings", json=bad).status_code == 422, bad
    assert c.get("/api/readings").json()["readings"] == []


def test_profile_and_target(tmp_path):
    c = _c(tmp_path)
    assert c.put("/api/readings/profile", json={"sex": "Male", "height_cm": 174}).status_code == 200
    assert c.put("/api/readings/target", json={"weight_kg": 66, "body_fat_pct": 18, "muscle_kg": 52}).status_code == 200
    doc = c.get("/api/readings").json()
    assert doc["profile"]["height_cm"] == 174 and doc["target"]["body_fat_pct"] == 18
    c.delete("/api/readings/target")
    assert c.get("/api/readings").json()["target"] is None
