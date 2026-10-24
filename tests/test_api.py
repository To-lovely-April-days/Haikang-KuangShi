"""接口清单逐条集成测试（P0：health / state / system；P2 补齐其余）。"""

from __future__ import annotations


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["version"]


def test_state_shape(client):
    r = client.get("/api/v1/state")
    assert r.status_code == 200
    s = r.json()
    for key in (
        "time",
        "uptime_total_s",
        "run_state",
        "deployed_model_id",
        "cameras",
        "mcu",
        "faults",
        "jobs",
        "stats",
    ):
        assert key in s, key
    assert s["run_state"] in {"stopped", "running", "paused", "sampling", "fault"}
    assert s["cameras"][0]["id"] == 1 and s["cameras"][0]["connected"] is True
    assert s["mcu"]["connected"] is True
    assert s["faults"] == []


def test_system_info_and_save(client):
    r = client.get("/api/v1/system/info")
    assert r.status_code == 200
    assert r.json()["disk_free_mb"] > 0
    r = client.post("/api/v1/system/save")
    assert r.status_code == 200 and r.json()["saved"] is True
    assert (client.ctx.data_dir / "config.backup.json").exists()


def test_404_is_structured(client):
    r = client.get("/api/v1/nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "NOT_FOUND"


def test_static_mount(client):
    p = client.ctx.data_dir / "snapshots" / "x.txt"
    p.write_text("hi")
    r = client.get("/static/snapshots/x.txt")
    assert r.status_code == 200 and r.text == "hi"
