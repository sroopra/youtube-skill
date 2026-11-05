import json

from ytu import cli, proc
from ytu.pipeline import doctor


def test_doctor_reports_without_key_and_never_prints_key(monkeypatch, cfg, capsys, tmp_path):
    monkeypatch.setattr(proc, "which", lambda n: None if n in ("ffmpeg", "deno") else f"/bin/{n}")
    monkeypatch.setattr(proc, "ytdlp_available_inprocess", lambda: False)
    out = doctor.run_doctor(cfg)
    assert out["yt_dlp"]["launcher"] == "uvx" and out["binaries"]["ffmpeg"] is None
    assert out["gemini_key"]["present"] is False and "network" not in out
    assert any("ffmpeg" in w for w in out["warnings"]) and any("no Gemini key" in w for w in out["warnings"])
    assert out["cache_writable"] is True

    monkeypatch.setenv("GEMINI_API_KEY", "sekret-value-123")
    assert cli.main(["doctor", "--cache-dir", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "sekret-value-123" not in text and json.loads(text)["gemini_key"]["source"] == "env:GEMINI_API_KEY"


def test_doctor_network_probe_failure_has_remediation(monkeypatch, cfg):
    monkeypatch.setattr(doctor, "probe_host", lambda h, t: {"ok": h.startswith("www"), "error": "DNS failed",
                                                             "remediation": doctor.SANDBOX_HINT})
    out = doctor.run_doctor(cfg, network=True)
    bad = out["network"]["generativelanguage.googleapis.com"]
    assert not bad["ok"] and "allowedDomains" in bad["remediation"]
    assert any("generativelanguage" in w for w in out["warnings"])


def test_probe_host_dns_failure(monkeypatch):
    def gai(*a, **k):
        raise OSError("nodename nor servname provided")
    monkeypatch.setattr(doctor.socket, "getaddrinfo", gai)
    r = doctor.probe_host("www.youtube.com", 1.0)
    assert r["ok"] is False and "remediation" in r
