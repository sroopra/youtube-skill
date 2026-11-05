import logging
import os
from pathlib import Path

import pytest

from ytu.config import (DEFAULT_MODELS, Config, get_gemini_key, key_status, parse_env_file,
                        require_gemini_key, resolve_cache_dir)
from ytu.errors import BadInput, MissingKey

SECRET = "AIzaTESTSECRETVALUE123"


def test_defaults(tmp_path):
    c = Config.load({"cache_dir": tmp_path})
    assert c.models == DEFAULT_MODELS
    assert c.transcript_providers == ["ytdlp", "yta", "asr"]
    assert c.max_cost_usd == 0.50 and c.max_input_tokens == 800_000
    assert c.strategy == "auto" and c.force_refresh is False
    assert c.out_dir_for("abc") == Path("youtube-knowledge") / "abc"
    assert c.timeout("metadata") == 60.0


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("YTU_GEMINI_MODELS", "m1, m2")
    monkeypatch.setenv("YTU_TRANSCRIPT_PROVIDERS", "yta,ytdlp")
    monkeypatch.setenv("YTU_MAX_COST_USD", "1.5")
    monkeypatch.setenv("YTU_TIMEOUT_METADATA", "5")
    c = Config.load({"cache_dir": tmp_path})
    assert c.models == ["m1", "m2"]
    assert c.transcript_providers == ["yta", "ytdlp"]
    assert c.max_cost_usd == 1.5
    assert c.timeout("metadata") == 5.0


def test_single_model_pins_first(monkeypatch, tmp_path):
    monkeypatch.setenv("YTU_GEMINI_MODEL", "gemini-3.5-flash-lite")
    c = Config.load({"cache_dir": tmp_path})
    assert c.models[0] == "gemini-3.5-flash-lite"
    assert c.models.count("gemini-3.5-flash-lite") == 1 and len(c.models) == 3


def test_precedence_file_env_cli(monkeypatch, tmp_path):
    cf = tmp_path / "config.toml"
    cf.write_text('max_cost_usd = 2.0\nmodels = ["a"]\nbogus = 1\n[timeouts]\ngemini = 10\n')
    monkeypatch.setenv("YTU_CONFIG_FILE", str(cf))
    c = Config.load({"cache_dir": tmp_path})
    assert c.max_cost_usd == 2.0 and c.models == ["a"] and c.timeout("gemini") == 10
    monkeypatch.setenv("YTU_MAX_COST_USD", "3")
    assert Config.load({"cache_dir": tmp_path}).max_cost_usd == 3.0
    assert Config.load({"cache_dir": tmp_path, "max_cost_usd": 4.0, "models": None}).max_cost_usd == 4.0


def test_invalid_values(monkeypatch, tmp_path):
    with pytest.raises(BadInput):
        Config.load({"cache_dir": tmp_path, "strategy": "nope"})
    with pytest.raises(BadInput):
        Config.load({"cache_dir": tmp_path, "transcript_providers": ["bogus"]})
    monkeypatch.setenv("YTU_MAX_COST_USD", "abc")
    with pytest.raises(BadInput):
        Config.load({"cache_dir": tmp_path})


def test_bad_toml(monkeypatch, tmp_path):
    cf = tmp_path / "c.toml"
    cf.write_text("not = = toml")
    monkeypatch.setenv("YTU_CONFIG_FILE", str(cf))
    with pytest.raises(BadInput):
        Config.load({"cache_dir": tmp_path})


def test_cache_dir_resolution(monkeypatch, tmp_path):
    monkeypatch.setenv("YTU_CACHE_DIR", str(tmp_path / "c1"))
    assert resolve_cache_dir() == tmp_path / "c1"
    monkeypatch.delenv("YTU_CACHE_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert resolve_cache_dir() == tmp_path / "xdg" / "youtube-understanding"


def test_cache_dir_fallback_when_unwritable(monkeypatch, tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        monkeypatch.setenv("YTU_CACHE_DIR", str(ro / "sub"))
        monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
        p = resolve_cache_dir()
        assert p.parent == tmp_path / "tmp" and p.name.startswith("youtube-understanding-")
    finally:
        ro.chmod(0o700)


def test_parse_env_file():
    text = """
# comment
export GEMINI_API_KEY="abc def"
GOOGLE_API_KEY='x'
PLAIN=value # trailing
EMPTY=
bad line
 SPACED = v
"""
    d = parse_env_file(text)
    assert d == {"GEMINI_API_KEY": "abc def", "GOOGLE_API_KEY": "x", "PLAIN": "value", "EMPTY": "", "SPACED": "v"}


def test_key_from_env_first(monkeypatch, tmp_path):
    ef = tmp_path / ".env"
    ef.write_text("GEMINI_API_KEY=fromfile\n")
    ef.chmod(0o600)
    monkeypatch.setenv("YTU_ENV_FILE", str(ef))
    assert get_gemini_key() == "fromfile"
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    assert get_gemini_key() == "google"
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    assert get_gemini_key() == SECRET
    assert key_status()["source"] == "env:GEMINI_API_KEY"


def test_key_default_file_location(tmp_path):
    ef = Path(os.environ["XDG_CONFIG_HOME"]) / "youtube-understanding" / ".env"
    ef.parent.mkdir(parents=True)
    ef.write_text(f"GEMINI_API_KEY={SECRET}\n")
    ef.chmod(0o600)
    assert get_gemini_key() == SECRET
    st = key_status()
    assert st["present"] and st["source"] == f"file:{ef}"
    assert SECRET not in repr(st)


def test_key_file_permission_warning(monkeypatch, tmp_path, caplog):
    ef = tmp_path / ".env"
    ef.write_text(f"GEMINI_API_KEY={SECRET}\n")
    ef.chmod(0o644)
    monkeypatch.setenv("YTU_ENV_FILE", str(ef))
    with caplog.at_level(logging.WARNING, logger="ytu"):
        assert get_gemini_key() == SECRET
    assert "chmod 600" in caplog.text
    assert SECRET not in caplog.text


def test_missing_key_error(monkeypatch, tmp_path):
    monkeypatch.setenv("YTU_ENV_FILE", str(tmp_path / "nope.env"))
    assert get_gemini_key() is None
    with pytest.raises(MissingKey) as e:
        require_gemini_key()
    assert e.value.exit_code == 5
    assert "missing" in e.value.message and "nope.env" in e.value.message
    assert "GEMINI_API_KEY" in e.value.message


def test_key_never_in_config_dict(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    assert SECRET not in repr(Config.load({"cache_dir": tmp_path}).to_dict())


def test_cache_dir_fallback_rejects_foreign_or_open_dir(monkeypatch, tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    planted = tmp_path / "tmp" / f"youtube-understanding-{os.getuid()}"
    planted.mkdir(parents=True)
    planted.chmod(0o777)  # pre-created world-writable by "someone else"
    try:
        monkeypatch.setenv("YTU_CACHE_DIR", str(ro / "sub"))
        monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
        p = resolve_cache_dir()
        assert p != planted and p.name.startswith("youtube-understanding-") and oct(p.stat().st_mode & 0o777) == "0o700"
    finally:
        ro.chmod(0o700)


def test_cache_tmp_root_refuses_symlink(tmp_path):
    from ytu.config import cache_tmp_root
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "tmp").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(BadInput):
        cache_tmp_root(tmp_path / "c")
    assert cache_tmp_root(tmp_path / "ok") == tmp_path / "ok" / "tmp"


def test_file_key_registered_for_redaction(monkeypatch, tmp_path):
    from ytu import proc
    ef = tmp_path / ".env"
    ef.write_text("GEMINI_API_KEY=AIzaFILEONLYKEY987654\n")
    ef.chmod(0o600)
    monkeypatch.setenv("YTU_ENV_FILE", str(ef))
    assert get_gemini_key() == "AIzaFILEONLYKEY987654"
    assert proc.redact("boom AIzaFILEONLYKEY987654 boom") == "boom *** boom"
