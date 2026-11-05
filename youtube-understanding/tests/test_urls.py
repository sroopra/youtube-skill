import pytest

from ytu.errors import BadInput
from ytu.urls import fmt_ts, parse_ts, parse_url, ts_link

VID = "jNQXAC9IVRw"


@pytest.mark.parametrize("url,start", [
    (VID, None),
    (f"https://www.youtube.com/watch?v={VID}", None),
    (f"http://youtube.com/watch?v={VID}&feature=share", None),
    (f"www.youtube.com/watch?v={VID}", None),
    (f"https://m.youtube.com/watch?v={VID}&t=42", 42.0),
    (f"https://music.youtube.com/watch?v={VID}&list=RDAMVM{VID}", None),
    (f"https://youtu.be/{VID}", None),
    (f"https://youtu.be/{VID}?si=abc&t=1m2s", 62.0),
    (f"https://www.youtube.com/shorts/{VID}", None),
    (f"https://www.youtube.com/embed/{VID}?start=90", 90.0),
    (f"https://www.youtube-nocookie.com/embed/{VID}", None),
    (f"https://www.youtube.com/live/{VID}?feature=share", None),
    (f"https://www.youtube.com/v/{VID}", None),
    (f"https://www.youtube.com/watch?v={VID}&t=1h2m3s", 3723.0),
    (f"https://www.youtube.com/watch?v={VID}#t=754", 754.0),
    (f"https://www.youtube.com/watch?v={VID}&list=PL123&index=2", None),
    (f"  https://www.youtube.com/watch?v={VID}  ", None),
    (f"https://www.youtube.com/watch?v={VID}&t=garbage", None),
])
def test_parse_url_forms(url, start):
    ref = parse_url(url)
    assert ref.video_id == VID
    assert ref.canonical_url == f"https://www.youtube.com/watch?v={VID}"
    assert ref.start_s == start


@pytest.mark.parametrize("url", [
    "", "https://www.youtube.com/playlist?list=PL123",
    "https://www.youtube.com/@karpathy", "https://www.youtube.com/@NASA/live",
    "https://www.youtube.com/channel/UC123", "https://www.youtube.com/c/foo", "https://www.youtube.com/user/foo",
    "https://vimeo.com/123", f"https://evil.com/watch?v={VID}", f"https://youtube.com.evil.com/watch?v={VID}",
    "https://www.youtube.com/watch?v=short", "https://youtu.be/", "ftp://youtube.com/watch?v=" + VID,
    "https://www.youtube.com/watch?v=" + VID + "x", "notavideo", "a" * 12,
])
def test_parse_url_rejects(url):
    with pytest.raises(BadInput) as e:
        parse_url(url)
    assert e.value.exit_code == 2
    assert e.value.remediation


def test_playlist_and_channel_messages():
    with pytest.raises(BadInput, match="playlist"):
        parse_url("https://www.youtube.com/playlist?list=PL1")
    with pytest.raises(BadInput, match="channel"):
        parse_url("https://www.youtube.com/@x")


@pytest.mark.parametrize("s,expected", [
    ("754", 754.0), ("754s", 754.0), ("754.5", 754.5), ("12:34", 754.0), ("1:02:03", 3723.0),
    ("12m34s", 754.0), ("1h2m3s", 3723.0), ("1h", 3600.0), ("2m", 120.0), ("0", 0.0), (90, 90.0), ("00:00:05", 5.0),
])
def test_parse_ts(s, expected):
    assert parse_ts(s) == expected


@pytest.mark.parametrize("s", ["", "abc", "1:2:3:4", "-5", "12:xx", "1h2x", -1])
def test_parse_ts_rejects(s):
    with pytest.raises(BadInput):
        parse_ts(s)


@pytest.mark.parametrize("sec,expected", [
    (0, "00:00"), (5.9, "00:05"), (754.2, "12:34"), (3599, "59:59"), (3600, "01:00:00"), (3723.7, "01:02:03"),
    (-3, "00:00"),
])
def test_fmt_ts(sec, expected):
    assert fmt_ts(sec) == expected


def test_ts_link():
    assert ts_link(VID, 754.9) == f"https://www.youtube.com/watch?v={VID}&t=754s"
    assert ts_link(VID, 0) == f"https://www.youtube.com/watch?v={VID}&t=0s"


def test_roundtrip_fmt_parse():
    for s in (0, 59, 61, 3599, 3600, 7322):
        assert parse_ts(fmt_ts(s)) == s


@pytest.mark.parametrize("bad", ["99:99:99", "1:75", "10:60", "1:60:00"])
def test_colon_fields_out_of_range(bad):
    with pytest.raises(BadInput):
        parse_ts(bad)
