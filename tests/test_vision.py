import pymupdf
import pytest

from src.ingestion.pdf_loader import PDFLoader
from src.models import PageImage
from src.vision.describer import VisionDescriber
from src.vision.providers import QuotaExceeded


class StubProvider:
    def __init__(self, name, replies):
        """`replies` is consumed one per call; an Exception instance in it is raised instead of returned."""
        self.name, self.replies, self.calls = name, list(replies), 0

    def describe(self, image, prompt):
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else "default description"
        if isinstance(reply, Exception):
            raise reply
        return reply


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def img(data=b"one", full_page=False):
    return PageImage(page=1, index=0, data=data, mime_type="image/png", is_full_page=full_page)


def describer(tmp_path, *providers, clock=None):
    return VisionDescriber(list(providers), cache_path=tmp_path / "cache.json", cooldown_s=60, clock=clock or Clock())


def test_first_provider_is_used_when_healthy(tmp_path):
    gemini, ollama = StubProvider("gemini", ["a chart"]), StubProvider("ollama", ["x"])
    d = describer(tmp_path, gemini, ollama)
    assert d.describe(img()) == "a chart"
    assert (gemini.calls, ollama.calls) == (1, 0)


def test_quota_error_falls_back_to_next_provider(tmp_path):
    gemini, ollama = StubProvider("gemini", [QuotaExceeded("429")]), StubProvider("ollama", ["from ollama"])
    d = describer(tmp_path, gemini, ollama)
    assert d.describe(img()) == "from ollama"
    assert d.provider_counts == {"ollama": 1}
    assert any("gemini quota exhausted" in e for e in d.events)


def test_exhausted_provider_is_skipped_during_cooldown_then_retried(tmp_path):
    clock = Clock()
    gemini = StubProvider("gemini", [QuotaExceeded("429"), "gemini is back"])
    ollama = StubProvider("ollama", ["o1", "o2"])
    d = describer(tmp_path, gemini, ollama, clock=clock)

    assert d.describe(img(b"1")) == "o1"       # gemini hits quota, ollama answers
    clock.now = 30
    assert d.describe(img(b"2")) == "o2"       # still cooling down: gemini not even called
    assert gemini.calls == 1
    clock.now = 61
    assert d.describe(img(b"3")) == "gemini is back"   # cooldown over: gemini tried first again


def test_other_errors_skip_provider_for_one_image_only(tmp_path):
    gemini = StubProvider("gemini", [RuntimeError("503 overloaded"), "second image ok"])
    ollama = StubProvider("ollama", ["fallback"])
    d = describer(tmp_path, gemini, ollama)
    assert d.describe(img(b"1")) == "fallback"
    assert d.describe(img(b"2")) == "second image ok"   # gemini was NOT blocked by the non-quota error


def test_all_providers_failing_returns_none_and_records_error(tmp_path):
    d = describer(tmp_path, StubProvider("gemini", [QuotaExceeded("x")]), StubProvider("ollama", [ConnectionError("down")]))
    assert d.describe(img()) is None
    assert "ollama" in d.last_error


def test_none_reply_means_decorative_and_is_cached_as_empty(tmp_path):
    p = StubProvider("gemini", ["NONE"])
    d = describer(tmp_path, p)
    assert d.describe(img()) == ""
    assert d.describe(img()) == ""       # served from cache, not asked again
    assert p.calls == 1


def test_empty_reply_counts_as_failure_and_tries_next(tmp_path):
    d = describer(tmp_path, StubProvider("gemini", ["   "]), StubProvider("ollama", ["real text"]))
    assert d.describe(img()) == "real text"


def test_cache_persists_across_runs_and_costs_nothing(tmp_path):
    first = StubProvider("gemini", ["cached description"])
    describer(tmp_path, first).describe(img())
    second = StubProvider("gemini", ["should not be used"])
    d2 = describer(tmp_path, second)
    assert d2.describe(img()) == "cached description"
    assert second.calls == 0 and d2.cache_hits == 1


def test_cache_still_serves_when_no_provider_is_available(tmp_path):
    describer(tmp_path, StubProvider("gemini", ["kept description"])).describe(img(b"known"))
    offline = describer(tmp_path)   # no providers at all
    assert offline.describe(img(b"known")) == "kept description"
    assert offline.describe(img(b"never seen")) is None
    assert offline.last_error == "no vision provider available"


def test_scan_and_figure_of_same_bytes_are_cached_separately(tmp_path):
    p = StubProvider("gemini", ["figure text", "transcribed text"])
    d = describer(tmp_path, p)
    assert d.describe(img(b"x", full_page=False)) == "figure text"
    assert d.describe(img(b"x", full_page=True)) == "transcribed text"


def make_pdf(path):
    doc = pymupdf.open()
    big = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 300))
    big.clear_with(200)
    small = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40))
    small.clear_with(90)

    page = doc.new_page()                                   # page 1: text + one real image + one icon
    page.insert_text((72, 72), "A page with a chart and a tiny icon.")
    page.insert_image(pymupdf.Rect(72, 100, 272, 300), pixmap=big)
    page.insert_image(pymupdf.Rect(300, 100, 330, 130), pixmap=small)

    page2 = doc.new_page()                                  # page 2: no text at all (scanned-style)
    page2.insert_image(pymupdf.Rect(0, 0, 400, 400), pixmap=big)

    doc.new_page().insert_text((72, 72), "Plain text page.")   # page 3: text only
    doc.save(path)


def test_loader_yields_figures_and_full_scanned_pages_but_skips_icons(tmp_path):
    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf)
    images = list(PDFLoader().iter_images(str(pdf)))

    assert [(i.page, i.is_full_page) for i in images] == [(1, False), (2, True)]
    assert all(i.data.startswith(b"\x89PNG") for i in images)
