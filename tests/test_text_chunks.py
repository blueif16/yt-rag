"""One text layer: pages and captions become chunks with the same envelope."""
import pytest

from yt_rag.chunks import chunk_meta, header
from yt_rag.ingest import build_items_for_video
from yt_rag.text_ingest import (
    YOUTUBE_RE,
    build_items_for_text,
    page_doc_id,
    split_markdown,
)

CORE = {"type", "origin", "source", "doc_id", "title", "author",
        "published_at", "locator", "chunk_index"}

PAGE = """# Expression examples

Intro paragraph about expressions.

## Wiggle randomly

wiggle(2,30)

Makes the layer wiggle 2 times per second up to 30 pixels.

## Overshoot

```
amp = .06; freq = 3; decay = 5;
n = 0;
```

Inertial bounce after the last keyframe.
"""


# ---- splitting ----

def test_sections_pack_into_one_chunk_when_small():
    chunks = split_markdown(PAGE)
    assert len(chunks) == 1
    # the single H1 is the page title, so it is not a section
    assert chunks[0]["sections"] == ["Wiggle randomly", "Overshoot"]
    assert "## Overshoot" in chunks[0]["text"]


def test_section_labels_keep_two_levels():
    md = ("# Speed\n\n## Control speed\n\nIntro text here.\n\n### Before\n\n### After\n\n"
          + "Body " * 60)
    (chunk,) = split_markdown(md)
    assert chunk["sections"] == ["Control speed", "Control speed › After"]
    assert chunk["anchor"] == "Control speed"


def test_figure_caption_headings_are_text():
    cap = ("On the left is the original DV footage (courtesy Creative License); on the right "
           "is the same shot with 720 degrees of motion blur added")
    md = f"## Motion blur\n\nText.\n\n### {cap}\n\nMore text."
    (chunk,) = split_markdown(md)
    assert chunk["sections"] == ["Motion blur"]
    assert cap in chunk["text"]


def test_tiny_tail_joins_previous_chunk():
    md = "## A\n\n" + "x" * 1400 + "\n\n## B\n\nshort tail."
    chunks = split_markdown(md)
    assert len(chunks) == 1 and chunks[0]["text"].endswith("short tail.")


def test_chunks_respect_max_chars_and_carry_context():
    md = "## Graph editor\n\n" + "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(12))
    chunks = split_markdown(md, max_chars=1000)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 1000 + 40 for c in chunks)  # + context line
    # a chunk that starts mid-section names its section on the first line
    assert chunks[1]["text"].startswith("(Graph editor)")


def test_code_fence_is_never_split():
    md = "## Loop\n\n```\nline1\n\nline2\n```\n"
    (chunk,) = split_markdown(md)
    assert "line1\n\nline2" in chunk["text"]


def test_oversized_paragraph_is_split_at_sentences():
    md = "## Long\n\n" + " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = split_markdown(md, max_chars=500)
    assert len(chunks) > 3
    assert all(len(c["text"]) <= 500 + 40 for c in chunks)


# ---- envelope ----

def test_page_items_share_the_envelope():
    items = build_items_for_text(PAGE, title="Expression examples",
                                 source_url="https://helpx.adobe.com/x.html",
                                 author="helpx.adobe.com", published_at="2024-08-29",
                                 type="doc", origin="web")
    meta = items[0]["meta"]
    assert CORE <= set(meta)
    assert meta["doc_id"] == page_doc_id("https://helpx.adobe.com/x.html")
    # the first chunk starts in the page intro (no section), so it links to the page top
    assert meta["source"] == "https://helpx.adobe.com/x.html"
    assert items[0]["content"].startswith('[helpx.adobe.com — "Expression examples" — 2024-08-29 — § ')


def test_video_items_share_the_envelope():
    vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nhello graph editor\n\n"
           "00:00:04.000 --> 00:00:08.000\nspeed graph now\n")
    info = {"id": "abcdefghijk", "title": "Easing", "channel": "Ben", "upload_date": "20250101"}
    (item,) = build_items_for_video(info, vtt, "@ben")
    meta = item["meta"]
    assert CORE <= set(meta)
    assert meta["type"] == "transcript" and meta["origin"] == "youtube"
    assert meta["doc_id"] == "abcdefghijk" and meta["author"] == "Ben"
    assert meta["source"] == "https://youtu.be/abcdefghijk?t=1"
    assert meta["locator"] == "00:01–00:08"
    assert item["content"].startswith('[Ben — "Easing" — 2025-01-01 — 00:01]\n')


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        chunk_meta(type="video", origin="x", source="s", doc_id="d", title="t",
                   author="a", published_at="", locator="", chunk_index=0)


def test_header_skips_empty_parts():
    assert header("", "T", "", "") == '["T"]'


@pytest.mark.parametrize("url,yt", [
    ("https://www.youtube.com/watch?v=zjkBMFhNj_g", True),
    ("https://youtu.be/zjkBMFhNj_g?t=3", True),
    ("https://www.youtube.com/shorts/abcdefghijk", True),
    ("https://helpx.adobe.com/after-effects/using/expression-examples.html", False),
    ("https://www.schoolofmotion.com/blog/youtube-tips", False),
])
def test_url_routing(url, yt):
    assert bool(YOUTUBE_RE.search(url)) is yt


# ---- caption track choice ----

from yt_rag.ingest import pick_caption_track  # noqa: E402


def test_manual_track_wins():
    info = {"subtitles": {"en-US": [], "live_chat": []}, "automatic_captions": {"en": [], "en-orig": []}}
    assert pick_caption_track(info, ["en.*"]) == ("en-US", "manual")


def test_auto_prefers_original_over_translation_endpoint():
    info = {"subtitles": {}, "automatic_captions": {"en": [], "en-orig": [], "de": []}}
    assert pick_caption_track(info, ["en.*"]) == ("en-orig", "auto")


def test_no_matching_track():
    info = {"subtitles": {}, "automatic_captions": {"de-orig": [], "de": []}}
    assert pick_caption_track(info, ["en.*"]) is None


# ---- boilerplate + sponsors ----

from yt_rag.ingest import chunk_cues, drop_segments  # noqa: E402
from yt_rag.text_ingest import strip_repeated_paragraphs  # noqa: E402


def test_repeated_site_paragraphs_are_dropped():
    footer = "**Create stunning 3D animations in After Effects**"
    a, b = strip_repeated_paragraphs([f"Speed text one.\n\n{footer}", f"Mattes text two.\n\n{footer}"])
    assert footer not in a and footer not in b
    assert a == "Speed text one." and b == "Mattes text two."


def test_sponsor_cues_are_dropped():
    cues = [(0, 4, "intro"), (60, 64, "this video is sponsored by"), (170, 174, "graph editor")]
    assert [c[2] for c in drop_segments(cues, [(53, 161)])] == ["intro", "graph editor"]


def test_transcript_tail_merges():
    cues = [(i * 10, i * 10 + 10, "word " * 40) for i in range(10)] + [(100, 101, "time.")]
    chunks = chunk_cues(cues)
    assert chunks[-1]["text"].endswith("time.") and len(chunks[-1]["text"]) > 300


# ---- tags + search shaping ----

from yt_rag.mcp_server import cap_per_source  # noqa: E402
from yt_rag.taxonomy import TagError, validate_tags  # noqa: E402


def test_tags_normalise():
    t = validate_tags({"topics": "easing", "scenarios": ["ui-promo", "ui-promo"],
                       "level": "advanced", "impact": "high", "tools": ["Deep-Glow-2 "]})
    assert t == {"topics": ["easing"], "scenarios": ["ui-promo"], "level": "advanced",
                 "impact": "high", "tools": ["deep-glow-2"]}


@pytest.mark.parametrize("bad", [
    {"topics": ["glow"]}, {"scenarios": ["promo"]}, {"level": "pro"},
    {"impact": "huge"}, {"tools": ["Deep Glow"]}, {"mood": "calm"},
])
def test_tags_reject_off_vocabulary(bad):
    with pytest.raises(TagError):
        validate_tags(bad)


def test_stock_only_is_kept_as_empty_list():
    assert validate_tags({"tools": []}) == {"tools": []}


def test_cap_per_source_keeps_order():
    rows = [{"id": i, "metadata": {"doc_id": d}} for i, d in enumerate("aaabca")]
    assert [r["id"] for r in cap_per_source(rows, 2)] == [0, 1, 3, 4]
    assert len(cap_per_source(rows, 0)) == 6
