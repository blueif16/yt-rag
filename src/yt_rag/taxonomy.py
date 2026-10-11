"""
Tags: a small fixed vocabulary that says what a source is good FOR.

Tags are set once per source (video or page) by whoever curates it and copied
onto every chunk, so search can filter and an agent can ask "what suits this
scenario?" Five fields, all optional:

    topics     what technique it teaches          (list, from TOPICS)
    scenarios  what kind of shot/video it serves  (list, from SCENARIOS)
    level      basic | intermediate | advanced
    impact     how big a visible jump it gives: high (premium look, people
               notice), medium (solid improvement), low (reference/mechanics)
    tools      third-party plugins it REQUIRES, kebab-case ("deep-glow-2");
               [] = stock After Effects only

Videos also record `views` and `likes` automatically at ingest.
"""
from __future__ import annotations
import re

TOPICS = {
    "easing": "graph editor, keyframe interpolation, timing and spacing",
    "text": "text animators, kinetic typography, type in motion",
    "shapes": "shape layers, trim paths, repeaters, UI and vector animation",
    "expressions": "expressions, rigs, sliders, procedural animation",
    "camera-3d": "3D layers, cameras, depth of field, 3D renderers",
    "motion-blur": "native motion blur, shutter angle, RSMB, pixel motion blur",
    "light-glow": "glows, light rays, flares, emissive and neon looks",
    "particles": "particle systems (Particular, Form, CC Particle World)",
    "mattes-masks": "masks, track mattes, reveals",
    "transitions": "shot-to-shot transitions, match cuts, whips and punches",
    "compositing": "keying, tracking, light wrap, shadows, integrating layers",
    "color-finish": "grading, grain, texture, bit depth, the final look",
    "workflow": "precomps, project organisation, templates, performance",
    "render-export": "render queue, aerender, codecs, color management, interpreting footage",
    "design": "motion design principles, composition, layout, pacing, storytelling",
}

SCENARIOS = {
    "ui-promo": "app, product, UI and website promo shots",
    "kinetic-type": "type-driven sequences, quotes, title text",
    "logo-ident": "logo stings, channel idents, intros and outros",
    "explainer": "diagrams, lecture and explainer graphics, callouts",
    "data-viz": "charts, counters, stat hits",
    "character": "character or host animation and integration",
    "titles": "lower thirds, titles, captions",
    "hero-fx": "hero moments: glows, particles, 3D, big reveals",
    "edit-pacing": "whole-video craft: pacing, rhythm, scene flow, retention",
}

LEVELS = ("basic", "intermediate", "advanced")
IMPACTS = ("high", "medium", "low")
TOOL_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class TagError(ValueError):
    pass


def validate_tags(tags: dict | None) -> dict:
    """Check and normalise a tag dict; returns only the tag fields that are set."""
    if not tags:
        return {}
    unknown = set(tags) - {"topics", "scenarios", "level", "impact", "tools"}
    if unknown:
        raise TagError(f"unknown tag fields {sorted(unknown)}")
    out: dict = {}
    for field, vocab in (("topics", TOPICS), ("scenarios", SCENARIOS)):
        vals = tags.get(field) or []
        if isinstance(vals, str):
            vals = [vals]
        bad = [v for v in vals if v not in vocab]
        if bad:
            raise TagError(f"{field} {bad} not in vocabulary: {sorted(vocab)}")
        if vals:
            out[field] = sorted(set(vals))
    if tags.get("level"):
        if tags["level"] not in LEVELS:
            raise TagError(f"level must be one of {LEVELS}")
        out["level"] = tags["level"]
    if tags.get("impact"):
        if tags["impact"] not in IMPACTS:
            raise TagError(f"impact must be one of {IMPACTS}")
        out["impact"] = tags["impact"]
    if "tools" in tags:
        tools = sorted({t.strip().lower() for t in (tags["tools"] or [])})
        bad = [t for t in tools if not TOOL_RE.match(t)]
        if bad:
            raise TagError(f"tools must be kebab-case plugin names, got {bad}")
        out["tools"] = tools
    return out


def vocabulary_text() -> str:
    """The vocabulary as text, for tool docstrings and curator prompts."""
    lines = ["topics:"] + [f"  {k}: {v}" for k, v in TOPICS.items()]
    lines += ["scenarios:"] + [f"  {k}: {v}" for k, v in SCENARIOS.items()]
    lines += [f"level: {' | '.join(LEVELS)}", f"impact: {' | '.join(IMPACTS)}",
              "tools: required third-party plugins, kebab-case; [] = stock AE only"]
    return "\n".join(lines)
