# promo-craft library: curator brief (shared by all curators)

`promo-craft` is a yt-rag collection that our agents load while they plan app launch and promo videos for our own apps. Its purpose: how top teams plan and pace launch and promo videos (Product Hunt, YouTube and X launch films, landing-page hero videos, App Store previews): story structures, hooks, copy and voice-over, length by platform, logo and CTA placement, captions for silent viewing, sound, and what separates premium from average.

It is not After Effects craft. AE technique lives in the `after-effects` collection. Here we want the decisions a creative director, launch marketer or editor makes before and around the animation.

Every source becomes text chunks (video captions or page text), so a source is only worth adding if its WORDS teach something an agent can act on.

## Quality bar
- **Videos:** a top practitioner explaining in speech (a launch-film director, studio, editor, founder who shipped a famous launch, or a marketer with data). Normally 20k+ views, or less when the speaker is clearly top tier (say why). 2022 or later preferred. 5–60 minutes of real teaching. An English caption track (manual or auto) is required. Reject silent showreels and "top 10 tips" fluff.
- **Pages:** official platform specs (Apple, Google, YouTube, X, Meta, LinkedIn, TikTok, Product Hunt), studio case studies and making-of articles, data studies (Wistia, Vidyard, etc.), and the best written guides. At least ~2000 characters of real text. Pages carry exact numbers, so include them generously.
- **Breadth over depth:** cover different questions (structure, hooks, length, CTA, captions, sound, platform rules, premium vs average). Don't stack five sources that say the same thing.
- **Recency:** 2023–2026 first. Older only when it is still the canonical reference (say why).

## Already in the database (do not repeat)
- Everything in `/home/ran/yt-rag/eval/after-effects/bulk/ALREADY_INGESTED.txt`.
- Promo-relevant sources already in the `after-effects` collection: themanifest.fyi "Launching Comet with Perplexity"; the-brandidentity.com "Art&Graft evolves Gemini"; iamp.at "Meet the new Microsoft Surface Pro"; quincemedia.com "Why Apple Still Launches in 3D, Not AI"; support.google.com/youtube/answer/9314415 (key moments for retention); BBC subtitle guidelines; YouTube videos eNgD1kg3U14 (MKBHD Method), sLgHqZSe2o0 (Learn By Leo), CdyTIng0M8Q (Inside The Edit pacing), ZbVmyH9NVnk (cutting music), JOm_EqBoW0M (3 sound effects), tzfzkTAQsnE (sound design secrets), Jccbzpicst8 (SaaS motion graphics), of-Nd8bnnD4 (Daniel Schiffer movement), xUK64UkTmW0 and FVR8zz8ci2k (StudioBinder), R3ablyDL__8 (Mhd Labs), Bw-ucrydyuw (Jake In Motion Figma mockup).
- The lead agent runs a final SQL dedupe, so you don't need DB access.

## Checks (do them for every pick; metadata only, never download media)
Videos, one at a time with ~4 s between calls (YouTube rate-limits; on HTTP 429 wait 60 s and retry once):

    /home/ran/yt-rag/.venv/bin/python -c "import sys; from yt_dlp import YoutubeDL; from yt_dlp.networking.impersonate import ImpersonateTarget; y=YoutubeDL({'quiet':True,'no_warnings':True,'skip_download':True,'impersonate':ImpersonateTarget.from_str('chrome')}); i=y.extract_info(sys.argv[1],download=False); print(i.get('title'),'|',i.get('channel'),'|',i.get('duration'),'s |',i.get('upload_date'),'| views',i.get('view_count'),'| likes',i.get('like_count'),'| manual:',[k for k in (i.get('subtitles') or {}) if k.startswith('en')],'| auto:',[k for k in (i.get('automatic_captions') or {}) if k.startswith('en')])" URL

Pages, with yt-rag's own fetcher (what ingest will use):

    /home/ran/yt-rag/.venv/bin/python -c "import sys; from yt_rag.text_ingest import fetch_page; p=fetch_page(sys.argv[1]); print(p['title'],'|',p['site'],'|',len(p['markdown']),'chars'); print([l for l in p['markdown'].splitlines() if l.startswith('#')][:10])" URL

If the fetcher fails or returns junk (JavaScript-only page, paywall, login), fetch the text with the Exa web_fetch tool and save the faithful Markdown, with its headings, to `/home/ran/yt-rag/eval/promo-craft/texts/<short-slug>.md`; then set `text_file` to that path. Don't summarise or rewrite the source.

Do NOT ingest anything. Windows has ~2 GB free: no media downloads, no large files.

## Tags (required; the ingest fails on any word outside this vocabulary)
- `topics` (1–3) from: easing, text, shapes, expressions, camera-3d, motion-blur, light-glow, particles, mattes-masks, transitions, compositing, color-finish, workflow, render-export, design. Most promo sources are `design` (story, pacing, layout, copy). Use `text` for captions and type, `transitions` for cut grammar, `color-finish` for look, `workflow` for production process, `render-export` for platform delivery specs.
- `scenarios` (0–3) from: ui-promo, kinetic-type, logo-ident, explainer, data-viz, character, titles, hero-fx, edit-pacing. Most fit `ui-promo` and/or `edit-pacing`; captions are `titles`; logo/end-card placement is `logo-ident`.
- `level`: basic | intermediate | advanced.
- `impact`: high = applying it visibly lifts the film or its results; medium = solid; low = reference.
- `tools`: [] unless a product is required.

## Output
Write your list to the JSON file named in your task, as a list of:

    {"url", "kind": "video"|"page", "title", "author",
     "type": "tutorial"|"doc"|"article"   (pages only; doc = official spec),
     "views", "likes", "published", "duration_sec", "captions": "manual"|"auto"   (videos),
     "approx_chars", "text_file"   (pages; text_file only if you saved one),
     "tags": {"topics": [...], "scenarios": [...], "level": "...", "impact": "...", "tools": []},
     "lesson": "the concrete, reusable rule or standard this source teaches, with its numbers",
     "example": "deep link to where it is stated or shown (YouTube &t=…s, or page #section)",
     "why": "one line: why an agent planning a premium launch film needs this"}
