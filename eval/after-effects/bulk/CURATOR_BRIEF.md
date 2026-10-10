# After Effects library — curator brief (shared by all curators)

We are building an After Effects library: a retrieval knowledge base our AI agents query while they build and finish videos for a tech YouTube channel (Ran Studio: AI apps, agents, devices, product/UI promos, explainer lectures with a character host, energetic channel idents). Every source becomes text chunks (video captions or page text), so a source is only worth adding if its WORDS teach something an agent can act on.

## Quality bar — only the best, not whatever is familiar
Pick sources that are high quality, widely valued, concretely functional, and that give the biggest visible jump in the result. For every candidate ask: "if our agent applies this, does the video look clearly more premium or work clearly better?"

- Videos: normally >= 50k views, or >= 15k for niche advanced or plugin topics where the instructor is top tier (say why). Likes/views around 2% or more is a good sign. Prefer 2021 or later, since AE changed a lot (2023 track mattes, Advanced 3D renderer, Properties panel). Older only if it is the canonical explanation and still accurate. 5–60 minutes of real teaching. Must explain in speech; a silent screen recording is useless.
- Reject: template or plugin sales pitches, "top 10 tips" fluff, beginner basics that repeat what we already have, livestreams over 90 min, anything whose value is only on screen.
- Pages: authoritative references and manuals (Adobe Help, plugin manuals, expression references), and the best written tutorials and articles. At least ~2000 characters of real text. Pages carry exact values, settings and code, which videos rarely do, so include them generously.
- Already in the library (do not repeat): see /home/ran/yt-rag/eval/after-effects/bulk/ALREADY_INGESTED.txt
- Aim for 25–35 sources in your area. Fewer excellent beats more average.

## Checks (do them for every pick)
Videos, one at a time with ~4 s between calls (YouTube rate-limits; on HTTP 429 wait 60 s, retry once):

    /home/ran/yt-rag/.venv/bin/python -c "import sys; from yt_dlp import YoutubeDL; from yt_dlp.networking.impersonate import ImpersonateTarget; y=YoutubeDL({'quiet':True,'no_warnings':True,'skip_download':True,'impersonate':ImpersonateTarget.from_str('chrome')}); i=y.extract_info(sys.argv[1],download=False); print(i.get('title'),'|',i.get('channel'),'|',i.get('duration'),'s |',i.get('upload_date'),'| views',i.get('view_count'),'| likes',i.get('like_count'),'| manual:',[k for k in (i.get('subtitles') or {}) if k.startswith('en')],'| auto:',[k for k in (i.get('automatic_captions') or {}) if k.startswith('en')])" URL

Need an English track (manual or auto).

Pages — run yt-rag's own fetcher (what ingest will use):

    /home/ran/yt-rag/.venv/bin/python -c "import sys; from yt_rag.text_ingest import fetch_page; p=fetch_page(sys.argv[1]); print(p['title'],'|',p['site'],'|',len(p['markdown']),'chars'); print([l for l in p['markdown'].splitlines() if l.startswith('#')][:10])" URL

If it fails or returns junk (JavaScript-only page such as Notion docs, a paywall, a login page), you may fetch the text yourself with the Exa web_fetch tool (load it via ToolSearch, query "exa web fetch") and save the faithful Markdown, with its headings, to /home/ran/yt-rag/eval/after-effects/bulk/texts/<short-slug>.md. Then set "text_file" to that path. Do not summarise or rewrite the source; keep it faithful.

Use the Exa web search tools to find candidates. Do NOT ingest anything into the database.

## Tags (required for every source)
Use ONLY the vocabulary in /home/ran/yt-rag/eval/after-effects/bulk/VOCAB.txt:
- topics (1–3) and scenarios (0–3, only real fits);
- level;
- impact: high = applying it visibly elevates the result; medium = solid improvement; low = reference or mechanics;
- tools: plugins the technique REQUIRES; [] = stock AE.

Our installed plugins are deep-glow-2, rsmb and the trapcode suite, but stock-only techniques are valued.

## Output
Write your list to the JSON file named in your task, as a list of:

    {"url", "kind": "video"|"page", "title", "author",
     "type": "tutorial"|"doc"|"article"   (pages only),
     "views", "likes", "published", "duration_sec", "captions": "manual"|"auto"   (videos),
     "approx_chars", "text_file"   (pages; text_file only if you saved one),
     "tags": {"topics": [...], "scenarios": [...], "level": "...", "impact": "...", "tools": [...]},
     "why": "one line: the concrete skill and the visible or functional payoff"}

Final message, under 250 words:
- counts (videos, pages, impact mix);
- the 5 strongest picks;
- notable rejections;
- what you could not find.
