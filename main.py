"""
Gemma-Merge backend
-------------------
Open-source contribution copilot for beginners.

  POST /api/demystify   GitHub issue  -> Summary, Action Plan, Target Areas, Difficulty
  POST /api/synthesize  git diff      -> Commit title, PR description, checklist, git steps

Design goals
  * Never exceeds a hard time budget (default 28 s) -> TOTAL_BUDGET_SECONDS
  * Never returns a half-finished answer: every section is validated, and any
    missing/truncated section is completed by a local rule-based analyzer.
  * Never fails with a 503: if both AI engines are down the local analyzer answers.
  * Secrets are masked locally BEFORE anything leaves the machine.
"""

import hashlib
import os
import re
import time
from collections import OrderedDict
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "gemma-4-26b-a4b-it")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")

TOTAL_BUDGET_SECONDS = float(os.getenv("TOTAL_BUDGET_SECONDS", "28"))  # hard ceiling per request
CLOUD_TIMEOUT_SECONDS = float(os.getenv("CLOUD_TIMEOUT_SECONDS", "15"))
CLOUD_MAX_TOKENS = int(os.getenv("CLOUD_MAX_TOKENS", "1400"))  # was 450 -> truncated answers
LOCAL_MAX_TOKENS = int(os.getenv("LOCAL_MAX_TOKENS", "700"))

MAX_ISSUE_CHARS = 6000
MAX_DIFF_CHARS = 14000

MASK = "[MASKED_SECRET_CREDENTIAL]"

app = FastAPI(title="Gemma-Merge Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # wildcard origin + credentials is invalid per the CORS spec
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------
# Zero-trust credential sanitization
# ----------------------------------------------------------------------------
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\beyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,}"),  # JWT
]
ASSIGNMENT_SECRET = re.compile(
    r"""(?i)([\w-]*(?:api[_-]?key|secret|token|passwd|password)[\w-]*["']?\s*[=:]\s*)(["'])([^"'\n]{4,})\2"""
)


def sanitize_text(text: str):
    """Mask credentials. Returns (masked_text, number_of_secrets_masked)."""
    count = 0
    for pattern in SECRET_PATTERNS:
        text, n = pattern.subn(MASK, text)
        count += n

    def _assign(m):
        nonlocal count
        if MASK in m.group(3):
            return m.group(0)
        count += 1
        return f"{m.group(1)}{m.group(2)}{MASK}{m.group(2)}"

    text = ASSIGNMENT_SECRET.sub(_assign, text)
    return text, count


def sanitize_diff(diff_text: str):
    """Backwards-compatible wrapper: (masked, had_secrets)."""
    masked, n = sanitize_text(diff_text)
    return masked, n > 0


# ----------------------------------------------------------------------------
# AI engines (cloud -> local Ollama), all under one shared deadline
# ----------------------------------------------------------------------------
class EngineError(Exception):
    pass


_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX = 64


def _cache_get(key):
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return dict(_CACHE[key])
    return None


def _cache_put(key, value):
    _CACHE[key] = dict(value)
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


def call_cloud(prompt: str, read_timeout: float):
    if not GEMINI_API_KEY:
        raise EngineError("GEMINI_API_KEY not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{CLOUD_MODEL}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": CLOUD_MAX_TOKENS,
            "temperature": 0.2,
            "topP": 0.9,
        },
    }
    # API key travels in a header (not the URL) so it never lands in logs.
    res = requests.post(
        url,
        json=payload,
        headers={"x-goog-api-key": GEMINI_API_KEY},
        timeout=(4, read_timeout),
    )
    if res.status_code != 200:
        raise EngineError(f"HTTP {res.status_code}: {res.text[:160]}")
    cand = res.json()["candidates"][0]
    parts = cand.get("content", {}).get("parts", [])
    # Skip "thinking" parts so planning notes never reach the user.
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        raise EngineError("empty response")
    truncated = cand.get("finishReason") == "MAX_TOKENS"
    return text, truncated


def call_ollama(prompt: str, read_timeout: float):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {"num_predict": LOCAL_MAX_TOKENS, "temperature": 0.2, "num_ctx": 4096},
    }
    res = requests.post(OLLAMA_URL, json=payload, timeout=(2, read_timeout))
    if res.status_code != 200:
        raise EngineError(f"HTTP {res.status_code}")
    data = res.json()
    text = data.get("response", "")
    if not text.strip():
        raise EngineError("empty response")
    return text, data.get("done_reason") == "length"


def generate_ai_response(prompt: str):
    """
    Try cloud, then local, inside TOTAL_BUDGET_SECONDS.
    Returns dict(output, engine, offline_mode, truncated) or None if both fail.
    """
    key = hashlib.sha256(prompt.encode()).hexdigest()
    cached = _cache_get(key)
    if cached:
        cached["cached"] = True
        return cached

    start = time.monotonic()

    def remaining():
        return TOTAL_BUDGET_SECONDS - (time.monotonic() - start)

    # Primary: Google AI Studio (Gemma 4)
    if GEMINI_API_KEY and remaining() > 3:
        print(f"[AI] Cloud attempt ({CLOUD_MODEL})")
        try:
            text, truncated = call_cloud(prompt, min(CLOUD_TIMEOUT_SECONDS, remaining() - 1))
            result = {
                "output": text,
                "engine": f"Cloud Gemma 4 ({CLOUD_MODEL})",
                "offline_mode": False,
                "truncated": truncated,
            }
            _cache_put(key, result)
            return result
        except Exception as e:  # noqa: BLE001
            print(f"[AI] Cloud failed: {type(e).__name__}: {e}")

    # Fallback: local Ollama
    if remaining() > 3:
        print(f"[AI] Local attempt ({OLLAMA_MODEL})")
        try:
            text, truncated = call_ollama(prompt, remaining() - 1)
            result = {
                "output": text,
                "engine": f"Local {OLLAMA_MODEL} via Ollama",
                "offline_mode": True,
                "truncated": truncated,
            }
            _cache_put(key, result)
            return result
        except Exception as e:  # noqa: BLE001
            print(f"[AI] Local failed: {type(e).__name__}: {e}")

    return None


# ----------------------------------------------------------------------------
# Section parsing & completion
# ----------------------------------------------------------------------------
def parse_sections(text: str, titles):
    """
    Split model output into {title: body}. Tolerates '### Title', '## Title',
    '**Title**', 'Title:' and any preamble/thinking text before the first heading.
    """
    alt = "|".join(re.escape(t) for t in sorted(titles, key=len, reverse=True))
    heading = re.compile(
        rf"^\s*(?:#{{1,4}}\s*)?(?:\*\*)?\s*(?:[^\w\s]{{1,3}}\s*)?({alt})\s*:?\s*(?:\*\*)?\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(heading.finditer(text))
    out = {}
    canon = {t.lower(): t for t in titles}
    for i, m in enumerate(matches):
        title = canon[m.group(1).lower()]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if title not in out and body:
            out[title] = body
    return out


def normalize_body(body: str):
    body = re.sub(r"\n{3,}", "\n\n", body.strip())
    # Steps glued onto one line ("... 2. Investigate: ...") -> one per line.
    body = re.sub(r"(?<=[.!?`)])[ \t]+(?=\d{1,2}\.\s)", "\n", body)
    # Drop stray code-fence wrappers around the whole section.
    return body.strip()


def complete_sections(parsed, titles, fallback, truncated):
    """
    Merge model sections with fallback ones so the answer is always complete.
    Returns (sections[list of {title, body, source}], partial_flag).
    """
    sections, partial = [], False
    last_parsed = None
    for t in titles:
        if t in parsed:
            last_parsed = t
    for t in titles:
        body = normalize_body(parsed.get(t, ""))
        too_short = len(body) < 12
        cut_off = truncated and t == last_parsed  # last section is probably unfinished
        if too_short or cut_off:
            sections.append({"title": t, "body": fallback[t], "source": "auto"})
            partial = True
        else:
            sections.append({"title": t, "body": body, "source": "ai"})
    return sections, partial


def sections_to_markdown(sections):
    return "\n\n".join(f"### {s['title']}\n{s['body']}" for s in sections)


# ----------------------------------------------------------------------------
# Issue analysis
# ----------------------------------------------------------------------------
ISSUE_TITLES = ["Summary", "Action Plan", "Target Codebase Areas", "Difficulty"]

PATH_RE = re.compile(
    r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.(?:py|js|jsx|ts|tsx|mjs|cjs|java|kt|go|rs|c|h|cpp|hpp|cs|rb|php|swift|vue|svelte|md|json|ya?ml|toml|html|css|sh))(?::\d+)?"
)
ERROR_LINE_RE = re.compile(r"^.*\b(?:\w*Error|Exception|Traceback|panic|FATAL|failed|cannot|can't|undefined)\b.*$", re.I | re.M)


def heuristic_issue(issue: str):
    lines = [l.strip() for l in issue.splitlines() if l.strip()]
    title = re.sub(r"^\[[^\]]*\]:?\s*", "", lines[0]) if lines else "the reported problem"
    errors = [e.strip() for e in ERROR_LINE_RE.findall(issue)][:2]
    paths = []
    for p in PATH_RE.findall(issue):
        if p not in paths and not p.startswith("node:"):
            paths.append(p)
    paths = paths[:3]

    summary = f"The issue reports: {title.rstrip('.')}."
    if errors:
        summary += f" The key symptom is `{errors[0][:140]}`."
    summary += " Read the steps in the issue to see how to trigger it, then find the code that produces that behavior."

    repro = "Follow the \"Steps to Reproduce\" in the issue and write a tiny script or test that triggers the problem."
    m = re.search(r"steps to reproduce:?\s*\n(.+?)(?:\n#{1,4}\s|\Z)", issue, re.I | re.S)
    if m:
        first = " ".join(m.group(1).strip().splitlines()[:2])
        repro = f"Reproduce it first. The issue says: {first[:200]}"
    where = ", ".join(f"`{p}`" for p in paths) if paths else "the module named in the error message or stack trace"
    plan = (
        f"1. Reproduce: {repro}\n"
        f"2. Investigate: Open {where} and search for the code path that runs just before the error appears.\n"
        "3. Resolve: Change the smallest piece of logic that causes the wrong behavior and keep the fix focused.\n"
        "4. Verify: Add a test that fails before your change and passes after, then run the project's full test suite."
    )
    targets = "\n".join(f"- `{p}` - mentioned in the issue; start reading here." for p in paths[:2]) or (
        "- Search the repository for the error message text or function names from the stack trace."
    )
    level = "Beginner" if len(issue) < 700 and not errors else "Intermediate"
    return {
        "Summary": summary,
        "Action Plan": plan,
        "Target Codebase Areas": targets,
        "Difficulty": f"{level} - estimated automatically from the issue length and error details.",
    }


def build_issue_prompt(issue: str):
    return f"""You are a friendly open-source mentor helping a first-time contributor understand a GitHub issue.
Use plain, beginner-friendly English. Do NOT show thinking, drafts or notes.
Output ONLY the four sections below with these exact headings, in this order, and finish every section.

### Summary
2-4 sentences: what is broken, why it most likely happens, and who or what it affects.

### Action Plan
1. Reproduce: how to trigger the bug (include a short command or snippet in backticks).
2. Investigate: which module or logic to read, and what to look for.
3. Resolve: the specific code change to make.
4. Verify: which test to add or run to prove the fix works.
(One or two sentences per step, each step on its own line.)

### Target Codebase Areas
- `path/or/module` - why to look here (maximum 2 bullets; only use paths mentioned in the issue or standard for the project)

### Difficulty
Beginner, Intermediate or Advanced, then " - " and one short reason.

ISSUE:
\"\"\"
{issue}
\"\"\"
"""


# ----------------------------------------------------------------------------
# Diff analysis
# ----------------------------------------------------------------------------
PR_TITLES = ["Commit Title", "Summary of Changes", "Verification Checklist"]
CONVENTIONAL_RE = re.compile(r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([\w./-]+\))?!?: \S.{2,}$")

TEST_CMD = {
    ".py": "pytest",
    ".js": "npm test",
    ".jsx": "npm test",
    ".ts": "npm test",
    ".tsx": "npm test",
    ".go": "go test ./...",
    ".rs": "cargo test",
    ".java": "mvn test",
    ".rb": "bundle exec rspec",
    ".php": "composer test",
}


def parse_diff(diff: str):
    files, cur = [], None

    def start(path):
        nonlocal cur
        cur = {"path": path, "added": 0, "removed": 0, "status": "modified", "contexts": []}
        files.append(cur)

    for line in diff.splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            start(m.group(2) if m else line[11:].strip())
        elif line.startswith("new file mode"):
            cur and cur.update(status="added")
        elif line.startswith("deleted file mode"):
            cur and cur.update(status="deleted")
        elif line.startswith("rename to "):
            cur and cur.update(status="renamed", path=line[10:].strip())
        elif line.startswith("+++ "):
            p = line[4:].strip()
            if p != "/dev/null":
                p = re.sub(r"^b/", "", p)
                if cur is None:
                    start(p)
        elif line.startswith("--- "):
            continue
        elif line.startswith("@@"):
            m = re.match(r"@@ .*? @@\s*(.*)$", line)
            ctx = m.group(1).strip() if m else ""
            if cur and ctx and ctx not in cur["contexts"]:
                cur["contexts"].append(ctx[:80])
        elif line.startswith("+") and cur:
            cur["added"] += 1
        elif line.startswith("-") and cur:
            cur["removed"] += 1
    return files


def compact_diff(diff: str):
    """Drop noise lines and cap size so the model answers faster."""
    keep = [l for l in diff.splitlines() if not re.match(r"^(index |similarity index|old mode|new mode)", l)]
    text = "\n".join(keep)
    if len(text) > MAX_DIFF_CHARS:
        text = text[:MAX_DIFF_CHARS] + "\n[... diff truncated for speed; summarize what is shown ...]"
    return text


def guess_commit(files):
    if not files:
        return "chore", "project", "update project files"
    paths = [f["path"] for f in files]
    exts = {os.path.splitext(p)[1].lower() for p in paths}
    is_test = lambda p: bool(re.search(r"(^|/)(tests?|__tests__|spec)(/|$)|(_test|\.test|\.spec|test_)", p))  # noqa: E731
    if all(is_test(p) for p in paths):
        ctype = "test"
    elif exts <= {".md", ".rst", ".txt"}:
        ctype = "docs"
    elif any(f["status"] == "added" and not is_test(f["path"]) for f in files):
        ctype = "feat"
    else:
        ctype = "fix"
    parts = [p.split("/") for p in paths]
    if len(files) == 1:
        scope = os.path.splitext(os.path.basename(paths[0]))[0]
    else:
        common = os.path.commonprefix(parts)
        scope = common[-1] if common else parts[0][0]
        scope = scope if scope and "." not in scope else parts[0][0]
    scope = re.sub(r"[^\w./-]", "", scope)[:24] or "core"
    ctx = next((c for f in files for c in f["contexts"]), "")
    ident = re.search(r"([A-Za-z_][\w]*)\s*\(", ctx)
    what = f"update {ident.group(1)} handling" if ident else f"update {os.path.basename(paths[0])}"
    return ctype, scope, what


def slugify(text: str):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "change"


def heuristic_pr(files):
    ctype, scope, what = guess_commit(files)
    title = f"{ctype}({scope}): {what}"
    bullets = []
    for f in files[:6]:
        ident = re.search(r"([A-Za-z_]\w*)\s*\(", f["contexts"][0]) if f["contexts"] else None
        extra = f" (in `{ident.group(1)}`)" if ident else ""
        bullets.append(f"- `{f['path']}`: {f['status']}, +{f['added']} / -{f['removed']} lines{extra}.")
    if not bullets:
        bullets = ["- No file changes could be detected in the pasted diff."]
    exts = [os.path.splitext(f["path"])[1].lower() for f in files]
    cmd = next((TEST_CMD[e] for e in exts if e in TEST_CMD), "the project's test command")
    checklist = (
        f"1. Unit Test: Add or update a test covering the changed code, then run `{cmd}`; it should pass.\n"
        "2. Regression Check: Run the full test suite and confirm nothing that passed before now fails.\n"
        "3. Manual Check: Run the affected feature once by hand and confirm the behavior matches the description."
    )
    return {"Commit Title": title, "Summary of Changes": "\n".join(bullets), "Verification Checklist": checklist}


def clean_title(raw: str, fallback: str):
    first = next((l for l in raw.splitlines() if l.strip()), "")
    t = re.sub(r"^[\s>*_`\"'#-]+|[\s*_`\"']+$", "", first)
    t = re.sub(r"^(commit title|title)\s*:\s*", "", t, flags=re.I)
    if CONVENTIONAL_RE.match(t) and len(t) <= 72:
        return t
    return fallback


def build_git_steps(title: str, files):
    ctype = title.split("(")[0].split(":")[0] or "fix"
    branch = f"{ctype}/{slugify(title.split(':', 1)[-1])}"
    paths = " ".join(f['path'] for f in files[:8]) or "."
    return [
        {"title": "Create a branch", "detail": "Never work on main. A branch keeps your change separate.", "command": f"git checkout -b {branch}"},
        {"title": "Review what you changed", "detail": "Make sure only the files you meant to change are listed.", "command": "git status\ngit diff"},
        {"title": "Stage your files", "detail": "Pick exactly the files that belong to this fix.", "command": f"git add {paths}"},
        {"title": "Commit with the generated title", "detail": "The title follows the Conventional Commits style most projects expect.", "command": f'git commit -m "{title}"'},
        {"title": "Push your branch", "detail": "Upload the branch to your fork.", "command": f"git push -u origin {branch}"},
        {"title": "Open the Pull Request", "detail": "On GitHub click 'Compare & pull request', paste the title and the copied description, and link the issue with 'Closes #<issue-number>'.", "command": ""},
    ]


def build_pr_markdown(sections, files, stats):
    by = {s["title"]: s["body"] for s in sections}
    checklist = []
    for line in by["Verification Checklist"].splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^(?:\d+[.)]|[-*])\s*", "", line)
        checklist.append(f"- [ ] {line}")
    changed = "\n".join(f"- `{f['path']}` (+{f['added']} / -{f['removed']})" for f in files) or "- (none detected)"
    return (
        f"# {by['Commit Title']}\n\n"
        f"## Summary of Changes\n{by['Summary of Changes']}\n\n"
        f"## Files Changed ({stats['files']} file{'s' if stats['files'] != 1 else ''}, +{stats['added']} / -{stats['removed']})\n{changed}\n\n"
        f"## Verification Checklist\n" + "\n".join(checklist) + "\n\n"
        "## Related Issue\nCloses #<issue-number>\n"
    )


def build_diff_prompt(diff: str, files, stats):
    names = ", ".join(f["path"] for f in files[:6]) or "unknown"
    return f"""You are a GitHub pull-request assistant helping a beginner. Do NOT show thinking, drafts or notes.
Output ONLY the three sections below with these exact headings, in this order, and finish every section.

### Commit Title
One line in Conventional Commits format, at most 72 characters, e.g. fix(http): pass config to timeout errors

### Summary of Changes
3-5 bullets. Name the exact functions, variables or files changed, say what changed and why it matters.

### Verification Checklist
1. Unit Test: name the test case and the exact assertion expected.
2. Regression Check: state which existing tests or integration check to run.
3. Manual Check: one quick manual step and the expected result.

FACTS: {stats['files']} file(s) changed ({names}); +{stats['added']} / -{stats['removed']} lines.

GIT DIFF:
{diff}
"""


# ----------------------------------------------------------------------------
# Request schemas & routes
# ----------------------------------------------------------------------------
class IssueRequest(BaseModel):
    issue_text: str = Field(..., max_length=200_000)


class DiffRequest(BaseModel):
    diff_text: str = Field(..., max_length=2_000_000)


@app.get("/")
def serve_frontend():
    index = BASE_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"error": "index.html not found."}


@app.get("/tailwind.js")
def serve_tailwind():
    # Kept for older copies of the frontend; the new index.html needs no Tailwind.
    tw = BASE_DIR / "tailwind.js"
    if tw.exists():
        return FileResponse(tw)
    return Response(status_code=204)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "cloud_configured": bool(GEMINI_API_KEY),
        "cloud_model": CLOUD_MODEL,
        "local_model": OLLAMA_MODEL,
        "budget_seconds": TOTAL_BUDGET_SECONDS,
    }


@app.post("/api/demystify")
def demystify_issue(req: IssueRequest):
    started = time.monotonic()
    if not req.issue_text.strip():
        raise HTTPException(status_code=400, detail="Issue text cannot be empty.")

    issue, n_secrets = sanitize_text(req.issue_text.strip())
    issue = issue[:MAX_ISSUE_CHARS]

    ai = generate_ai_response(build_issue_prompt(issue))
    fallback = heuristic_issue(issue)

    if ai:
        parsed = parse_sections(ai["output"], ISSUE_TITLES)
        sections, partial = complete_sections(parsed, ISSUE_TITLES, fallback, ai["truncated"])
        engine, offline = ai["engine"], ai["offline_mode"]
    else:
        sections = [{"title": t, "body": fallback[t], "source": "auto"} for t in ISSUE_TITLES]
        partial, engine, offline = True, "Offline rule-based analyzer (no AI engine reachable)", True

    return {
        "output": sections_to_markdown(sections),
        "sections": sections,
        "engine": engine,
        "offline_mode": offline,
        "partial": partial,
        "secret_flagged": n_secrets > 0,
        "secrets_masked": n_secrets,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }


@app.post("/api/synthesize")
def synthesize_pr(req: DiffRequest):
    started = time.monotonic()
    if not req.diff_text.strip():
        raise HTTPException(status_code=400, detail="Diff text cannot be empty.")

    clean_diff, n_secrets = sanitize_text(req.diff_text)
    files = parse_diff(clean_diff)
    if not files:
        raise HTTPException(
            status_code=400,
            detail="This doesn't look like a git diff. Paste the output of `git diff` (it starts with 'diff --git').",
        )
    stats = {
        "files": len(files),
        "added": sum(f["added"] for f in files),
        "removed": sum(f["removed"] for f in files),
    }

    fallback = heuristic_pr(files)
    ai = generate_ai_response(build_diff_prompt(compact_diff(clean_diff), files, stats))

    if ai:
        parsed = parse_sections(ai["output"], PR_TITLES)
        sections, partial = complete_sections(parsed, PR_TITLES, fallback, ai["truncated"])
        engine, offline = ai["engine"], ai["offline_mode"]
    else:
        sections = [{"title": t, "body": fallback[t], "source": "auto"} for t in PR_TITLES]
        partial, engine, offline = True, "Offline rule-based analyzer (no AI engine reachable)", True

    # Validate the commit title (Conventional Commits, <= 72 chars).
    for s in sections:
        if s["title"] == "Commit Title":
            s["body"] = clean_title(s["body"], fallback["Commit Title"])
    title = next(s["body"] for s in sections if s["title"] == "Commit Title")

    return {
        "output": sections_to_markdown(sections),
        "sections": sections,
        "pr_markdown": build_pr_markdown(sections, files, stats),
        "git_steps": build_git_steps(title, files),
        "files": [{k: f[k] for k in ("path", "status", "added", "removed")} for f in files],
        "stats": stats,
        "engine": engine,
        "offline_mode": offline,
        "partial": partial,
        "secret_flagged": n_secrets > 0,
        "secrets_masked": n_secrets,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }
