import os
import re
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Cloud API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
CLOUD_MODEL = "gemma-4-26b-a4b-it"

app = FastAPI(title="Gemma-Merge Backend")

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- SECURITY & SANITIZATION -----------------
SECRET_PATTERNS = [
    r'ghp_[a-zA-Z0-9]{20,}',
    r'AKIA[0-9A-Z]{16}',
    r'sk-[a-zA-Z0-9]{20,}',
    r'AIza[0-9A-Za-z-_]{35}',
    r'(?i)(api[_-]?key|secret|token|password)\s*=\s*[\'"][^\'"]+[\'"]'
]

def sanitize_diff(diff_text: str):
    masked_diff = diff_text
    secrets_found = []
    for pattern in SECRET_PATTERNS:
        matches = re.findall(pattern, masked_diff)
        if matches:
            secrets_found.extend(matches)
            masked_diff = re.sub(pattern, '[MASKED_SECRET_CREDENTIAL]', masked_diff)
    return masked_diff, len(secrets_found) > 0

# ----------------- CORE AI ROUTING LOGIC -----------------
def generate_ai_response(prompt: str) -> dict:
    cloud_err = None
    ollama_err = None

    # 1. Cloud Inference: Gemma via Google AI Studio
    print(f"\n[AI Engine] Attempting Cloud API ({CLOUD_MODEL})...")
    try:
        cloud_url = f"https://generativelanguage.googleapis.com/v1beta/models/{CLOUD_MODEL}:generateContent?key={GEMINI_API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        res = requests.post(cloud_url, json=payload, timeout=5.0)
        
        if res.status_code == 200:
            data = res.json()
            text_output = data["candidates"][0]["content"]["parts"][0]["text"]
            print(f"[AI Engine] Cloud Success! Powered by {CLOUD_MODEL}")
            return {
                "output": text_output,
                "engine": f"Cloud Gemma ({CLOUD_MODEL})",
                "offline_mode": False
            }
        else:
            cloud_err = f"HTTP {res.status_code}: {res.text}"
            print(f"[Cloud API Failed] {cloud_err}")
    except Exception as e:
        cloud_err = str(e)
        print(f"[Cloud API Exception] {cloud_err}")

    # 2. Offline Fallback: Local Gemma 2 (2B) via Ollama
    print("[AI Engine] Falling back to local Gemma 2 (2B) on 127.0.0.1...")
    try:
        ollama_url = "http://127.0.0.1:11434/api/generate"
        payload = {
            "model": "gemma2:2b",
            "prompt": prompt,
            "stream": False
        }
        res = requests.post(ollama_url, json=payload, timeout=90.0)
        
        if res.status_code == 200:
            print("[AI Engine] Local Ollama Success!")
            return {
                "output": res.json().get("response", "No response generated."),
                "engine": "Local Gemma 2 (2B Quantized via Ollama)",
                "offline_mode": True
            }
        else:
            ollama_err = f"Ollama HTTP {res.status_code}: {res.text}"
            print(f"[Ollama Failed] {ollama_err}")
    except Exception as e:
        ollama_err = str(e)
        print(f"[Ollama Error] {ollama_err}")

    error_summary = f"Cloud Error: ({cloud_err}) | Ollama Error: ({ollama_err})"
    print(f"\n[CRITICAL FAILURE] {error_summary}\n")
    raise HTTPException(status_code=503, detail=error_summary)

# ----------------- SCHEMAS & ENDPOINTS -----------------
class IssueRequest(BaseModel):
    issue_text: str

class DiffRequest(BaseModel):
    diff_text: str

# SERVES STATIC ASSETS
@app.get("/")
def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"error": "index.html not found in project directory."}

@app.get("/tailwind.js")
def serve_tailwind():
    if os.path.exists("tailwind.js"):
        return FileResponse("tailwind.js")
    return {"error": "tailwind.js not found."}

@app.post("/api/demystify")
def demystify_issue(req: IssueRequest):
    if not req.issue_text.strip():
        raise HTTPException(status_code=400, detail="Issue text cannot be empty.")
    
    prompt = f"""You are an open-source maintainer helping a newcomer.
Provide ONLY the final structured guidance in clean Markdown based strictly on the issue text.
Do NOT output drafting steps, internal reasoning, or placeholders like 'path/to/file'.

Format strictly with these exact sections:
### 📌 Summary
(2 clear sentences explaining the core bug and consequence)

### 🛠️ Action Plan
1. **Reproduce:** (Exact steps to trigger the bug)
2. **Investigate:** (Where the issue originates)
3. **Resolve:** (The specific logical fix)

### 📂 Target Codebase Areas
- List 1 to 2 exact file paths mentioned in the stack trace or environment (e.g. adapters, middleware) with a 1-sentence explanation of what to inspect there.

GITHUB ISSUE:
{req.issue_text}
"""
    return generate_ai_response(prompt)

@app.post("/api/synthesize")
def synthesize_pr(req: DiffRequest):
    if not req.diff_text.strip():
        raise HTTPException(status_code=400, detail="Diff text cannot be empty.")
    
    clean_diff, had_secrets = sanitize_diff(req.diff_text)
    
    prompt = f"""You are a senior repository maintainer generating a production-ready Pull Request from this git diff.
Write concrete, highly detailed markdown. Do NOT repeat instructions, do NOT use placeholders like '(scope)' or '(test logic)'.

Format strictly with these exact sections:

### 🏷️ Commit Title
Provide one exact Conventional Commit line with an inferred scope (e.g. feat(ci): ..., fix(auth): ..., refactor(core): ...).

### 📝 Summary of Changes
- Detail the exact methods, variables, or endpoints added or modified.
- Explain the architectural purpose and why this change is necessary.

### ✅ Verification Checklist
1. **Unit Test:** Name the specific test case, mock object, and expected assertion status.
2. **Regression Check:** State the exact upstream integration or security check to run before merging.

GIT DIFF:
{clean_diff}
"""
    result = generate_ai_response(prompt)
    result["secret_flagged"] = had_secrets
    return result