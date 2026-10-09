# ⚡ Gemma-Merge

> **Dual-Engine Open-Source Contributor Copilot & Zero-Trust Privacy Harness**  
> Built for Hacktoberfest Hack Day Bengaluru '26 (IEEE CIS × RIT)

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Track 1: Best Use of Gemma 4 (PS 02)](https://img.shields.io/badge/Track%201-Best%20Use%20of%20Gemma%204%20(PS%2002)-blue.svg)](#-track-alignment)

---

## 📌 Problem Statement

1. **Cryptic Maintainer Jargon:** New open-source contributors struggle to interpret technical shorthand and stack traces in GitHub issues.
2. **Pull Request Anxiety:** Junior developers often fail to format Conventional Commit titles, structured summaries, and testing checklists expected by maintainers.
3. **Accidental Credential Exposure:** Developers frequently hardcode local test tokens and API keys during rapid debugging, inadvertently pushing them to public repositories.
4. **Cloud-Only Brittleness:** Existing AI copilot extensions fail completely under spotty hackathon Wi-Fi or air-gapped development environments.

---

## 💡 What Gemma-Merge Does

* **Tab 1: Issue Demystifier:** Ingests raw GitHub bug reports and translates them into plain-English explanations, a 3-step action plan (Reproduce, Investigate, Resolve), and specific target files to inspect.
* **Tab 2: Diff-to-PR Synthesizer:** Ingests raw `git diff` code modifications and automatically produces standardized Conventional Commit titles, architectural summaries, and QA testing checklists.
* **Zero-Trust Security Shield:** Uses in-memory regex scanning to intercept and sanitize API tokens (`ghp_`, `AKIA`, `sk-`, Google keys) into `[MASKED_SECRET_CREDENTIAL]` before model tokenization.
* **Dual-Engine Resilience:** Queries **Cloud Gemma 4 (26B)** via Google AI Studio as the primary engine and automatically fails over to **Local Gemma 2 (2B Quantized)** on Ollama (`127.0.0.1`) if offline or latency spikes.

---

## 🏆 Track Alignment

### Track 1: Best Use of Gemma 4 — PS 02 (Intelligent Personal / Professional Copilot)

Gemma-Merge is purpose-built as an intelligent developer copilot powered centrally by `gemma-4-26b-a4b-it`:

* **Unstructured Input Reasoning:** Understands unstructured bug tickets, stack traces, and raw diffs, breaking them down into actionable steps.
* **Structured Output Generation:** Enforces strict Conventional Commit schemas, architectural justifications, and verification checklists.
* **Human-in-the-Loop Agency:** Acts strictly as an advisory copilot; all changes are staged in an interactive review dashboard to ensure human validation before any PR submission.
* **Workflow Reliability Layer:** Integrates local in-memory secret sanitization and edge fallback to local Gemma 2 to guarantee uninterrupted developer workflows.

---

## 🛠️ Tech Stack

* **Frontend:** HTML5, Local Tailwind CSS (100% offline styling)
* **Backend:** FastAPI, Uvicorn, Python Requests
* **Primary AI Engine:** Gemma 4 (`gemma-4-26b-a4b-it`) via Google AI Studio API
* **Edge Fallback Engine:** Quantized Gemma 2 (`gemma2:2b`) via local Ollama instance
* **Security:** In-memory deterministic regex credential masking engine

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/devmousin/Gemma-Merge.git](https://github.com/devmousin/Gemma-Merge.git)
cd Gemma-Merge