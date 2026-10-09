# ⚡ Gemma-Merge

> **Dual-Engine Open-Source Contributor Copilot & Zero-Trust Privacy Harness**  
> Built for Hacktoberfest Hack Day Bengaluru '26 (IEEE CIS × RIT)[cite: 9].

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Track 1](https://img.shields.io/badge/Track%201-Best%20Use%20of%20Gemma%204%20(PS%2002)-blue.svg)](#-track-alignment)
[![Track 2](https://img.shields.io/badge/Track%202-Best%20Open--Source%20AI%20Project%20(PS%2004)-purple.svg)](#-track-alignment)

---

## 📌 Problem Statement

1. **Cryptic Maintainer Jargon:** New open-source contributors struggle to understand technical shorthand and stack traces in GitHub issues.
2. **Pull Request Anxiety:** Junior developers often fail to format Conventional Commit titles, structured summaries, and testing checklists expected by maintainers[cite: 4, 5, 9].
3. **Accidental Credential Exposure:** Developers frequently hardcode local test tokens and API keys during debugging, inadvertently pushing them to public repositories.
4. **Cloud-Only Dependency:** Existing AI copilots fail completely under spotty hackathon Wi-Fi or air-gapped development environments[cite: 5, 10].

---

## 💡 What Gemma-Merge Does

* **Tab 1: Issue Demystifier:** Ingests raw GitHub bug reports and translates them into a plain-English explanation, a 3-step action plan (Reproduce, Investigate, Resolve), and specific target files to inspect.
* **Tab 2: Diff-to-PR Synthesizer:** Ingests raw `git diff` code modifications and automatically produces standardized Conventional Commit titles, architectural summaries, and QA testing checklists.
* **Zero-Trust Security Shield:** Uses in-memory regex scanning to intercept and sanitize API tokens (`ghp_`, `AKIA`, `sk-`, Google keys) into `[MASKED_SECRET_CREDENTIAL]` before model tokenization.
* **Dual-Engine Resilience:** Queries **Cloud Gemma 4 (26B)** via Google AI Studio as the primary engine and automatically fails over to **Local Gemma 2 (2B Quantized)** on Ollama (`127.0.0.1`) if offline or latency spikes[cite: 5, 10].

---

## 🏆 Track Alignment

### Track 1: Best Use of Gemma 4 — PS 02 (Intelligent Copilot)[cite: 9, 10]
* Translates unstructured bug reports into multi-step reasoning and actionable plans[cite: 4, 9].
* Generates structured outputs (Conventional PR format, QA verification) requiring human-in-the-loop developer approval before merging[cite: 4, 5, 9].
* Uses `gemma-4-26b-a4b-it` as the primary reasoning engine[cite: 5, 10].

### Track 2: Best Open-Source AI Project — PS 04 (AI System Behind the AI)
* Implements dynamic model routing between cloud endpoints and local edge inference[cite: 5, 10].
* Acts as a privacy and workflow reliability layer that prevents credential leaks and eliminates cloud downtime[cite: 5, 10, 11].

---

## 🛠️ Tech Stack

* **Frontend:** HTML5, Local Tailwind CSS (100% offline styling)[cite: 4, 5]
* **Backend:** FastAPI, Uvicorn, Python Requests
* **Cloud AI:** Gemma 4 (`gemma-4-26b-a4b-it`) via Google AI Studio API[cite: 5, 10]
* **Local Edge AI:** Quantized Gemma 2 (`gemma2:2b`) via local Ollama instance[cite: 5]
* **Security:** In-memory deterministic regex credential masking engine[cite: 5]

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/devmousin/Gemma-Merge.git](https://github.com/devmousin/Gemma-Merge.git)
cd Gemma-Merge