# 📚 Study Buddy — The 100% Offline Active-Recall Study Companion

> Built for the **AppBuildersPH Hackathon**: Delivering reliable, high-yield, error-free AI study tools directly on standard consumer laptops with zero cloud dependencies.

---

## 🌟 The Problem & Real-World Utility
In the Philippines and many regions worldwide, students face:
- **Unreliable internet & high mobile data costs:** Students cannot rely on always-on cloud AI during daily commutes, in dorms, or during power/network interruptions.
- **Expensive subscriptions:** $20/month cloud subscriptions or pay-per-token API fees are out of reach for average students.
- **Passive reading vs. Active retention:** Standard AI chatbots produce generic, passive bullet points that create the illusion of competence but lead to poor exam retention.

**Study Buddy** solves this by running an **insightful, active-recall study engine 100% locally on standard consumer hardware** for ₱0, with no internet required.

---

## 🚀 Key Features

- **🧠 Offline Quiz Tutor ("Quiz me" tab):** Upload notes and get your first question in seconds. Type an answer in your own words and the local model grades it against the exact passage in your notes, with a one-line hint and a highlighted page/slide citation. Missed questions come back for review, and can be exported as Quizlet flashcards. Every model call is short (a few seconds on a CPU), so it stays responsive with Wi-Fi off.
- **⚡ 100% Offline Generation:** Powered by local quantized models (Qwen2.5 1.5B via Ollama) running on standard CPU with under 4GB RAM footprint.
- **📖 Dual Reviewer Modes:**
  - **Comprehensive Reviewer:** Deep concept breakdowns, "Don't Mix Up" distinctions, worked examples, formulas with LaTeX/Unicode math rendering, and practice questions.
  - **Easy Mode:** Intuitive real-world analogies, narrative walkthroughs, and visual explanations for difficult concepts.
- **🧠 Active Recall & Flashcard Export:** Generates ready-to-drill flashcards exported directly to Quizlet/Anki text formats and structured JSON.
- **🔍 On-Device Quality Auditor:** Automatically inspects flashcards for clarity, removes duplicate concepts, and validates question rigor.
- **📑 Multi-Format Ingestion:** Reads PDFs, PowerPoint (`.pptx`), Word (`.docx`), Excel (`.xlsx`), Markdown, and raw text notes.
- **🖨️ One-Click Printable PDFs:** Produces clean, publication-ready study guides with styled headers, math typography, and answer keys.
- **🌐 Hybrid Architecture:** Seamlessly switches to Google Gemini when online for multi-file cross-chapter synthesis, but remains fully functional offline.

---

## 🛠️ Technical Architecture

```
[Student Notes (PDF, PPTX, DOCX, TXT)]
                  │
                  ▼
         [Local Extraction Layer]
       (pypdfium2 / python-pptx / docx)
                  │
                  ▼
        [Local AI Engine (Ollama)]
    ├── Model: Qwen2.5-Coder:1.5B (Quantized)
    ├── Memory Pinning: keep_alive persistent cache
    ├── Tuned Context: 2048-token low-latency attention
    └── Strict Structured Outputs (Pydantic Schema)
                  │
                  ▼
         [Local Quality Auditor]
    (Heuristic deduplication & quality check)
                  │
                  ▼
        [Deliverable Outputs]
    ├── Styled Study Guide (PDF via xhtml2pdf)
    ├── Easy Analogy Guide (PDF)
    └── Active Recall Flashcards (Quizlet / JSON)
```

---

## 💻 Hardware Requirements
Study Buddy is deliberately engineered to run smoothly on budget consumer laptops:
- **CPU:** Standard Intel / AMD dual-core or quad-core processor (no dedicated GPU required).
- **RAM:** 4GB to 8GB total system memory.
- **OS:** Windows (WSL2), Linux, or macOS.

---

## 🏁 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/study-buddy.git
cd study-buddy
```

### 2. Set Up Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Install & Start Ollama (For 100% Offline Mode)
Install Ollama from [ollama.com](https://ollama.com), then pull the lightweight 1.5B model:
```bash
ollama pull qwen2.5:1.5b        # quiz tutor (set STUDY_QUIZ_MODEL to change)
ollama pull qwen2.5-coder:1.5b  # offline reviewers (set STUDY_MODEL to change)
```

### 4. (Optional) Cloud Mode Configuration
If you want to use the optional Gemini online mode, copy `.env.example` to `.env` and add your API key:
```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

### 5. Launch the Application
```bash
./start.sh
# Or directly via streamlit:
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🏆 Hackathon Alignment (AppBuildersPH)

| Category | Weight | How Study Buddy Delivers |
| :--- | :---: | :--- |
| **Problem & Usefulness** | **25%** | Delivers zero-cost, high-retention study materials for students facing unstable internet and exam stress. |
| **Local AI Implementation** | **25%** | Meaningful on-device inference using quantized local LLMs, strict structured Pydantic schemas, and local quality auditing. |
| **Technical Execution** | **20%** | Pre-warmed model caching, low-latency CPU attention windows, and resilient JSON fallback parsers ensuring 100% error-free live demos. |
| **Innovation** | **15%** | Moves beyond passive ChatGPT chat summaries into cognitive-science active recall (distinctions, worked examples, flashcards). |
| **Product & Demo Quality** | **15%** | Clean Streamlit UI with one-click PDF generation, live card previews, and verifiable "Airplane Mode" live testing. |

---

## 📄 License
MIT License. Free to use, adapt, and share.
