# NutriMind AI — Multi-Agent Clinical Wellness Suite

NutriMind AI is a state-of-the-art, medically-aligned multi-agent RAG chatbot and dashboard designed for automated clinical health planning, blood panel analysis, biological longevity estimation, and sports-science training prescription. 

It is built with a premium **Clinical-Minimal aesthetic** (Notion-minimal, #F7F5F0 warm alabaster, ink-black elements, and gold-amber accents), using a fully **decoupled, two-terminal React + TypeScript + Flask REST API architecture**.

---

## 🏗️ Decoupled System Architecture

The ecosystem separates the UI client from the orchestration engine, maintaining clean routing, modular microservices, and sub-2 second turnaround speeds:

```
                  ┌───────────────────────────────────────────┐
                  │          React + TypeScript client        │
                  │              (Vite | Port 5173)           │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        │ (HTTP REST / JSON)
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │           Python Flask REST API           │
                  │             (WSGI | Port 5000)            │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │       CoordinatorAgent Orchestrator       │
                  └─────────────────────┬─────────────────────┘
                                        │
                  ┌─────────────────────┴─────────────────────┐
                  ▼ (Parallel Threads Execution)              ▼
       ┌─────────────────────┐                     ┌─────────────────────┐
       │   RetrievalAgent    │                     │ PubMedSearchAgent   │
       │ (ChromaDB + PubMed) │                     │ (PubMed Articles)   │
       └─────────────────────┘                     └─────────────────────┘
       ┌─────────────────────┐                     ┌─────────────────────┐
       │  BioAgeCalculator   │                     │ ClinicalKinesiology │
       │ (Biomarker Engine)  │                     │ (Sports Medicine)   │
       └─────────────────────┘                     └─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │           SafetyAuditorAgent              │
                  │  (Meal Allergies & Biomechanical Caps)    │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │          ClinicalCritiqueAgent            │
                  │   (Medical Board Peer Review & Grade)     │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │           Unified Response (JSON)         │
                  └───────────────────────────────────────────┘
```

---

## ⚡ Concurrent Agent Orchestration Pipeline

To maximize accuracy while sustaining high turnaround rates under local execution, `CoordinatorAgent` uses a Python `ThreadPoolExecutor` to dispatch **four specialized clinical routines concurrently** inside `process_health_query`:

1. **`RetrievalAgent` (Semantic Vector RAG):**
   * Uses biomedical-specific **`NeuML/pubmedbert-base-embeddings`** (trained on 14M+ PubMed publications) for clinical-grade semantic vector searching inside ChromaDB.
   * Employs memory-based **LRU term caching** to skip neural network encoding for common health inquiries.
2. **`PubMedSearchAgent` (Clinical Web Scraper):**
   * Directs real-time queries to PubMed database hubs to retrieve clinical papers matching current symptoms or biomarker anomalies.
3. **`BioAgeCalculatorAgent` (Longevity Math):**
   * Parses blood panel results (LDL, Blood Pressure, Glucose, BMI) to estimate the patient's **Biological Age** vs. **Chronological Age**.
   * Identifies biological pathway priorities (e.g. AMPK activation, Autophagy trigger, Nitric oxide synthesis) and details specific longevity score offsets.
4. **`ClinicalKinesiologyAgent` (Exercise Prescription):**
   * Prescribes custom weekly training divisions (aerobic intervals, muscular hypertrophy, cardiovascular conditioning) mapped against specific biomarker abnormalities.

### Unified Safety & Medical Peer Review
Once the concurrent agents deliver their payloads, they are audited before rendering to the client:
* **`SafetyAuditorAgent` (Dual Safety Gate):** Screens nutritional recommendations against known patient allergens, and audits physical workouts to attach RPE intensity ceiling restrictions (e.g. preventing high-intensity lifting or Valsalva actions for hypertensive patients).
* **`ClinicalCritiqueAgent` (Medical Peer Review Board):** Grades the combined health strategy (e.g. **A+**), evaluates cellular adaptation mechanics (e.g., how resistance training uregulates GLUT4 receptors to clear circulating blood glucose), and formats a clean JSON output.

---

## 🛡️ 3-Tier Open-Source LLM Fallback Pipeline

To guarantee 100% service uptime even when the primary Gemini API is rate-limited or hits quota caps (e.g., `429 Quota Exceeded`), your LLM connector employs a robust, automatic **three-tier fallback system**:

```
                       [ Primary Gemini LLM API Call ]
                                      │
                                      ▼ (Sustained 429 Quota Exception?)
                                      │
                      ┌───────────────┴───────────────┐
                      ▼ (Tier 1)                      │
             ┌──────────────────┐                     │
             │   Local Ollama   │◄────────────────────┤
             │ (Ollama Offline) │                     │
             └──────────────────┘                     │
                      ┌───────────────────────────────┤
                      ▼ (Tier 2)                      │
             ┌──────────────────┐                     │
             │   Hugging Face   │◄────────────────────┤
             │  Serverless API  │                     │
             └──────────────────┘                     │
                      ▼ (Tier 3)                      ▼
             ┌────────────────────────────────────────┐
             │       Local CPU Transformers Pipeline   │
             │  (Qwen 2.5 0.5B Instruct - ~900MB)     │
             └────────────────────────────────────────┘
```

1. **Tier 1: Local Ollama Server (Fully Private & Offline)**
   * Detects if an Ollama local endpoint is active (`http://localhost:11434`) and attempts fast queries to `qwen2.5`, `llama3`, `mistral`, or `phi3`.
2. **Tier 2: Hugging Face Serverless API (Zero Credentials, Compact Models)**
   * Avoids heavy models (Qwen 72B is excluded to prevent latency). Instead, it targets fast, high-performance, and publicly available open-source options:
     * `Qwen/Qwen2.5-7B-Instruct`
     * `Qwen/Qwen2.5-1.5B-Instruct`
     * `meta-llama/Meta-Llama-3-8B-Instruct`
     * `mistralai/Mistral-7B-Instruct-v0.3`
3. **Tier 3: Local CPU Transformers (100% Private, Secure Fallback)**
   * Seamlessly spins up a local Hugging Face `transformers` pipeline to load **`Qwen/Qwen2.5-0.5B-Instruct` (900MB)** into your local cache.
   * **Optimized for CPU:** Runs directly on pure CPU without requiring external dependencies like `accelerate`.
   * **Pre-Loaded Singleton:** Caches the pipeline in memory so subsequent query steps run near-instantly!

---

## 🚀 Quick Start (2-Terminal Execution)

Ensure you have your environment set up and the virtual environment active.

### Terminal 1: Python Flask Backend REST API
```bash
# Navigate to the root directory
cd Agentic-RAG-Chatbot-for-Multi-Format-Document-QA-using-Model-Context-Protocol-MCP--main

# Activate the virtual environment
.venv\Scripts\activate

# Install dependencies (if not done already)
pip install -r requirements-local.txt
pip install flask-cors

# Run the Flask backend
python app.py
```
*Backend runs on `http://127.0.0.1:5000`.*

### Terminal 2: React + TS Frontend Client
```bash
# Navigate to the frontend directory
cd frontend

# Install package modules
npm install

# Start Vite dev server
npm run dev
```
*Frontend runs on `http://localhost:5173`.*

---

## 🎨 Premium Visual Elements

* **Concentric Fueling Rings:** real-time animated macro status indicators representing Caloric Target (Gold), Protein (Coral), Carbs (Teal), and Fats (Green).
* **InsideTracker-Style Gauges:** visual biomarker tables featuring sliding needles mapped dynamically to your bio-indicator levels.
* **Longevity Age Badges:** real-time Biological Age offsets and longevity score progress dials.
* **Agent Diagnostics Console:** live telemetry monitoring vector engine performance and MCP pipeline raw JSON traces.
