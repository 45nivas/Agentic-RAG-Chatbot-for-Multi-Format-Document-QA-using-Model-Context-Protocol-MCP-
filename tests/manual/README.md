# Manual Clinical Diagnostics Suite

This directory contains standalone, manual verification and diagnostic scripts that test the end-to-end integration of NutriMind AI clinical agents against live APIs.

> [!WARNING]
> These scripts cost real Gemini API tokens and should be run deliberately. Do NOT run them as part of automated CI/CD pipelines. Pytest is explicitly configured in `pytest.ini` to ignore this directory.

## Prerequisite Setup
Ensure that you have an active `.env` file at the repository root containing your live Google GenAI API key:
```env
GEMINI_API_KEY=your_live_api_key_here
```

## Available Scripts

### 1. E2E Integration Suite (`manual_gemini_server_e2e.py`)
This script launches the Flask backend app in a background subprocess, uploads the PDF clinical report (`MR. N SATYENDRA.pdf`), queries the chatbot (triggering parallel agent execution), and downloads the compiled ReportLab PDF.
*   **Run command**:
    ```bash
    .venv\Scripts\python tests/manual/manual_gemini_server_e2e.py
    ```

### 2. Gemini Response Truncation Tester (`manual_nutri_planner_debug.py`)
This script validates Gemini generation length behaviors by running plain text prompts with different token configurations against the real Gemini API.
*   **Run command**:
    ```bash
    .venv\Scripts\python tests/manual/manual_nutri_planner_debug.py
    ```
