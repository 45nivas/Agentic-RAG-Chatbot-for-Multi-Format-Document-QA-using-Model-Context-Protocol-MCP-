import pytest
import json
from unittest.mock import patch, MagicMock

# Ensure we can import app and agents correctly
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from agents.llm_response_agent import clean_and_repair_json
from agents.health_agents import SafetyAuditorAgent, WebResearchAgent

def mifflin_st_jeor(weight, height, age, gender):
    if gender.lower() == "male":
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

# 1. Mifflin-St Jeor BMR Verification
def test_mifflin_bmr():
    # Male: 42 years, 85kg, 180cm
    assert mifflin_st_jeor(85, 180, 42, "Male") == 85 * 10 + 6.25 * 180 - 5 * 42 + 5
    # Female: 30 years, 60kg, 165cm
    assert mifflin_st_jeor(60, 165, 30, "Female") == 60 * 10 + 6.25 * 165 - 5 * 30 - 161

# 2. JSON Clean and Repair Utility Verification
def test_clean_and_repair_json():
    # Test 1: standard json markdown fenced
    fenced_json = "```json\n{\"key\": \"value\"}\n```"
    assert json.loads(clean_and_repair_json(fenced_json)) == {"key": "value"}
    
    # Test 2: standard code markdown fenced
    fenced_json_2 = "```\n{\"key\": \"value\"}\n```"
    assert json.loads(clean_and_repair_json(fenced_json_2)) == {"key": "value"}
    
    # Test 3: conversational wraps
    conversational_json = "Here is the parsed profile:\n{\n  \"longevity_score\": 90\n}\nHope this helps!"
    assert json.loads(clean_and_repair_json(conversational_json)) == {"longevity_score": 90}

# 3. Safety Auditor Allergen Detection Verification
def test_safety_auditor_flags_allergen():
    auditor = SafetyAuditorAgent()
    auditor.client = MagicMock()
    
    # Mock Response from the consolidated call method
    mock_audit_json = {
        "audit_report": "CRITICAL WARNING: The proposed meal plan contained peanut butter which violates the patient peanut allergy. Corrected peanut butter to almond butter.",
        "is_cleared": True,
        "corrections_made": ["Substituted peanut butter with almond butter due to patient peanuts allergy."],
        "final_meal_plan": {
            "breakfast": {
                "name": "Almond Butter Toast",
                "ingredients": ["1 slice sourdough", "1 tbsp almond butter"]
            }
        },
        "final_training_plan": {}
    }
    
    auditor.client.call.return_value = json.dumps(mock_audit_json)
    
    profile = {"allergies": ["Peanuts"], "medical_conditions": [], "biomarkers": []}
    targets = {"calories": 2000}
    meal_plan = {
        "breakfast": {
            "name": "Peanut Butter Toast",
            "ingredients": ["1 slice sourdough", "1 tbsp peanut butter"]
        }
    }
    
    result_msg = auditor.audit_and_correct(profile, targets, meal_plan, {})
    payload = result_msg.payload
    
    assert payload["is_cleared"] is True
    assert "peanut" in payload["audit_report"].lower()
    assert len(payload["corrections_made"]) > 0
    assert "almond butter" in payload["corrections_made"][0].lower()

# 4. PubMed NCBI E-Utilities Mock Search Engine Verification
@patch('agents.health_agents.requests.get')
def test_pubmed_search_returns_results(mock_get):
    agent = WebResearchAgent()
    
    # Mock responses for esearch and esummary
    mock_search_resp = MagicMock()
    mock_search_resp.status_code = 200
    mock_search_resp.json.return_value = {
        "esearchresult": {
            "idlist": ["123456", "789012"]
        }
    }
    
    mock_summary_resp = MagicMock()
    mock_summary_resp.status_code = 200
    mock_summary_resp.json.return_value = {
        "result": {
            "uids": ["123456", "789012"],
            "123456": {
                "title": "Clinical Study on Longevity Pathways",
                "source": "Journal of Endocrinology",
                "pubdate": "2024 May"
            },
            "789012": {
                "title": "Effects of Cardiorespiratory Splits on Hypertension",
                "source": "Hypertension Journal",
                "pubdate": "2023 Nov"
            }
        }
    }
    
    # Mocking sequential GET calls
    mock_get.side_effect = [mock_search_resp, mock_summary_resp]
    
    articles = agent.search_pubmed("longevity", max_results=2)
    
    assert len(articles) == 2
    assert articles[0]["uid"] == "123456"
    assert articles[0]["title"] == "Clinical Study on Longevity Pathways"
    assert articles[0]["journal"] == "Journal of Endocrinology"
    assert articles[1]["uid"] == "789012"

# 5. REST API Chat Endpoint Payload Guards Verification
def test_api_chat_missing_field():
    client = app.test_client()
    
    # Test 1: Empty JSON payload
    resp = client.post("/api/chat", json={})
    assert resp.status_code == 400
    assert "error" in resp.json
    assert "message field required" in resp.json["error"].lower()
    
    # Test 2: Message field is not a string
    resp_2 = client.post("/api/chat", json={"message": 12345})
    assert resp_2.status_code == 400
    assert "must be a string" in resp_2.json["error"].lower()
    
    # Test 3: Message is empty space
    resp_3 = client.post("/api/chat", json={"message": "   "})
    assert resp_3.status_code == 400
    assert "cannot be empty" in resp_3.json["error"].lower()
    
    # Test 4: Message length exceeds 4000 chars
    resp_4 = client.post("/api/chat", json={"message": "A" * 4001})
    assert resp_4.status_code == 400
    assert "too long" in resp_4.json["error"].lower()

# 6. PDF Clinical Report Generation Verification
def test_report_generation():
    from agents.report_generator import ClinicalReportGenerator
    
    profile = {
        "demographics": {"age": 42, "weight_kg": 85, "height_cm": 180, "gender": "Male", "activity_level": "Moderate"},
        "goals": ["Optimize longevity"],
        "allergies": ["Peanuts"],
        "medical_conditions": ["Hypertension"],
        "biomarkers": [
            {
                "name": "Glucose",
                "value": 110,
                "unit": "mg/dL",
                "status": "Elevated",
                "normal_range": "70-99 mg/dL",
                "clinical_significance": "Mild insulin resistance."
            }
        ]
    }
    
    meal_plan = {
        "targets": {"calories": 2100, "protein": 140, "carbs": 180, "fats": 70},
        "breakfast": {"name": "Eggs and Avocado", "calories": 400, "protein": 30, "carbs": 10, "fats": 25, "ingredients": ["3 eggs", "avocado"], "instructions": "Scramble and slice."}
    }
    
    training_plan = {
        "weekly_split": "3-Day Strength",
        "exercises": [{"name": "Goblet Squats", "sets": "3", "reps": "12", "intensity": "RPE 7", "instructions": "Hold weight and squat."}],
        "safety_precautions": ["Avoid breath holding."]
    }
    
    bio_age_results = {
        "chronological_age": 42,
        "biological_age": 39,
        "longevity_score": 92,
        "pathway_focus": "AMPK Activation",
        "longevity_tips": ["12-hour fast."]
    }
    
    critique = {
        "clinical_grade": "A+",
        "mechanics_explanation": "Hypertrophy helps clear glucose.",
        "peer_review_notes": "Perfect protocol.",
        "advanced_optimizations": ["Pair D3 with fat."],
        "scientific_citations": ["PubMed reference"]
    }
    
    pdf_bytes = ClinicalReportGenerator.generate_pdf(
        profile=profile,
        meal_plan=meal_plan,
        training_plan=training_plan,
        bio_age_results=bio_age_results,
        critique=critique,
        audit_report="Peanuts allergen corrected to Almonds.",
        corrections=["Substituted peanut butter with almond butter."]
    )
    
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF")

# 7. Gemini Timeout Fast-Fallback Verification
def test_gemini_timeout_fallback():
    from agents.llm_response_agent import LLMClient
    
    client = LLMClient()
    
    # Mock self.gemini.generate_content to raise a timeout-like exception
    class DeadlineExceeded(Exception):
        pass
        
    mock_gen = MagicMock(side_effect=DeadlineExceeded("Deadline Exceeded"))
    client.gemini.generate_content = mock_gen
    
    # Mock get_installed_ollama_models to return empty list to force Gemini to run
    # and call_opensource_fallback to return a successful mock string
    with patch('agents.llm_response_agent.get_installed_ollama_models', return_value=[]), \
         patch('agents.llm_response_agent.call_opensource_fallback', return_value="fallback_response") as mock_fallback:
         
        res = client.call("test query")
        
        # 1. Confirm that it did not retry 4 times (generate_content should have been called exactly once)
        assert mock_gen.call_count == 1
        
        # 2. Confirm it fell back to opensource_fallback and returned the fallback response
        assert res == "fallback_response"
        mock_fallback.assert_called_once_with("test query")

