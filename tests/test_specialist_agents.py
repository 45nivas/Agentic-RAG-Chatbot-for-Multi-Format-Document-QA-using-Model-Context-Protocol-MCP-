"""
Baseline unit tests for 5 specialist clinical agents that previously had zero test coverage:
  - WebResearchAgent
  - NutriPlannerAgent
  - BioAgeCalculatorAgent
  - ClinicalKinesiologyAgent
  - ClinicalCritiqueAgent

Each agent gets two tests:
  1. Happy path: mock LLM returns well-formed output, verify MCPMessage structure.
  2. Malformed response: mock LLM returns garbage/empty, verify graceful fallback (no crash).

NOTE: ClinicalAnalyzerAgent and SafetyAuditorAgent already have coverage in
test_clinical_suite.py and test_demographics.py — intentionally skipped here.
RetrievalAgent, IngestionAgent, and LLMResponseAgent are infrastructure agents
with lower clinical-decision priority — not covered in this pass.
"""
import json
from unittest.mock import MagicMock
from agents.mcp import MCPMessage
from agents.health_agents import (
    WebResearchAgent,
    NutriPlannerAgent,
    BioAgeCalculatorAgent,
    ClinicalKinesiologyAgent,
    ClinicalCritiqueAgent,
)

# Common mock health profile for testing inputs
SAMPLE_PROFILE = {
    "demographics": {
        "name": "N Satyendra",
        "age": 23,
        "weight_kg": 75.0,
        "height_cm": 175.0,
        "gender": "Male",
        "activity_level": "Moderate",
    },
    "goals": ["Optimize overall health"],
    "allergies": ["Peanuts"],
    "medical_conditions": ["Mild Hypertension"],
    "biomarkers": [
        {"name": "Systolic BP", "value": 130, "unit": "mmHg", "status": "Normal"}
    ],
}


# =====================================================================
# 1. WebResearchAgent Tests
# =====================================================================

def test_web_researcher_happy_path():
    """Verify WebResearchAgent wraps a well-formed LLM research note into an MCPMessage
    with correct sender/receiver/type and the raw text in the payload."""
    agent = WebResearchAgent()
    agent.search_pubmed = MagicMock(return_value=[
        {"uid": "123456", "title": "Hypertension and Sodium Restriction", "journal": "JAMA", "pubdate": "2024"}
    ])
    agent.client.call = MagicMock(return_value="PubMed Research Mandate: Limit sodium to 1500mg.")

    msg = agent.perform_clinical_search(SAMPLE_PROFILE["medical_conditions"], "test query")

    assert isinstance(msg, MCPMessage)
    assert msg.sender == "WebResearchAgent"
    assert msg.receiver == "NutriPlannerAgent"
    assert msg.type == "RESEARCH_NOTE"
    assert msg.payload["research_note"] == "PubMed Research Mandate: Limit sodium to 1500mg."


def test_web_researcher_malformed():
    """Verify WebResearchAgent does not crash when the LLM returns an empty string.

    SCOPE LIMITATION: This test ONLY confirms that the agent's perform_clinical_search
    method handles an empty-string LLM response without crashing. This works because
    WebResearchAgent treats the LLM output as raw text — it never calls json.loads(),
    so there is no parse-crash vector for malformed content.

    This test does NOT cover the case where self.client.call() itself raises an
    exception (e.g. network timeout, API auth failure, rate-limit 429). That failure
    mode would propagate as an unhandled exception from perform_clinical_search and
    is currently caught at the coordinator level (tested in test_coordinator_resilience.py),
    not within WebResearchAgent itself.
    """
    agent = WebResearchAgent()
    agent.search_pubmed = MagicMock(return_value=[])
    agent.client.call = MagicMock(return_value="")

    msg = agent.perform_clinical_search([], "test query")

    assert isinstance(msg, MCPMessage)
    assert msg.sender == "WebResearchAgent"
    assert msg.payload["research_note"] == ""


# =====================================================================
# 2. NutriPlannerAgent Tests
# =====================================================================

def test_nutri_planner_happy_path():
    """Verify NutriPlannerAgent parses a well-formed JSON meal plan from the LLM
    and calculates correct Mifflin-St Jeor calorie targets from SAMPLE_PROFILE.

    Deterministic calorie derivation (Male, Moderate):
      BMR = 10*75 + 6.25*175 - 5*23 + 5 = 750 + 1093.75 - 115 + 5 = 1733.75
      TDEE = 1733.75 * 1.55 (Moderate) = 2687.3125
      calorie_target = int(2687.3125) = 2687   (no goal adjustment, clamped to [1200,4000])
    """
    agent = NutriPlannerAgent()
    happy_response = {
        "breakfast": {"name": "Protein Oats", "calories": 400, "protein": 30, "carbs": 50, "fats": 8, "ingredients": [], "instructions": ""},
        "lunch": {"name": "Salad", "calories": 500, "protein": 40, "carbs": 20, "fats": 20, "ingredients": [], "instructions": ""},
        "dinner": {"name": "Chicken", "calories": 600, "protein": 50, "carbs": 40, "fats": 15, "ingredients": [], "instructions": ""},
        "snack": {"name": "Yogurt", "calories": 150, "protein": 15, "carbs": 10, "fats": 2, "ingredients": [], "instructions": ""},
    }
    agent.client.call = MagicMock(return_value=json.dumps(happy_response))

    msg = agent.generate_plan(SAMPLE_PROFILE, "Mock Research Note")

    assert isinstance(msg, MCPMessage)
    assert msg.sender == "NutriPlannerAgent"
    assert msg.receiver == "SafetyAuditorAgent"
    assert msg.type == "MEAL_PLAN"
    assert msg.payload["meal_plan"]["breakfast"]["name"] == "Protein Oats"
    assert msg.payload["targets"]["calories"] == 2687


def test_nutri_planner_malformed():
    """Verify NutriPlannerAgent falls back to a hardcoded default meal plan
    when the LLM returns unparseable garbage instead of valid JSON."""
    agent = NutriPlannerAgent()
    agent.client.call = MagicMock(return_value="INVALID JSON GARBAGE {]")

    msg = agent.generate_plan(SAMPLE_PROFILE, "Mock Research Note")

    assert isinstance(msg, MCPMessage)
    assert msg.sender == "NutriPlannerAgent"
    # Fallback breakfast and snack from the hardcoded default in generate_plan
    assert msg.payload["meal_plan"]["breakfast"]["name"] == "Protein Oats"
    assert msg.payload["meal_plan"]["snack"]["name"] == "Almonds and Apple"


# =====================================================================
# 3. BioAgeCalculatorAgent Tests
# =====================================================================

def test_bio_age_calculator_happy_path():
    """Verify BioAgeCalculatorAgent parses a well-formed biological age JSON response."""
    agent = BioAgeCalculatorAgent()
    happy_response = {
        "chronological_age": 23,
        "biological_age": 21,
        "longevity_score": 92,
        "pathway_focus": "Insulin Sensitivity & AMPK Activation",
        "longevity_tips": ["Tip 1", "Tip 2"],
    }
    agent.client.call = MagicMock(return_value=json.dumps(happy_response))

    msg = agent.calculate_bio_age(SAMPLE_PROFILE)

    assert isinstance(msg, MCPMessage)
    assert msg.sender == "BioAgeCalculatorAgent"
    assert msg.receiver == "CoordinatorAgent"
    assert msg.type == "BIO_AGE_CALCULATION"
    assert msg.payload["bio_age_results"]["biological_age"] == 21
    assert msg.payload["bio_age_results"]["longevity_score"] == 92


def test_bio_age_calculator_malformed():
    """Verify BioAgeCalculatorAgent falls back to chronological-age defaults
    when the LLM returns an empty/unparseable response."""
    agent = BioAgeCalculatorAgent()
    agent.client.call = MagicMock(return_value="")

    msg = agent.calculate_bio_age(SAMPLE_PROFILE)

    assert isinstance(msg, MCPMessage)
    # Fallback: biological_age == chronological_age from profile (23)
    assert msg.payload["bio_age_results"]["chronological_age"] == 23
    assert msg.payload["bio_age_results"]["biological_age"] == 23
    assert msg.payload["bio_age_results"]["pathway_focus"] == "General Mitochondrial Health"


# =====================================================================
# 4. ClinicalKinesiologyAgent Tests
# =====================================================================

def test_clinical_kinesiologist_happy_path():
    """Verify ClinicalKinesiologyAgent parses a well-formed exercise prescription JSON."""
    agent = ClinicalKinesiologyAgent()
    happy_response = {
        "weekly_split": "4-Day Upper/Lower Hypertrophy Split",
        "exercises": [
            {"name": "Goblet Squats", "sets": "4", "reps": "8", "intensity": "RPE 8", "instructions": "Breathe steadily."}
        ],
        "safety_precautions": ["Avoid Valsalva maneuver under heavy load."],
    }
    agent.client.call = MagicMock(return_value=json.dumps(happy_response))

    msg = agent.prescribe_exercise(SAMPLE_PROFILE)

    assert isinstance(msg, MCPMessage)
    assert msg.sender == "ClinicalKinesiologyAgent"
    assert msg.receiver == "CoordinatorAgent"
    assert msg.type == "TRAINING_PLAN"
    assert msg.payload["training_plan"]["weekly_split"] == "4-Day Upper/Lower Hypertrophy Split"
    assert msg.payload["training_plan"]["exercises"][0]["name"] == "Goblet Squats"


def test_clinical_kinesiologist_malformed():
    """Verify ClinicalKinesiologyAgent falls back to a default 3-Day conditioning
    split when the LLM returns truncated/invalid JSON."""
    agent = ClinicalKinesiologyAgent()
    agent.client.call = MagicMock(return_value="{ truncated JSON...")

    msg = agent.prescribe_exercise(SAMPLE_PROFILE)

    assert isinstance(msg, MCPMessage)
    assert msg.payload["training_plan"]["weekly_split"] == "3-Day General Conditioning Split"
    assert msg.payload["training_plan"]["exercises"][0]["name"] == "Bodyweight Squats"


# =====================================================================
# 5. ClinicalCritiqueAgent Tests
# =====================================================================

def test_clinical_critique_happy_path():
    """Verify ClinicalCritiqueAgent parses a well-formed peer review JSON response."""
    agent = ClinicalCritiqueAgent()
    happy_response = {
        "clinical_grade": "A+",
        "mechanics_explanation": "Detailed biological pathway mechanics.",
        "peer_review_notes": "Senior board review confirmed clinical safety.",
        "advanced_optimizations": ["Pair Vitamin D3 with fat-soluble meal."],
        "scientific_citations": ["AHA Cardiovascular Guidelines (2024)"],
    }
    agent.client.call = MagicMock(return_value=json.dumps(happy_response))

    msg = agent.critique_plan(
        profile=SAMPLE_PROFILE,
        targets={"calories": 2687},
        meal_plan={"breakfast": {}},
        training_plan={"weekly_split": "Test"},
        research_note="Mock Research",
        audit_report="Mock Audit",
    )

    assert isinstance(msg, MCPMessage)
    assert msg.sender == "ClinicalCritiqueAgent"
    assert msg.receiver == "CoordinatorAgent"
    assert msg.type == "CLINICAL_CRITIQUE_RESULT"
    assert msg.payload["critique"]["clinical_grade"] == "A+"
    assert len(msg.payload["critique"]["scientific_citations"]) == 1


def test_clinical_critique_malformed():
    """Verify ClinicalCritiqueAgent falls back to a default grade-A critique
    when the LLM returns plain text instead of valid JSON."""
    agent = ClinicalCritiqueAgent()
    agent.client.call = MagicMock(return_value="Random text string with no JSON formatting")

    msg = agent.critique_plan(
        profile=SAMPLE_PROFILE,
        targets={},
        meal_plan={},
        training_plan={},
        research_note="",
        audit_report="",
    )

    assert isinstance(msg, MCPMessage)
    assert msg.payload["critique"]["clinical_grade"] == "A"
    assert "WHO General Dietary and Physical Activity Guidelines" in msg.payload["critique"]["scientific_citations"]
