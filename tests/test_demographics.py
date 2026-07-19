import json
from unittest.mock import MagicMock
from agents.health_agents import NutriPlannerAgent, ClinicalAnalyzerAgent

def test_demographics_casting():
    """Verify that NutriPlannerAgent safely casts demographics strings to numeric values during BMR target calculation"""
    agent = NutriPlannerAgent()
    
    # Mock LLM Client call to return dummy JSON meal plan and prevent real calls
    dummy_json = '{"breakfast": {"name": "Test Meal", "calories": 500, "protein": 30, "carbs": 50, "fats": 10, "ingredients": [], "instructions": ""}}'
    agent.client.call = MagicMock(return_value=dummy_json)
    
    # Define demographics dict containing numeric values formatted as strings
    profile = {
        "demographics": {
            "age": "23",
            "weight_kg": "75",
            "height_cm": "175",
            "gender": "Male",
            "activity_level": "Moderate"
        },
        "goals": ["Optimize overall health"],
        "allergies": [],
        "medical_conditions": []
    }
    
    # Call generate_plan, which calculates calorie/macro targets
    agent.generate_plan(profile, "Research note details")
    
    # Verify the mocked call was made and that correct macro calculation limits were passed in prompt
    prompt_arg = agent.client.call.call_args[0][0]
    
    # Target macros calculated with Mifflin-St Jeor:
    # BMR = 10 * 75 + 6.25 * 175 - 5 * 23 + 5 = 1733.75 kcal
    # TDEE for Moderate = 1733.75 * 1.55 = 2687 kcal
    assert "2687kcal" in prompt_arg or "2687.31kcal" in prompt_arg or "2687" in prompt_arg
    assert "167g" in prompt_arg or "167" in prompt_arg # protein
    assert "302g" in prompt_arg or "302" in prompt_arg # carbs
    assert "89g" in prompt_arg or "89" in prompt_arg # fats


def test_demographics_validation():
    """Verify extraction-layer validation of demographics, confirming correct type conversions and warnings tracking"""
    agent = ClinicalAnalyzerAgent()
    
    # Test Scenario 1: Mixed inputs (valid strings, garbage, missing fields)
    mock_payload_1 = {
        "demographics": {
            "name": "N Satyendra",
            "age": "23",
            "weight_kg": "Unknown",
            "height_cm": "",
            "activity_level": "Active"
        },
        "goals": ["Optimize overall health"],
        "allergies": [],
        "medical_conditions": [],
        "biomarkers": []
    }
    
    agent.client.call = MagicMock(return_value=json.dumps(mock_payload_1))
    
    msg = agent.analyze_health_data(["Dummy text chunk"])
    payload = msg.payload
    profile = payload["profile"]
    demographics = profile["demographics"]
    
    # Assertions
    assert isinstance(demographics["age"], int)
    assert demographics["age"] == 23
    assert isinstance(demographics["weight_kg"], float)
    assert demographics["weight_kg"] == 75.0  # defaulted
    assert isinstance(demographics["height_cm"], float)
    assert demographics["height_cm"] == 175.0  # defaulted
    assert demographics["gender"] == "Male"  # defaulted from missing
    
    warning_text = payload["extraction_error"]
    assert "weight_kg" in warning_text
    assert "height_cm" in warning_text
    assert "gender" in warning_text
    assert "age" not in warning_text
    assert payload["extraction_incomplete"] is True

    # Test Scenario 2: Genuinely non-numeric garbage for all fields
    mock_payload_2 = {
        "demographics": {
            "name": "Jane Doe",
            "age": "N/A",
            "weight_kg": "N/A",
            "height_cm": "N/A"
        }
    }
    
    agent.client.call = MagicMock(return_value=json.dumps(mock_payload_2))
    
    msg = agent.analyze_health_data(["Dummy text chunk"])
    payload = msg.payload
    demographics = payload["profile"]["demographics"]
    
    assert isinstance(demographics["age"], int)
    assert demographics["age"] == 30
    assert isinstance(demographics["weight_kg"], float)
    assert demographics["weight_kg"] == 75.0
    assert isinstance(demographics["height_cm"], float)
    assert demographics["height_cm"] == 175.0
    
    warning_text = payload["extraction_error"]
    assert "age" in warning_text
    assert "weight_kg" in warning_text
    assert "height_cm" in warning_text
    assert payload["extraction_incomplete"] is True
