from unittest.mock import MagicMock
from agents.coordinator_agent import CoordinatorAgent
from agents.mcp import MCPMessage

def test_coordinator_resilience():
    """Verify CoordinatorAgent recovers from parallel agent failures and logs degraded statuses correctly"""
    coordinator = CoordinatorAgent()
    
    # 1. Mock BioAgeCalculatorAgent to raise an exception
    coordinator.bio_age_calculator.calculate_bio_age = MagicMock(
        side_effect=RuntimeError("Simulated BioAge calculator database timeout")
    )
    
    # 2. Mock other parallel agents to return successful messages
    coordinator.retrieval_agent.embed_and_retrieve = MagicMock(
        return_value=MCPMessage(
            sender="RetrievalAgent",
            receiver="CoordinatorAgent",
            type="RETRIEVED",
            payload={"retrieved_context": ["Mock vector text"]}
        )
    )
    coordinator.web_researcher.perform_clinical_search = MagicMock(
        return_value=MCPMessage(
            sender="WebResearchAgent",
            receiver="CoordinatorAgent",
            type="RESEARCH",
            payload={"research_note": "Mocked clinical research details"}
        )
    )
    coordinator.clinical_kinesiologist.prescribe_exercise = MagicMock(
        return_value=MCPMessage(
            sender="ClinicalKinesiologyAgent",
            receiver="CoordinatorAgent",
            type="TRAINING",
            payload={"training_plan": {"weekly_split": "Mock exercise program"}}
        )
    )
    
    # 3. Mock sequential agents to process normally
    coordinator.nutri_planner.generate_plan = MagicMock(
        return_value=MCPMessage(
            sender="NutriPlannerAgent",
            receiver="CoordinatorAgent",
            type="MEALS",
            payload={"targets": {"calories": 2000}, "meal_plan": {"breakfast": "Oatmeal"}}
        )
    )
    coordinator.safety_auditor.audit_and_correct = MagicMock(
        return_value=MCPMessage(
            sender="SafetyAuditorAgent",
            receiver="CoordinatorAgent",
            type="AUDIT",
            payload={
                "final_meal_plan": {"breakfast": "Oatmeal"},
                "final_training_plan": {"weekly_split": "Mock exercise program"},
                "audit_report": "Audit passed cleanly",
                "corrections_made": []
            }
        )
    )
    coordinator.clinical_critique.critique_plan = MagicMock(
        return_value=MCPMessage(
            sender="ClinicalCritiqueAgent",
            receiver="CoordinatorAgent",
            type="CRITIQUE",
            payload={"critique": {"grade": "A"}}
        )
    )
    coordinator.llm_agent.generate_response = MagicMock(
        return_value=MCPMessage(
            sender="LLMResponseAgent",
            receiver="CoordinatorAgent",
            type="RESPONSE",
            payload={"answer": "Here is your integrated health review, but Biological Age metrics were unavailable."}
        )
    )
    
    profile = {
        "demographics": {"name": "Test User", "age": 30, "weight_kg": 75, "height_cm": 170},
        "biomarkers": []
    }
    
    result = coordinator.process_health_query("Generate my health plan", profile)
    
    assert result["success"] is True
    assert "BioAgeCalculatorAgent" in result["degraded_agents"]
    assert result["bio_age_results"] is None
    assert "Simulated BioAge" not in result["answer"]


def test_clinical_grounding_flag():
    """Verify that coordinator detects empty research notes when WebResearchAgent fails and populates grounding alerts"""
    coordinator = CoordinatorAgent()
    
    # Mock retrieval, kinesiology, nutritional planners, safety audit, and critique
    coordinator.retrieval_agent.embed_and_retrieve = MagicMock(
        return_value=MCPMessage(sender="R", receiver="C", type="T", payload={"retrieved_context": []})
    )
    coordinator.clinical_kinesiologist.prescribe_exercise = MagicMock(
        return_value=MCPMessage(sender="K", receiver="C", type="T", payload={"training_plan": {}})
    )
    coordinator.nutri_planner.generate_plan = MagicMock(
        return_value=MCPMessage(sender="N", receiver="C", type="T", payload={"targets": {}, "meal_plan": {}})
    )
    coordinator.safety_auditor.audit_and_correct = MagicMock(
        return_value=MCPMessage(sender="S", receiver="C", type="T", payload={"final_meal_plan": {}, "final_training_plan": {}})
    )
    coordinator.clinical_critique.critique_plan = MagicMock(
        return_value=MCPMessage(sender="Cr", receiver="C", type="T", payload={"critique": {}})
    )
    coordinator.llm_agent.generate_response = MagicMock(
        return_value=MCPMessage(sender="L", receiver="C", type="T", payload={"answer": "Mock Answer"})
    )
    
    profile = {
        "demographics": {"name": "Test User", "age": 30, "weight_kg": 75, "height_cm": 170},
        "biomarkers": []
    }
    
    # Test Scenario 1: WebResearchAgent fails -> empty research_note
    coordinator.web_researcher.perform_clinical_search = MagicMock(
        side_effect=RuntimeError("Web Search Error")
    )
    coordinator.bio_age_calculator.calculate_bio_age = MagicMock(
        return_value=MCPMessage(sender="B", receiver="C", type="T", payload={"bio_age_results": {}})
    )
    
    result1 = coordinator.process_health_query("How is my blood?", profile)
    
    assert result1["reduced_clinical_grounding"] is True
    assert result1["clinical_grounding_explanation"] is not None
    
    # Test Scenario 2: WebResearchAgent succeeds -> real research_note
    coordinator.web_researcher.perform_clinical_search = MagicMock(
        return_value=MCPMessage(sender="W", receiver="C", type="T", payload={"research_note": "Found NCBI PubMed guidelines"})
    )
    coordinator.bio_age_calculator.calculate_bio_age = MagicMock(
        return_value=MCPMessage(sender="B", receiver="C", type="T", payload={"bio_age_results": {}})
    )
    
    result2 = coordinator.process_health_query("How is my blood?", profile)
    
    assert result2["reduced_clinical_grounding"] is False
    assert result2["clinical_grounding_explanation"] is None
