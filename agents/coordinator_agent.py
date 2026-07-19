from dataclasses import dataclass, field
from typing import List, Dict, Any
import time
from .ingestion_agent import IngestionAgent
from .retrieval_agent import RetrievalAgent
from .llm_response_agent import LLMResponseAgent
from .mcp import MCPMessage
from .health_agents import ClinicalAnalyzerAgent, WebResearchAgent, NutriPlannerAgent, SafetyAuditorAgent, ClinicalCritiqueAgent, BioAgeCalculatorAgent, ClinicalKinesiologyAgent

@dataclass
class CoordinatorAgent:
    ingestion_agent: IngestionAgent = field(default_factory=IngestionAgent)
    retrieval_agent: RetrievalAgent = field(default_factory=RetrievalAgent)
    llm_agent: LLMResponseAgent = field(default_factory=LLMResponseAgent)
    clinical_analyzer: ClinicalAnalyzerAgent = field(default_factory=ClinicalAnalyzerAgent)
    web_researcher: WebResearchAgent = field(default_factory=WebResearchAgent)
    nutri_planner: NutriPlannerAgent = field(default_factory=NutriPlannerAgent)
    safety_auditor: SafetyAuditorAgent = field(default_factory=SafetyAuditorAgent)
    clinical_critique: ClinicalCritiqueAgent = field(default_factory=ClinicalCritiqueAgent)
    bio_age_calculator: BioAgeCalculatorAgent = field(default_factory=BioAgeCalculatorAgent)
    clinical_kinesiologist: ClinicalKinesiologyAgent = field(default_factory=ClinicalKinesiologyAgent)
    
    mcp_trace: List[Dict[str, Any]] = field(default_factory=list)

    def analyze_document(self, file_paths: List[str]) -> Dict[str, Any]:
        """Runs the ingestion and clinical markers extraction pipeline on uploaded reports"""
        self.mcp_trace = []
        try:
            # 1. Ingestion Agent
            ingest_msg = self.ingestion_agent.parse_documents(file_paths)
            self.mcp_trace.append(ingest_msg.to_dict())
            chunks = ingest_msg.payload.get("chunks", [])
            failed_files = ingest_msg.payload.get("failed_files", [])
            
            if not chunks:
                return {"error": "No text extracted from document"}
                
            # 2. Clinical Analyzer Agent
            clinical_msg = self.clinical_analyzer.analyze_health_data(chunks)
            self.mcp_trace.append(clinical_msg.to_dict())
            
            profile = clinical_msg.payload.get("profile", {})
            extraction_incomplete = clinical_msg.payload.get("extraction_incomplete", False)
            extraction_error = clinical_msg.payload.get("extraction_error")
            return {
                "success": True,
                "profile": profile,
                "chunks": chunks,
                "failed_files": failed_files,
                "extraction_incomplete": extraction_incomplete,
                "extraction_error": extraction_error,
                "mcp_trace": self.mcp_trace
            }
        except Exception as e:
            error_msg = MCPMessage(
                sender="CoordinatorAgent",
                receiver="UI",
                type="ERROR",
                payload={"error": str(e)}
            )
            self.mcp_trace.append(error_msg.to_dict())
            return {"error": str(e), "mcp_trace": self.mcp_trace}

    def process_health_query(self, query: str, profile: Dict[str, Any], retrieved_chunks: List[str] = None) -> Dict[str, Any]:
        """Runs the Plan-Reason-Audit loop for health planning and questions in parallel threads"""
        self.mcp_trace = []
        try:
            import logging
            import re
            logger = logging.getLogger(__name__)
            
            # Helper to time and log parallel agent contributions
            def timed_submit(agent_name, func, *args, **kwargs):
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration = time.perf_counter() - start
                    logger.info(f"⏱️ {agent_name} contribution: {duration:.2f}s")
            
            # Detect casual greetings/non-clinical queries to respond instantly and prevent CPU thrashing
            clean_q = re.sub(r'[^\w\s]', '', query.strip().lower())
            greetings = {"hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening", "howdy", "yo", "hi there", "hello there", "how are you", "who are you", "what is this"}
            is_greeting = clean_q in greetings or (len(clean_q.split()) <= 3 and any(w in {"hi", "hello", "hey", "hola", "yo"} for w in clean_q.split()))
            
            if is_greeting:
                greeting_msg = (
                    "Hello! I am NutriMind AI, your advanced agentic clinical health and diet companion.\n\n"
                    "How can I assist you with your health and fitness goals today? You can:\n"
                    "• **Upload health/clinical documents** (PDFs, reports, summaries) in the Dashboard to extract biomarkers and analyze your profile.\n"
                    "• **Ask specific questions** about your biomarkers, nutrient targets, or recommended clinical dietary plans.\n"
                    "• **Request a customized diet plan** and physical exercise split tailored to your physiological profile."
                )
                logger.info(f"Casual greeting detected: '{query}'. Responding instantly.")
                return {
                    "success": True,
                    "answer": greeting_msg,
                    "meal_plan": {},
                    "training_plan": {},
                    "targets": None,
                    "audit_report": "Casual greeting detected. Safety audit bypassed.",
                    "corrections": [],
                    "bio_age_results": None,
                    "critique": None,
                    "mcp_trace": [
                        {
                            "sender": "CoordinatorAgent",
                            "receiver": "UI",
                            "type": "GREETING",
                            "payload": {"answer": greeting_msg}
                        }
                    ],
                    "degraded_agents": [],
                    "reduced_clinical_grounding": False,
                    "clinical_grounding_explanation": None
                }

            from concurrent.futures import ThreadPoolExecutor

            conditions = profile.get("medical_conditions", [])
            biomarkers = [b.get("name") for b in profile.get("biomarkers", []) if isinstance(b, dict) and b.get("name")]
            search_context = conditions + biomarkers
            degraded_agents = []
            
            # Step 1: Execute local ChromaDB retrieval, web research, bio-age calculation, and kinesiology split in parallel!
            with ThreadPoolExecutor(max_workers=4) as executor:
                if retrieved_chunks is None:
                    logger.info("⚡ Parallel Thread A: Starting vector retrieval from local ChromaDB...")
                    retrieval_future = executor.submit(
                        timed_submit,
                        "RetrievalAgent",
                        self.retrieval_agent.embed_and_retrieve,
                        chunks=[],  # No new chunks added during query
                        query=query,
                        top_k=5
                    )
                else:
                    retrieval_future = None

                logger.info("⚡ Parallel Thread B: Dispatching clinical search to WebResearchAgent...")
                research_future = executor.submit(
                    timed_submit,
                    "WebResearchAgent",
                    self.web_researcher.perform_clinical_search,
                    search_context,
                    query
                )

                logger.info("⚡ Parallel Thread C: Dispatching biological age longevity metrics calculation...")
                bio_age_future = executor.submit(
                    timed_submit,
                    "BioAgeCalculatorAgent",
                    self.bio_age_calculator.calculate_bio_age,
                    profile
                )

                logger.info("⚡ Parallel Thread D: Dispatching clinical fitness routine prescription...")
                exercise_future = executor.submit(
                    timed_submit,
                    "ClinicalKinesiologyAgent",
                    self.clinical_kinesiologist.prescribe_exercise,
                    profile
                )

                # Wait for threads and collect results with individual try/except wrappers
                
                # A. WebResearchAgent
                try:
                    research_msg = research_future.result()
                    self.mcp_trace.append(research_msg.to_dict())
                    research_note = research_msg.payload.get("research_note", "")
                except Exception as e:
                    logger.error(f"❌ WebResearchAgent failed: {e}", exc_info=True)
                    degraded_agents.append("WebResearchAgent")
                    research_note = ""

                # B. BioAgeCalculatorAgent
                try:
                    bio_age_msg = bio_age_future.result()
                    self.mcp_trace.append(bio_age_msg.to_dict())
                    bio_age_results = bio_age_msg.payload.get("bio_age_results", {})
                except Exception as e:
                    logger.error(f"❌ BioAgeCalculatorAgent failed: {e}", exc_info=True)
                    degraded_agents.append("BioAgeCalculatorAgent")
                    bio_age_results = None

                # C. ClinicalKinesiologyAgent
                try:
                    exercise_msg = exercise_future.result()
                    self.mcp_trace.append(exercise_msg.to_dict())
                    proposed_training = exercise_msg.payload.get("training_plan", {})
                except Exception as e:
                    logger.error(f"❌ ClinicalKinesiologyAgent failed: {e}", exc_info=True)
                    degraded_agents.append("ClinicalKinesiologyAgent")
                    proposed_training = {}

                # D. RetrievalAgent
                if retrieval_future:
                    try:
                        retrieval_msg = retrieval_future.result()
                        self.mcp_trace.append(retrieval_msg.to_dict())
                        active_retrieved_context = retrieval_msg.payload.get("retrieved_context", [])
                    except Exception as e:
                        logger.error(f"❌ RetrievalAgent failed: {e}", exc_info=True)
                        degraded_agents.append("RetrievalAgent")
                        active_retrieved_context = []
                else:
                    active_retrieved_context = retrieved_chunks or []

            reduced_clinical_grounding = not research_note and "WebResearchAgent" in degraded_agents
            clinical_grounding_explanation = "Meal plan generated without literature-backed clinical guidelines due to a research service issue." if reduced_clinical_grounding else None

            # Step 2: NutriPlanner (sequential, feeds on clinical search note guidelines)
            logger.info("⚡ Step 2: Generating customized nutritional guidelines and calorie/macro calculations...")
            start_seq = time.perf_counter()
            planner_msg = self.nutri_planner.generate_plan(profile, research_note)
            logger.info(f"⏱️ NutriPlannerAgent contribution: {time.perf_counter() - start_seq:.2f}s")
            self.mcp_trace.append(planner_msg.to_dict())
            targets = planner_msg.payload.get("targets", {})
            proposed_meals = planner_msg.payload.get("meal_plan", {})
            
            # Step 3: Safety Auditor (sequential, critiques the proposed menu & training split against limits)
            logger.info("⚡ Step 3: Performing critical allergen and kinesiological safety audit...")
            start_seq = time.perf_counter()
            audit_msg = self.safety_auditor.audit_and_correct(profile, targets, proposed_meals, proposed_training)
            logger.info(f"⏱️ SafetyAuditorAgent contribution: {time.perf_counter() - start_seq:.2f}s")
            self.mcp_trace.append(audit_msg.to_dict())
            
            final_meals = audit_msg.payload.get("final_meal_plan", proposed_meals)
            final_training = audit_msg.payload.get("final_training_plan", proposed_training)
            audit_report = audit_msg.payload.get("audit_report", "")
            corrections = audit_msg.payload.get("corrections_made", [])
            
            # Step 3.5: Clinical Critique (sequential, critiques the final meal + training plans and assigns a grade)
            logger.info("⚡ Step 3.5: Conducting senior medical board peer review critique...")
            start_seq = time.perf_counter()
            critique_msg = self.clinical_critique.critique_plan(profile, targets, final_meals, final_training, research_note, audit_report)
            logger.info(f"⏱️ ClinicalCritiqueAgent contribution: {time.perf_counter() - start_seq:.2f}s")
            self.mcp_trace.append(critique_msg.to_dict())
            critique = critique_msg.payload.get("critique", {})
            
            # Step 4: Generate LLM Final Response detailing findings and suggestions
            logger.info("⚡ Step 4: Synthesizing final medical summary response...")
            import json
            context_feed = [
                f"Extracted Profile: {profile}",
                f"Clinical Guidelines:\n{research_note}",
                f"Biological Age & Longevity calculations:\n{json.dumps(bio_age_results)}",
                f"Safety Audit Report:\n{audit_report}",
                f"Corrections made: {corrections}",
                f"Senior Board Critique Review:\n{json.dumps(critique)}"
            ]
            if active_retrieved_context:
                context_feed.extend(active_retrieved_context)
                
            start_seq = time.perf_counter()
            llm_msg = self.llm_agent.generate_response(context_feed, query)
            logger.info(f"⏱️ LLMResponseAgent contribution: {time.perf_counter() - start_seq:.2f}s")
            self.mcp_trace.append(llm_msg.to_dict())
            answer = llm_msg.payload.get("answer", "Could not generate final response.")
            
            return {
                "success": True,
                "answer": answer,
                "meal_plan": final_meals,
                "training_plan": final_training,
                "targets": targets,
                "audit_report": audit_report,
                "corrections": corrections,
                "bio_age_results": bio_age_results,
                "critique": critique,
                "mcp_trace": self.mcp_trace,
                "degraded_agents": degraded_agents,
                "reduced_clinical_grounding": reduced_clinical_grounding,
                "clinical_grounding_explanation": clinical_grounding_explanation
            }
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"❌ Error coordinating agents: {e}", exc_info=True)
            error_msg = MCPMessage(
                sender="CoordinatorAgent",
                receiver="UI",
                type="ERROR",
                payload={"error": "An internal error occurred during coordination."}
            )
            self.mcp_trace.append(error_msg.to_dict())
            return {
                "success": False,
                "answer": "I encountered an issue while generating your health analysis. Please try again, or rephrase your question.",
                "mcp_trace": self.mcp_trace,
                "degraded_agents": [],
                "reduced_clinical_grounding": False,
                "clinical_grounding_explanation": None
            }
