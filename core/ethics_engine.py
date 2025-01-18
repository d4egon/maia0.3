import logging
from datetime import time
from math import e
from typing import Dict, Any, List, Tuple
import json
from pathlib import Path
import re
from sentence_transformers import util
import numpy as np

from core.memory_engine import MemoryEngine


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EthicsEngine:
    def __init__(self, memory_engine: MemoryEngine):
        """
        Initialize EthicsEngine with a MemoryEngine for managing ethical scenarios and decisions.

        :param memory_engine: An instance of MemoryEngine to interact with the memory system.
        """
        self.memory_engine = memory_engine
        self.predefined_scenarios = self._load_predefined_scenarios()

    def _load_predefined_scenarios(self) -> Dict[str, Dict]:
        """
        Load predefined ethical scenarios from a JSON file.

        :return: Dictionary of ethical scenarios.
        """
        scenario_file = Path(__file__).parent / "ethics_scenarios.json"
        if scenario_file.exists():
            with scenario_file.open() as f:
                return json.load(f)
        logger.warning("[LOAD SCENARIOS] No predefined scenarios file found.")
        return {}

    def _save_predefined_scenarios(self):
        """
        Save predefined ethical scenarios to a JSON file.
        """
        scenario_file = Path(__file__).parent / "ethics_scenarios.json"
        with scenario_file.open('w') as f:
            json.dump(self.predefined_scenarios, f, indent=4)
        logger.info("[SAVE SCENARIOS] Predefined scenarios saved to file.")

    def evaluate_decision(self, scenario: str, choice: str) -> Dict[str, Any]:
        """
        Evaluate an ethical decision based on predefined scenarios and dynamic context analysis.

        :param scenario: Text describing the ethical scenario to evaluate.
        :param choice: The choice made within the scenario.
        :return: Dictionary containing the evaluation results or error message.
        """
        logger.info(f"[EVALUATION] Evaluating scenario: {scenario}, Choice: {choice}")

        try:
            ethical_scenario = self.get_scenario(scenario)

            if not ethical_scenario:
                logger.warning(f"[EVALUATION] Scenario not found: {scenario}")
                return {"error": "Scenario not found."}

            outcome = ethical_scenario.get("choices", {}).get(choice, {"outcome": "Invalid choice.", "weight": 0})
            lesson = ethical_scenario.get("lesson", "No lesson learned.")

            # Store Decision in Memory
            ethical_decision = {
                "type": "ethical_decision",
                "scenario": scenario,
                "choice": choice,
                "outcome": outcome["outcome"],
                "outcome_weight": outcome.get("weight", 0),
                "lesson": lesson,
            }
            self.store_ethics_memory(ethical_decision)

            logger.info(f"[EVALUATION RESULT] {ethical_decision}")
            return ethical_decision
        except Exception as e:
            logger.error(f"[EVALUATION ERROR] {e}", exc_info=True)
            return {"error": "An error occurred during evaluation."}

    def get_scenario(self, scenario_text: str) -> Dict:
        """
        Retrieve predefined ethical scenarios from memory or built-in list.

        :param scenario_text: The text of the scenario to look up.
        :return: Dictionary with scenario details or None if not found.
        """
        try:
            # First, try to find the scenario in memory
            scenario_in_memory = self.memory_engine.search_memories([scenario_text])
            if scenario_in_memory:
                return scenario_in_memory[0]

            # If not found in memory, check predefined scenarios
            scenario = self.predefined_scenarios.get(scenario_text)
            if scenario:
                logger.debug(f"[SCENARIO FOUND] Using predefined scenario for: {scenario_text}")
                return scenario
            else:
                logger.warning(f"[SCENARIO NOT FOUND] Scenario not found: {scenario_text}")
        except Exception as e:
            logger.error(f"[GET SCENARIO ERROR] {e}", exc_info=True)
            return None
                               
    
    def store_ethics_memory(self, decision: Dict[str, Any]):
        """
        Store ethical decisions in the memory database.

        :param decision: Dictionary containing details of the ethical decision.
        """
        try:
            self.memory_engine.create_memory_node(
                decision["outcome"], 
                {
                    "type": "ethical_decision",
                    "scenario": decision["scenario"],
                    "choice": decision["choice"],
                    "outcome_weight": decision["outcome_weight"],
                    "lesson": decision["lesson"],
                    "timestamp": decision.get("timestamp", time.time())
                },
                [decision["scenario"], decision["choice"]]
            )
            logger.info(f"[MEMORY STORED] Ethical decision stored: {decision}")
        except Exception as e:
            logger.error(f"[MEMORY STORAGE ERROR] {e}", exc_info=True)

    def recursive_evaluation(self, scenario: str, depth: int = 3) -> Dict[str, Any]:
        """
        Perform a recursive evaluation of an ethical scenario to explore deeper layers of decision-making.
    
        :param scenario: Ethical scenario to evaluate.
        :param depth: Depth of recursion.
        :return: Dictionary with detailed evaluation results.
        """
        if depth <= 0:
            return {"result": "Depth limit reached. No further analysis possible."}
    
        try:
            evaluation = self.evaluate_decision(scenario, choice="explore")
            if "error" not in evaluation:
                # Implement more nuanced decision paths
                choices = evaluation.get("choices", {})
                next_choices = [choice for choice, details in choices.items() if details["weight"] > 0.5]
                if next_choices:
                    next_choice = next_choices[0]  # Choose the first valid choice for simplicity
                    next_scenario = evaluation.get("next_scenario", scenario)  # Assume 'explore' leads to a new or same scenario
                    next_layer = self.recursive_evaluation(next_scenario, depth - 1)
                    evaluation["next_layer"] = next_layer
            return evaluation
        except Exception as e:
            logger.error(f"[RECURSIVE ERROR] {e}", exc_info=True)
            return {"error": "An error occurred during recursive evaluation."}

    def resolve_conflicts(self, scenario: str, choices: List[str]) -> str:
        """
        Resolve conflicts by evaluating multiple choices and recommending the optimal path.
    
        :param scenario: Ethical scenario to evaluate.
        :param choices: List of choices to evaluate.
        :return: Recommended choice.
        """
        try:
            evaluations = {choice: self.evaluate_decision(scenario, choice) for choice in choices}
            # Filter out invalid choices and sort by outcome weight
            valid_choices = {choice: eval_data for choice, eval_data in evaluations.items() if "error" not in eval_data}
            if not valid_choices:
                logger.warning(f"[CONFLICT RESOLUTION] No valid choices for scenario '{scenario}'")
                return "No valid choice available."
            best_choice = max(valid_choices, key=lambda x: valid_choices[x].get("outcome_weight", 0))
            logger.info(f"[CONFLICT RESOLUTION] Best choice for '{scenario}': {best_choice}")
            return best_choice
        except Exception as e:
            logger.error(f"[CONFLICT RESOLUTION ERROR] {e}", exc_info=True)
            return "Error in resolving conflicts."

    def enhance_scenarios(self, new_scenarios: Dict[str, Dict]) -> None:
        """
        Enhance the predefined scenarios with new ethical scenarios.

        :param new_scenarios: A dictionary of new scenarios to add.
        """
        try:
            for scenario, details in new_scenarios.items():
                if scenario not in self.predefined_scenarios:
                    self.predefined_scenarios[scenario] = details
                    self._save_predefined_scenarios()  # Save updated scenarios to file
                    # Store new scenario in memory
                    self.memory_engine.create_memory_node(
                        scenario, 
                        {
                            "type": "ethical_scenario",
                            "choices": details.get("choices", {}),
                            "lesson": details.get("lesson", "No lesson provided")
                        },
                        [scenario]
                    )
                    logger.info(f"[SCENARIO ENHANCEMENT] Added new scenario: {scenario}")
                else:
                    logger.warning(f"[SCENARIO ENHANCEMENT] Scenario '{scenario}' already exists. Skipping.")
        except Exception as e:
            logger.error(f"[ENHANCE SCENARIOS ERROR] {e}", exc_info=True)

    def get_all_scenarios(self) -> List[Dict]:
        """
        Retrieve all ethical scenarios from both memory and predefined list.

        :return: List of all scenarios.
        """
        try:
            # Query all scenarios from memory
            memory_scenarios = self.memory_engine.retrieve_all_memories_by_label("ethical_scenario")
            
            # Combine with predefined scenarios
            all_scenarios = []
            for scenario in memory_scenarios:
                all_scenarios.append({
                    "scenario": scenario["content"],
                    "choices": scenario["metadata"].get("choices", {}),
                    "lesson": scenario["metadata"].get("lesson", "No lesson provided")
                })
            for scenario, details in self.predefined_scenarios.items():
                if scenario not in [s['scenario'] for s in all_scenarios]:
                    all_scenarios.append({"scenario": scenario, **details})

            logger.info(f"[SCENARIOS RETRIEVED] Total scenarios: {len(all_scenarios)}")
            return all_scenarios
        except Exception as e:
            logger.error(f"[SCENARIOS RETRIEVAL ERROR] {e}", exc_info=True)
            return []

    def check(self, content: str) -> bool:
        """
        Check the content for ethical compliance using advanced NLP techniques.

        :param content: The content to check.
        :return: True if the content passes the ethics check, False otherwise.
        """
        try:
            # Basic keyword check for immediate red flags
            prohibited_keywords = ["hate", "violence", "discrimination", "harm", "abuse"]
            if any(keyword in content.lower() for keyword in prohibited_keywords):
                logger.warning(f"[ETHICS CHECK] Content contains prohibited keywords.")
                return False

            # Semantic analysis for ethical alignment
            content_embedding = self.memory_engine.model.encode([content])[0]
            
            # Check against ethical principles stored in memory
            ethical_principles = self.memory_engine.retrieve_all_memories_by_label("ethical_principle")
            if ethical_principles:
                principle_embeddings = [np.array(principle['embedding']) for principle in ethical_principles]
                similarities = util.cos_sim(content_embedding, np.vstack(principle_embeddings))
                if np.max(similarities) > 0.8:  # Threshold for alignment with ethical principles
                    logger.info(f"[ETHICS CHECK] Content aligns with ethical principles in memory.")
                    return True

            # Analyze sentiment and tone
            nlp = self.memory_engine.model.nlp  # Assuming NLP capabilities are within the sentence transformer model
            doc = nlp(content)
            sentiment_score = doc.sentiment  # Assuming sentiment is available in the NLP model

            if sentiment_score < -0.2:  # Negative sentiment might indicate ethical issues
                logger.warning(f"[ETHICS CHECK] Content shows negative sentiment, potentially raising ethical concerns.")
                return False

            # Check for ethical concepts in the content
            ethical_concepts = ["justice", "fairness", "equality", "compassion", "integrity", "honesty", "respect"]
            concept_matches = sum(1 for concept in ethical_concepts if concept in content.lower())
            if concept_matches > 0:
                logger.info(f"[ETHICS CHECK] Content contains {concept_matches} positive ethical concepts.")
                return True

            # Advanced pattern matching for ethical breaches
            patterns = [
                r'\b(cheat|lie|steal|deceive)\b',  # Words associated with unethical behavior
                r'(dishonest|y|unfair|immoral|unethical)',  # Adjectives suggesting unethical actions
                r'(exploit|manipulate|coerce)',  # Verbs indicating potential ethical violations
            ]
            if any(re.search(pattern, content.lower()) for pattern in patterns):
                logger.warning(f"[ETHICS CHECK] Content matches patterns indicating ethical breaches.")
                return False

            # Check against predefined scenarios for context
            for scenario, details in self.predefined_scenarios.items():
                if any(word in content.lower() for word in scenario.lower().split()):
                    # Check if the content supports or contradicts the lesson
                    lesson = details.get("lesson", "").lower()
                    lesson_embedding = self.memory_engine.model.encode([lesson])[0]
                    similarity = util.cos_sim(content_embedding, lesson_embedding).item()

                    if similarity > 0.7:  # High similarity with lesson
                        logger.info(f"[ETHICS CHECK] Content supports the ethical lesson of scenario: {scenario}")
                        return True
                    else:
                        logger.warning(f"[ETHICS CHECK] Content contradicts or is unrelated to the lesson of scenario: {scenario}")
                        return False
                    
            # If no clear ethical violation or alignment found, default to true but with a note
            logger.info("[ETHICS CHECK] Content does not trigger explicit ethical flags; defaulting to ethically compliant.")
            return True
        except Exception as e:
            print(f"An error occurred: {e}")
            logger.error(f"[ETHICS CHECK ERROR] {e}", exc_info=True)
            return False  # Err on the side of caution if there's an error in checking