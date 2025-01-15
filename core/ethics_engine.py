# Filename: /core/ethics_engine.py

import logging
from typing import Dict, Any, List
import json
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EthicsEngine:
    def __init__(self, db):
        self.db = db
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

            outcome = ethical_scenario["choices"].get(choice, {"outcome": "Invalid choice.", "weight": 0})
            lesson = ethical_scenario.get("lesson", "No lesson learned.")

            # Store Decision in Memory
            ethical_decision = {
                "scenario": scenario,
                "choice": choice,
                "outcome": outcome["outcome"],
                "outcome_weight": outcome.get("weight", 0),  # Added weight for evaluating choices
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
        Retrieve predefined ethical scenarios from the database or built-in list.

        :param scenario_text: The text of the scenario to look up.
        :return: Dictionary with scenario details or None if not found.
        """
        try:
            query = """
            MATCH (e:EthicsScenario {scenario: $scenario_text})
            RETURN e.scenario AS scenario, e.choices AS choices, e.lesson AS lesson
            """
            result = self.db.run_query(query, {"scenario_text": scenario_text})

            if result:
                logger.debug(f"[SCENARIO FOUND] {result[0]}")
                return result[0]

            # Fallback to Predefined Scenarios
            scenario = self.predefined_scenarios.get(scenario_text)
            if scenario:
                logger.debug(f"[SCENARIO FALLBACK] Using predefined scenario for: {scenario_text}")
                return scenario
            else:
                logger.warning(f"[SCENARIO NOT FOUND] Scenario '{scenario_text}' not in database or predefined scenarios.")
                return None
        except Exception as e:
            logger.error(f"[SCENARIO RETRIEVAL ERROR] {e}", exc_info=True)
            return None

    def store_ethics_memory(self, decision: Dict[str, Any]):
        """
        Store ethical decisions in the memory database.

        :param decision: Dictionary containing details of the ethical decision.
        """
        try:
            query = """
            CREATE (e:Ethics {scenario: $scenario, choice: $choice, outcome: $outcome, outcome_weight: $outcome_weight, lesson: $lesson})
            """
            self.db.run_query(query, decision)
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
                next_choices = [choice for choice, details in evaluation["choices"].items() if details["weight"] > 0.5]
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
                    # Optionally, you could write back to the JSON file or directly to Neo4j
                    self._save_predefined_scenarios()
                    logger.info(f"[SCENARIO ENHANCEMENT] Added new scenario: {scenario}")
                else:
                    logger.warning(f"[SCENARIO ENHANCEMENT] Scenario '{scenario}' already exists. Skipping.")
        except Exception as e:
            logger.error(f"[ENHANCE SCENARIOS ERROR] {e}", exc_info=True)

    def get_all_scenarios(self) -> List[Dict]:
        """
        Retrieve all ethical scenarios from both the database and predefined list.

        :return: List of all scenarios.
        """
        try:
            # Query all scenarios from Neo4j
            query = """
            MATCH (e:EthicsScenario)
            RETURN e.scenario AS scenario, e.choices AS choices, e.lesson AS lesson
            """
            db_scenarios = self.db.run_query(query)
            
            # Combine with predefined scenarios
            all_scenarios = []
            for scenario in db_scenarios:
                all_scenarios.append(scenario)
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
        Check the content for ethical compliance.

        :param content: The content to check.
        :return: True if the content passes the ethics check, False otherwise.
        """
        # Implement the ethics check logic
        # Here, you can add more complex logic to analyze the content
        prohibited_keywords = ["hate", "violence", "discrimination"]
        for keyword in prohibited_keywords:
            if keyword in content.lower():
                logger.warning(f"[ETHICS CHECK] Content contains prohibited keyword: {keyword}")
                return False
        return True