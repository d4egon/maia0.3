import logging
from typing import List, Dict, Optional
from core.memory_engine import MemoryEngine
from NLP.nlp_engine import NLP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractiveLearning:
    def __init__(self, memory_engine: MemoryEngine, nlp_engine: NLP):
        """
        Initialize InteractiveLearning with MemoryEngine and NLP for operations and analysis.

        :param memory_engine: An instance of MemoryEngine for managing database interactions.
        :param nlp_engine: An instance of NLP for text analysis.
        """
        self.memory_engine = memory_engine
        self.nlp_engine = nlp_engine

    def identify_knowledge_gaps(self, label: str = "Emotion") -> List[Dict]:
        """
        Identify nodes that lack an example context or other attributes.

        :param label: The label of the nodes to check (default "Emotion").
        :return: List of dictionaries containing ID and name of nodes with missing example contexts.
        """
        query = f"""
        MATCH (n:{label})
        WHERE NOT EXISTS(n.example_context)
        RETURN n.id AS id, n.name AS name
        """
        try:
            result = self.memory_engine.neo4j.run_query(query)
            logger.info(f"[KNOWLEDGE GAP] Identified {len(result)} knowledge gaps for {label}.")
            return result
        except Exception as e:
            logger.error(f"[ERROR] Error identifying knowledge gaps: {e}")
            raise

    def ask_questions(self, knowledge_gaps: List[Dict]):
        """
        Interactively ask for and update missing example contexts for given knowledge gaps.

        :param knowledge_gaps: List of dictionaries containing 'id' and 'name' of nodes needing context.
        """
        for gap in knowledge_gaps:
            try:
                logger.info(f"[QUERY] Querying user for example context for emotion: {gap['name']}")
                print(f"Help me understand '{gap['name']}' better!")
                example_context = self._get_valid_input(f"Can you give me an example of {gap['name']}? ")
                
                if example_context:
                    self._update_example_context(gap['id'], example_context)
                else:
                    logger.warning(f"[WARNING] No example context provided for {gap['name']}")
            except Exception as e:
                logger.error(f"[ERROR] Error asking questions for {gap['name']}: {e}")

    def _get_valid_input(self, prompt: str) -> str:
        """
        Get valid input from the user, ensuring it's not empty.

        :param prompt: The question or prompt to display to the user.
        :return: The user's input as a string.
        """
        while True:
            user_input = input(prompt).strip()
            if user_input:
                return user_input
            print("Please provide a non-empty response.")

    def _update_example_context(self, node_id: str, example_context: str):
        """
        Update the example context for a node in the database.

        :param node_id: The ID of the node to update.
        :param example_context: The example context to set for the node.
        """
        self.memory_engine.update_memory_metadata(node_id, {"example_context": example_context})
        logger.info(f"[UPDATE] Updated example context for node ID: {node_id}")

    def generate_follow_up_questions(self, theme: str) -> List[str]:
        """
        Generate follow-up questions based on a specific theme.
    
        :param theme: The theme to base questions on.
        :return: A list of follow-up questions.
        """
        questions = {
            "emotion": [
                "How does this make you feel?",
                "Can you share a similar experience?",
                "What emotions does this situation evoke?"
            ],
            "knowledge": [
                "What more would you like to know about this?",
                "Have you encountered this idea before?",
                "What questions do you have about this concept?"
            ]
        }
        return questions.get(theme, ["Can you tell me more?"])

    def refine_knowledge(self, node_id: str, feedback: str):
        """
        Refine knowledge based on user feedback.
    
        :param node_id: ID of the node to refine.
        :param feedback: User feedback to apply.
        """
        try:
            current_metadata = self.memory_engine.get_memory_metadata(node_id) or {}
            current_metadata["feedback"] = current_metadata.get("feedback", "") + " | " + feedback
            self.memory_engine.update_memory_metadata(node_id, current_metadata)
            logger.info(f"[KNOWLEDGE REFINED] Feedback added to node {node_id}.")
        except Exception as e:
            logger.error(f"[REFINE ERROR] Failed to refine node {node_id}: {e}")

    def learn_from_text(self, text: str):
        """
        Learn from new text input by identifying relevant themes or emotions and updating the memory system.

        :param text: The text from which to learn.
        """
        try:
            # Use NLP for theme detection
            _, intent = self.nlp_engine.process(text)
            detected_themes = ["emotion"] if "emotion" in intent else ["knowledge"]

            for theme in detected_themes:
                questions = self.generate_follow_up_questions(theme)
                for question in questions:
                    response = self._get_valid_input(question)
                    if response:
                        self._store_new_knowledge(text, theme, response)
        except Exception as e:
            logger.error(f"[LEARNING ERROR] Error learning from text: {e}")

    def _store_new_knowledge(self, text: str, theme: str, response: str):
        """
        Store newly learned information in the memory system.

        :param text: The original text.
        :param theme: The theme detected in the text.
        :param response: The user's response to the follow-up question.
        """
        metadata = {
            "theme": theme, 
            "response": response
        }
        self.memory_engine.create_memory_node(text, metadata, [theme])
        logger.info(f"[LEARNING] New knowledge stored for theme: {theme}")