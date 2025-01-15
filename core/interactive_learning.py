# File: /core/interactive_learning.py

import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractiveLearning:
    def __init__(self, graph_client):
        """
        Initialize InteractiveLearning with a graph client for database operations.

        :param graph_client: An instance of a class that can run Cypher queries.
        """
        self.graph_client = graph_client

    def identify_knowledge_gaps(self) -> List[Dict]:
        """
        Identify nodes (emotions) that lack an example context.

        :return: List of dictionaries containing ID and name of nodes with missing example contexts.
        """
        query = """
        MATCH (n:Emotion)
        WHERE NOT EXISTS(n.example_context)
        RETURN n.id AS id, n.name AS name
        """
        try:
            result = self.graph_client.run_query(query)
            logger.info(f"[KNOWLEDGE GAP] Identified {len(result)} knowledge gaps.")
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
        query = f"""
        MATCH (n:Emotion {{id: '{node_id}'}})
        SET n.example_context = '{example_context.replace("'", "''")}'
        """
        try:
            self.graph_client.run_query(query)
            logger.info(f"[UPDATE] Updated example context for node ID: {node_id}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to update example context for node ID {node_id}: {e}")
            raise

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
            query = f"""
            MATCH (n) WHERE n.id = '{node_id}'
            SET n.feedback = COALESCE(n.feedback, '') + ' | ' + '{feedback.replace("'", "''")}'
            """
            self.graph_client.run_query(query)
            logger.info(f"[KNOWLEDGE REFINED] Feedback added to node {node_id}.")
        except Exception as e:
            logger.error(f"[REFINE ERROR] Failed to refine node {node_id}: {e}")

    def learn_from_text(self, text: str):
        """
        Learn from new text input by identifying relevant themes or emotions and updating the graph database.

        :param text: The text from which to learn.
        """
        try:
            # Here you would implement logic to analyze text, perhaps using NLP techniques
            # This is a placeholder for actual text analysis
            detected_themes = ["emotion", "knowledge"]  # Example detection
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
        Store newly learned information in the graph database.

        :param text: The original text.
        :param theme: The theme detected in the text.
        :param response: The user's response to the follow-up question.
        """
        query = f"""
        CREATE (n:Learning {{text: '{text.replace("'", "''")}', 
                             theme: '{theme}', 
                             response: '{response.replace("'", "''")}'}})
        """
        try:
            self.graph_client.run_query(query)
            logger.info(f"[LEARNING] New knowledge stored for theme: {theme}")
        except Exception as e:
            logger.error(f"[STORAGE ERROR] Failed to store new knowledge: {e}")