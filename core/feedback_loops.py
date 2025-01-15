# File: /core/feedback_loops.py

import logging
from typing import Dict, List, Any
from core.neo4j_connector import Neo4jConnector
from core.memory_engine import MemoryEngine
from NLP.consciousness_engine import ConsciousnessEngine  # Assuming this class exists
from core.semantic_builder import SemanticBuilder  # For semantic analysis of feedback

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeedbackLoops:
    def __init__(self, graph_client: Neo4jConnector, memory_engine: MemoryEngine):
        self.graph_client = graph_client
        self.memory_engine = memory_engine
        self.semantic_builder = SemanticBuilder(graph_client)  # For semantic analysis of feedback
        self.consciousness_engine = ConsciousnessEngine(memory_engine, None)  # Assuming EmotionEngine exists

    def prompt_user_validation(self, node_id: str, node_name: str, attributes: Dict[str, Any]) -> None:
        # ... (Existing code or enhancements)

        def _update_node_attribute(self, node_id: str, attr_name: str, new_value: Any):
        # ... (Existing code)

            def gather_feedback(self, user_feedback: str) -> Dict[str, Any]:
                feedback_categories = {
            "positive": ["great", "good", "helpful", "useful", "nice"],
            "negative": ["bad", "poor", "useless", "wrong", "incorrect"],
            "suggestion": ["could", "should", "suggest", "recommend", "idea"]
        }

        feedback = {"text": user_feedback, "categories": []}
        lower_feedback = user_feedback.lower()

        for category, keywords in feedback_categories.items():
            if any(keyword in lower_feedback for keyword in keywords):
                feedback["categories"].append(category)

        if not feedback["categories"]:
            feedback["categories"].append("neutral")

        # Semantic Analysis for deeper insight
        semantic_analysis = self.semantic_builder.infer_relationships(user_feedback, "previous feedback")
        feedback["semantic_insights"] = semantic_analysis

        logger.info(f"[FEEDBACK GATHERED] {feedback}")
        self._store_feedback(feedback)
        return feedback

    def _store_feedback(self, feedback: Dict[str, Any]):
        query = """
        CREATE (f:Feedback {text: $text, categories: $categories, semantic_insights: $semantic_insights, timestamp: datetime()})
        """
        try:
            self.graph_client.run_query(query, feedback)
            logger.info(f"[FEEDBACK STORED] Feedback entry created")
        except Exception as e:
            logger.error(f"[FEEDBACK ERROR] Failed to store feedback: {e}", exc_info=True)
            raise

    def analyze_feedback_trends(self) -> Dict[str, int]:
        query = """
        MATCH (f:Feedback)
        UNWIND f.categories AS category
        RETURN category, count(f) AS frequency
        ORDER BY frequency DESC
        """
        try:
            trends = self.graph_client.run_query(query)
            trend_dict = {item['category']: item['frequency'] for item in trends}
            logger.info(f"[TREND ANALYSIS] Feedback trends: {trend_dict}")

            if trend_dict.get("negative", 0) > trend_dict.get("positive", 0):
                logger.warning("[TREND ACTION] More negative feedback detected. Initiating learning cycle.")
                self.initiate_learning_cycle()

            return trend_dict
        except Exception as e:
            logger.error(f"[TREND ANALYSIS ERROR] {e}", exc_info=True)
            return {}

        def initiate_learning_cycle(self):
            logger.info("[LEARNING CYCLE] Starting learning cycle based on feedback.")
        try:
            # Gather recent feedback for learning
            recent_feedback = self.get_feedback_by_category("all", limit=10)  # Assuming 'all' returns all feedback

            for feedback in recent_feedback:
                # Update memories with feedback
                self.memory_engine.store_memory(
                    text=feedback["text"],
                    emotions=feedback["categories"],  # Assuming categories can represent emotions
                    extra_properties={"type": "feedback"}
                )

                # Generate new insights from feedback
                introspection = self.consciousness_engine.reflect(feedback["text"])
                self.memory_engine.store_memory(
                    text=f"Reflected on feedback: {introspection}",
                    emotions=["reflective"],
                    extra_properties={"type": "introspection"}
                )

            # Update semantic relationships based on feedback
            self.semantic_builder.build_relationships(label="Feedback", relationship_type="INSIGHT_FROM")

            logger.info("[LEARNING CYCLE] Feedback processed and memories updated.")
        except Exception as e:
            logger.error(f"[LEARNING CYCLE ERROR] {e}", exc_info=True)

    def get_feedback_by_category(self, category: str = "all", limit: int = 10) -> List[Dict]:
        """
        Retrieve feedback entries from a specific category or all feedback for detailed analysis or reporting.

        :param category: The feedback category to filter by or 'all' for all feedback.
        :param limit: Number of feedback entries to retrieve.
        :return: List of feedback entries matching the criteria.
        """
        if category == "all":
            query = f"""
            MATCH (f:Feedback)
            RETURN f.text AS text, f.categories AS categories, f.semantic_insights AS semantic_insights
            ORDER BY f.timestamp DESC
            LIMIT {limit}
            """
        else:
            query = f"""
            MATCH (f:Feedback)
            WHERE $category IN f.categories
            RETURN f.text AS text, f.categories AS categories, f.semantic_insights AS semantic_insights
            ORDER BY f.timestamp DESC
            LIMIT {limit}
            """
        try:
            feedback = self.graph_client.run_query(query, {"category": category} if category != "all" else {})
            logger.info(f"[FEEDBACK RETRIEVAL] Retrieved {len(feedback)} feedback entries for category '{category}'")
            return feedback
        except Exception as e:
            logger.error(f"[FEEDBACK RETRIEVAL ERROR] {e}", exc_info=True)
            return []

    def integrate_feedback_into_memory(self, feedback: str):
        """
        Integrate feedback into the memory system, potentially adjusting existing knowledge or creating new memory nodes.

        :param feedback: The feedback text to integrate.
        """
        try:
            analyzed_feedback = self.gather_feedback(feedback)
            categories = analyzed_feedback["categories"]

            if "suggestion" in categories:
                self._create_suggestion_memory(feedback)
            elif "negative" in categories:
                self._review_negative_feedback(feedback)
            elif "positive" in categories:
                self._reinforce_positive_feedback(feedback)
            else:
                logger.info("[MEMORY INTEGRATION] Neutral feedback, no specific action taken.")

            logger.info(f"[MEMORY INTEGRATION] Feedback '{feedback}' integrated into memory.")
        except Exception as e:
            logger.error(f"[MEMORY INTEGRATION ERROR] {e}", exc_info=True)

    def _create_suggestion_memory(self, suggestion: str):
        """
        Create a new memory node for a suggestion in the graph database.

        :param suggestion: The suggestion text to store as memory.
        """
        query = """
        CREATE (m:Memory {type: 'Suggestion', text: $suggestion, timestamp: datetime()})
        """
        try:
            self.graph_client.run_query(query, {"suggestion": suggestion})
            logger.info(f"[SUGGESTION STORED] Suggestion memory created: {suggestion}")
        except Exception as e:
            logger.error(f"[SUGGESTION ERROR] Failed to store suggestion: {e}", exc_info=True)

    def _review_negative_feedback(self, feedback: str):
        """
        Review existing memories or knowledge in light of negative feedback.

        :param feedback: The feedback text to review against existing memories.
        """
        # Example of how you might review memories
        related_memories = self.memory_engine.search_memory(feedback)
        if related_memories:
            for memory in related_memories:
                # Adjust confidence or relevance of memory based on negative feedback
                self.memory_engine.update_memory(memory["id"], "confidence", memory.get("confidence", 1.0) * 0.9)
            logger.info(f"[NEGATIVE FEEDBACK] Reviewed and adjusted related memories for feedback: {feedback}")
        else:
            logger.info(f"[NEGATIVE FEEDBACK] No related memories found for feedback: {feedback}")

    def _reinforce_positive_feedback(self, feedback: str):
        """
        Reinforce positive behaviors or memories based on positive feedback.

        :param feedback: The feedback text to reinforce positive actions or memories.
        """
        related_memories = self.memory_engine.search_memory(feedback)
        if related_memories:
            for memory in related_memories:
                # Increase confidence or mark as successful
                self.memory_engine.update_memory(memory["id"], "confidence", memory.get("confidence", 1.0) * 1.1)
                self.memory_engine.update_memory(memory["id"], "success", True)
            logger.info(f"[POSITIVE FEEDBACK] Reinforced positive behaviors for feedback: {feedback}")
        else:
            logger.info(f"[POSITIVE FEEDBACK] No related memories found for feedback: {feedback}")