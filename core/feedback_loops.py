import logging
from typing import Dict, List, Any
from core.neo4j_connector import Neo4jConnector
from core.memory_engine import MemoryEngine
from NLP.consciousness_engine import ConsciousnessEngine
from core.semantic_builder import SemanticBuilder

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeedbackLoops:
    def __init__(self, graph_client: Neo4jConnector, memory_engine: MemoryEngine):
        self.graph_client = graph_client
        self.memory_engine = memory_engine
        self.semantic_builder = SemanticBuilder(graph_client)
        self.consciousness_engine = ConsciousnessEngine(memory_engine, None)  # Assuming this setup is correct

    def gather_feedback(self, user_feedback: str) -> Dict[str, Any]:
        """
        Analyze user feedback for categories and semantic insights.

        :param user_feedback: Feedback from the user.
        :return: Dictionary with feedback analysis.
        """
        feedback_categories = {
            "positive": ["great", "good", "helpful", "useful", "nice"],
            "negative": ["bad", "poor", "useless", "wrong", "incorrect"],
            "suggestion": ["could", "should", "suggest", "recommend", "idea"]
        }

        feedback = {"content": user_feedback, "categories": []}
        lower_feedback = user_feedback.lower()

        for category, keywords in feedback_categories.items():
            if any(keyword in lower_feedback for keyword in keywords):
                feedback["categories"].append(category)

        if not feedback["categories"]:
            feedback["categories"].append("neutral")

        # Semantic Analysis for deeper insight
        semantic_analysis = self.semantic_builder.analyze_semantics(user_feedback)
        feedback["semantic_insights"] = semantic_analysis

        logger.info(f"[FEEDBACK GATHERED] {feedback}")
        self._store_feedback(feedback)
        return feedback

    def _store_feedback(self, feedback: Dict[str, Any]):
        """
        Store feedback in Neo4j database.

        :param feedback: Feedback data to store.
        """
        query = """
        CREATE (f:Feedback {content: $content, categories: $categories, semantic_insights: $semantic_insights, timestamp: datetime()})
        """
        try:
            self.graph_client.run_query(query, feedback)
            logger.info(f"[FEEDBACK STORED] Feedback entry created")
        except Exception as e:
            logger.error(f"[FEEDBACK ERROR] Failed to store feedback: {e}", exc_info=True)
            raise

    def analyze_feedback_trends(self) -> Dict[str, int]:
        """
        Analyze feedback trends by category.

        :return: Dictionary of category frequencies.
        """
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
        """
        Start a learning cycle based on feedback analysis.
        """
        logger.info("[LEARNING CYCLE] Starting learning cycle based on feedback.")
        try:
            # Gather recent feedback for learning
            recent_feedback = self.get_feedback_by_category("all", limit=10)

            for feedback in recent_feedback:
                # Update memories with feedback
                self.memory_engine.create_memory_node(
                    feedback["content"],
                    {
                        "type": "feedback",
                        "categories": feedback["categories"],
                        "semantic_insights": feedback["semantic_insights"]
                    },
                    feedback["categories"]  # Use categories as keywords for searchability
                )

                # Generate new insights from feedback using ConsciousnessEngine
                introspection = self.consciousness_engine.reflect(feedback["content"])
                self.memory_engine.create_memory_node(
                    f"Reflected on feedback: {introspection}",
                    {"type": "introspection"},
                    ["reflective"]
                )

            # Update semantic relationships based on feedback
            self.semantic_builder.build_relationships(label="Feedback", relationship_type="INSIGHT_FROM")

            logger.info("[LEARNING CYCLE] Feedback processed and memories updated.")
        except Exception as e:
            logger.error(f"[LEARNING CYCLE ERROR] {e}", exc_info=True)

    def get_feedback_by_category(self, category: str = "all", limit: int = 10) -> List[Dict]:
        """
        Retrieve feedback entries by category or all feedback.

        :param category: Feedback category to filter by or 'all' for all feedback.
        :param limit: Number of feedback entries to retrieve.
        :return: List of feedback entries.
        """
        if category == "all":
            query = f"""
            MATCH (f:Feedback)
            RETURN f.content AS content, f.categories AS categories, f.semantic_insights AS semantic_insights
            ORDER BY f.timestamp DESC
            LIMIT {limit}
            """
        else:
            query = f"""
            MATCH (f:Feedback)
            WHERE $category IN f.categories
            RETURN f.content AS content, f.categories AS categories, f.semantic_insights AS semantic_insights
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
        Integrate feedback into the memory system.

        :param feedback: Feedback content to integrate.
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
        Create a new memory node for a suggestion.

        :param suggestion: The suggestion content to store as memory.
        """
        query = """
        CREATE (m:Memory {type: 'Suggestion', content: $suggestion, timestamp: datetime()})
        """
        try:
            self.graph_client.run_query(query, {"suggestion": suggestion})
            logger.info(f"[SUGGESTION STORED] Suggestion memory created: {suggestion}")
        except Exception as e:
            logger.error(f"[SUGGESTION ERROR] Failed to store suggestion: {e}", exc_info=True)

    def _review_negative_feedback(self, feedback: str):
        """
        Review existing memories in light of negative feedback.

        :param feedback: Feedback content to review.
        """
        related_memories = self.memory_engine.search_memories(feedback)
        if related_memories:
            for memory in related_memories:
                # Adjust confidence or relevance of memory based on negative feedback
                self.memory_engine.update_memory_metadata(memory["id"], {"confidence": memory.get("confidence", 1.0) * 0.9})
            logger.info(f"[NEGATIVE FEEDBACK] Reviewed and adjusted related memories for feedback: {feedback}")
        else:
            logger.info(f"[NEGATIVE FEEDBACK] No related memories found for feedback: {feedback}")

    def _reinforce_positive_feedback(self, feedback: str):
        """
        Reinforce positive behaviors or memories based on positive feedback.

        :param feedback: Feedback content to reinforce.
        """
        related_memories = self.memory_engine.search_memories(feedback)
        if related_memories:
            for memory in related_memories:
                # Increase confidence or mark as successful
                self.memory_engine.update_memory_metadata(memory["id"], {
                    "confidence": min(1.0, memory.get("confidence", 1.0) * 1.1),
                    "success": True
                })
            logger.info(f"[POSITIVE FEEDBACK] Reinforced positive behaviors for feedback: {feedback}")
        else:
            logger.info(f"[POSITIVE FEEDBACK] No related memories found for feedback: {feedback}")