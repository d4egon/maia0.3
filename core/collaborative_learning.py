import logging
from typing import Optional
from NLP.nlp_engine import NLP
from core.memory_engine import MemoryEngine

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CollaborativeLearning:
    def __init__(self, conversation_engine, nlp_engine: NLP, memory_engine: MemoryEngine):
        """
        Initialize CollaborativeLearning with references to ConversationEngine, NLP, and MemoryEngine.

        :param conversation_engine: An instance of ConversationEngine for conversation management.
        :param nlp_engine: An instance of NLP for language processing tasks.
        :param memory_engine: An instance of MemoryEngine for memory operations.
        """
        # We don't import ConversationEngine here, we just accept it as a parameter
        self.conversation_engine = conversation_engine
        self.nlp_engine = nlp_engine
        self.memory_engine = memory_engine

    def detect_doubt(self, thought: str) -> bool:
        """
        Detect doubt or uncertainty in a thought using sentiment analysis and keyword checking.

        :param thought: The thought to analyze for doubt.
        :return: True if doubt is detected, False otherwise.
        """
        sentiment_scores = self.nlp_engine.analyze_emotions(thought)
        if any(emotion in sentiment_scores for emotion in ["uncertain", "doubt", "confused"]) or "?" in thought:
            return True
        return False

    def generate_query(self, thought: str) -> Optional[str]:
        """
        Generate a query for user input if doubt is detected, with interactive options.

        :param thought: The thought where doubt was detected.
        :return: A query string for clarification or None if no doubt detected.
        """
        if self.detect_doubt(thought):
            return f"I'm unsure about '{thought}'. Can you clarify?\n" \
                   "1. Yes, I can clarify.\n" \
                   "2. No, let's move on.\n" \
                   "Please choose an option or provide more details."
        return None

    def handle_user_feedback(self, feedback: str):
        """
        Process user feedback to refine knowledge or behavior, updating the model and memory.

        :param feedback: User's feedback text.
        """
        try:
            logger.info(f"[COLLABORATIVE LEARNING] User feedback received: {feedback}")
            
            # Process feedback through NLP for deeper insights
            analyzed_feedback = self.nlp_engine.process(feedback)
            intent, emotions = analyzed_feedback[1], analyzed_feedback[0]["emotions"]

            # Store feedback as memory
            memory_id = self.memory_engine.create_memory_node(feedback, {"type": "feedback"}, intent)
            self.memory_engine.enrich_attributes(memory_id, {"emotions": emotions})

            # Update conversation model with feedback
            self.conversation_engine.update_conversation_model([feedback], intent, emotions)

            logger.info(f"[COLLABORATIVE LEARNING] Feedback processed and integrated.")
        except Exception as e:
            logger.error(f"[COLLABORATIVE LEARNING ERROR] {e}", exc_info=True)

    def explore_ambiguity(self, thought: str) -> str:
        """
        Collaboratively explore ambiguous or unclear thoughts.

        :param thought: The ambiguous thought to explore.
        :return: Query for clarification or a message if no ambiguity detected.
        """
        if self.detect_doubt(thought):
            query = self.generate_query(thought)
            logger.info(f"[AMBIGUITY DETECTION] Query generated: {query}")
            return query
        return "No ambiguity detected."

    def integrate_feedback_loop(self, feedback: str) -> Optional[str]:
        """
        Integrate user feedback into a refinement loop.

        :param feedback: Feedback to integrate.
        :return: Refined knowledge or an error message if integration fails.
        """
        try:
            logger.info(f"[FEEDBACK INTEGRATION] Integrating feedback: {feedback}")
            refined_knowledge = self.conversation_engine.process_feedback(feedback)
            if refined_knowledge:
                logger.info(f"[FEEDBACK INTEGRATION] Knowledge refined: {refined_knowledge}")
                return refined_knowledge
            else:
                logger.warning("[FEEDBACK INTEGRATION] No refinement from feedback.")
        except Exception as e:
            logger.error(f"[FEEDBACK ERROR] {e}", exc_info=True)
            return "An error occurred while integrating feedback. Please try again."