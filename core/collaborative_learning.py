# core/collaborative_learning.py

import nltk # type: ignore
from nltk.sentiment import SentimentIntensityAnalyzer # type: ignore
from typing import Optional
import logging
from NLP.nlp_engine import NLP

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CollaborativeLearning:
    def __init__(self, conversation_engine):
        """
        Initialize CollaborativeLearning with a reference to the ConversationEngine.

        :param conversation_engine: An instance of ConversationEngine.
        """
        self.conversation_engine = conversation_engine
        # Download necessary NLTK data if not already present
        try:
            nltk.download('vader_lexicon', quiet=True)
        except Exception as e:
            logger.error(f"[NLTK ERROR] Failed to download VADER lexicon: {e}", exc_info=True)

    def detect_doubt(self, thought: str) -> bool:
        """
        Detect doubt or uncertainty in a thought using NLTK sentiment analysis and keyword checking.
        """
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(thought)
        # Assuming lower compound scores might indicate uncertainty or negativity
        if scores['compound'] < 0.2 or "?" in thought or any(word in thought.lower() for word in ["uncertain", "doubt", "wonder", "confusion"]):
            return True
        return False

    def generate_query(self, thought: str) -> Optional[str]:
        """
        Generate a query for user input if doubt is detected, with interactive options.
        """
        if self.detect_doubt(thought):
            return f"I'm unsure about '{thought}'. Can you clarify? \n1. Yes, I can clarify. \n2. No, let's move on. \nPlease choose an option or provide more details."
        return None

    def handle_user_feedback(self, feedback: str):
        """
        Process user feedback to refine knowledge or behavior, and optionally update the model.
        """
        from core.conversation_engine import ConversationEngine  # Lazy import
        if isinstance(self.conversation_engine, ConversationEngine):
            logger.info(f"[COLLABORATIVE LEARNING] User feedback received: {feedback}")
            self.conversation_engine.process_feedback(feedback)
            # Optionally, trigger model update here
            self.conversation_engine.update_conversation_model([feedback])

    def explore_ambiguity(self, thought: str):
        """
        Collaboratively explore ambiguous or unclear thoughts.
    
        :param thought: The ambiguous thought to explore.
        """
        if self.detect_doubt(thought):
            question = self.generate_query(thought)
            logger.info(f"[AMBIGUITY DETECTION] {question}")
            return question
        return "No ambiguity detected."
    
    def integrate_feedback_loop(self, feedback: str):
        """
        Integrate user feedback into a refinement loop.
    
        :param feedback: Feedback to integrate.
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

# core/feedback_loops.py
from core.collaborative_learning import CollaborativeLearning
from NLP.nlp_engine import NLP

class FeedbackLoops:
    def __init__(self, collaborative_learning: CollaborativeLearning, nlp_engine: NLP):
        self.collaborative_learning = collaborative_learning
        self.nlp_engine = nlp_engine
    
    def integrate_feedback(self):
        """
        Integrate feedback into the learning process, potentially affecting model retraining.
        """
        feedback_data = self.collaborative_learning.analyze_feedback()
        if feedback_data['negative'] > feedback_data['positive']:
            logger.info("Negative feedback predominant, considering model adjustment or retraining.")
            # Here you could trigger a retraining or adjust model parameters
            # self.nlp_engine.adjust_model_based_on_feedback(feedback_data)
        else:
            logger.info("Feedback suggests model is performing well, maintaining current state.")