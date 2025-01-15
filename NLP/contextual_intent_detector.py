#NLP/contextual_intent_detector.py
import logging
from typing import Dict, List
import nltk # type: ignore
from nltk.corpus import wordnet  # type: ignore
from core.memory_engine import MemoryEngine

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ContextualIntentDetector:
    def __init__(self, memory_engine: MemoryEngine):
        """
        Initialize the ContextualIntentDetector with memory integration for context-aware intent detection.

        :param memory_engine: An instance of MemoryEngine for memory-based context analysis.
        """
        nltk.download('wordnet')  # Download wordnet data if not already present
        
        self.memory_engine = memory_engine
        self.keywords = {
            "greeting": ["hello", "hi", "hey", "good morning", "morning", "good evening"],
            "negation": ["i don't", "i disagree", "not really", "never mind", "i refuse"],
            "confirmation": ["yes", "correct", "right", "of course", "sure", "yeah"],
            "emotion_positive": ["happy", "joyful", "excited", "optimistic", "grateful"],
            "emotion_negative": ["sad", "angry", "frustrated", "lonely", "upset"],
            "ethical": ["virtue", "justice", "morality", "values", "is it right"],
            "thematic": ["faith", "hope", "love", "truth", "redemption", "purpose"],
        }
        self.expanded_keywords = self.expand_keywords(self.keywords)

    @staticmethod
    def get_synonyms(word: str) -> List[str]:
        """
        Get synonyms for a word using WordNet.

        :param word: The word to find synonyms for.
        :return: A list of synonyms.
        """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower())  # Convert to lowercase for consistency
        return list(synonyms)

    def expand_keywords(self, keywords: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Expand the keywords dictionary with synonyms for each word.

        :param keywords: The original keywords dictionary.
        :return: The expanded keywords dictionary with unique synonyms.
        """
        expanded_keywords = {}
        for intent, words in keywords.items():
            expanded_keywords[intent] = list(set(words + [
                synonym for word in words for synonym in self.get_synonyms(word)
            ]))  # Use set to remove duplicates
        logger.info(f"[EXPANDED KEYWORDS] Keywords expanded with synonyms.")
        return expanded_keywords

    def detect_intent(self, text: str) -> Dict[str, float]:
        """
        Detect the intent from the given text using keyword matching and memory context.

        :param text: The text to analyze for intent.
        :return: A dictionary containing the detected intent and its confidence score.
        """
        try:
            text_lower = text.lower().strip()

            # Initialize scoring for all intents
            intent_scores: Dict[str, int] = {intent: 0 for intent in self.expanded_keywords}

            # Score based on keyword matches
            for intent, words in self.expanded_keywords.items():
                for word in words:
                    if word in text_lower:
                        intent_scores[intent] += 1

            # Incorporate memory context
            memory = self.memory_engine.search_memory(text)
            if memory:
                logger.info(f"[MEMORY CONTEXT] Found memory: {memory['text']} for input: '{text}'")
                for intent, words in self.expanded_keywords.items():
                    if any(word in memory['text'].lower() for word in words):
                        intent_scores[intent] += 2  # Higher weight for memory matches

            # Determine the best matching intent and calculate confidence
            best_intent = max(intent_scores, key=intent_scores.get)
            total_score = sum(intent_scores.values())
            confidence = intent_scores[best_intent] / total_score if total_score > 0 else 0

            # Fallback if no significant score is found
            if confidence < 0.2:  # Threshold for low confidence
                logger.info(f"[INTENT DETECTION] No significant intent detected for text: '{text}'")
                return {"intent": "unknown", "confidence": 0.0}
            logger.info(f"[INTENT DETECTION] Detected intent: {best_intent} with confidence {confidence:.2f}")
            return {"intent": best_intent, "confidence": round(confidence, 2)}
        except Exception as e:
            logger.error(f"[INTENT DETECTION ERROR] Error detecting intent for text: '{text}': {e}", exc_info=True)
            return {"intent": "unknown", "confidence": 0.0}