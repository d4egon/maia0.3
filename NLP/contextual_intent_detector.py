import logging
from typing import Dict, List
from sentence_transformers import util, SentenceTransformer
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
        self.memory_engine = memory_engine
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
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

    def expand_keywords(self, keywords: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Expand the keywords dictionary with synonyms for each word using semantic similarity.

        :param keywords: The original keywords dictionary.
        :return: The expanded keywords dictionary with unique synonyms based on semantic closeness.
        """
        expanded_keywords = {}
        for intent, words in keywords.items():
            expanded_set = set(words)
            for word in words:
                word_embedding = self.model.encode([word])[0]
                # Find similar words in a predefined vocabulary or from a corpus
                # Here we're simulating this with a small list for simplicity
                similar_words = self._get_similar_words(word_embedding, ["happy", "joy", "sad", "angry", "justice", "morality", "faith", "hope"])
                expanded_set.update(similar_words)
            expanded_keywords[intent] = list(expanded_set)
        logger.info(f"[EXPANDED KEYWORDS] Keywords expanded with semantic similarity.")
        return expanded_keywords

    def _get_similar_words(self, word_embedding: List[float], vocabulary: List[str], threshold: float = 0.7) -> List[str]:
        """
        Find words in the vocabulary that are semantically similar to the given word embedding.

        :param word_embedding: Embedding of the word to match against.
        :param vocabulary: List of words to check for similarity.
        :param threshold: Similarity threshold for considering words as synonyms.
        :return: List of words from vocabulary with similarity above the threshold.
        """
        similar_words = []
        for word in vocabulary:
            word_vec = self.model.encode([word])[0]
            similarity = util.cos_sim(word_embedding, word_vec).item()
            if similarity > threshold:
                similar_words.append(word)
        return similar_words

    def detect_intent(self, text: str) -> Dict[str, float]:
        """
        Detect the intent from the given text using keyword matching, semantic similarity, and memory context.

        :param text: The text to analyze for intent.
        :return: A dictionary containing the detected intent and its confidence score.
        """
        try:
            text_lower = text.lower().strip()
            text_embedding = self.model.encode([text])[0]

            # Initialize scoring for all intents
            intent_scores: Dict[str, float] = {intent: 0.0 for intent in self.expanded_keywords}

            # Score based on semantic similarity and keyword matches
            for intent, words in self.expanded_keywords.items():
                for word in words:
                    word_embedding = self.model.encode([word])[0]
                    similarity = util.cos_sim(text_embedding, word_embedding).item()
                    intent_scores[intent] += similarity

            # Incorporate memory context using semantic similarity
            memory = self.memory_engine.search_memory_by_embedding(text_embedding)
            if memory:
                logger.info(f"[MEMORY CONTEXT] Found memory: {memory['text']} for input: '{text}'")
                memory_embedding = memory.get('embedding', [])
                if memory_embedding:
                    similarity = util.cos_sim(text_embedding, memory_embedding).item()
                    for intent, words in self.expanded_keywords.items():
                        if any(word in memory['text'].lower() for word in words):
                            intent_scores[intent] += similarity * 2  # Higher weight for memory matches

            # Determine the best matching intent and calculate confidence
            if intent_scores:
                best_intent = max(intent_scores, key=intent_scores.get)
                total_score = sum(intent_scores.values())
                confidence = intent_scores[best_intent] / total_score if total_score > 0 else 0

                # Fallback if no significant score is found
                if confidence < 0.2:  # Threshold for low confidence
                    logger.info(f"[INTENT DETECTION] No significant intent detected for text: '{text}'")
                    return {"intent": "unknown", "confidence": 0.0}
                logger.info(f"[INTENT DETECTION] Detected intent: {best_intent} with confidence {confidence:.2f}")
                return {"intent": best_intent, "confidence": round(confidence, 2)}
            else:
                return {"intent": "unknown", "confidence": 0.0}

        except Exception as e:
            logger.error(f"[INTENT DETECTION ERROR] Error detecting intent for text: '{text}': {e}", exc_info=True)
            return {"intent": "unknown", "confidence": 0.0}