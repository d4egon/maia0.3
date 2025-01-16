import logging
from typing import List, Dict
from core.memory_engine import MemoryEngine
from sentence_transformers import util

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IntentDetector:
    def __init__(self, memory_engine: MemoryEngine):
        """
        Initialize the IntentDetector with a comprehensive list of intents, associated keywords,
        and memory integration for enhanced context.

        :param memory_engine: An instance of MemoryEngine to access stored memories.
        """
        self.memory_engine = memory_engine
        self.intents: Dict[str, List[str]] = {
            "greeting": ["hello", "hi", "hey", "greetings", "good morning", "good evening"],
            "identity": ["who", "what", "name", "you", "your purpose", "identity", "self", "origin"],
            "exit": ["bye", "exit", "quit", "goodbye", "see you later", "farewell"],
            "ethical_question": ["is it right", "should I", "ethics", "morals", "values", "virtue", "justice"],
            "thematic_query": ["meaning", "purpose", "faith", "hope", "redemption", "love", "truth"],
            "question": ["what", "why", "how", "when", "where", "which", "could", "do"],
            "confirmation": ["yes", "yeah", "sure", "okay", "indeed", "affirmative", "right", "correct"],
            "negation": ["no", "nope", "not", "never", "negative", "disagree", "wrong"],
        }
        self.model = memory_engine.model  # Assuming MemoryEngine has a model attribute for sentence embeddings

    def detect_intent(self, tokens: List[str], sentence_embedding: List[float]) -> str:
        """
        Detect the intent based on the presence of keywords in the tokenized text, memory relevance, 
        and semantic similarity.

        :param tokens: List of tokens (words) to check against intents.
        :param sentence_embedding: Precomputed embedding for the entire sentence.
        :return: The detected intent or 'unknown' if no match found.
        """
        try:
            lower_tokens = [token.lower() for token in tokens]
            
            # Keyword-based matching
            for intent, keywords in self.intents.items():
                if self._check_keywords(lower_tokens, keywords):
                    if intent in ["question", "ethical_question", "thematic_query"] and self._is_question(tokens):
                        return intent
                    elif intent not in ["question", "ethical_question", "thematic_query"]:
                        return intent

            # Semantic memory search
            memory = self.memory_engine.search_memory_by_embedding(sentence_embedding)
            if memory:
                logger.info(f"[MEMORY CONTEXT] Found related memory: {memory['text']} for tokens: {tokens}")
                for intent, keywords in self.intents.items():
                    if intent in ["ethical_question", "thematic_query"] and self._check_keywords(memory['text'].lower().split(), keywords):
                        return intent

            # Semantic intent matching
            for intent in self.intents.keys():
                intent_embedding = self.model.encode([intent])[0]
                similarity = util.cos_sim(sentence_embedding, intent_embedding).item()
                if similarity > 0.7:  # Threshold for intent matching
                    return intent

            logger.info(f"[INTENT DETECTION] No specific intent detected for tokens: {tokens}")
            return "unknown"
        except Exception as e:
            logger.error(f"[INTENT DETECTION ERROR] Error detecting intent for tokens: {tokens}: {e}", exc_info=True)
            return "unknown"

    def _check_keywords(self, tokens: List[str], keywords: List[str]) -> bool:
        """
        Helper method to check if any keyword is present in the given tokens.

        :param tokens: List of tokens to check.
        :param keywords: List of keywords to match against.
        :return: True if any keyword matches, False otherwise.
        """
        return any(keyword in ' '.join(tokens) for keyword in keywords)

    def _is_question(self, tokens: List[str]) -> bool:
        """
        Helper method to detect if the tokens form a question.

        :param tokens: List of tokens to analyze.
        :return: True if the tokens suggest a question, False otherwise.
        """
        return '?' in tokens or any(token.lower() in ["what", "why", "how", "when", "where", "which", "could", "do"] for token in tokens)

    def update_intents(self, new_intents: Dict[str, List[str]]):
        """
        Update or add new intents with their associated keywords.

        :param new_intents: Dictionary of intents with lists of keywords.
        """
        try:
            self.intents.update(new_intents)
            logger.info(f"[INTENT UPDATE] Updated intents: {new_intents}")
        except Exception as e:
            logger.error(f"[INTENT UPDATE ERROR] Failed to update intents: {e}", exc_info=True)

    def search_memory_by_embedding(self, embedding: List[float]) -> Dict:
        """
        Search for memory based on semantic similarity rather than keyword matching.

        :param embedding: The embedding to compare against stored memories.
        :return: The memory with the highest similarity or None if no match found.
        """
        # This method might need to be implemented in MemoryEngine to support semantic search
        # Here's a placeholder for how it might work
        memories = self.memory_engine.get_all_memories()
        best_match = None
        highest_similarity = 0
        for memory in memories:
            memory_embedding = memory.get('embedding', [])
            if memory_embedding:
                similarity = util.cos_sim(embedding, memory_embedding).item()
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = memory
        return best_match if best_match else {}