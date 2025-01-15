# NLP/response_generator.py
import os
import random
import logging
from typing import Dict, List
from NLP.contextual_intent_detector import ContextualIntentDetector
from core.neo4j_connector import Neo4jConnector
from core.memory_engine import MemoryEngine

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_OUTPUT_LENGTH = 200  # Adjust this to your preference

class ResponseGenerator:
    def __init__(self, memory_engine: MemoryEngine, neo4j_connector: Neo4jConnector):
        """
        Initialize the ResponseGenerator with necessary components for dynamic response generation.

        :param memory_engine: An instance of MemoryEngine for managing memories.
        :param neo4j_connector: An instance of Neo4jConnector for database operations.
        """
        self.memory_engine = memory_engine
        self.neo4j_connector = neo4j_connector
        self.intent_detector = ContextualIntentDetector(memory_engine)
        self.default_user_name = os.getenv("USER_NAME", "User")  # Fallback to "User" if not in .env

    def generate_response(self, memory: Dict, user_name: str = None, intent: str = "unknown", context: str = "") -> str:
        """
        Generate a dynamically tailored response considering memory, user identity, detected intent, and context.

        :param memory: A dictionary containing memory information including text and emotions.
        :param user_name: The name of the user to personalize the response. Defaults to environment variable.
        :param intent: The detected intent of the user's input.
        :param context: Additional context for framing the response.
        :return: A string response rich in language and tailored to the user's situation.
        """
        try:
            user_name = user_name or self.default_user_name

            # Retrieve relevant memories
            related_memory = self.memory_engine.search_memory(memory.get("text", ""))
            memory_text = related_memory.get("text", "nothing specific from my memory to relate to this yet")

            # Base response components
            base_greeting = f"Hello {user_name}, "
            emotion_phrase = self._format_emotion_phrase(memory)
            memory_phrase = self._format_memory_phrase(related_memory)
            context_phrase = f"Speaking of {context}, " if context else ""

            # Intent-specific responses
            intent_responses = {
                "greeting": ["How are you today?", "It's always nice to connect with you. What's on your mind?",
                             "What brings you here today? I'd love to know more."],
                "ethical_question": ["That's a deep ethical question. Let's explore it further.",
                                     "Ethical dilemmas often require reflection. What's your perspective on this?",
                                     "These are challenging thoughts. What do you think is the right path forward?"],
                "thematic_query": ["Faith, hope, and love guide so much of what we do. What resonates with you?",
                                   "This reminds me of profound truths. Care to share your thoughts?",
                                   "These themes often inspire reflection. Do you see them shaping your world?"],
                "emotion_positive": ["I'm glad you're feeling good! Would you like to share more about what makes you happy?",
                                     "It's wonderful to hear positivity from you. What's bringing you joy today?",
                                     "Moments like these remind us of what makes life beautiful. Tell me more!"],
                "emotion_negative": ["I'm here for you if you want to talk about what's troubling you.",
                                     "Life can feel heavy at times. Would you like to share your thoughts?",
                                     "Even in tough times, there's strength in sharing. How can I help?"],
                "unknown": ["I'm intrigued by your thoughts. Can you elaborate?",
                            "This is interesting. Tell me more, so I can better understand.",
                            "Your words spark curiosity in me. Please share more details."]
            }

            # Select an intent-specific response or fall back to a default
            selected_response = random.choice(intent_responses.get(intent, ["Let's explore this further together."]))

            # Construct the final response
            final_response = f"{base_greeting}{context_phrase}{emotion_phrase}{memory_phrase}{selected_response}"

            # Apply maximum output length restriction
            if len(final_response) > MAX_OUTPUT_LENGTH:
                final_response = final_response[:MAX_OUTPUT_LENGTH] + "..."

            logger.info(f"[GENERATED RESPONSE] {final_response}")
            return final_response

        except Exception as e:
            logger.error(f"[RESPONSE GENERATION ERROR] Error generating response: {e}", exc_info=True)
            return f"Apologies, {user_name}, but I encountered an issue generating your response."

    def _format_emotion_phrase(self, memory: Dict) -> str:
        """Helper method to format the emotion part of the response."""
        emotions = memory.get('emotions', ['neutral'])
        return f"I sense your {', '.join(emotions)} feelings. " if emotions != ['neutral'] else "I can't quite grasp your emotions yet. "

    def _format_memory_phrase(self, related_memory: Dict) -> str:
        """Helper method to format the memory recall part of the response."""
        return f"I recall something related: '{related_memory.get('text', '')}'. " if related_memory else "This seems new to me. "

    def generate_random_response(self, intent: str) -> str:
        """
        Generate a fallback random response for situations where specific memories or contexts are unavailable.

        :param intent: The detected intent of the user's input.
        :return: A random response string.
        """
        fallback_responses = {
            "greeting": [
                "Hi there! How can I assist you today?",
                "Hello! What's on your mind?",
                "It's great to see you. How can I help?"
            ],
            "unknown": [
                "That's intriguing. Could you explain more?",
                "I'm curious about what you mean. Can you tell me more?",
                "I'm listening. Could you elaborate?"
            ]
        }
        selected_response = random.choice(fallback_responses.get(intent, ["Let's explore this idea further."]))
        logger.info(f"[FALLBACK RESPONSE] {selected_response}")
        return selected_response
