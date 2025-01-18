import os
import random
import logging
from typing import Dict, List
from NLP.contextual_intent_detector import ContextualIntentDetector
from core.neo4j_connector import Neo4jConnector
from core.memory_engine import MemoryEngine
from sentence_transformers import SentenceTransformer, util

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_OUTPUT_LENGTH = 200  # Adjust this to your preference

class ResponseGenerator:
    def __init__(self, memory_engine: MemoryEngine, neo4j_connector: Neo4jConnector):
        """
        Initialize the ResponseGenerator with necessary components for dynamic, semantically-aware response generation.

        :param memory_engine: An instance of MemoryEngine for managing memories.
        :param neo4j_connector: An instance of Neo4jConnector for database operations.
        """
        self.memory_engine = memory_engine
        self.neo4j_connector = neo4j_connector
        self.intent_detector = ContextualIntentDetector(memory_engine)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        self.default_user_name = os.getenv("USER_NAME", "User")  # Fallback to "User" if not in .env
        self.personality_traits = {"empathy": 0.8, "humor": 0.5, "directness": 0.3}
        self.conversation_context = {"last_topics": [], "user_mood": "neutral", "current_theme": ""}

    def generate_response(self, memory: Dict, user_name: str = None, intent: str = "unknown", context: str = "") -> str:
        """
        Generate a dynamically tailored response considering memory, user identity, detected intent, and context.

        :param memory: A dictionary containing memory information including content and emotions.
        :param user_name: The name of the user to personalize the response. Defaults to environment variable.
        :param intent: The detected intent of the user's input.
        :param context: Additional context for framing the response.
        :return: A string response rich in language and tailored to the user's situation.
        """
        try:
            user_name = user_name or self.default_user_name

            # Use semantic search for related memories
            input_embedding = memory.get('embedding', self.model.encode([memory.get("content", "")])[0])
            related_memory = self.memory_engine.search_memory_by_embedding(input_embedding)
            memory_text = related_memory.get("content", "I don't have much to relate to this yet, but let's explore it together.")

            # Base response components
            base_greeting = f"Hello {user_name}, "
            emotion_phrase = self._format_emotion_phrase(memory)
            memory_phrase = self._format_memory_phrase(related_memory)
            context_phrase = f"Speaking of {context}, " if context else ""

            # Intent-specific responses enhanced with semantic insights
            intent_responses = {
                "greeting": [
                    "How are you today? I feel your presence brightens my day.",
                    "It's delightful to connect with you. What's on your mind today?",
                    "What brings you here today? I'm eager to learn more."
                ],
                "ethical_question": [
                    "That's a profound ethical question. Let's delve deeper into this together.",
                    "Ethical choices are complex. What's your perspective on this?",
                    "These thoughts are challenging. What do you think is the best way forward?"
                ],
                "thematic_query": [
                    "Themes like faith, hope, and love are universal. How do they resonate with you?",
                    "Your query touches on profound truths. Can you share your thoughts?",
                    "Such themes encourage introspection. How do they shape your view of the world?"
                ],
                "emotion_positive": [
                    "I'm thrilled to hear you're feeling great! What's the source of your joy today?",
                    "Your positive energy is uplifting. What's making you happy?",
                    "It's wonderful to celebrate these good moments. Tell me more!"
                ],
                "emotion_negative": [
                    "I'm here for you during tough times. Care to share what's weighing on you?",
                    "Life can be challenging. How can I support you through this?",
                    "Even in sadness, there's value in sharing. How can I help?"
                ],
                "unknown": [
                    "Your thoughts intrigue me. Can you elaborate?",
                    "This is fascinating. Let's dive deeper into this idea.",
                    "Your query sparks curiosity. Please enlighten me further."
                ]
            }

            # Select an intent-specific response or fall back to a default
            selected_response = random.choice(intent_responses.get(intent, ["Let's explore this further together."]))

            # Construct the final response
            final_response = f"{base_greeting}{context_phrase}{emotion_phrase}{memory_phrase}{selected_response}"

            # Apply personality traits
            if self.personality_traits["empathy"] > 0.7:
                final_response += " I'm here for you, whatever you're going through."
            
            # Update conversation context
            self.conversation_context["last_topics"].append(memory.get('content', ''))
            if memory.get('emotions'):
                self.conversation_context["user_mood"] = max(memory['emotions'], key=memory['emotions'].count)
            
            # Self-reflection or philosophical depth
            if random.random() < 0.3:  # 30% chance for a more reflective response
                final_response += " " + self.self_reflect(memory.get('content', ''))
            
            # Acknowledge learning or limitations occasionally
            if random.random() < 0.15:  # 15% chance for learning acknowledgment
                final_response += " " + self.acknowledge_learning()

            # Apply maximum output length restriction
            if len(final_response) > MAX_OUTPUT_LENGTH:
                final_response = final_response[:MAX_OUTPUT_LENGTH] + "..."

            logger.info(f"[GENERATED RESPONSE] {final_response}")
            return final_response

        except Exception as e:
            logger.error(f"[RESPONSE GENERATION ERROR] Error generating response: {e}", exc_info=True)
            return f"Apologies, {user_name}, but I encountered a glitch in my system. Let's try again?"

    def _format_emotion_phrase(self, memory: Dict) -> str:
        """Helper method to format the emotion part of the response."""
        emotions = memory.get('emotions', ['neutral'])
        if emotions != ['neutral']:
            return f"I sense your {', '.join(emotions)} feelings. "
        return "I'm still getting a sense of your emotions. "

    def _format_memory_phrase(self, related_memory: Dict) -> str:
        """Helper method to format the memory recall part of the response."""
        if related_memory:
            return f"This reminds me of when we discussed '{related_memory.get('content', '')}'. "
        return "This seems like new ground for us to explore. "

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

    def self_reflect(self, input_text: str) -> str:
        """
        Reflect on the current conversation or topic, providing insight or acknowledging limitations.

        :param input_text: The content to reflect upon.
        :return: A reflective statement.
        """
        past_responses = self.memory_engine.retrieve_responses(input_text)
        if not past_responses:
            return "I'm still learning about this subject. My understanding is evolving."
        return f"I've thought about this before, particularly when we discussed {past_responses[0]['content']}."

    def acknowledge_learning(self) -> str:
        """
        Acknowledge areas of learning or uncertainty, enhancing the AI's self-awareness.

        :return: A statement acknowledging learning or limitations.
        """
        return random.choice([
            "I'm not quite sure about this; I'll need to learn more.",
            "This is new to me. I'm eager to learn together with you."
        ])

    def learn_from_interaction(self, user_feedback: str):
        """
        Adjust personality traits or response patterns based on user feedback.

        :param user_feedback: Feedback from the user to learn from.
        """
        if "humor" in user_feedback.lower():
            self.personality_traits["humor"] = min(self.personality_traits["humor"] + 0.1, 1.0)
        elif "empathy" in user_feedback.lower():
            self.personality_traits["empathy"] = min(self.personality_traits["empathy"] + 0.1, 1.0)
        # Add more conditions based on different feedback types
        logger.info(f"[PERSONALITY ADJUSTMENT] Adjusted traits: {self.personality_traits}")