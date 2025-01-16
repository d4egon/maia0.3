import logging
from collections import defaultdict
from transformers import pipeline
from typing import Dict, List, Tuple, Union
import numpy as np
from core.memory_engine import MemoryEngine

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmotionEngine:
    def __init__(self, memory_engine: MemoryEngine):
        """Initialize EmotionEngine with emotion analysis capabilities."""
        self.memory_engine = memory_engine
        self.emotion_keywords = {
            "happy": {"keywords": ["joy", "excited", "love", "cheerful", "elated"], "weight": 2.0},
            "sad": {"keywords": ["sad", "down", "heartbroken", "depressed"], "weight": 1.8},
            "angry": {"keywords": ["angry", "mad", "furious", "rage", "annoyed"], "weight": 2.2},
            "fearful": {"keywords": ["afraid", "scared", "fear", "terrified", "anxious"], "weight": 2.0},
            "surprised": {"keywords": ["surprised", "shocked", "amazed", "startled"], "weight": 1.5},
            "neutral": {"keywords": ["okay", "fine", "alright", "normal"], "weight": 1.0},
        }
        
        self.dynamic_emotional_state = {"neutral": 100.0}
        self.emotion_transitions = defaultdict(int)
        self.emotion_history = []
        self.emotion_classifier = pipeline("sentiment-analysis")

    def analyze_emotion(self, text: str) -> Tuple[str, float]:
        """
        Analyzes the emotion of a given text using keyword matching, sentiment analysis, and an emotion classifier.

        :param text: The text to analyze.
        :return: A tuple of (emotion, confidence).
        """
        text_lower = text.lower()
        emotion_scores = defaultdict(float)

        try:
            # Keyword Matching with Scoring
            for emotion, data in self.emotion_keywords.items():
                weight = data["weight"]
                for keyword in data["keywords"]:
                    if keyword in text_lower:
                        emotion_scores[emotion] += weight
                        logger.debug(f"[KEYWORD MATCH] '{keyword}' → '{emotion}' (Weight: {weight})")

            # Emotion Classification
            classified_emotions = self.emotion_classifier(text)
            for result in classified_emotions:
                emotion = result['label'].lower()
                if emotion in self.emotion_keywords:
                    emotion_scores[emotion] += result['score'] * self.emotion_keywords[emotion]['weight']

            # Handle case where no keywords or classifier matches
            if not emotion_scores:
                sentiment_result = self.emotion_classifier(text)[0]  # Use the most likely emotion from classifier
                emotion_scores[sentiment_result['label'].lower()] = sentiment_result['score']

            if emotion_scores:
                detected_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = round(emotion_scores[detected_emotion], 2)
                logger.info(f"[DETECTED EMOTION] {detected_emotion} (Confidence: {confidence})")

                # Update dynamic emotional state
                self.update_emotional_state(detected_emotion, confidence)

                # Update history for transition matrix
                if self.emotion_history:
                    last_emotion = self.emotion_history[-1]
                    if last_emotion != detected_emotion:
                        self.emotion_transitions[(last_emotion, detected_emotion)] += 1
                self.emotion_history.append(detected_emotion)

                # Store emotion analysis in memory
                self.memory_engine.create_memory_node(
                    text, 
                    {
                        "type": "emotion_analysis",
                        "emotion": detected_emotion,
                        "confidence": confidence,
                        "timestamp": np.datetime64('now').item()
                    },
                    []
                )

                return detected_emotion, confidence

            logger.info("[DEFAULT] No matching emotion found. Defaulting to 'neutral'.")
            return "neutral", 0.0
        except Exception as e:
            logger.error(f"[EMOTION ANALYSIS ERROR] {e}", exc_info=True)
            return "neutral", 0.0  # Default to neutral emotion with zero confidence if an error occurs

    def contextual_emotion_analysis(self, text: str) -> Tuple[str, float]:
        """Analyze text for emotional content with context from memory."""
        try:
            # Get base sentiment
            sentiment = self.emotion_classifier(text)[0]
            emotion = sentiment['label']
            confidence = sentiment['score']
            
            # Check for context from memory
            memory = self.memory_engine.search_memory_by_embedding(self.memory_engine.model.encode([text])[0].tolist())
            if memory:
                prev_emotion = memory['metadata'].get('emotion', 'neutral')
                prev_confidence = memory['metadata'].get('confidence', 0.0)
                
                # Adjust confidence based on previous emotion if it matches
                if prev_emotion == emotion:
                    confidence = min(1.0, confidence + (prev_confidence * 0.1))  # Example adjustment
                else:
                    confidence = max(0.0, confidence - (prev_confidence * 0.05))  # Reduce confidence if emotions differ

            # Update state
            self.update_emotional_state(emotion, confidence)
            
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"[EMOTION ANALYSIS ERROR] {e}", exc_info=True)
            return "neutral", 0.0

    def update_emotional_state(self, emotion: str, confidence: float):
        """Update emotional state with new emotion."""
        try:
            # Update state
            current_state = self.dynamic_emotional_state.get(emotion, 0.0)
            self.dynamic_emotional_state[emotion] = min(100.0, current_state + (confidence * 10))
            
            # Decay other emotions
            for e in self.dynamic_emotional_state:
                if e != emotion:
                    self.dynamic_emotional_state[e] = max(0.0, self.dynamic_emotional_state[e] - 5.0)
                    
            # Store transition
            if self.emotion_history:
                transition = (self.emotion_history[-1], emotion)
                self.emotion_transitions[transition] += 1
                
            self.emotion_history.append(emotion)
            
            # Store in memory
            self.memory_engine.update_memory_metadata(
                self.emotion_history[-1],  # Assuming the last emotion is stored as memory
                {"emotional_state": self.dynamic_emotional_state, "timestamp": np.datetime64('now').item()}
            )
            
        except Exception as e:
            logger.error(f"[EMOTION UPDATE ERROR] {e}", exc_info=True)

    def get_emotional_state(self) -> Dict[str, float]:
        """
        Retrieve the current dynamic emotional state.

        :return: A dictionary representing the current emotional state percentages.
        """
        return self.dynamic_emotional_state

    def emotional_transition_matrix(self) -> np.array:
        """
        Generate an emotional transition matrix based on historical data.

        :return: A numpy array representing the probability of transitioning between emotions.
        """
        emotions = list(self.emotion_keywords.keys())
        matrix = np.zeros((len(emotions), len(emotions)))

        # Fill the matrix with counts from transitions
        for (from_emo, to_emo), count in self.emotion_transitions.items():
            if from_emo in emotions and to_emo in emotions:
                from_index = emotions.index(from_emo)
                to_index = emotions.index(to_emo)
                matrix[from_index][to_index] = count

        # Convert counts to probabilities
        for i in range(len(emotions)):
            row_sum = matrix[i].sum()
            if row_sum > 0:
                matrix[i] /= row_sum

        logger.info(f"[TRANSITION MATRIX] Generated matrix for emotions: {emotions}")
        return matrix

    def predict_next_emotion(self) -> str:
        """
        Predict the next emotion based on the current emotional state and the transition matrix.

        :return: The predicted next emotion based on probability.
        """
        if not self.emotion_history:
            logger.warning("[PREDICTION] Cannot predict next emotion: no history available.")
            return "neutral"

        current_emotion = self.emotion_history[-1]
        transition_matrix = self.emotional_transition_matrix()
        emotions = list(self.emotion_keywords.keys())
        current_index = emotions.index(current_emotion)

        # If we have no transitions from the current emotion, we can't predict
        if np.all(transition_matrix[current_index] == 0):
            logger.warning(f"[PREDICTION] No transition data for '{current_emotion}'. Defaulting to neutral.")
            return "neutral"

        # Predict based on probabilities
        next_emotion_index = np.random.choice(len(emotions), p=transition_matrix[current_index])
        next_emotion = emotions[next_emotion_index]
        logger.info(f"[PREDICTION] Predicted next emotion from '{current_emotion}' to '{next_emotion}'")
        return next_emotion