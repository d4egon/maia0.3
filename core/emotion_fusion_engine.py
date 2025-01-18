import logging
from typing import Dict, List, Tuple
from core.emotion_engine import EmotionEngine
from core.memory_engine import MemoryEngine
from PIL import Image
import numpy as np
from transformers import pipeline
from config.utils import get_visual_emotion_model

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmotionFusionEngine:
    def __init__(self, memory_engine: MemoryEngine, nlp_engine):
        """
        Initialize the EmotionFusionEngine with memory and NLP engines, along with visual emotion analysis.

        :param memory_engine: An instance for memory operations.
        :param nlp_engine: An instance for NLP operations.
        """
        self.memory_engine = memory_engine
        self.nlp_engine = nlp_engine
        self.emotion_engine = EmotionEngine(memory_engine)

        # Visual emotion analysis model
        try:
            self.visual_emotion_model = get_visual_emotion_model()
            logger.info("[INIT] Visual emotion model loaded successfully.")
        except Exception as e:
            logger.error(f"[INIT ERROR] Failed to load visual emotion model: {e}")
            raise

    def fuse_emotions(self, visual_input: str, text_input: str) -> Tuple[str, float]:
        """
        Fuse emotions from visual analysis, content context, and memory search results.

        :param visual_input: Path to visual input file.
        :param text_input: Textual input to analyze.
        :return: A tuple of (final emotion, total confidence).
        """
        logger.info("[FUSION] Starting emotion fusion process.")

        try:
            # Analyze emotions separately
            visual_emotion, visual_confidence = self.analyze_visual_emotion(visual_input)
            text_emotion, text_confidence = self.analyze_text_emotion(text_input)

            # Search contextual memory by embedding for more accurate matches
            text_embedding = self.memory_engine.model.encode([text_input])[0].tolist()
            context_data = self.memory_engine.search_memory_by_embedding(text_embedding)
            context_emotion, context_confidence = self.analyze_context_emotion(context_data)

            # Decision Fusion Logic
            final_emotion, total_confidence = self.determine_final_emotion(
                visual_emotion, visual_confidence,
                text_emotion, text_confidence,
                context_emotion, context_confidence
            )

            # Store the fused memory with semantic embedding for future searches
            fused_memory = f"Image emotion: {visual_emotion}, NLP emotion: {text_emotion}, Context emotion: {context_emotion}"
            self.memory_engine.create_memory_node(
                fused_memory,
                {
                    "type": "emotion_fusion",
                    "visual_input": visual_input,
                    "text_input": text_input,
                    "final_emotion": final_emotion,
                    "confidence": total_confidence
                },
                [final_emotion]  # Use emotion as keyword for searching
            )

            logger.info(f"[FUSION RESULT] Final Emotion: {final_emotion} (Total Confidence: {total_confidence})")
            return final_emotion, total_confidence
        except Exception as e:
            logger.error(f"[FUSION ERROR] {e}", exc_info=True)
            return "neutral", 0.0  # Default to neutral if fusion fails

    def analyze_visual_emotion(self, visual_input: str) -> Tuple[str, float]:
        """
        Analyze emotions from visual data using a pre-trained model.

        :param visual_input: Path to the image file.
        :return: Tuple of (emotion, confidence).
        """
        try:
            image = Image.open(visual_input).convert('RGB')
            predictions = self.visual_emotion_model(image)
            emotion = self._map_prediction_to_emotion(predictions[0]['label'])
            confidence = predictions[0]['score']
            logger.info(f"[VISUAL] Detected Emotion: {emotion} (Confidence: {confidence})")
            return emotion, confidence
        except Exception as e:
            logger.error(f"[VISUAL FAILED] {e}")
            return "neutral", 0.0

    def _map_prediction_to_emotion(self, prediction: str) -> str:
        """
        Map a prediction label to a more generic emotion if possible.

        :param prediction: The label predicted by the visual model.
        :return: A mapped emotion or the prediction if no mapping exists.
        """
        emotion_map = {
            "happy": ["smile", "happiness"],
            "sad": ["sadness", "frown"],
            "angry": ["anger", "furious"],
            "fearful": ["fear", "scared"],
            "surprised": ["surprise", "astonished"],
            "neutral": ["neutral", "calm"]
        }
        for emotion, keywords in emotion_map.items():
            if prediction.lower() in keywords:
                return emotion
        return "neutral"  # If no match, return neutral

    def analyze_text_emotion(self, text_input: str) -> Tuple[str, float]:
        """
        Analyze emotions from content data using the Emotion Engine.

        :param text_input: Content to analyze for emotion.
        :return: Tuple of (emotion, confidence).
        """
        try:
            text_emotion, confidence = self.emotion_engine.analyze_emotion(text_input)
            logger.info(f"[CONTENT] Detected Emotion: {text_emotion} (Confidence: {confidence})")
            return text_emotion, confidence
        except Exception as e:
            logger.error(f"[CONTENT FAILED] {e}")
            return "neutral", 0.0

    def analyze_context_emotion(self, context_data: List[Dict]) -> Tuple[str, float]:
        """
        Analyze emotions from context-related memory search results.

        :param context_data: List of memory contexts to analyze.
        :return: Tuple of (emotion, confidence).
        """
        try:
            if not context_data:
                logger.info("[CONTEXT] No relevant context found.")
                return "neutral", 0.0

            context_texts = [m["content"] for m in context_data]
            context_emotion, confidence = self.emotion_engine.contextual_emotion_analysis(" ".join(context_texts))
            logger.info(f"[CONTEXT] Detected Emotion: {context_emotion} (Confidence: {confidence})")
            return context_emotion, confidence
        except Exception as e:
            logger.error(f"[CONTEXT FAILED] {e}")
            return "neutral", 0.0

    def determine_final_emotion(
        self,
        visual_emotion: str, visual_confidence: float,
        text_emotion: str, text_confidence: float,
        context_emotion: str, context_confidence: float
    ) -> Tuple[str, float]:
        """
        Determine the final emotion based on the highest cumulative confidence score with weights.

        :param visual_emotion: Emotion from visual analysis.
        :param visual_confidence: Confidence score for visual emotion.
        :param text_emotion: Emotion from content analysis.
        :param text_confidence: Confidence score for content emotion.
        :param context_emotion: Emotion from context analysis.
        :param context_confidence: Confidence score for context emotion.
        :return: A tuple of (final emotion, total confidence).
        """
        emotions = {
            visual_emotion: visual_confidence * 1.5,  # Higher weight for visual as it's less common
            text_emotion: text_confidence * 1.2,
            context_emotion: context_confidence
        }

        final_emotion = max(emotions, key=emotions.get)
        total_confidence = round(sum(emotions.values()), 2)
        logger.info(f"[FINAL EMOTION] {final_emotion} (Total Confidence: {total_confidence})")
        return final_emotion, total_confidence

    def decay_emotional_states(self) -> Dict[str, float]:
        """
        Apply time-based decay to emotional confidence scores.
    
        :return: Updated emotional states.
        """
        try:
            for emotion in list(self.emotion_engine.dynamic_emotional_state.keys()):
                self.emotion_engine.dynamic_emotional_state[emotion] *= 0.9  # 10% decay
            logger.info(f"[DECAY APPLIED] {self.emotion_engine.dynamic_emotional_state}")
            return self.emotion_engine.dynamic_emotional_state
        except Exception as e:
            logger.error(f"[DECAY ERROR] {e}", exc_info=True)
            return {}

    def prioritize_context_emotions(self, context_emotions: List[Dict]) -> str:
        """
        Prioritize emotions based on contextual confidence and thematic relevance.
    
        :param context_emotions: List of context emotions with confidence scores.
        :return: Dominant emotion.
        """
        try:
            if not context_emotions:
                return "neutral"
            
            # Sort by confidence, then by thematic relevance if available
            prioritized = sorted(context_emotions, 
                                 key=lambda x: (x.get("confidence", 0), 
                                                x.get("thematic_relevance", 0)),
                                 reverse=True)[0]
            logger.info(f"[PRIORITIZED EMOTION] {prioritized}")
            return prioritized.get("emotion", "neutral")
        except Exception as e:
            logger.error(f"[PRIORITIZATION ERROR] {e}", exc_info=True)
            return "neutral"