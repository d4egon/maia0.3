# File: /core/thought_loop.py

import logging
import random
import time
from typing import List, Dict, Optional
from core.memory_engine import MemoryEngine
from core.emotion_engine import EmotionEngine
import spacy   # type: ignore

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy model for NLP tasks
nlp = spacy.load("en_core_web_sm")

class ThoughtLoop:
    def __init__(self, memory_engine: MemoryEngine, thought_engine):
        """
        Initialize ThoughtLoop with necessary engines for autonomous thought processing.

        :param memory_engine: Handles memory operations.
        :param thought_engine: Processes and generates thoughts.
        """
        self.memory_engine = memory_engine
        self.thought_engine = thought_engine
        self.running = False
        self.emotion_engine = EmotionEngine(memory_engine)

    def run(self):
        """
        Continuously run uninitiated thought loops.
        """
        self.running = True
        logger.info("[THOUGHT LOOP] Starting autonomous thought loops.")
        while self.running:
            try:
                # Generate a thought
                thought = self.thought_engine.generate_thought()
                logger.info(f"[THOUGHT LOOP] Generated Thought: {thought}")

                # Process thought for new vocabulary
                new_words = self.thought_engine.detect_new_words(thought)
                for word, meaning in new_words.items():
                    self.memory_engine.store_new_word(word, meaning)

                # Analyze emotion
                emotion = self.emotion_engine.process_emotion(thought)
                self.memory_engine.tag_memory(thought, emotion)

                # Simulate interaction patterns
                self.thought_engine.learn_interaction_patterns(thought)

                # Pause before next loop iteration
                time.sleep(random.uniform(5, 15))

            except Exception as e:
                logger.error(f"[THOUGHT LOOP ERROR] {e}", exc_info=True)

    def stop(self):
        """
        Stop the thought loop.
        """
        self.running = False
        logger.info("[THOUGHT LOOP] Thought loop has been stopped.")