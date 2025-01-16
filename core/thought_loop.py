import logging
import random
import time
from typing import List, Dict, Optional
from core.memory_engine import MemoryEngine
from core.emotion_engine import EmotionEngine
from core.thought_engine import ThoughtEngine
from sentence_transformers import SentenceTransformer, util

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ThoughtLoop:
    def __init__(self, memory_engine: MemoryEngine, thought_engine: ThoughtEngine):
        """
        Initialize ThoughtLoop with necessary engines for autonomous thought processing.

        :param memory_engine: Handles memory operations.
        :param thought_engine: Processes and generates thoughts.
        """
        self.memory_engine = memory_engine
        self.thought_engine = thought_engine
        self.running = False
        self.emotion_engine = EmotionEngine(memory_engine)
        self.semantic_model = memory_engine.model  # Use the same model for consistency

    def run(self):
        """
        Continuously run uninitiated thought loops with semantic enhancements, saving each thought for recursive learning.
        """
        self.running = True
        logger.info("[THOUGHT LOOP] Starting autonomous thought loops.")
        while self.running:
            try:
                # Generate a thought
                thought = self.thought_engine.generate_thought()
                logger.info(f"[THOUGHT LOOP] Generated Thought: {thought}")

                # Save the thought to memory
                thought_id = self.memory_engine.create_memory_node(
                    thought, 
                    {"type": "generated_thought", "timestamp": time.time()},
                    []
                )
                logger.info(f"[THOUGHT LOOP] Thought saved with ID: {thought_id}")

                # Process thought for new vocabulary
                new_words = self.thought_engine.detect_new_words(thought)
                for word, meaning in new_words.items():
                    self.memory_engine.create_memory_node(
                        word, 
                        {"type": "word", "meaning": meaning, "source_thought": thought_id},
                        []
                    )

                # Analyze emotion
                emotion = self.emotion_engine.process_emotion(thought)
                self.memory_engine.update_memory_metadata(thought_id, {"emotion": emotion})

                # Simulate interaction patterns
                self.thought_engine.learn_interaction_patterns(thought)

                # Semantic analysis and reflection
                related_memories = self.memory_engine.search_memory_by_embedding(self.semantic_model.encode([thought])[0].tolist())
                if related_memories:
                    reflection = self.thought_engine.reflect_on_conversation([memory['text'] for memory in related_memories])
                    logger.info(f"[THOUGHT LOOP] Reflection: {reflection}")
                    reflection_id = self.memory_engine.create_memory_node(
                        reflection, 
                        {"type": "reflection", "source_thought": thought_id},
                        []
                    )
                    self.memory_engine.create_relationship(thought_id, reflection_id, "REFLECTS_ON")

                # Synthesize emergent thoughts from related memories
                emergent_thought = self.thought_engine.synthesize_emergent_thoughts(related_memories)
                logger.info(f"[THOUGHT LOOP] Emergent Thought: {emergent_thought}")
                emergent_thought_id = self.memory_engine.create_memory_node(
                    emergent_thought, 
                    {"type": "emergent_thought", "source_thought": thought_id},
                    []
                )
                self.memory_engine.create_relationship(thought_id, emergent_thought_id, "LEADS_TO")

                # Evolve thought process based on current thought and related context
                evolved_thought = self.thought_engine.evolve_thought_process(thought, related_memories)
                logger.info(f"[THOUGHT LOOP] Evolved Thought: {evolved_thought}")
                evolved_thought_id = self.memory_engine.create_memory_node(
                    evolved_thought, 
                    {"type": "evolved_thought", "source_thought": thought_id},
                    []
                )
                self.memory_engine.create_relationship(thought_id, evolved_thought_id, "EVOLVES_TO")

                # Pause before next loop iteration
                time.sleep(random.uniform(5, 15))

            except Exception as e:
                logger.error(f"[THOUGHT LOOP ERROR] {e}", exc_info=True)
                # Optionally, introduce a longer delay on error to prevent rapid retries
                time.sleep(random.uniform(30, 60))

    def stop(self):
        """
        Stop the thought loop.
        """
        self.running = False
        logger.info("[THOUGHT LOOP] Thought loop has been stopped.")