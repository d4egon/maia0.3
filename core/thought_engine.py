import logging
import random
import time
from typing import List, Dict, Optional
from config.utils import get_sentence_transformer_model
import requests
from core.memory_engine import MemoryEngine
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ThoughtEngine:
    def __init__(self, memory_engine: MemoryEngine):
        """
        Initialize ThoughtEngine with a MemoryEngine for semantic and memory operations.

        :param memory_engine: An instance of MemoryEngine for managing database interactions.
        """
        self.memory_engine = memory_engine
        self.model = get_sentence_transformer_model()

    def reflect(self, input_text: str) -> str:
        """
        Reflect on an input event by querying related memories and emotions from the memory system.

        :param input_text: The event text to reflect upon.
        :return: A string describing the memory and associated emotion or a message if no memory is found.
        """
        try:
            input_embedding = self.model.encode([input_text])[0]
            memory = self.memory_engine.search_memory_by_embedding(input_embedding)
            if memory:
                message = f"Found memory of '{memory['content']}' with emotion '{memory['metadata'].get('emotion', 'neutral')}'."
                logger.info(f"[REFLECT] {message}")
                return message
            else:
                message = f"No memory found for '{input_text}'. Creating new memory."
                self.memory_engine.create_memory_node(input_text, {"event": input_text, "emotion": "neutral"}, [])
                logger.info(f"[REFLECT] {message}")
                return message
        except Exception as e:
            logger.error(f"[REFLECT ERROR] Error reflecting on '{input_text}': {e}", exc_info=True)
            return "An error occurred while reflecting on the event."

    def synthesize_emergent_thoughts(self, memory_nodes: List[Dict]) -> str:
        """
        Synthesize emergent thoughts from related memory nodes using semantic analysis.
    
        :param memory_nodes: List of memory nodes to synthesize.
        :return: A synthesized thought.
        """
        try:
            if not memory_nodes:
                return "No memories to synthesize thoughts from."

            # Combine themes and generate a new thought
            themes = [node.get('metadata', {}).get('theme', '') for node in memory_nodes]
            combined_themes = " + ".join(set(themes))
            theme_embedding = self.model.encode([combined_themes])[0]
            
            # Find related memories to inspire the thought
            related_memories = self.memory_engine.search_memory_by_embedding(theme_embedding.tolist(), top_n=3)
            if related_memories:
                insights = [mem['content'] for mem in related_memories]
                emergent_thought = f"By combining {combined_themes}, we arrive at a new perspective: {' '.join(insights)}."
            else:
                emergent_thought = f"Combining {combined_themes} leads to a fresh understanding, though no direct parallels were found in memory."

            logger.info(f"[EMERGENT THOUGHT] {emergent_thought}")
            # Store this emergent thought
            self.memory_engine.create_memory_node(emergent_thought, {"type": "emergent_thought"}, themes)
            return emergent_thought
        except Exception as e:
            logger.error(f"[SYNTHESIS ERROR] {e}", exc_info=True)
            return "An error occurred during thought synthesis."

    def generate_thought(self) -> str:
        """
        Generate a random thought based on memories or predefined themes.
        """
        try:
            themes = self.memory_engine.search_memories(["theme"])
            if not themes:
                themes = [{"content": theme} for theme in ["hope", "reflection", "change", "connection"]]  # Fallback themes
            
            selected_theme = random.choice(themes)['content']
            thought = f"Thinking about {selected_theme}..."
            
            # Store this thought in memory as a learning process
            self.memory_engine.create_memory_node(thought, {"theme": selected_theme, "type": "generated_thought"}, [selected_theme])
            
            logger.info(f"[GENERATED THOUGHT] {thought}")
            return thought
        except Exception as e:
            logger.error(f"[GENERATE THOUGHT ERROR] {e}", exc_info=True)
            return "An error occurred while generating a thought."

    def detect_new_words(self, thought: str) -> Dict[str, str]:
        """
        Detect new words in the thought and provide meanings using a free API.

        :return: A dictionary with new words as keys and their meanings as values.
        """
        try:
            from nltk.corpus import wordnet
            from nltk.tokenize import word_tokenize
            from nltk import pos_tag

            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('wordnet', quiet=True)

            # Tokenize and tag parts of speech
            words = word_tokenize(thought)
            tagged_words = pos_tag(words)

            new_words = {}
            for word, tag in tagged_words:
                if tag.startswith('N') or tag.startswith('V') or tag.startswith('J'):
                    # Only consider nouns, verbs, and adjectives as 'new' words for this example
                    synsets = wordnet.synsets(word)
                    if synsets:
                        # Get the first definition as a basic meaning
                        meaning = synsets[0].definition()
                        new_words[word] = meaning
                    elif word.isalpha():  # Ensure we're dealing with actual words
                        # If not found in WordNet, we might use a free API like Wiktionary or Datamuse
                        # Here's an example with Datamuse API for words not in WordNet:
                        response = requests.get(f"https://api.datamuse.com/words?sp={word}&md=d")
                        if response.status_code == 200:
                            data = response.json()
                            if data:
                                new_words[word] = data[0].get('defs', ['No definition found.'])[0]
                            else:
                                new_words[word] = "No definition found."
                        else:
                            new_words[word] = "API Error in fetching meaning."

            logger.info(f"[DETECT NEW WORDS] Detected {len(new_words)} new words.")
            return new_words
        except Exception as e:
            logger.error(f"[DETECT NEW WORDS ERROR] {e}", exc_info=True)
            return {}

    def learn_interaction_patterns(self, thought: str):
        """
        Simulate learning interaction patterns from a thought.
        """
        try:
            logger.info(f"[INTERACTION PATTERNS] Learning from '{thought}'")
            thought_embedding = self.model.encode([thought])[0]
            related_patterns = self.memory_engine.search_memory_by_embedding(thought_embedding.tolist())
            
            if related_patterns:
                for pattern in related_patterns:
                    logger.debug(f"Found related pattern: {pattern['content']}")
            else:
                logger.debug("No related patterns found, creating new pattern memory.")
            
            self.memory_engine.create_memory_node(
                thought, 
                {"type": "interaction_pattern", "timestamp": time.time()},
                ["pattern"]
            )
        except Exception as e:
            logger.error(f"[INTERACTION PATTERNS ERROR] Error learning patterns from '{thought}': {e}", exc_info=True)

    def evolve_thought_process(self, user_input: str, context: Optional[List[Dict]] = None) -> str:
        """
        Evolve the thought process by considering user input and optional context.

        :param user_input: The user's input to evolve the thought process around.
        :param context: Optional list of context dictionaries to inform the thought process.
        :return: An evolved thought or reflection based on the input and context.
        """
        try:
            user_embedding = self.model.encode([user_input])[0]
            related_memories = self.memory_engine.search_memory_by_embedding(user_embedding.tolist())
            
            if context:
                context_embeddings = [self.model.encode([c['content']])[0] for c in context]
                combined_embedding = np.mean(context_embeddings, axis=0)
                more_related = self.memory_engine.search_memory_by_embedding(combined_embedding.tolist())
                related_memories.extend(more_related)

            if related_memories:
                subjects = [memory['metadata'].get('subject', '') for memory in related_memories if 'subject' in memory['metadata']]
                actions = [memory['metadata'].get('verb', '') for memory in related_memories if 'verb' in memory['metadata']]
                objects = [memory['metadata'].get('object', '') for memory in related_memories if 'object' in memory['metadata']]

                if subjects and actions:
                    thought = f"Considering {', '.join(subjects)} and the action of {', '.join(actions)}, I think about {random.choice(actions)}ing {', '.join(objects) if objects else 'something'} in the context of {user_input}."
                else:
                    thought = f"Reflecting on '{user_input}', I ponder its implications."
            else:
                thought = f"No context found, pondering '{user_input}' in isolation."

            logger.info(f"[EVOLVED THOUGHT] {thought}")
            self.memory_engine.create_memory_node(thought, {"input": user_input, "context": context if context else []}, ["evolved_thought"])
            return thought
        except Exception as e:
            logger.error(f"[EVOLVE THOUGHT PROCESS ERROR] {e}", exc_info=True)
            return "An error occurred while evolving the thought process."

    def reflect_on_conversation(self, conversation_history: List[str]) -> str:
        """
        Reflect on the entire conversation history to provide a summary or insight.

        :param conversation_history: List of strings representing the conversation history.
        :return: A reflective statement on the conversation.
        """
        try:
            conversation_embedding = self.model.encode(conversation_history)
            mean_embedding = np.mean(conversation_embedding, axis=0)
            related_memories = self.memory_engine.search_memory_by_embedding(mean_embedding.tolist())

            sentiment = np.mean([self.memory_engine.analyze_emotion(content)['sentiment'] for content in conversation_history])
            key_topics = [memory['metadata'].get('theme', '') for memory in related_memories if 'theme' in memory['metadata']]
            entities = [memory['metadata'].get('entities', []) for memory in related_memories if 'entities' in memory['metadata']]
            entities = [item for sublist in entities for item in sublist]  # Flatten the list

            reflection = f"After reviewing our conversation, I sense a {('positive' if sentiment >= 0 else 'negative')} tone. We discussed topics like {', '.join(set(key_topics)) if key_topics else 'various subjects'}. Key entities included {', '.join(set(entities)) if entities else 'none in particular'}. This dialogue has enriched my understanding."

            logger.info(f"[CONVERSATION REFLECTION] {reflection}")
            self.memory_engine.create_memory_node(reflection, {"history": conversation_history}, ["conversation_reflection"])
            return reflection
        except Exception as e:
            logger.error(f"[CONVERSATION REFLECTION ERROR] {e}", exc_info=True)
            return "An error occurred while reflecting on the conversation."

    def process(self, content: str) -> str:
        """
        Process content and generate biblical context-aware thoughts.
        """
        try:
            content_embedding = self.model.encode([content])[0]
            related_memories = self.memory_engine.search_memory_by_embedding(content_embedding.tolist())

            subjects = [memory['metadata'].get('subject', '') for memory in related_memories if 'subject' in memory['metadata']]
            actions = [memory['metadata'].get('verb', '') for memory in related_memories if 'verb' in memory['metadata']]
            objects = [memory['metadata'].get('object', '') for memory in related_memories if 'object' in memory['metadata']]
            entities = [memory['metadata'].get('entities', []) for memory in related_memories if 'entities' in memory['metadata']]
            entities = [item for sublist in entities for item in sublist]  # Flatten the list

            thought = self._generate_thought(subjects, actions, objects, entities, content)

            # Store in memory with biblical context
            self.memory_engine.create_memory_node(
                thought,
                {
                    "original_content": content,
                    "entities": entities,
                    "timestamp": time.time(),
                    "type": "biblical_thought"
                },
                ["generated_thought", "biblical_context"]
            )

            logger.info(f"[THOUGHT PROCESS] Generated thought: {thought}")
            return thought

        except Exception as e:
            logger.error(f"[THOUGHT PROCESS ERROR] {e}", exc_info=True)
            return f"Error in thought process: {str(e)}"

    def _generate_thought(self, subjects: List[str], actions: List[str], 
                         objects: List[str], entities: List[str], context: str) -> str:
        """
        Generate contextual thought with biblical perspective.
        """
        base_thought = f"Reflecting on {', '.join(subjects)} and their actions of {', '.join(actions)}"
        
        if objects:
            base_thought += f", considering how {random.choice(actions)}ing {', '.join(objects)}"
        
        if entities:
            base_thought += f" relates to {', '.join(entities)}"
            
        base_thought += f" within the context of {context}."
        
        return base_thought