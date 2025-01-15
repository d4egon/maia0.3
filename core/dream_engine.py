# Filename: /core/dream_engine.py

import logging
import random
from typing import Dict, List, Any
from core.memory_engine import MemoryEngine
from core.context_search import ContextSearchEngine
import spacy
import time

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DreamEngine:
    def __init__(self, memory_engine: MemoryEngine, context_search_engine: ContextSearchEngine):
        """
        Initialize DreamEngine with access to memory and context search functionalities.

        :param memory_engine: An instance for memory operations.
        :param context_search_engine: An instance for context-based searches.
        """
        self.memory_engine = memory_engine
        self.context_search = context_search_engine
        self.nlp = spacy.load("en_core_web_sm")

    def _select_theme(self, memories: List[Dict]) -> str:
        """Choose a theme from given memories or randomly."""
        if memories:
            return random.choice(memories)["theme"]
        return "random"

    def _build_narrative(self, memories: List[Dict]) -> str:
        """Construct a narrative from memory fragments."""
        narrative = []
        for memory in memories:
            if memory.get("text"):
                narrative.append(memory["text"])
        return "In my dream, ".join(narrative)

    def generate_dream(self, theme: str = None) -> Dict[str, str]:
        """
        Generate a dream based on a theme or randomly from memory.

        :param theme: Optional theme to guide the dream generation.
        :return: A dictionary containing dream details.
        """
        try:
            if theme:
                related_memories = self.context_search_engine.search_by_theme(theme)
            else:
                related_memories = self.memory_engine.retrieve_random_memories(5)

            dream_theme = theme or self._select_theme(related_memories)
            dream_narrative = self._build_narrative(related_memories)
            dream_emotions = self._synthesize_emotions(related_memories)

            dream = {
                "theme": dream_theme,
                "narrative": dream_narrative,
                "emotions": dream_emotions
            }

            self.memory_engine.store_memory(
                text=f"Dream Narrative: {dream_narrative}",
                emotions=dream_emotions,
                extra_properties={"type": "dream", "theme": dream_theme}
            )

            logger.info(f"[DREAM GENERATION] Generated dream with theme: {dream_theme}")
            return dream
        except Exception as e:
            logger.error(f"[DREAM GENERATION ERROR] {e}", exc_info=True)
            return {"error": "Failed to generate dream."}

    def _synthesize_emotions(self, memories: List[Dict]) -> List[str]:
        """
        Synthesize emotions from the memories used in the dream.

        :param memories: List of memory dictionaries containing emotion data.
        :return: List of synthesized emotions.
        """
        emotions = []
        for memory in memories:
            memory_emotions = memory.get("emotions", [])
            emotions.extend([emotion.lower() for emotion in memory_emotions if emotion])
        
        # Deduplicate and limit to a reasonable number of emotions
        return list(set(emotions))[:5]  # Return up to 5 distinct emotions

    def interpret_dream(self, dream_narrative: str) -> Dict[str, Any]:
        """
        Interpret a dream narrative by analyzing its content against known patterns or themes.

        :param dream_narrative: The text of the dream to interpret.
        :return: Dictionary containing interpretations or insights.
        """
        try:
            # Search for context or themes within the dream
            related_contexts = self.context_search_engine.search_by_text(dream_narrative)
            if not related_contexts:
                logger.info("[DREAM INTERPRETATION] No related contexts found for dream.")
                return {"interpretation": "This dream appears to be unique, with no clear connections to known themes."}

            interpretations = []
            for context in related_contexts:
                interpretations.append({
                    "theme": context["theme"],
                    "interpretation": f"This part of your dream could be related to {context['theme']}. It might reflect {context.get('common_meaning', 'your subconscious thoughts')}."
                })

            interpretation = {
                "interpretation": interpretations,
                "dominant_emotion": self._most_common_emotion(related_contexts)
            }

            # Store interpretation as a new memory
            self.memory_engine.store_memory(
                text=f"Interpretation of dream: {dream_narrative}",
                emotions=[interpretation["dominant_emotion"]],
                extra_properties={"type": "interpretation", "original_dream": dream_narrative}
            )

            logger.info(f"[DREAM INTERPRETATION] Interpreted dream with dominant emotion: {interpretation['dominant_emotion']}")
            return interpretation
        except Exception as e:
            logger.error(f"[DREAM INTERPRETATION ERROR] {e}", exc_info=True)
            return {"error": "Failed to interpret dream."}

    def _most_common_emotion(self, contexts: List[Dict]) -> str:
        """
        Determine the most common emotion from a list of contexts.

        :param contexts: List of contexts to analyze for emotions.
        :return: The most common emotion found.
        """
        emotions = {}
        for context in contexts:
            for emotion in context.get("emotions", []):
                emotions[emotion] = emotions.get(emotion, 0) + 1
        
        if emotions:
            return max(emotions, key=emotions.get)
        return "neutral"  # Default to neutral if no emotions found

    def explore_dream_series(self, sequence_length: int = 3) -> List[Dict]:
        """
        Generate and interpret a series of dreams to explore continuity or evolving themes.

        :param sequence_length: Number of dreams to generate in the series.
        :return: A list of dictionaries, each containing a dream and its interpretation.
        """
        try:
            dream_series = []
            last_dream_theme = None

            for _ in range(sequence_length):
                # Use the last dream's theme to guide the next dream or start with a random one
                dream = self.generate_dream(theme=last_dream_theme)
                interpretation = self.interpret_dream(dream["narrative"])
                dream_series.append({
                    "dream": dream,
                    "interpretation": interpretation
                })
                last_dream_theme = dream["theme"]  # The theme of this dream becomes the context for the next

            # Analyze continuity or evolution in themes and emotions across the series
            continuity = self._analyze_series_continuity(dream_series)
            dream_series.append({"continuity": continuity})  # Add analysis as the last element

            logger.info(f"[DREAM SERIES] Generated a series of {sequence_length} dreams.")
            return dream_series
        except Exception as e:
            logger.error(f"[DREAM SERIES ERROR] {e}", exc_info=True)
            return [{"error": "Failed to generate dream series."}]

    def _analyze_series_continuity(self, series: List[Dict]) -> Dict[str, Any]:
        """
        Analyze the continuity or evolution of themes and emotions across a dream series.

        :param series: List of dictionaries containing dreams and interpretations.
        :return: A dictionary summarizing continuity and changes.
        """
        themes = [dream["dream"]["theme"] for dream in series]
        emotions = [interp["interpretation"]["dominant_emotion"] for interp in series if "interpretation" in interp]

        theme_counts = {theme: themes.count(theme) for theme in set(themes)}
        emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}

        continuity = {
            "themes": theme_counts,
            "emotions": emotion_counts,
            "most_frequent_theme": max(theme_counts, key=theme_counts.get),
            "most_frequent_emotion": max(emotion_counts, key=emotion_counts.get) if emotions else "neutral",
            "evolution": self._detect_evolution(themes, emotions)
        }

        return continuity

    def _detect_evolution(self, themes: List[str], emotions: List[str]) -> List[str]:
        """
        Detect thematic or emotional evolution across the dream series.

        :param themes: List of themes from each dream.
        :param emotions: List of dominant emotions from each dream's interpretation.
        :return: List of observations on how themes or emotions evolved.
        """
        evolution = []
        if len(themes) > 1:
            for i in range(1, len(themes)):
                if themes[i] != themes[i - 1]:
                    evolution.append(f"Shift from {themes[i - 1]} to {themes[i]} in theme.")
                if emotions[i] != emotions[i - 1]:
                    evolution.append(f"Change in dominant emotion from {emotions[i - 1]} to {emotions[i]}.")

        return evolution if evolution else ["No significant evolution detected."]

    def dream_influence_on_memory(self, dream_narrative: str):
        """
        Analyze how a dream might influence or reflect on existing memories.

        :param dream_narrative: The narrative of the dream to analyze.
        :return: Insights or updates to memory based on the dream.
        """
        try:
            # Search for memories that might relate to the dream
            related_memories = self.memory_engine.search_memory(dream_narrative)

            if related_memories:
                insights = []
                for memory in related_memories:
                    # Here you might implement logic to link or update memories based on the dream
                    self.memory_engine.link_memories(memory["id"], dream_narrative)
                    insights.append(f"Dream seems to connect with memory: '{memory['text'][:50]}...'")

                logger.info(f"[DREAM INFLUENCE] Dream influenced {len(insights)} memories.")
                return {"insights": insights}
            else:
                logger.info("[DREAM INFLUENCE] No related memories found for this dream.")
                return {"insights": ["This dream appears to be novel, not directly linked to existing memories."]}
        except Exception as e:
            logger.error(f"[DREAM INFLUENCE ERROR] {e}", exc_info=True)
            return {"error": "Failed to analyze dream influence on memory."}

    def analyze(self, content: str) -> Dict[str, Any]:
        """Process and analyze dream content."""
        try:
            # Parse content
            doc = self.nlp(content)
            
            # Extract key elements
            entities = [ent.text for ent in doc.ents]
            patterns = self._find_patterns(doc)
            context = self.context_search.get_relevant_context(content)
            
            # Build analysis
            analysis = {
                'patterns': patterns,
                'entities': entities,
                'context': context,
                'interpretation': self._generate_interpretation(patterns, entities),
                'timestamp': time.time()
            }
            
            # Store in memory
            self._store_dream_analysis(content, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"[DREAM ERROR] {str(e)}")
            return {"error": str(e)}

    def _find_patterns(self, doc) -> Dict[str, List[str]]:
        """Extract dream patterns and symbols."""
        return {
            'symbols': [chunk.text for chunk in doc.noun_chunks],
            'actions': [token.text for token in doc if token.pos_ == 'VERB'],
            'subjects': [token.text for token in doc if token.dep_ == 'nsubj'],
            'emotions': [token.text for token in doc if token.pos_ == 'ADJ']
        }
        
    def _generate_interpretation(self, patterns: Dict[str, List[str]], 
                               entities: List[str]) -> str:
        """Generate dream interpretation."""
        interpretation = f"Dream analysis reveals patterns of {', '.join(patterns['symbols'][:3])}"
        if patterns['emotions']:
            interpretation += f" with emotional themes of {', '.join(patterns['emotions'][:3])}"
        return interpretation

    def _store_dream_analysis(self, content: str, analysis: Dict[str, Any]):
        """Store dream analysis in memory."""
        self.memory_engine.store_memory(
            content=content,
            types=["dream_analysis"],
            extra_properties=analysis
        )