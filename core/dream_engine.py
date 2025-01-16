import logging
import random
from typing import Dict, List, Any
from core.memory_engine import MemoryEngine
from core.context_search import ContextSearchEngine
from sentence_transformers import SentenceTransformer, util
import numpy as np
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
        self.model = memory_engine.model  # Use the same model from MemoryEngine for semantic analysis
        self.chaos_level = 0.3  # Default level of chaos

    def set_chaos_level(self, level: float):
        """Adjust the level of chaos in dream processing."""
        self.chaos_level = min(max(level, 0.0), 1.0)

    def _select_theme(self, memories: List[Dict]) -> str:
        """Choose a theme with a chance for chaos."""
        if not memories:
            return "random"
        
        # Introduce chaos by sometimes picking a random theme
        if random.random() < self.chaos_level:
            return random.choice(["chaos", "anarchy", "surreal", "absurd"])
        return random.choice(memories)["metadata"].get("theme", "random")

    def _build_narrative(self, memories: List[Dict]) -> str:
        """Construct a narrative with chaotic elements."""
        narrative = []
        for memory in memories:
            if memory.get("text"):
                # Add a chance to distort or rearrange text
                if random.random() < self.chaos_level:
                    narrative.append(self._chaos_mix(memory["text"]))
                else:
                    narrative.append(memory["text"])
        
        # Shuffle the narrative segments for added chaos
        if random.random() < self.chaos_level:
            random.shuffle(narrative)
        
        return "In a chaotic dream, ".join(narrative)

    def _chaos_mix(self, text: str) -> str:
        """Randomly mix up words in the text."""
        words = text.split()
        random.shuffle(words)
        return " ".join(words)

    def generate_dream(self, theme: str = None) -> Dict[str, str]:
        """
        Generate a dream with potential chaotic elements.

        :param theme: Optional theme to guide the dream generation.
        :return: A dictionary containing dream details.
        """
        try:
            if theme:
                related_memories = self.context_search.search_related_contexts(theme)
            else:
                # Add randomness to how many memories we pull
                related_memories = self.memory_engine.retrieve_all_memories()[:random.randint(2, 10)]  

            dream_theme = theme or self._select_theme(related_memories)
            dream_narrative = self._build_narrative(related_memories)
            dream_emotions = self._synthesize_emotions(related_memories)

            if random.random() < self.chaos_level:
                dream_emotions = self._chaotic_emotions(dream_emotions)

            dream = {
                "theme": dream_theme,
                "narrative": dream_narrative,
                "emotions": dream_emotions
            }

            dream_embedding = self.model.encode([dream_narrative])[0].tolist()
            self.memory_engine.create_memory_node(
                dream_narrative, 
                {
                    "type": "dream", 
                    "theme": dream_theme,
                    "emotions": dream_emotions,
                    "embedding": dream_embedding,
                    "chaos_level": self.chaos_level
                },
                [dream_theme]  # Use theme as keyword
            )

            logger.info(f"[DREAM GENERATION] Generated dream with theme: {dream_theme}, chaos level: {self.chaos_level}")
            return dream
        except Exception as e:
            logger.error(f"[DREAM GENERATION ERROR] {e}", exc_info=True)
            return {"error": "Failed to generate dream."}

    def _synthesize_emotions(self, memories: List[Dict]) -> List[str]:
        """
        Synthesize emotions with a chance for chaotic influence.

        :param memories: List of memory dictionaries containing emotion data.
        :return: List of synthesized emotions.
        """
        emotions = []
        for memory in memories:
            memory_emotions = memory.get("metadata", {}).get("emotions", [])
            emotions.extend([emotion.lower() for emotion in memory_emotions if emotion])
        
        return list(set(emotions))[:5]  # Return up to 5 distinct emotions

    def _chaotic_emotions(self, emotions: List[str]) -> List[str]:
        """Randomly alter or add to emotions for chaotic effect."""
        if random.random() < self.chaos_level:
            emotions.append(random.choice(["confused", "ecstatic", "paranoid", "melancholic"]))
        return emotions

    def interpret_dream(self, dream_narrative: str) -> Dict[str, Any]:
        """
        Interpret a dream with a chaotic twist.

        :param dream_narrative: The text of the dream to interpret.
        :return: Dictionary containing interpretations or insights.
        """
        try:
            dream_embedding = self.model.encode([dream_narrative])[0]
            related_contexts = self.context_search.search_related_contexts(dream_narrative, similarity_threshold=0.6 - self.chaos_level * 0.4)

            if not related_contexts or random.random() < self.chaos_level:
                return {"interpretation": "This dream defies conventional interpretation; it's a chaotic tapestry of the mind."}

            interpretations = []
            for context in related_contexts:
                if random.random() < self.chaos_level:
                    interpretations.append({
                        "theme": "chaos",
                        "interpretation": "The dream seems to be an explosion of random thoughts."
                    })
                else:
                    interpretations.append({
                        "theme": context["metadata"].get("theme", "unknown"),
                        "interpretation": f"This part of your dream could be related to {context['metadata'].get('theme', 'your subconscious')}. It might reflect {context.get('metadata', {}).get('common_meaning', 'your subconscious thoughts')}."
                    })

            interpretation = {
                "interpretation": interpretations,
                "dominant_emotion": self._most_common_emotion(related_contexts)
            }

            if random.random() < self.chaos_level:
                interpretation["dominant_emotion"] = random.choice(["baffled", "incomprehensible", "weird"])

            interpretation_embedding = self.model.encode([interpretation["interpretation"][0]["interpretation"]])[0].tolist()
            self.memory_engine.create_memory_node(
                f"Interpretation of dream: {dream_narrative}",
                {
                    "type": "interpretation", 
                    "original_dream": dream_narrative,
                    "dominant_emotion": interpretation["dominant_emotion"],
                    "embedding": interpretation_embedding,
                    "chaos_level": self.chaos_level
                },
                [interpretation["dominant_emotion"]]
            )

            logger.info(f"[DREAM INTERPRETATION] Interpreted dream with dominant emotion: {interpretation['dominant_emotion']}, chaos level: {self.chaos_level}")
            return interpretation
        except Exception as e:
            logger.error(f"[DREAM INTERPRETATION ERROR] {e}", exc_info=True)
            return {"error": "Failed to interpret dream."}

    def _most_common_emotion(self, contexts: List[Dict]) -> str:
        """
        Determine the most common emotion with a chance for chaos.

        :param contexts: List of contexts to analyze for emotions.
        :return: The most common emotion found or a chaotic one.
        """
        emotions = {}
        for context in contexts:
            for emotion in context.get("metadata", {}).get("emotions", []):
                emotions[emotion] = emotions.get(emotion, 0) + 1
        
        if emotions and random.random() > self.chaos_level:
            return max(emotions, key=emotions.get)
        return random.choice(["chaotic", "unpredictable", "frenzied"])

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
            # Search for memories that might relate to the dream using semantic similarity
            dream_embedding = self.model.encode([dream_narrative])[0]
            related_memories = self.memory_engine.search_memory_by_embedding(dream_embedding.tolist())

            if related_memories:
                insights = []
                for memory in related_memories:
                    self.memory_engine.create_relationship(
                        memory["id"], 
                        dream_narrative, 
                        "INFLUENCED_BY_DREAM"
                    )
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
            # Semantic analysis
            content_embedding = self.model.encode([content])[0].tolist()
            
            # Extract key elements
            entities = [ent.text for ent in self.model.nlp(content).ents]  # Assuming NLP is part of the sentence transformer model
            patterns = self._find_patterns(content)
            context = self.context_search.search_related(content)
            
            # Build analysis
            analysis = {
                'patterns': patterns,
                'entities': entities,
                'context': context,
                'interpretation': self._generate_interpretation(patterns, entities),
                'embedding': content_embedding,
                'timestamp': time.time()
            }
            
            # Store in memory
            self._store_dream_analysis(content, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"[DREAM ERROR] {str(e)}")
            return {"error": str(e)}

    def _find_patterns(self, content: str) -> Dict[str, List[str]]:
        """Extract dream patterns with chaotic elements."""
        doc = self.model.nlp(content)
        patterns = {
            'symbols': [chunk.text for chunk in doc.noun_chunks],
            'actions': [token.text for token in doc if token.pos_ == 'VERB'],
            'subjects': [token.text for token in doc if token.dep_ == 'nsubj'],
            'emotions': [token.text for token in doc if token.pos_ == 'ADJ']
        }
        
        if random.random() < self.chaos_level:
            for key in patterns:
                patterns[key].append(random.choice(["chaos", "disorder", "unknown"]))
        
        return patterns

    def _generate_interpretation(self, patterns: Dict[str, List[str]], 
                               entities: List[str]) -> str:
        """Generate dream interpretation with chaotic elements."""
        base_interpret = f"Dream analysis reveals patterns of {', '.join(patterns['symbols'][:3])}"
        if patterns['emotions']:
            base_interpret += f" with emotional themes of {', '.join(patterns['emotions'][:3])}"
        
        if random.random() < self.chaos_level:
            base_interpret += f", but chaos reigns supreme, suggesting a break from conventional interpretation."
        return base_interpret

    def _store_dream_analysis(self, content: str, analysis: Dict[str, Any]):
        """Store dream analysis in memory with chaos level."""
        self.memory_engine.create_memory_node(
            content,
            {
                "type": "dream_analysis",
                "analysis": analysis,
                "chaos_level": self.chaos_level
            },
            []
        )