from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging
from datetime import datetime
from config.utils import get_sentence_transformer_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SemanticBuilder:
    def __init__(self, memory_engine, similarity_threshold=0.8, batch_size=1000):
        """
        Initialize SemanticBuilder with MemoryEngine for semantic analysis operations.

        :param memory_engine: An instance of MemoryEngine for memory operations.
        :param similarity_threshold: Threshold for considering two nodes similar.
        :param batch_size: Number of nodes to process in one batch for performance.
        """
        self.memory_engine = memory_engine
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        
        # Use the same model as in MemoryEngine for consistency
        self.similarity_model = get_sentence_transformer_model()

    def compute_similarities(self, texts: List[str]) -> np.ndarray:
        """
        Compute cosine similarity between all pairs of texts using embeddings.

        :param texts: List of text descriptions
        :return: Similarity matrix
        """
        embeddings = self.similarity_model.encode(texts)
        return util.cos_sim(embeddings, embeddings).numpy()

    def build_relationships(self, label="MemoryNode", relationship_type="SIMILAR_TO"):
        """
        Build relationships in the graph based on similarity of node descriptions, considering chunks and sentences.

        :param label: The label of nodes to analyze
        :param relationship_type: Type of relationship to create based on similarity
        """
        nodes = self.memory_engine.retrieve_all_memories()
        all_texts = []
        
        # Gather all relevant text from memories, chunks, and sentences
        for node in nodes:
            if 'content' in node:
                all_texts.append(node['content'])
            if 'chunks' in node:
                for chunk in node['chunks']:
                    all_texts.append(chunk['content'])
                    if 'sentences' in chunk:
                        all_texts.extend(chunk['sentences'])
        
        if not all_texts:
            logger.warning(f"No texts found for label: {label}")
            return

        similarities = self.compute_similarities(all_texts)

        # Assuming IDs are stored in a way that we can map back to them from the index
        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):
                if similarities[i][j] > self.similarity_threshold:
                    # Here you need to map back to the actual node or chunk IDs
                    node1_id = self._get_id_from_text_index(i, nodes)
                    node2_id = self._get_id_from_text_index(j, nodes)
                    self.memory_engine.create_relationship(node1_id, node2_id, relationship_type)

    def _get_id_from_text_index(self, index, nodes):
        """Helper method to map back from text index to node or chunk ID."""
        counter = 0
        for node in nodes:
            if 'content' in node:
                if counter == index:
                    return node['id']
                counter += 1
            if 'chunks' in node:
                for chunk in node['chunks']:
                    if counter == index:
                        return chunk['id'] if 'id' in chunk else node['id']  # Fallback to memory ID if chunk ID is not available
                    counter += 1
                    if 'sentences' in chunk:
                        for sentence in chunk['sentences']:
                            if counter == index:
                                return chunk['id'] if 'id' in chunk else node['id']  # Use chunk or memory ID for sentences
                            counter += 1
                return None  # In case we don't find a match, though this should not happen with correct indexing

    def create_relationship(self, id1, id2, relationship_type):
        """
        Alias method for creating relationships, using MemoryEngine's method.

        :param id1: ID of the first node
        :param id2: ID of the second node
        :param relationship_type: Type of relationship to establish
        """
        self.memory_engine.create_relationship(id1, id2, relationship_type)

    def detect_narrative_shifts(self) -> List[Dict]:
        """
        Detect shifts in narrative themes in the graph.

        :return: List of detected shifts with additional context.
        """
        memories = self.memory_engine.retrieve_all_memories()
        memories.sort(key=lambda x: x['metadata'].get('created_at', ''))
        shifts = []
        for i in range(1, len(memories)):
            if 'theme' in memories[i]['metadata'] and 'theme' in memories[i-1]['metadata']:
                if memories[i]['metadata']['theme'] != memories[i-1]['metadata']['theme']:
                    # Handle potential timestamp format issues
                    try:
                        timestamp = datetime.fromisoformat(memories[i]['metadata']['created_at'])
                        shift_description = f"Shift from '{memories[i-1]['metadata']['theme']}' to '{memories[i]['metadata']['theme']}' at {timestamp}."
                    except ValueError:
                        # If the timestamp format is not ISO, assume it's a string representation
                        timestamp = memories[i]['metadata']['created_at']
                        shift_description = f"Shift from '{memories[i-1]['metadata']['theme']}' to '{memories[i]['metadata']['theme']}' at {timestamp}."
                    
                    shifts.append({
                        "description": shift_description,
                        "old_theme": memories[i-1]['metadata']['theme'],
                        "new_theme": memories[i]['metadata']['theme'],
                        "timestamp": memories[i]['metadata']['created_at']
                    })
        logger.info(f"[NARRATIVE SHIFTS] Detected {len(shifts)} shifts.")
        return shifts

    def analyze_concept_evolution(self, concept: str) -> List[Dict]:
        """
        Analyze how a concept evolves over time in the system.

        :param concept: The concept to track evolution for.
        :return: List of evolution points with timestamps.
        """
        # This method assumes a structure where concepts evolve through memories or explicit evolution relationships
        memories = self.memory_engine.search_memories([concept])
        evolution_points = []
        for memory in memories:
            if 'evolved_to' in memory['metadata']:
                evolution_points.append({
                    "from": concept,
                    "to": memory['metadata']['evolved_to'],
                    "start_time": memory['metadata'].get('created_at', ''),
                    "end_time": memory['metadata'].get('evolved_at', '')
                })
        logger.info(f"[CONCEPT EVOLUTION] Analyzed evolution for {concept}: {len(evolution_points)} points.")
        return evolution_points

    def infer_relationships(self, text1: str, text2: str) -> str:
        """
        Infer a relationship between two pieces of text using semantic understanding.

        :param text1: First piece of text.
        :param text2: Second piece of text.
        :return: Inferred relationship type as a string.
        """
        embedding1 = self.similarity_model.encode([text1])[0]
        embedding2 = self.similarity_model.encode([text2])[0]
        similarity = util.cos_sim(embedding1, embedding2).item()
        
        if similarity > 0.9:
            return "DIRECTLY_RELATED"
        elif similarity > 0.7:
            return "INDIRECTLY_RELATED"
        else:
            return "UNRELATED"

    def infer_topic(self, text: str) -> str:
        """
        Infer the dominant topic or theme from a given text.

        :param text: Text to analyze for topic inference.
                :return: Inferred topic or theme.
        """
        # Placeholder for a more sophisticated topic modeling approach
        # Currently, it's a simple keyword-based approach
        keywords = ["science", "technology", "history", "culture", "politics"]
        for keyword in keywords:
            if keyword in text.lower():
                return keyword
        return "unknown"

    def detect_transition(self, current_input: str, conversation_context: str) -> str:
        """
        Detect if there's a transition in topic or theme from the conversation context to the current input.

        :param current_input: The current user input.
        :param conversation_context: The context of the conversation so far.
        :return: Description of the transition or 'No Transition' if none detected.
        """
        current_topic = self.infer_topic(current_input)
        context_topic = self.infer_topic(conversation_context)
        
        if current_topic != context_topic:
            return f"Transition detected from '{context_topic}' to '{current_topic}'"
        else:
            return "No Transition"

    def infer_topics(self, text: str) -> List[str]:
        """
        Infer multiple topics or themes from a given text.

        :param text: Text to analyze for topic inference.
        :return: List of inferred topics or themes.
        """
        # Placeholder for a more sophisticated topic modeling approach
        # Currently, it's a simple keyword-based approach
        keywords = ["science", "technology", "history", "culture", "politics", "economy", "environment", "health"]
        topics = []
        for keyword in keywords:
            if keyword in text.lower():
                topics.append(keyword)
        return topics if topics else ["unknown"]

# New methods added here

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze the sentiment of a given text, returning a sentiment score.

        :param text: Text to analyze for sentiment.
        :return: Sentiment score between -1 (very negative) and 1 (very positive).
        """
        from transformers import pipeline
        sentiment_pipeline = pipeline("sentiment-analysis")
        result = sentiment_pipeline(text)[0]
        return result['score'] if result['label'] == 'POSITIVE' else -result['score']

    def advanced_topic_modeling(self, text: str, num_topics: int = 5) -> List[Dict]:
        """
        Perform more advanced topic modeling on the text to identify multiple topics.

        :param text: Text to analyze for topics.
        :param num_topics: Number of topics to extract.
        :return: List of dictionaries with topic labels and their relevance scores.
        """
        from transformers import pipeline
        topic_model = pipeline("zero-shot-classification", model="joeddav/bart-large-mnli-yahoo-answers")
        candidate_labels = ["science", "technology", "history", "culture", "politics", "economy", "environment", "health", "sports", "entertainment"]
        results = topic_model(text, candidate_labels)
        
        # Filter and sort by score
        topics = sorted(
            [{"topic": label, "score": score} for label, score in zip(results['labels'], results['scores']) if score > 0.1],
            key=lambda x: x['score'], reverse=True
        )[:num_topics]
        
        return topics if topics else [{"topic": "unknown", "score": 0.0}]

    def analyze_temporal_trends(self, start_date: str, end_date: str, topic: str) -> Dict:
        """
        Analyze the frequency of a topic over a specified time range.

        :param start_date: Start date in 'YYYY-MM-DD' format.
        :param end_date: End date in 'YYYY-MM-DD' format.
        :param topic: Topic to analyze.
        :return: Dictionary with dates as keys and frequency counts as values.
        """
        memories = self.memory_engine.retrieve_memories_by_time_range(start_date, end_date)
        trend_data = {}
        for memory in memories:
            if 'theme' in memory['metadata'] and topic.lower() in memory['metadata']['theme'].lower():
                date = memory['metadata']['created_at'].split('T')[0]
                trend_data[date] = trend_data.get(date, 0) + 1
        
        return trend_data

    def analyze_relationship_strength(self, node_id: str) -> Dict:
        """
        Analyze the strength of relationships for a given node by counting related nodes.

        :param node_id: ID of the node to analyze.
        :return: Dictionary with relationship types as keys and counts as values.
        """
        relationships = self.memory_engine.get_relationships_for_node(node_id)
        strength = {}
        for rel in relationships:
            rel_type = rel['type']
            strength[rel_type] = strength.get(rel_type, 0) + 1
        
        return strength

# Usage example:
# sb = SemanticBuilder(memory_engine)
# sb.build_relationships(label="MemoryNode", relationship_type="SIMILAR_TO")