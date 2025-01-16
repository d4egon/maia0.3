from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging
from datetime import datetime

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
        self.similarity_model = memory_engine.model

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
        Build relationships in the graph based on similarity of node descriptions.

        :param label: The label of nodes to analyze
        :param relationship_type: Type of relationship to create based on similarity
        """
        nodes = self.memory_engine.retrieve_all_memories()
        texts = [node["text"] for node in nodes if "text" in node]
        
        if not texts:
            logger.warning(f"No texts found for label: {label}")
            return

        similarities = self.compute_similarities(texts)

        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):
                if similarities[i][j] > self.similarity_threshold:
                    self.memory_engine.create_relationship(nodes[i]['id'], nodes[j]['id'], relationship_type)

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
                    timestamp = datetime.fromisoformat(memories[i]['metadata']['created_at'])
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

# Usage example:
# sb = SemanticBuilder(memory_engine)
# sb.build_relationships(label="MemoryNode", relationship_type="SIMILAR_TO")