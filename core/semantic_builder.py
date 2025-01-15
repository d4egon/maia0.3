# File: /core/semantic_builder.py

from transformers import pipeline
from sentence_transformers import SentenceTransformer # type: ignore
import torch
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SemanticBuilder:
    def __init__(self, graph_client, similarity_threshold=0.8, batch_size=1000):
        """
        Initialize SemanticBuilder with necessary components for semantic analysis.

        :param graph_client: Client to interact with the graph database.
        :param similarity_threshold: Threshold for considering two nodes similar.
        :param batch_size: Number of nodes to process in one batch for performance.
        """
        self.graph_client = graph_client
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size

        # Use SentenceTransformer for potentially better performance in similarity tasks
        self.similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Example model, can be changed
        
        # Additional model for semantic analysis if needed
        self.semantic_model = pipeline("fill-mask", model="bert-base-uncased")

    def compute_similarities(self, texts):
        """
        Compute cosine similarity between all pairs of texts using embeddings.

        :param texts: List of text descriptions
        :return: Similarity matrix
        """
        embeddings = self.similarity_model.encode(texts)
        return cosine_similarity(embeddings)

    def build_relationships(self, label="Emotion", relationship_type="SIMILAR_TO"):
        """
        Build relationships in the graph based on similarity of node descriptions.

        :param label: The label of nodes to analyze
        :param relationship_type: Type of relationship to create based on similarity
        """
        query = f"""
        MATCH (e:{label})
        WHERE e.description IS NOT NULL
        RETURN e.id AS id, e.name AS name, e.description AS description
        """
        try:
            nodes = self.graph_client.run_query(query)
            descriptions = [node["description"] for node in nodes]
            
            if not descriptions:
                logger.warning(f"No descriptions found for label: {label}")
                return

            similarities = self.compute_similarities(descriptions)

            for i in range(len(similarities)):
                for j in range(i + 1, len(similarities)):
                    if similarities[i][j] > self.similarity_threshold:
                        self.create_relationship(nodes[i]['id'], nodes[j]['id'], relationship_type)

        except Exception as e:
            logger.error(f"Error building relationships: {e}")

    def create_relationship(self, id1, id2, relationship_type):
        """
        Create a relationship between two nodes in the graph.

        :param id1: ID of the first node
        :param id2: ID of the second node
        :param relationship_type: Type of relationship to establish
        """
        query = f"""
        MATCH (n1), (n2)
        WHERE n1.id = '{id1}' AND n2.id = '{id2}'
        MERGE (n1)-[:{relationship_type}]->(n2)
        """
        try:
            self.graph_client.run_query(query)
            logger.info(f"Relationship {relationship_type} created between {id1} and {id2}")
        except Exception as e:
            logger.error(f"Failed to create relationship between {id1} and {id2}: {e}")

    def detect_narrative_shifts(self):
        """
        Detect shifts in narrative themes in the graph.

        :return: List of detected shifts with additional context.
        """
        query = """
        MATCH (m:Memory)
        WHERE m.theme IS NOT NULL AND m.timestamp IS NOT NULL
        RETURN m.text AS text, m.theme AS theme, m.timestamp AS timestamp
        ORDER BY m.timestamp
        """
        try:
            nodes = self.graph_client.run_query(query)
            shifts = []
            for i in range(1, len(nodes)):
                if nodes[i]["theme"] != nodes[i - 1]["theme"]:
                    timestamp = datetime.strptime(nodes[i]['timestamp'], "%Y-%m-%d %H:%M:%S")  # Adjust format if needed
                    shift_description = f"Shift from '{nodes[i - 1]['theme']}' to '{nodes[i]['theme']}' at {timestamp}."
                    shifts.append({
                        "description": shift_description,
                        "old_theme": nodes[i - 1]["theme"],
                        "new_theme": nodes[i]["theme"],
                        "timestamp": nodes[i]["timestamp"]
                    })
            logger.info(f"[NARRATIVE SHIFTS] Detected {len(shifts)} shifts.")
            return shifts
        except Exception as e:
            logger.error(f"[SHIFT DETECTION ERROR] {e}")
            return []

    def analyze_concept_evolution(self, concept):
        """
        Analyze how a concept evolves over time in the graph.

        :param concept: The concept to track evolution for.
        :return: List of evolution points with timestamps.
        """
        query = f"""
        MATCH (c:Concept {{name: '{concept}'}})-[r:EVOLVES_TO]->(next)
        RETURN c.name AS name, c.timestamp AS timestamp, next.name AS evolved_to, next.timestamp AS evolved_timestamp
        ORDER BY c.timestamp
        """
        try:
            evolutions = self.graph_client.run_query(query)
            evolution_points = []
            for evolution in evolutions:
                evolution_points.append({
                    "from": evolution["name"],
                    "to": evolution["evolved_to"],
                    "start_time": evolution["timestamp"],
                    "end_time": evolution["evolved_timestamp"]
                })
            logger.info(f"[CONCEPT EVOLUTION] Analyzed evolution for {concept}: {len(evolution_points)} points.")
            return evolution_points
        except Exception as e:
            logger.error(f"[EVOLUTION ANALYSIS ERROR] {e}")
            return []

    def infer_relationships(self, text1, text2):
        """
        Infer a relationship between two pieces of text using semantic understanding.

        :param text1: First piece of text.
        :param text2: Second piece of text.
        :return: Inferred relationship type.
        """
        combined_text = f"{text1} [MASK] {text2}"
        result = self.semantic_model(combined_text)
        # Here, you'd need logic to interpret the mask predictions into relationships
        # For now, let's just return the top prediction
        return result[0]['token_str']

# Usage example:
# sb = SemanticBuilder(graph_client)
# sb.build_relationships(label="Emotion", relationship_type="SIMILAR_TO")