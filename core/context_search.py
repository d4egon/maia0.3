# Filename: /core/context_search.py

import logging
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Union
import numpy as np

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ContextSearchEngine:
    def __init__(self, neo4j_connector):
        """
        Initialize ContextSearchEngine with Neo4j connector and sentence embedding model.

        :param neo4j_connector: An instance of Neo4jConnector for database operations.
        """
        self.db = neo4j_connector
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')  # Updated model

    def search_related_contexts(self, text: str, similarity_threshold: float = 0.7, top_n: int = 20) -> List[Dict]:
        """
        Search for related contexts in the database based on semantic similarity.

        :param text: Text to find related contexts for.
        :param similarity_threshold: Minimum cosine similarity score to consider a match.
        :param top_n: Maximum number of results to return.
        :return: List of dictionaries with related memories and their similarity scores.
        """
        try:
            # First, get potentially related contexts via full-text search
            query = """
            CALL db.index.fulltext.queryNodes('memoryIndex', $text)
            YIELD node, score
            RETURN node.text AS Memory, node.weight AS Weight, score
            ORDER BY score DESC LIMIT 100  # Fetch more to filter with embeddings
            """
            results = self.db.run_query(query, {"text": text.lower()})

            if not results:
                logger.info(f"[CONTEXT SEARCH] No memories found for '{text}'")
                return []

            input_embedding = self.embedding_model.encode(text.lower(), convert_to_tensor=True)
            filtered_results = []
            for record in results:
                memory_text = record["Memory"]
                memory_embedding = self.embedding_model.encode(memory_text.lower(), convert_to_tensor=True)
                similarity_score = util.cos_sim(input_embedding, memory_embedding).item()

                if similarity_score >= similarity_threshold:
                    filtered_results.append({
                        "memory": memory_text,
                        "weight": record["Weight"],
                        "similarity": round(similarity_score, 4)
                    })

            # Sort by similarity and limit to top_n
            sorted_results = sorted(filtered_results, key=lambda x: x['similarity'], reverse=True)
            final_results = sorted_results[:top_n]

            logger.info(f"[CONTEXT SEARCH] Found {len(final_results)} relevant contexts.")
            return final_results
        except Exception as e:
            logger.error(f"[SEARCH FAILED] {e}", exc_info=True)
            return []

    def search_related(self, text: str) -> List[Dict]:
        """
        Wrapper for `search_related_contexts` to align with ConversationEngine calls.
        """
        return self.search_related_contexts(text)

    def create_dynamic_links(self, source_text: str, related_memories: List[Dict], link_type: str = "RELATED_TO"):
        """
        Create dynamic relationships in the graph database based on contextual matching.

        :param source_text: The source memory text.
        :param related_memories: List of dictionaries containing related memory texts.
        :param link_type: Type of relationship to create.
        """
        try:
            for memory in related_memories:
                query = f"""
                MATCH (s:Memory {{text: $source_text}})
                MERGE (r:Memory {{text: $related_text}})
                MERGE (s)-[:{link_type} {{similarity: $similarity}}]->(r)
                """
                self.db.run_query(query, {
                    "source_text": source_text.lower(),
                    "related_text": memory["memory"].lower(),
                    "similarity": memory["similarity"]
                })
                logger.info(f"[LINK CREATED] '{source_text}' ↔ '{memory['memory']}' with similarity {memory['similarity']}")
        except Exception as e:
            logger.error(f"[LINK CREATION FAILED] {e}", exc_info=True)

    def advanced_context_matching(self, input_text: str, similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Match contexts using semantic embeddings and deeper similarity thresholds.

        :param input_text: Text to match against the graph.
        :param similarity_threshold: Minimum similarity score for matching.
        :return: List of matching contexts.
        """
        try:
            input_embedding = self.embedding_model.encode(input_text, convert_to_tensor=True)
            query = """
            MATCH (m:Memory)
            RETURN m.text AS text
            """
            matches = self.db.run_query(query)
            
            scored_matches = []
            for match in matches:
                match_embedding = self.embedding_model.encode(match["text"], convert_to_tensor=True)
                score = util.cos_sim(input_embedding, match_embedding).item()
                if score > similarity_threshold:
                    scored_matches.append({
                        "text": match["text"],
                        "score": round(score, 4)
                    })

            # Sort matches by score to show most relevant first
            sorted_matches = sorted(scored_matches, key=lambda x: x['score'], reverse=True)
            logger.info(f"[ADVANCED MATCHING] Found {len(sorted_matches)} matches above threshold {similarity_threshold}.")
            return sorted_matches
        except Exception as e:
            logger.error(f"[MATCHING ERROR] {e}", exc_info=True)
            return []

    def thematic_context_search(self, theme: str, similarity_threshold: float = 0.6) -> List[Dict]:
        """
        Search for contexts related to a specific theme.

        :param theme: The theme to search for.
        :param similarity_threshold: Minimum similarity score for thematic relevance.
        :return: List of contexts related to the theme.
        """
        try:
            # Assuming themes are stored in a property of Memory nodes
            query = f"""
            MATCH (m:Memory)-[:THEME_OF]->(t:Theme {{name: '{theme.lower()}'}})
            RETURN m.text AS text
            """
            memories = self.db.run_query(query)

            if not memories:
                logger.info(f"[THEMATIC SEARCH] No memories found for theme '{theme}'")
                return []

            theme_embedding = self.embedding_model.encode(theme.lower(), convert_to_tensor=True)
            thematic_contexts = []
            for memory in memories:
                memory_embedding = self.embedding_model.encode(memory["text"].lower(), convert_to_tensor=True)
                similarity = util.cos_sim(theme_embedding, memory_embedding).item()
                if similarity >= similarity_threshold:
                    thematic_contexts.append({
                        "text": memory["text"],
                        "similarity_to_theme": round(similarity, 4)
                    })

            logger.info(f"[THEMATIC SEARCH] Found {len(thematic_contexts)} contexts for theme '{theme}'.")
            return thematic_contexts
        except Exception as e:
            logger.error(f"[THEMATIC SEARCH ERROR] {e}", exc_info=True)
            return []

    def context_evolution(self, start_date: str, end_date: str, theme: str = None) -> Dict:
        """
        Analyze how contexts evolve over time within a specified date range and optional theme.

        :param start_date: Start date for analysis in 'YYYY-MM-DD' format.
        :param end_date: End date for analysis in 'YYYY-MM-DD' format.
        :param theme: Optional theme to filter contexts by
        :return: Dictionary summarizing context evolution.
        """
        try:
            dates_query = f"""
            MATCH (m:Memory)
            WHERE m.created_at >= datetime($start_date) AND m.created_at <= datetime($end_date)
            """
            theme_query = f" MATCH (m)-[:THEME_OF]->(t:Theme {{name: '{theme.lower()}'}})" if theme else ""
            query = f"""
            {dates_query}
            {theme_query}
            RETURN m.text AS text, m.created_at AS date, m.theme AS theme
            ORDER BY m.created_at
            """

            results = self.db.run_query(query, {"start_date": start_date, "end_date": end_date})

            if not results:
                logger.info(f"[CONTEXT EVOLUTION] No contexts found for the given date range and theme.")
                return {"error": "No data found for the specified criteria."}

            # Analyze evolution
            evolution = {
                "total_contexts": len(results),
                "themes": {},
                "dates": {}
            }

            for result in results:
                date_str = result['date'].strftime('%Y-%m-%d')
                if result['theme']:
                    for theme in result['theme'].split(','):
                        theme = theme.strip().lower()
                        if theme not in evolution['themes']:
                            evolution['themes'][theme] = 0
                        evolution['themes'][theme] += 1

                if date_str not in evolution['dates']:
                    evolution['dates'][date_str] = 0
                evolution['dates'][date_str] += 1

            logger.info(f"[CONTEXT EVOLUTION] Analyzed {evolution['total_contexts']} contexts.")
            return evolution
        except Exception as e:
            logger.error(f"[CONTEXT EVOLUTION ERROR] {e}", exc_info=True)
            return {"error": "An error occurred during context evolution analysis."}

    def multi_modal_search(self, text: str, image_embedding: Union[List[float], np.ndarray] = None) -> List[Dict]:
        """
        Perform a search combining text and potential image embeddings for more nuanced context matching.

        :param text: Text to search for.
        :param image_embedding: Optional image embedding for multi-modal search.
        :return: List of matched contexts with scores.
        """
        try:
            text_embedding = self.embedding_model.encode(text, convert_to_tensor=True)
            
            # Combine text and image embeddings if image_embedding is provided
            if image_embedding is not None:
                if isinstance(image_embedding, list):
                    image_embedding = np.array(image_embedding)
                combined_embedding = np.concatenate([text_embedding.cpu().numpy(), image_embedding])

                query = """
                MATCH (m:Memory)
                RETURN m.text AS text, m.image_embedding AS image_embedding
                """
                matches = self.db.run_query(query)

                scored_matches = []
                for match in matches:
                    # Assuming image_embedding in Neo4j is stored as a list for compatibility with numpy array
                    memory_embedding = np.concatenate([
                        self.embedding_model.encode(match["text"], convert_to_tensor=True).cpu().numpy(),
                        np.array(match["image_embedding"]) if match["image_embedding"] else np.zeros_like(image_embedding)
                    ])
                    score = util.cos_sim(torch.tensor([combined_embedding]), torch.tensor([memory_embedding])).item()
                    if score > 0.7:  # Adjust threshold as needed
                        scored_matches.append({
                            "text": match["text"],
                            "score": round(score, 4)
                        })

            else:
                # Fallback to text-only search if no image embedding provided
                scored_matches = self.advanced_context_matching(text)

            logger.info(f"[MULTI-MODAL SEARCH] Found {len(scored_matches)} matches.")
            return scored_matches
        except Exception as e:
            logger.error(f"[MULTI-MODAL SEARCH ERROR] {e}", exc_info=True)
            return []