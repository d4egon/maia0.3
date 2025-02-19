# context_search.py
import logging
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Union
import numpy as np
from config.utils import get_sentence_transformer_model

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ContextSearchEngine:
    def __init__(self, neo4j):
        """
        Initialize ContextSearchEngine with Neo4j connection and fetch the centralized model.

        :param neo4j: An instance of Neo4jConnector for database operations.
        """
        self.neo4j = neo4j
        self.embedding_model = get_sentence_transformer_model()

    def search_related_contexts(self, content: str, similarity_threshold: float = 0.7, top_n: int = 20) -> List[Dict]:
        """
        Search for related contexts in the database based on semantic similarity, considering the new memory structure.

        :param content: Text to find related contexts for.
        :param similarity_threshold: Minimum cosine similarity score to consider a match.
        :param top_n: Maximum number of results to return.
        :return: List of dictionaries with related memories, chunks, or sentences and their similarity scores.
        """
        try:
            # Initial keyword search at the content level
            initial_results = self.memory_engine.search_memories([content.lower()])

            if not initial_results:
                logger.info(f"[CONTEXT SEARCH] No memories found for '{content}'")
                return []

            input_embedding = self.embedding_model.encode(content.lower(), convert_to_tensor=True)
            filtered_results = []

            for record in initial_results:
                # Check if the memory has chunks
                if 'chunks' in record:
                    for chunk in record['chunks']:
                        chunk_embedding = torch.tensor(chunk['embedding'])  # Assuming each chunk has an embedding
                        similarity_score = util.cos_sim(input_embedding, chunk_embedding).item()

                        if similarity_score >= similarity_threshold:
                            filtered_results.append({
                                "memory": record["content"],
                                "chunk": chunk['content'],
                                "similarity": round(similarity_score, 4)
                            })
                else:
                    # If no chunks, fall back to memory level
                    memory_embedding = torch.tensor(record["embedding"])
                    similarity_score = util.cos_sim(input_embedding, memory_embedding).item()

                    if similarity_score >= similarity_threshold:
                        filtered_results.append({
                            "memory": record["content"],
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

    def search_related(self, content: str) -> List[Dict]:
        """
        Wrapper for `search_related_contexts` to align with ConversationEngine calls.
        """
        return self.search_related_contexts(content)

    def create_dynamic_links(self, source_text: str, related_memories: List[Dict], link_type: str = "RELATED_TO"):
        """
        Create dynamic relationships in the memory system based on contextual matching.

        :param source_text: The source memory text.
        :param related_memories: List of dictionaries containing related memory texts.
        :param link_type: Type of relationship to create.
        """
        try:
            for memory in related_memories:
                if 'chunk' in memory:
                    self.memory_engine.create_relationship(source_text, memory['chunk'], link_type)
                else:
                    self.memory_engine.create_relationship(source_text, memory['memory'], link_type)
                logger.info(f"[LINK CREATED] '{source_text}' ↔ '{memory.get('chunk', memory['memory'])}' with similarity {memory['similarity']}")
        except Exception as e:
            logger.error(f"[LINK CREATION FAILED] {e}", exc_info=True)

    def advanced_context_matching(self, input_text: str, similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Match contexts using semantic embeddings and deeper similarity thresholds.

        :param input_text: Text to match against the memory.
        :param similarity_threshold: Minimum similarity score for matching.
        :return: List of matching contexts.
        """
        try:
            input_embedding = self.embedding_model.encode(input_text, convert_to_tensor=True)
            memories = self.memory_engine.retrieve_all_memories()
            
            scored_matches = []
            for memory in memories:
                match_embedding = torch.tensor(memory['embedding'])
                score = util.cos_sim(input_embedding, match_embedding).item()
                if score > similarity_threshold:
                    scored_matches.append({
                        "content": memory["content"],
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
            # Assuming themes are stored in memory metadata or chunk metadata
            theme_embedding = self.embedding_model.encode(theme.lower(), convert_to_tensor=True)
            thematic_contexts = []
            for memory in self.memory_engine.retrieve_all_memories():
                if 'chunks' in memory:
                    for chunk in memory['chunks']:
                        if 'theme' in chunk['metadata'] and theme.lower() in chunk['metadata']['theme']:
                            chunk_embedding = torch.tensor(chunk['embedding'])
                            similarity = util.cos_sim(theme_embedding, chunk_embedding).item()
                            if similarity >= similarity_threshold:
                                thematic_contexts.append({
                                    "content": chunk["content"],
                                    "similarity_to_theme": round(similarity, 4)
                                })
                elif 'theme' in memory['metadata'] and theme.lower() in memory['metadata']['theme']:
                    memory_embedding = torch.tensor(memory['embedding'])
                    similarity = util.cos_sim(theme_embedding, memory_embedding).item()
                    if similarity >= similarity_threshold:
                        thematic_contexts.append({
                            "content": memory["content"],
                            "similarity_to_theme": round(similarity, 4)
                        })

            logger.info(f"[THEMATIC SEARCH] Found {len(thematic_contexts)} contexts for theme '{theme}'.")
            return thematic_contexts
        except Exception as e:
            logger.error(f"[THEMATIC SEARCH ERROR] {e}", exc_info=True)
            return []

    def context_evolution(self, start_date: str, end_date: str, theme: str = None) -> Dict:
        """
        Analyze how contexts evolve over time within a specified
                date range and optional theme.

        :param start_date: Start date for analysis in 'YYYY-MM-DD' format.
        :param end_date: End date for analysis in 'YYYY-MM-DD' format.
        :param theme: Optional theme to filter contexts by
        :return: Dictionary summarizing context evolution.
        """
        try:
            memories = self.memory_engine.retrieve_memories_by_time_range(start_date, end_date)
            
            if not memories:
                logger.info(f"[CONTEXT EVOLUTION] No contexts found for the given date range and theme.")
                return {"error": "No data found for the specified criteria."}

            evolution = {
                "total_contexts": len(memories),
                "themes": {},
                "dates": {}
            }

            for memory in memories:
                date_str = memory['metadata']['created_at'].split('T')[0]
                if theme:
                    # Check for theme in memory or chunk metadata
                    if 'chunks' in memory:
                        relevant_chunks = [chunk for chunk in memory['chunks'] if 'theme' in chunk['metadata'] and theme.lower() in chunk['metadata']['theme']]
                        if not relevant_chunks:
                            continue
                    elif 'theme' not in memory['metadata'] or theme.lower() not in memory['metadata']['theme']:
                        continue

                if 'chunks' in memory:
                    for chunk in memory['chunks']:
                        if 'theme' in chunk['metadata']:
                            for t in chunk['metadata']['theme'].split(','):
                                t = t.strip().lower()
                                evolution['themes'][t] = evolution['themes'].get(t, 0) + 1
                elif 'theme' in memory['metadata']:
                    for t in memory['metadata']['theme'].split(','):
                        t = t.strip().lower()
                        evolution['themes'][t] = evolution['themes'].get(t, 0) + 1

                evolution['dates'][date_str] = evolution['dates'].get(date_str, 0) + 1

            logger.info(f"[CONTEXT EVOLUTION] Analyzed {evolution['total_contexts']} contexts.")
            return evolution
        except Exception as e:
            logger.error(f"[CONTEXT EVOLUTION ERROR] {e}", exc_info=True)
            return {"error": "An error occurred during context evolution analysis."}

    def multi_modal_search(self, content: str, image_embedding: Union[List[float], np.ndarray] = None) -> List[Dict]:
        """
        Perform a search combining text and potential image embeddings for more nuanced context matching.

        :param content: Text to search for.
        :param image_embedding: Optional image embedding for multi-modal search.
        :return: List of matched contexts with scores.
        """
        try:
            text_embedding = self.embedding_model.encode(content, convert_to_tensor=True)
            
            if image_embedding is not None:
                if not isinstance(image_embedding, np.ndarray):
                    image_embedding = np.array(image_embedding)
                combined_embedding = np.concatenate([text_embedding.cpu().numpy(), image_embedding])

                scored_matches = []
                for memory in self.memory_engine.retrieve_all_memories():
                    if 'chunks' in memory:
                        for chunk in memory['chunks']:
                            if 'image_embedding' in chunk['metadata']:
                                memory_embedding = np.concatenate([
                                    chunk['embedding'],  # Assuming this is the text embedding
                                    chunk['metadata']['image_embedding']
                                ])
                                score = util.cos_sim(torch.tensor([combined_embedding]), torch.tensor([memory_embedding])).item()
                                if score > 0.7:  # Adjust threshold as needed
                                    scored_matches.append({
                                        "content": chunk["content"],
                                        "score": round(score, 4)
                                    })
                    elif 'image_embedding' in memory['metadata']:
                        memory_embedding = np.concatenate([
                            memory['embedding'],  # Assuming this is the text embedding
                            memory['metadata']['image_embedding']
                        ])
                        score = util.cos_sim(torch.tensor([combined_embedding]), torch.tensor([memory_embedding])).item()
                        if score > 0.7:  # Adjust threshold as needed
                            scored_matches.append({
                                "content": memory["content"],
                                "score": round(score, 4)
                            })

            else:
                # Fallback to text-only search if no image embedding provided
                scored_matches = self.advanced_context_matching(content)

            logger.info(f"[MULTI-MODAL SEARCH] Found {len(scored_matches)} matches.")
            return scored_matches
        except Exception as e:
            logger.error(f"[MULTI-MODAL SEARCH ERROR] {e}", exc_info=True)
            return []