import logging
from typing import List, Dict
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT # type: ignore
from core.memory_engine import MemoryEngine
from config.utils import get_sentence_transformer_model, get_keyword_extractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttributeEnrichment:
    def __init__(self, memory_engine: MemoryEngine):
        """
        Initialize AttributeEnrichment with MemoryEngine and miniLM-L12 model.

        :param memory_engine: An instance of MemoryEngine for managing database interactions.
        """
        self.memory_engine = memory_engine
        self.model = get_sentence_transformer_model()
        self.keyword_extractor = get_keyword_extractor(self.model)

    def get_metadata_suggestions(self) -> Dict:
        """
        Generate a dictionary of suggested metadata fields for various contexts.

        :return: Dictionary with suggested metadata fields.
        """
        return {
        "basic": {
            "author": "Creator or source of the content",
            "date_created": "Date when the content was created",
            "language": "Language of the content",
            "format": "Format of the content (e.g., text, image, video)"
        },
        "content": {
            "tone": "Emotional tone (e.g., formal, casual, sarcastic)",
            "sentiment": "Positive, negative, or neutral sentiment",
            "keywords": "Important keywords for indexing and search",
            "summary": "Brief summary of the content",
            "category": "Broad category or topic (e.g., science, literature)"
        },
        "contextual": {
            "geography": "Geographical relevance or origin",
            "era": "Historical period or relevance",
            "culture": "Cultural context or significance",
            "event": "Associated event if applicable"
        },
        "technical": {
            "complexity": "Complexity level (e.g., beginner, advanced)",
            "accuracy": "How factually accurate or reliable the content is",
            "source_type": "Primary, secondary, or tertiary source",
            "verification_status": "Has the content been verified?"
        },
        "semantic": {
            "concepts": "Core concepts discussed",
            "entities": "Named entities like people, organizations, locations",
            "relations": "Relationships between entities or concepts",
            "theme": "Overarching theme or narrative"
        },
        "usage": {
            "intended_use": "Purpose of the content (e.g., education, entertainment)",
            "audience": "Target demographic or audience",
            "accessibility": "Accessibility features or considerations",
            "license": "License under which the content is shared"
        },
        "temporal": {
            "time_relevance": "Is the content time-sensitive?",
            "updatability": "How often should this content be updated?",
            "valid_until": "Date until which the content is valid or relevant"
        },
        "ai_specific": {
            "model_compatibility": "Which AI models can process this data?",
            "training_use": "Can this be used for AI training?",
            "inference_potential": "Usefulness for AI inference tasks",
            "bias_check": "Checked for bias?"
        }
    }

    def enrich_attributes(self, memory_id: str, attributes: Dict):
        """
        Update memory attributes in the database.

        :param memory_id: ID of the memory to update.
        :param attributes: Dictionary of attributes to set.
        :raises Exception: If there's an error updating attributes.
        """
        current_metadata = self.get_memory_metadata(memory_id) or {}
        current_metadata.update(attributes)
        self.update_memory_metadata(memory_id, current_metadata)
        logger.info(f"Updated attributes for memory ID: {memory_id}")

    def auto_enrichment(self, memory_id: str, content: str):
        """
        Automatically enrich memory attributes based on content analysis using miniLM-L12.

        :param memory_id: ID of the memory to enrich.
        :param content: The content of the memory for analysis.
        """
        metadata_suggestions = self.get_metadata_suggestions()
        enriched_metadata = {}

        # Embed the content
        embedding = self.model.encode([content])[0]

        # Keywords extraction using KeyBERT with miniLM-L12
        keywords = self.keyword_extractor.extract_keywords(content, keyphrase_ngram_range=(1, 1), top_n=5)
        keywords = [keyword[0] for keyword in keywords]  # Extract just the words

        # Sentiment analysis - Here we use a simple cosine similarity to predefined sentiment embeddings
        sentiment = self._estimate_sentiment(content)

        # Category inference by comparing to pre-defined category embeddings
        category = self._infer_category(embedding)

        # Complexity estimation might be based on sentence length and embedding diversity
        complexity = self._estimate_complexity(content)

        enriched_metadata.update({
            "date_created": datetime.now().isoformat(),
            "language": "English",  # More sophisticated language detection could be used here
            "format": "text",
            "keywords": keywords,
            "sentiment": sentiment,
            "category": category,
            "complexity": complexity,
            "intended_use": "Education",  # This would typically be inferred from context or user input
            "audience": "General Adult",  # This could be inferred from content or user profile
            "model_compatibility": ["NLP", "Text Analysis"],
            "training_use": "Yes",
        })

        self.enrich_attributes(memory_id, enriched_metadata)

    def _estimate_sentiment(self, text: str) -> str:
        """
        Estimate sentiment by comparing the text embedding to known sentiment embeddings.

        :param text: The text to analyze.
        :return: Sentiment label.
        """
        positive_embedding = self.model.encode(["This is great!"])
        negative_embedding = self.model.encode(["This is terrible!"])
        neutral_embedding = self.model.encode(["This is okay."])

        text_embedding = self.model.encode([text])[0]
        similarities = {
            "positive": util.cos_sim(text_embedding, positive_embedding)[0][0].item(),
            "negative": util.cos_sim(text_embedding, negative_embedding)[0][0].item(),
            "neutral": util.cos_sim(text_embedding, neutral_embedding)[0][0].item()
        }

        return max(similarities, key=similarities.get)

    def _infer_category(self, embedding: np.ndarray) -> str:
        """
        Infer category by comparing the embedding to pre-computed category embeddings.

        :param embedding: The embedding of the content.
        :return: Predicted category.
        """
        if not hasattr(self, 'category_centroids'):
            # Predefined category embeddings - you'd typically compute these from a training set
            self.category_centroids = {
                'science': self.model.encode(["Science deals with facts and theories."]),
                'literature': self.model.encode(["Literature encompasses stories and poetry."]),
                'technology': self.model.encode(["Technology involves innovation and computers."])
            }
        
        distances = {k: util.cos_sim(embedding, v)[0][0].item() for k, v in self.category_centroids.items()}
        return max(distances, key=distances.get)

    def _estimate_complexity(self, text: str) -> str:
        """
        Estimate complexity based on sentence count and embedding diversity.

        :param text: The text to analyze.
        :return: Complexity level.
        """
        sentences = text.split('.')
        sentence_embeddings = self.model.encode(sentences)
        
        # Diversity could be measured by variance or spread of embeddings
        diversity = np.var(sentence_embeddings, axis=0).mean()
        
        if diversity > 0.5 or len(sentences) > 5:
            return "High"
        elif diversity > 0.3 or len(sentences) > 3:
            return "Medium"
        else:
            return "Low"

    # Helper methods for metadata operations
    def get_memory_metadata(self, memory_id: str) -> Dict:
        """Retrieve metadata for a given memory ID."""
        query = f"""
        MATCH (m:MemoryNode {{id: '{memory_id}'}})
        RETURN m.metadata AS metadata
        """
        result = self.memory_engine.neo4j.run_query(query)
        return result[0]['metadata'] if result else {}

    def update_memory_metadata(self, memory_id: str, metadata: Dict):
        """Update metadata for a given memory ID."""
        query = f"""
        MATCH (m:MemoryNode {{id: '{memory_id}'}})
        SET m.metadata = $metadata
        """
        self.memory_engine.neo4j.run_query(query, {"metadata": metadata})