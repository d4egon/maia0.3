import logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

class SentenceParser:
    def __init__(self):
        """
        Initialize SentenceParser with a SentenceTransformer model for semantic analysis.
        """
        self.model = model

    def parse(self, tokenized_data: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[str]]:
        """
        Parse a dictionary containing tokenized text and its embedding to identify grammatical components.

        :param tokenized_data: Dictionary with 'tokens' and 'embedding' keys, where 'tokens' is a list of token dictionaries.
        :return: A dictionary containing parsed grammatical components.

        :raises ValueError: If the input data structure is not as expected.
        """
        parsed = {
            "subject": [],
            "verb": [],
            "object": [],
            "modifiers": [],
            "phrases": {
                "noun_phrases": [],
                "verb_phrases": [],
                "prepositional_phrases": []
            },
            "semantic_embedding": []
        }

        try:
            if not isinstance(tokenized_data, dict) or 'tokens' not in tokenized_data or 'embedding' not in tokenized_data:
                raise ValueError("Input must be a dictionary with 'tokens' and 'embedding' keys.")

            tokens = tokenized_data['tokens']
            if not isinstance(tokens, list) or not tokens:
                raise ValueError("Tokens must be a non-empty list of token dictionaries.")

            text = ' '.join([token['value'] for token in tokens])
            doc = self.model.encode([text])[0].tolist()  # Here we're using our own model to get sentence embedding
            parsed['semantic_embedding'] = doc

            # Basic grammatical parsing - this is simplified due to not using spaCy
            for token in tokens:
                if token['type'] == 'word':
                    word = token['value']
                    if word in ['am', 'is', 'are', 'was', 'were', 'be', 'been']:
                        parsed['verb'].append(word)
                    elif word in ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that']:
                        parsed['subject'].append(word)
                    elif word in ['the', 'a', 'an']:
                        parsed['modifiers'].append(word)
                    # Note: This is very basic and doesn't capture objects or more complex structures accurately

            # Phrase extraction is simplified due to lack of spaCy's detailed analysis
            phrases = ' '.join([token['value'] for token in tokens])
            parsed['phrases']['noun_phrases'] = [phrases]  # Placeholder for noun phrases without detailed parsing
            parsed['phrases']['verb_phrases'] = [phrases]  # Placeholder for verb phrases
            parsed['phrases']['prepositional_phrases'] = []  # Placeholder for prepositional phrases

            logger.info(f"[PARSE] Tokens parsed into structure: {parsed}")
            return parsed
        except ValueError as ve:
            logger.error(f"[PARSE ERROR] {ve}")
            raise
        except Exception as e:
            logger.error(f"[PARSE ERROR] Unexpected error during parsing: {e}", exc_info=True)
            raise