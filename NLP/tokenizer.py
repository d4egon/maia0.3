import logging
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

class Tokenizer:
    # Define a set of keywords to be tagged during tokenization
    KEYWORDS = {"grace", "faith", "sin", "redemption", "virtue", "justice", "mercy", "truth", "love", "evil"}

    def __init__(self):
        """
        Initialize Tokenizer with a SentenceTransformer model for semantic understanding.
        """
        self.model = model

    def tokenize(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Tokenize the input text into words, numbers, punctuation marks, and specific keywords using regex and semantic analysis.

        :param text: The text string to be tokenized.
        :return: A dictionary containing 'tokens' and 'embedding' for the entire text.

        :raises ValueError: If the input text is not a string or is empty.
        """
        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Input must be a non-empty string.")

            # Tokenize using regex for simplicity and speed
            words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            categorized_tokens = []
            for word in words:
                if word in self.KEYWORDS:
                    token_type = "keyword"
                elif word.isdigit():
                    token_type = "number"
                elif word in '.,;:!?':
                    token_type = "punctuation"
                else:
                    token_type = "word"

                categorized_tokens.append({"type": token_type, "value": word})

            # Generate embedding for the entire text
            embedding = self.model.encode([text])[0].tolist()

            logger.info(f"[TOKENIZATION] Text '{text[:50]}{'...' if len(text) > 50 else ''}' tokenized into {len(categorized_tokens)} tokens.")
            return {"tokens": categorized_tokens, "embedding": embedding}
        except ValueError as ve:
            logger.error(f"[TOKENIZATION ERROR] {ve}")
            raise
        except Exception as e:
            logger.error(f"[TOKENIZATION ERROR] Unexpected error: {e}", exc_info=True)
            raise

    def tokenize_batch(self, texts: List[str]) -> List[Dict[str, List[Dict[str, str]]]]:
        """
        Tokenize a batch of texts, including generating embeddings.

        :param texts: A list of text strings to be tokenized.
        :return: A list of dictionaries, each containing 'tokens' and 'embedding' for each text.

        :raises ValueError: If any text in the batch is not a string or is empty.
        """
        try:
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                raise ValueError("Input must be a list of strings.")
            
            # Generate embeddings for all texts at once for efficiency
            embeddings = self.model.encode(texts).tolist()

            return [
                {
                    "tokens": self.tokenize(text)["tokens"],
                    "embedding": embedding
                } 
                for text, embedding in zip(texts, embeddings) if text.strip()
            ]
        except ValueError as ve:
            logger.error(f"[BATCH TOKENIZATION ERROR] {ve}")
            raise
        except Exception as e:
            logger.error(f"[BATCH TOKENIZATION ERROR] Unexpected error: {e}", exc_info=True)
            raise