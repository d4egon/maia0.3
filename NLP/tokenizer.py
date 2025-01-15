import re
import logging
from typing import List, Dict
import spacy

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

class Tokenizer:
    # Define a set of keywords to be tagged during tokenization
    KEYWORDS = {"grace", "faith", "sin", "redemption", "virtue", "justice", "mercy", "truth", "love", "evil"}

    def tokenize(self, text: str) -> List[Dict[str, str]]:
        """
        Tokenize the input text into words, numbers, punctuation marks, and specific keywords using spaCy.

        :param text: The text string to be tokenized.
        :return: A list of dictionaries where each token includes the 'type' and 'value'.

        :raises ValueError: If the input text is not a string or is empty.
        """
        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Input must be a non-empty string.")

            doc = nlp(text)
            categorized_tokens = []
            for token in doc:
                token_value = token.text
                if token_value.lower() in self.KEYWORDS:
                    token_type = "keyword"
                elif token.pos_ == 'NUM':
                    token_type = "number"
                elif token.pos_ in ['PUNCT']:
                    token_type = "punctuation"
                else:
                    token_type = token.pos_

                categorized_tokens.append({"type": token_type, "value": token_value})

            logger.info(f"[TOKENIZATION] Text '{text[:50]}{'...' if len(text) > 50 else ''}' tokenized into {len(categorized_tokens)} categorized tokens.")
            return categorized_tokens
        except ValueError as ve:
            logger.error(f"[TOKENIZATION ERROR] {ve}")
            raise
        except Exception as e:
            logger.error(f"[TOKENIZATION ERROR] Unexpected error: {e}", exc_info=True)
            raise

    def tokenize_batch(self, texts: List[str]) -> List[List[Dict[str, str]]]:
        """
        Tokenize a batch of texts.

        :param texts: A list of text strings to be tokenized.
        :return: A list of lists of token dictionaries, one for each input text.

        :raises ValueError: If any text in the batch is not a string or is empty.
        """
        try:
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                raise ValueError("Input must be a list of strings.")
            
            return [self.tokenize(text) for text in texts if text.strip()]
        except ValueError as ve:
            logger.error(f"[BATCH TOKENIZATION ERROR] {ve}")
            raise
        except Exception as e:
            logger.error(f"[BATCH TOKENIZATION ERROR] Unexpected error: {e}", exc_info=True)
            raise