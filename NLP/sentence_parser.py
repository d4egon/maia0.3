import logging
import spacy
from typing import List, Dict

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

class SentenceParser:
    def parse(self, tokens: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """
        Parse a list of tokenized dictionaries to identify grammatical components using spaCy.

        :param tokens: List of dictionaries, where each dictionary represents a token with 'type' and 'value'.
        :return: A dictionary containing parsed grammatical components.
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
            }
        }

        try:
            if not isinstance(tokens, list) or not tokens:
                raise ValueError("Input must be a non-empty list of token dictionaries.")

            text = ' '.join([token['value'] for token in tokens])
            doc = nlp(text)

            for token in doc:
                if token.dep_ == 'nsubj':
                    parsed['subject'].append(token.text)
                elif token.pos_ == 'VERB':
                    parsed['verb'].append(token.text)
                elif token.dep_ == 'dobj':
                    parsed['object'].append(token.text)
                elif token.pos_ in ['ADJ', 'ADV']:
                    parsed['modifiers'].append(token.text)

            # Extract noun phrases
            parsed['phrases']['noun_phrases'] = [np.text for np in doc.noun_chunks]

            # Extract verb phrases (this is a basic approach, consider expanding)
            verb_phrases = []
            for sent in doc.sents:
                vp = []
                for token in sent:
                    if token.pos_ == 'VERB':
                        vp.append(token.text)
                        for child in token.children:
                            if child.dep_ in ['aux', 'neg', 'advmod']:
                                vp.append(child.text)
                if vp:
                    verb_phrases.append(' '.join(vp))
            parsed['phrases']['verb_phrases'] = verb_phrases

            # Extract prepositional phrases (this is a basic approach, consider expanding)
            prep_phrases = []
            for token in doc:
                if token.dep_ == 'prep':
                    phrase = [token.text]
                    for child in token.children:
                        phrase.append(child.text)
                    prep_phrases.append(' '.join(phrase))
            parsed['phrases']['prepositional_phrases'] = prep_phrases

            logger.info(f"[PARSE] Tokens parsed into structure: {parsed}")
            return parsed
        except ValueError as ve:
            logger.error(f"[PARSE ERROR] {ve}")
            raise
        except Exception as e:
            logger.error(f"[PARSE ERROR] Unexpected error during parsing: {e}", exc_info=True)
            raise