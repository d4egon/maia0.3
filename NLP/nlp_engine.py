import logging
from typing import Dict, List, Tuple
from sentence_transformers import InputExample, SentenceTransformer, util
from NLP.sentence_parser import SentenceParser
from NLP.tokenizer import Tokenizer
from NLP.intent_detector import IntentDetector
import requests
from core.file_parser import FileParser
from core.memory_engine import MemoryEngine
from core.attribute_enrichment import AttributeEnrichment

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NLP:
    def __init__(self, memory_engine: MemoryEngine, response_generator, neo4j_connector):
        """
        Initialize the NLP engine with necessary components.

        :param memory_engine: An instance of MemoryEngine for memory operations.
        :param response_generator: An instance for generating responses.
        :param neo4j_connector: An instance of Neo4jConnector for database operations.
        """
        self.memory_engine = memory_engine
        self.response_generator = response_generator
        self.neo4j_connector = neo4j_connector
        self.tokenizer = Tokenizer()
        self.sentence_parser = SentenceParser()
        self.file_parser = FileParser(memory_engine)
        self.intent_detector = IntentDetector(memory_engine)
        self.attribute_enrichment = AttributeEnrichment(memory_engine)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

    def process(self, text: str, user_name: str = "User", context: str = "general conversation") -> Tuple[str, str]:
        """
        Process input text, detect intent, analyze emotions, and generate a response.

        :param text: User input to analyze.
        :param user_name: The user's name for personalization.
        :param context: Additional context for the response.
        :return: A tuple of (response, intent).
        """
        try:
            tokens = self.tokenizer.tokenize(text)
            parsed_data = self.sentence_parser.parse(tokens)

            # Use miniLM-L12 for semantic analysis and intent detection
            word_tokens = [token['value'] for token in tokens if token['type'] == 'word']
            sentence_embedding = self.model.encode([text])[0]
            intent = self.intent_detector.detect_intent(word_tokens, sentence_embedding)
            emotions = self.analyze_emotions(text)

            # Enrich context data with more detailed information
            context_data = {
                "text": text, 
                "tokens": tokens, 
                "parsed": parsed_data, 
                "emotions": emotions,
                "subjects": parsed_data.get("subject", []),
                "verbs": parsed_data.get("verb", []),
                "objects": parsed_data.get("object", []),
                "embedding": sentence_embedding.tolist()  # Store embedding for later use
            }

            response = self.response_generator.generate_response(context_data, user_name, intent, context)

            # Store or update memory with enriched attributes
            memory_id = self.memory_engine.create_memory_node(text, context_data, [])
            self.attribute_enrichment.enrich_attributes(memory_id, context_data)

            logger.info(f"[NLP PROCESS] Text: '{text}', Intent: {intent}, Emotions: {emotions}, Response: {response}")
            return response, intent

        except Exception as e:
            logger.error(f"[NLP PROCESS ERROR] {e}", exc_info=True)
            return "I'm sorry, I encountered an error.", "error"

    def analyze_emotions(self, text: str) -> List[str]:
        """
        Analyze emotions based on text using semantic similarity with miniLM-L12.

        :param text: Text to analyze for emotions.
        :return: A list of detected emotions.
        """
        emotion_templates = {
            "happy": "I'm feeling joyful, content, or satisfied.",
            "sad": "I feel down, disappointed, or sorrowful.",
            "angry": "I am furious, irritated, or frustrated.",
            "neutral": "I feel okay, normal, or indifferent."
        }
        
        text_embedding = self.model.encode([text])[0]
        emotions_detected = []
        for emotion, template in emotion_templates.items():
            template_embedding = self.model.encode([template])[0]
            similarity = util.cos_sim(text_embedding, template_embedding).item()
            if similarity > 0.7:  # Threshold can be adjusted based on desired sensitivity
                emotions_detected.append(emotion)
        
        if not emotions_detected:
            emotions_detected = ["neutral"]  # Default to neutral if no emotions detected
        
        logger.info(f"[EMOTION ANALYSIS] Detected emotions: {emotions_detected}")
        return emotions_detected

    def update_intent_keywords(self, new_keywords: Dict[str, List[str]]):
        """
        Update the intent keywords dictionary with new or modified keywords.

        :param new_keywords: A dictionary of intents with their corresponding keyword lists.
        """
        try:
            self.intent_detector.update_intents(new_keywords)
            logger.info(f"[INTENT UPDATE] Updated intent keywords: {new_keywords}")
        except Exception as e:
            logger.error(f"[INTENT UPDATE ERROR] Failed to update keywords: {e}", exc_info=True)

    def fetch_internet_data(self, url: str) -> str:
        """
        Fetch data from the internet.

        :param url: URL to fetch data from.
        :return: Fetched data as string or error message if failed.
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Error fetching internet data: {e}", exc_info=True)
            return f"Failed to fetch data: {str(e)}"

    def process_file(self, file_path: str) -> Tuple[str, str]:
        """
        Process a file and generate a response.

        :param file_path: Path to the file to be processed.
        :return: A tuple of (response, intent).
        """
        try:
            file_content = self.file_parser.parse(file_path)
            if file_content:
                return self.process(file_content)
            else:
                logger.warning("File content could not be processed.")
                return "I'm sorry, I couldn't process the file content.", "error"
        except Exception as e:
            logger.error(f"Error processing file: {e}", exc_info=True)
            return "An error occurred while processing the file.", "error"

    def train_model(self, train_examples: List[InputExample]):
        """
        Train or fine-tune the model with provided examples.

        :param train_examples: List of InputExample objects for training.
        """
        from torch.utils.data import DataLoader
        from sentence_transformers import losses

        try:
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
            train_loss = losses.ContrastiveLoss(model=self.model)
            
            self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
            logger.info("Model training completed.")
        except Exception as e:
            logger.error(f"Error during model training: {e}", exc_info=True)