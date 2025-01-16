import logging
import mimetypes
import os
from config.settings import CONFIG
from core.neo4j_connector import Neo4jConnector
from core.memory_engine import MemoryEngine
from core.file_parser import FileParser
from core.emotion_engine import EmotionEngine
from core.ethics_engine import EthicsEngine
from core.thought_engine import ThoughtEngine
from core.conversation_engine import ConversationEngine
#from core.deduplication_engine import DeduplicationEngine
from core.dream_engine import DreamEngine
from core.context_search import ContextSearchEngine
from NLP.response_generator import ResponseGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class FilePipeline:
    def __init__(self):
        """
        Initialize FilePipeline with core components for file processing.
        """
        try:
            self.neo4j = Neo4jConnector(
                CONFIG["NEO4J_URI"],
                CONFIG["NEO4J_USER"],
                CONFIG["NEO4J_PASSWORD"]
            )
            self.memory_engine = MemoryEngine(self.neo4j)  # Pass the Neo4j connection to MemoryEngine
            self.file_parser = FileParser()  # No dependencies required
            self.emotion_engine = EmotionEngine(self.neo4j)  # Pass the Neo4j connection to EmotionEngine
            self.ethics_engine = EthicsEngine(self.neo4j)
            self.thought_engine = ThoughtEngine(self.neo4j)
            self.context_search_engine = ContextSearchEngine(self.neo4j)  # Pass the Neo4j connection to ContextSearchEngine
            self.response_generator = ResponseGenerator(self.memory_engine, self.neo4j)  # Assuming no dependencies required
            self.conversation_engine = ConversationEngine(self.memory_engine, self.response_generator, self.context_search_engine)
            #self.deduplication_engine = DeduplicationEngine()  # No dependencies required
            self.dream_engine = DreamEngine(self.memory_engine, self.context_search_engine)
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def process_file(self, filepath):
        """
        Process the uploaded file.

        :param filepath: Path to the file.
        :return: Result of file processing.
        """
        try:
            # Validate MIME Type
            mime_type, _ = mimetypes.guess_type(filepath)
            if not mime_type:
                raise ValueError("Could not determine file MIME type.")
            logger.info(f"[VALIDATION] MIME type validated: {mime_type}")

            # Parse File Content
            content = self.file_parser.parse(filepath, mime_type)
            if content is None:
                raise ValueError("File parsing returned None, possibly unsupported file type or parsing error.")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Parsed content is empty or not a string.")
            logger.info(f"[PARSING] Successfully parsed file content ({len(content)} characters).")

            # Analyze and Store Memory
            emotion, confidence = self.emotion_engine.contextual_emotion_analysis(content)
            self.emotion_engine.update_emotional_state(emotion, confidence)
            
            # Ethics Check
            if not self.ethics_engine.check(content):
                raise ValueError("Content failed ethics check.")

            # Thought Processing
            thoughts = self.thought_engine.process(content)
            logger.info(f"[THOUGHTS] Generated thoughts: {thoughts}")

            # Conversation Analysis
            conversation = self.conversation_engine.analyze(content)
            logger.info(f"[CONVERSATION] Analyzed conversation: {conversation}")

            # Deduplication
            if self.deduplication_engine.is_duplicate(content):
                raise ValueError("Content is a duplicate.")

            # Dream Analysis
            dreams = self.dream_engine.analyze(content)
            logger.info(f"[DREAMS] Analyzed dreams: {dreams}")

            # Store in Memory
            self.memory_engine.store_memory(
                content,
                [emotion],
                extra_properties={
                    "file_type": mime_type,
                    "filename": os.path.basename(filepath),
                    "thoughts": thoughts,
                    "conversation": conversation,
                    "dreams": dreams
                }
            )

            logger.info(f"[MEMORY] File '{os.path.basename(filepath)}' stored in memory with emotion '{emotion}' and confidence '{confidence}'.")
            return {
                "mime_type": mime_type,
                "content": content,
                "emotion": emotion,
                "confidence": confidence,
                "thoughts": thoughts,
                "conversation": conversation,
                "dreams": dreams
            }
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            raise

    def handle_file_upload(self, file_path):
        """
        Handle file upload and process the file.

        :param file_path: Path to the uploaded file.
        :return: Result of file processing.
        """
        try:
            if not os.path.exists(file_path):
                raise ValueError(f"File does not exist: {file_path}")

            result = self.process_file(file_path)
            return "File processed successfully."
        except Exception as e:
            logger.error(f"File upload handling failed: {e}")
            raise