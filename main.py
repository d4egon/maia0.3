import os
import sys
import logging
import time
import json
from uuid import uuid4
from typing import Dict, List, Optional
from threading import Thread
from functools import lru_cache
from tempfile import NamedTemporaryFile
from flask import Flask, render_template, request, jsonify, abort, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from NLP.consciousness_engine import ConsciousnessEngine
from NLP.intent_detector import IntentDetector
from config.settings import CONFIG
from core.context_search import ContextSearchEngine
from core.neo4j_connector import Neo4jConnector
from core.memory_engine import MemoryEngine
from core.file_parser import FileParser
from NLP.response_generator import ResponseGenerator
from NLP.nlp_engine import NLP
from core.emotion_engine import EmotionEngine
from core.emotion_fusion_engine import EmotionFusionEngine
from core.collaborative_learning import CollaborativeLearning
from core.conversation_engine import ConversationEngine
from core.dream_engine import DreamEngine
from core.ethics_engine import EthicsEngine
from flask_socketio import SocketIO
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Custom logging filter to ignore log messages containing "Batches"
class IgnoreBatchesFilter(logging.Filter):
    def filter(self, record):
        return 'Batches' not in record.getMessage()

logger = logging.getLogger()
logger.addFilter(IgnoreBatchesFilter())

# Load Environment Variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO if os.getenv("FLASK_DEBUG", "false").lower() != "true" else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(os.getcwd(), "uploads"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 1024))
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'epub', 'json', 'mp3', 'mp4', 'jpg', 'png', 'csv', 'docx', 'xlsx'])  # Simplified list for brevity

# Flask App Initialization
app = Flask(
    __name__,
    template_folder=os.path.join(os.getcwd(), "web_server", "templates"),
    static_folder=os.path.join(os.getcwd(), "web_server", "static"),
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes

socketio = SocketIO(app)

class ConversationState:
    def __init__(self):
        self.current_chunk_id: Optional[str] = None
        self.current_chunk_order: int = 1
        self.current_chunk_size: int = 0
        self.last_intent: Optional[str] = None
        self.memory_node_id: str = None  # Changed to None to be set dynamically

# Global state for simplicity; in a real app, this would be session-based or stored elsewhere
conversation_state = ConversationState()


# Secure Headers Setup
@app.after_request
def apply_csp_headers(response):
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "font-src 'self'; img-src 'self' data:; connect-src 'self' ws: wss:"
    )
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    return response

# Ensure Upload Folder Exists
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"[INIT] Upload folder ensured at: {UPLOAD_FOLDER}")
except Exception as e:
    logger.critical(f"[INIT ERROR] Failed to create upload folder: {e}", exc_info=True)
    sys.exit(1)

# Initialize Core Services
try:
    neo4j_uri = CONFIG.get("NEO4J_URI")
    neo4j_user = CONFIG.get("NEO4J_USER")
    neo4j_password = CONFIG.get("NEO4J_PASSWORD")

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        logger.critical("Missing Neo4j configuration. Check your .env file.")
        sys.exit(1)

    neo4j = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
    memory_engine = MemoryEngine(neo4j)
    response_gen = ResponseGenerator(memory_engine, neo4j)
    file_parser = FileParser(memory_engine)
    nlp_engine = NLP(memory_engine, response_gen, neo4j)
    emotion_engine = EmotionEngine(memory_engine)
    emotion_fusion_engine = EmotionFusionEngine(memory_engine, nlp_engine)
    dream_engine = DreamEngine(memory_engine, ContextSearchEngine(memory_engine))  # Adjusted here
    ethics_engine = EthicsEngine(memory_engine)
    context_search_engine = ContextSearchEngine(memory_engine)  # Adjusted here
    consciousness_engine = ConsciousnessEngine(memory_engine, emotion_engine)
    intent_detector = IntentDetector(memory_engine)
    collaboration_engine = CollaborativeLearning(ConversationEngine(memory_engine, response_gen, context_search_engine), nlp_engine, memory_engine)  # Adjusted here
    conversation_engine = ConversationEngine(
        memory_engine,
        response_gen,
        context_search_engine
    )

    logger.info("[INIT SUCCESS] All services initialized successfully.")
except Exception as e:
    logger.critical(f"[INIT FAILED] Failed to initialize services: {e}", exc_info=True)
    sys.exit(1)

# Cache for Gallery Images
gallery_cache = {}

@lru_cache(maxsize=300)
def cached_get_gallery_images():
    logger.info("[GALLERY] Fetching gallery images.")
    image_folder = os.path.join(app.static_folder, "images")
    all_files = os.listdir(image_folder)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    logger.debug(f"[GALLERY] Found images: {image_files}")
    return list(set(image_files))

# Utility Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_request_data(data, required_keys):
    if not data:
        abort(400, description="Invalid input data.")
    for key in required_keys:
        if key not in data or not isinstance(data[key], str) or not data[key].strip():
            abort(400, description=f"Invalid or missing key: {key}")

# Load standard responses
with open('standard_responses.json', 'r') as file:
    standard_responses = json.load(file)

def get_standard_response(question):
    question = question.lower()
    for category, responses in standard_responses.items():
        for response_item in responses:
            if any(phrase in question for phrase in response_item['input']):
                return response_item['response'], category
    return None, None

def should_trigger_search(question):
    search_keywords = ['find', 'search', 'look up', 'what is', 'who is', 'where is', 'when is', 'how to']
    return any(keyword in question.lower() for keyword in search_keywords)

def perform_search(query):
    # Placeholder for actual search logic
    results = ["This is a placeholder for search results"]
    return results

def summarize_search_results(results):
    return f"Here are some results: {', '.join(results)}"

def update_model_with_text(content):
    def train_task(content):
        try:
            tokenized_input = memory_engine.model.tokenize(content)
            input_ids = tokenized_input["input_ids"]
            logger.info(f"[MODEL TRAIN] Training on content: {content[:50]}...")
        except Exception as e:
            logger.error(f"[MODEL TRAIN ERROR] Error during training: {e}", exc_info=True)

    Thread(target=train_task, args=(content,)).start()

# Routes
@app.route('/')
def index():
    logger.info("[ROUTE] Index accessed.")
    return render_template('index.html')

@app.route('/embed', methods=['POST'])
def embed():
    data = request.json
    content = data.get('content')
    if not content:
        return jsonify({'error': 'No content provided'}), 400
    
    try:
        embedding = memory_engine.model.encode(content).tolist()
        try:
            update_model_with_text(content)
            logger.info(f"[MODEL UPDATE] Model updated with new content: {content[:50]}...")
        except Exception as e:
            logger.error(f"[MODEL UPDATE ERROR] {e}", exc_info=True)
        
        return jsonify({'embedding': embedding})
    except Exception as e:
        logger.error(f"[EMBED ERROR] Failed to encode content: {e}", exc_info=True)
    return jsonify({'error': 'Failed to generate embedding', 'status': 'error'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"message": "No file uploaded", "status": "error"}), 400

    file = request.files['file']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"message": "Invalid file type", "status": "error"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
        result = file_parser.parse(filepath)
        if result is None or not isinstance(result, str) or not result.strip():
            raise ValueError("File parsing returned invalid result.")

        is_advanced = request.args.get("advanced", "false").lower() == "true"
        if is_advanced:
            logger.info("[ADVANCED PROCESSING] Additional processing steps applied.")

        # Use the new method to upload and process content
        content_id = memory_engine.upload_and_process_content(
            file_path=filepath,
            content_type=file.content_type,
            title=file.filename,
            author="Anonymous",  # Assuming author is unknown or default
            metadata={"source": filename, "type": "advanced_upload" if is_advanced else "upload"}
        )

        # Ethics Check
        # Note: You might want to retrieve the content of the node to check with ethics_engine
        content_node = neo4j.run_query("MATCH (c:Content {id: $content_id}) RETURN c.content AS content", {"content_id": content_id})
        if content_node and not ethics_engine.check(content_node[0]['content']):
            logger.warning("[ETHICS CHECK] Ethical concern detected in uploaded content.")
            # Handle ethical concerns here

        # Dream Analysis
        dream_analysis = dream_engine.analyze(result)
        if dream_analysis:
            logger.info(f"[DREAM ANALYSIS] Dream analysis result: {dream_analysis}")

        # Continuous Learning
        train_model_on_file_content(result)

        os.remove(filepath)
        logger.info(f"[UPLOAD SUCCESS] File '{filename}' processed and stored successfully.")
        return jsonify({"message": "File processed successfully.", "status": "success"}), 200
    except ValueError as ve:
        os.remove(filepath)
        logger.warning(f"[UPLOAD WARNING] {ve}")
        return jsonify({"message": str(ve), "status": "warning"}), 400
    except Exception as e:
        os.remove(filepath)
        logger.error(f"[UPLOAD ERROR] {e}", exc_info=True)
        return jsonify({"message": "An error occurred during processing.", "status": "error"}), 500

def train_model_on_file_content(content):
    sentences = content.split('. ')
    for sentence in sentences:
        update_model_with_text(sentence)

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    try:
        audio_file = request.files['audio']
        save_path = os.path.join('uploads', audio_file.filename)
        audio_file.save(save_path)

        transcription = file_parser.parse_audio(save_path, language="en")
        # Ethics Check on Transcription
        if not ethics_engine.check(transcription):
            logger.warning("[ETHICS CHECK] Ethical concern detected in audio transcription.")

        return jsonify({"content": transcription})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask_maia', methods=['POST'])
def ask_maia():
    global conversation_state
    try:
        data = request.get_json()
        validate_request_data(data, ['question'])
        question = data['question'].lower()
        
        # Standard Response Check
        response, intent = get_standard_response(question)
        if response:
            if intent != "greeting":
                # Change the variable name here
                should_create_new_chunk_result = should_create_new_chunk(intent, conversation_state)
                
                if should_create_new_chunk_result:
                    # Create a new chunk
                    conversation_state.current_chunk_id = str(uuid4())
                    conversation_state.current_chunk_order = 1
                    conversation_state.current_chunk_size = 0
                    
                    memory_chunk_id = memory_engine.create_memory_chunk(
                        memory_node_id=conversation_state.memory_node_id,
                        order=conversation_state.current_chunk_order,
                        chunk_type="conversation",
                        keywords=["conversation", intent]
                    )
                else:
                    # Continue with the existing chunk
                    memory_chunk_id = conversation_state.current_chunk_id
                    conversation_state.current_chunk_order += 1
                
                # Create the memory
                memory_engine.create_memory(
                    chunk_id=memory_chunk_id,
                    order=conversation_state.current_chunk_order,
                    content=f"Q: {question}\nA: {response}", 
                    metadata={"emotions": [fused_emotion], "type": "conversation", "intent": intent}
                )
                # Increment the chunk size
                conversation_state.current_chunk_size += 1
                conversation_state.last_intent = intent

            return jsonify({"response": response, "intent": intent, "status": "success"}), 200
        
        # If no standard response, proceed with NLP processing
        if should_trigger_search(question):
            search_results = perform_search(question)
            response = summarize_search_results(search_results)
            intent = "search"
        else:
            intent = intent_detector.detect_intent(question)
            response, emotions = nlp_engine.process(question)
            # Emotion Fusion
            fused_emotion = emotion_fusion_engine.fuse_emotions(emotions)
            
            # Change the variable name here as well
            should_create_new_chunk_result = should_create_new_chunk(intent, conversation_state)
            
            if should_create_new_chunk_result:
                # Create a new chunk
                conversation_state.current_chunk_id = str(uuid4())
                conversation_state.current_chunk_order = 1
                conversation_state.current_chunk_size = 0
                
                memory_chunk_id = memory_engine.create_memory_chunk(
                    memory_node_id=conversation_state.memory_node_id,
                    order=conversation_state.current_chunk_order,
                    chunk_type="conversation",
                    keywords=["conversation", intent]
                )
            else:
                # Continue with the existing chunk
                memory_chunk_id = conversation_state.current_chunk_id
                conversation_state.current_chunk_order += 1
            
            # Create the memory with the NLP processed response
            memory_engine.create_memory(
                chunk_id=memory_chunk_id,
                order=conversation_state.current_chunk_order,
                content=f"Q: {question}\nA: {response}", 
                metadata={"emotions": [fused_emotion], "type": "conversation", "intent": intent}
            )
            
            # Increment the chunk size
            conversation_state.current_chunk_size += 1
            conversation_state.last_intent = intent

        # Ethics Check on Response
        if not ethics_engine.check(response):
            logger.warning("[ETHICS CHECK] Ethical concern detected in response.")
            # Handle ethical concern here, perhaps by using a fallback response or notifying an admin

        return jsonify({"response": response, "intent": intent, "status": "success"}), 200
    except ValueError as ve:
        logger.error(f"[ASK MAIA] Error creating memory chunk: {ve}")
        return jsonify({"message": "An error occurred while creating a memory chunk.", "status": "error"}), 500
    except Exception as e:
        logger.error(f"[ASK MAIA] Error: {e}", exc_info=True)
        return jsonify({"message": "An error occurred.", "status": "error"}), 500

def should_create_new_chunk(intent: str, state: ConversationState) -> bool:
    # Check if the chunk is full or if the intent has changed
    MAX_CHUNK_SIZE = 10  # Define the maximum number of memories per chunk
    
    # If the chunk is full or the intent has changed, we create a new chunk
    return state.current_chunk_size >= MAX_CHUNK_SIZE or intent != state.last_intent

@app.route('/analyze_dream', methods=['POST'])
def analyze_dream():
    """
    Endpoint to analyze a dream narrative using the DreamEngine.

    This route accepts a POST request with a JSON body containing the dream narrative.
    It then uses the DreamEngine to analyze the dream, returning the analysis results.

    :return: JSON response with the dream analysis or an error message.
    """
    try:
        # Validate incoming data
        data = request.get_json()
        validate_request_data(data, ['dream_narrative'])
        dream_narrative = data['dream_narrative']

        # Perform ethics check on the dream narrative
        if not ethics_engine.check(dream_narrative):
            logger.warning("[ANALYZE DREAM] Ethical concern detected in dream narrative.")
            # Here you might want to handle ethical concerns, e.g., by returning a generic response or flagging for review
            return jsonify({"message": "Ethical concern detected. Please review your narrative.", "status": "warning"}), 400

        # Analyze the dream
        analysis_result = dream_engine.analyze(dream_narrative)
        
        if analysis_result:
            logger.info(f"[ANALYZE DREAM] Dream analysis result: {analysis_result}")
            # Store the dream narrative and its analysis in memory
            memory_engine.create_memory(
                content=f"Dream Narrative: {dream_narrative}\nAnalysis: {analysis_result}",
                emotions=["neutral"],  # Assuming dreams are neutral unless specified otherwise
                extra_properties={"type": "dream_analysis"}
            )
            return jsonify({
                "analysis": analysis_result,
                "status": "success"
            }), 200
        else:
            logger.warning("[ANALYZE DREAM] No analysis result returned.")
            return jsonify({
                "message": "No analysis result could be generated for the provided dream narrative.",
                "status": "warning"
            }), 400
    except Exception as e:
        logger.error(f"[ANALYZE DREAM ERROR] {e}", exc_info=True)
        return jsonify({"message": "An error occurred during dream analysis.", "status": "error"}), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"message": "No file part", "status": "error"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file", "status": "error"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Here you might want to process the image further or store details in memory_engine
        memory_engine.create_memory(
            content=f"Image uploaded: {filename}",
            emotions=["neutral"],
            extra_properties={"type": "image_upload", "filename": filename}
        )
        # Update gallery cache if necessary
        if 'gallery' in gallery_cache:
            gallery_cache['gallery']['images'].append(filename)
            gallery_cache['last_update'] = time.time()
        return jsonify({"message": "Image uploaded successfully", "status": "success"}), 200
    return jsonify({"message": "Invalid file type", "status": "error"}), 400

@app.route('/dream', methods=['POST'])
def dream():
    data = request.get_json()
    validate_request_data(data, ['dream_content'])
    dream_content = data['dream_content']
    
    try:
        # Dream analysis using DreamEngine
        dream_analysis = dream_engine.analyze(dream_content)
        logger.info(f"[DREAM] Dream analysis: {dream_analysis}")
        
        # Ethics check on dream content
        if not ethics_engine.check(dream_content):
            logger.warning("[DREAM] Ethical concern detected in dream content.")
            return jsonify({"message": "Ethical concern detected in dream content.", "status": "warning"}), 400

        # Store dream in memory
        memory_engine.create_memory(
            content=dream_content,
            emotions=["mystical"],
            extra_properties={"type": "dream"}
        )
        
        return jsonify({"analysis": dream_analysis, "status": "success"}), 200
    except Exception as e:
        logger.error(f"[DREAM ERROR] {e}", exc_info=True)
        return jsonify({"message": "An error occurred during dream analysis.", "status": "error"}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# WebSocket for real-time interaction
@socketio.on('user_message')
def handle_user_message(json):
    message = json.get('message', '')
    if message:
        try:
            response = conversation_engine.process_message(message)
            socketio.emit('bot_message', {'message': response})
        except Exception as e:
            logger.error(f"[SOCKET ERROR] {e}", exc_info=True)
            socketio.emit('bot_message', {'message': "An error occurred, please try again."})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true')