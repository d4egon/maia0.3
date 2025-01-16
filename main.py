import os
import sys
import logging
import time
import json
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
from core.conversation_engine import ConversationEngine
from core.dream_engine import DreamEngine
from core.ethics_engine import EthicsEngine
from flask_socketio import SocketIO
from sentence_transformers import SentenceTransformer, util
import numpy as np

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
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

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
    file_parser = FileParser()
    nlp_engine = NLP(memory_engine, response_gen, neo4j)
    emotion_engine = EmotionEngine(db=neo4j)
    emotion_fusion_engine = EmotionFusionEngine(memory_engine, nlp_engine)
    dream_engine = DreamEngine(memory_engine, ContextSearchEngine(neo4j))
    ethics_engine = EthicsEngine(memory_engine)
    context_search_engine = ContextSearchEngine(neo4j)
    consciousness_engine = ConsciousnessEngine(memory_engine, emotion_engine)
    intent_detector = IntentDetector(memory_engine)
    
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

def update_model_with_text(text):
    def train_task(text):
        try:
            tokenized_input = model.tokenize(text)
            input_ids = tokenized_input["input_ids"]
            logger.info(f"[MODEL TRAIN] Training on text: {text[:50]}...")
        except Exception as e:
            logger.error(f"[MODEL TRAIN ERROR] Error during training: {e}", exc_info=True)

    Thread(target=train_task, args=(text,)).start()

# Routes
@app.route('/')
def index():
    logger.info("[ROUTE] Index accessed.")
    return render_template('index.html')

@app.route('/embed', methods=['POST'])
def embed():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        embedding = model.encode(text).tolist()
        try:
            update_model_with_text(text)
            logger.info(f"[MODEL UPDATE] Model updated with new text: {text[:50]}...")
        except Exception as e:
            logger.error(f"[MODEL UPDATE ERROR] {e}", exc_info=True)
        
            return jsonify({'embedding': embedding})
    except Exception as e:
        logger.error(f"[EMBED ERROR] Failed to encode text: {e}", exc_info=True)
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

        # Store the parsed content in memory
        memory_engine.store_memory(
            text=result,
            emotions=["neutral"],
            extra_properties={"source": filename, "type": "advanced_upload" if is_advanced else "upload"}
        )

        # Ethics Check
        if not ethics_engine.check(result):
            logger.warning("[ETHICS CHECK] Ethical concern detected in uploaded content.")
            # Here you might want to handle ethical concerns, e.g., by notifying an admin or flagging the content

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

        return jsonify({"text": transcription})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask_maia', methods=['POST'])
def ask_maia():
    try:
        data = request.get_json()
        validate_request_data(data, ['question'])
        question = data['question'].lower()
        
        # Standard Response Check
        response, intent = get_standard_response(question)
        if response:
            if intent != "greeting":
                memory_engine.store_memory(text=f"Q: {question}\nA: {response}", emotions=["neutral"], extra_properties={"type": "conversation"})
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
            # Store interaction with emotion data
            memory_engine.store_memory(text=f"Q: {question}\nA: {response}", emotions=[fused_emotion], extra_properties={"type": "conversation", "intent": intent})

        # Ethics Check on Response
        if not ethics_engine.check(response):
            logger.warning("[ETHICS CHECK] Ethical concern detected in response.")
            # Handle ethical concern here, perhaps by using a fallback response or notifying an admin

        return jsonify({"response": response, "intent": intent, "status": "success"}), 200
    except Exception as e:
        logger.error(f"[ASK MAIA] Error: {e}", exc_info=True)
        return jsonify({"message": "An error occurred.", "status": "error"}), 500

@app.route('/get_gallery_images', methods=['GET'])
def get_gallery_images():
    if 'gallery' not in gallery_cache or 'last_update' not in gallery_cache or (time.time() - gallery_cache['last_update']) > 3600:
        gallery_cache['gallery'] = {
            'images': cached_get_gallery_images(),
            'last_update': time.time()
        }
    return jsonify({"images": gallery_cache['gallery']['images']})

# Serve static files (like images)
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('web_server/static', path)

if __name__ == "__main__":
    logger.info("[START] MAIA Server is starting...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")