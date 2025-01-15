# main.py

import os
import sys
import logging
import time
import json
from threading import Thread
from functools import lru_cache
from tempfile import NamedTemporaryFile
from flask import Flask, render_template, request, jsonify, abort
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from config.settings import CONFIG
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
#from core.memory_linker import MemoryLinker    #dud for now
from core.context_search import ContextSearchEngine
#from core.deduplication_engine import DeduplicationEngine   #dud for now
from NLP.consciousness_engine import ConsciousnessEngine
from NLP.intent_detector import IntentDetector
from core.self_initiated_conversation import SelfInitiatedConversation
from flask_socketio import SocketIO
from sentence_transformers import SentenceTransformer

# Load Environment Variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO if os.getenv("FLASK_DEBUG", "false").lower() != "true" else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("main.log"),
    ],
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Constants
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(os.getcwd(), "uploads"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 1024))
ALLOWED_EXTENSIONS = {
    'tar', 'xz', 'zip', '7z', 'txt', 'pdf', 'epub', 'lit', 'md', 'doc', 'docx', 'csv', 'xml', 'html', 'odt', 'raw', 'jpg', 'png', 'jpeg', 'mp3', 'mp4', 'wav',
    'gif', 'bmp', 'tiff', 'svg', 'json', 'yaml', 'rtf', 'ppt', 'pptx', 'xls', 'xlsx', 'ods', 'ogg', 'flac', 'aac', 'm4a', 'mov', 'avi', 'wmv', 'mkv', 'webm',
    'mobi', 'azw', 'azw3', 'fb2', 'djvu', 'ps', 'tex', 'log', 'ini', 'conf', 'properties', 'bat', 'cmd', 'sh', 'py', 'js', 'java', 'cpp', 'c', 'h', 'hpp',
    'cs', 'vb', 'sql', 'php', 'rb', 'pl', 'pm', 'css', 'scss', 'less', 'sass', 'coffee', 'go', 'swift', 'kotlin', 'scala', 'rust', 'dart', 'lua', 'r', 'matlab',
    'm', 'asm', 's', 'f', 'f90', 'fortran', 'lisp', 'clojure', 'erl', 'hs', 'm4', 'pro', 'pas', 'dpr', 'dfm', 'vhd', 'vhdl', 'v', 'vh', 'vhf', 'vbs', 'asp',
    'aspx', 'jsp', 'cfm', 'cgi', 'fcgi', 'htaccess', 'htpasswd', 'dat', 'db', 'dbf', 'mdb', 'accdb', 'sqlite', 'sqlite3', 'sqlitedb', 'db3', 'fdb',
    'frm', 'myd', 'myi', 'ibd', 'odp', 'odg', 'odf', 'sxw', 'stw', 'sxc', 'stc', 'sxd', 'std', 'sxi', 'sti', 'sxm', 'sxg', 'otg', 'otp', 'ots', 'ott', 'otm',
    'ico', 'cur', 'ani', 'tga', 'pcx', 'psd', 'ai', 'eps', 'dxf', 'dwg', 'stl', 'obj', 'fbx', 'blend', 'dae', 'x', '3ds', 'wrl', 'vrml', 'x3d', 'gcode', 'iges',
    'igs', 'step', 'stp', 'mid', 'midi', 'rm', 'rmvb', 'flv', '3gp', '3g2', 'asf', 'wma', 'ra', 'ram', 'm3u', 'pls', 'm3u8', 'cue', 'kar', 'xm', 'mod', 's3m', 'it',
    'iso', 'img', 'nrg', 'mds', 'mdf', 'bin', 'ccd', 'sub', 'c2d', 'cdi', 'pdi', 'vcd', 'dmg', 'sparseimage', 'smi', 'smil', 'ass', 'srt', 'ssa', 'idx', 'vob', 'ifo',
    'bup', 'mpls', 'ttf', 'otf', 'woff', 'woff2', 'eot', 'fon', 'fnt', 'dfont', 'pfb', 'pfm', 'afm', 'otb', 'ttc', 'psf', 'pcf', 'bdf', 'snf', 'apk', 'ipa', 'deb',
    'rpm', 'pkg', 'msi', 'cab', 'exe', 'dll', 'ocx', 'sys', 'drv', 'com', 'key', 'numbers', 'pages', 'keynote'
}

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
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'  # Prevent clickjacking
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

    if not neo4j_uri or not neo4j_user or not neo4j_password:
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
    ethics_engine = EthicsEngine(neo4j)
    #memory_linker = MemoryLinker(neo4j)
    context_search_engine = ContextSearchEngine(neo4j)
    #deduplication_engine = DeduplicationEngine(neo4j_uri, neo4j_user, neo4j_password)
    consciousness_engine = ConsciousnessEngine(memory_engine, emotion_engine)
    intent_detector = IntentDetector(memory_engine)

    conversation_engine = ConversationEngine(
        memory_engine,
        response_gen,
        context_search_engine
    )
    self_initiated_conversation = SelfInitiatedConversation(memory_engine, conversation_engine, context_search_engine)

    logger.info("[INIT SUCCESS] All services initialized successfully.")
except Exception as e:
    logger.critical(f"[INIT FAILED] Failed to initialize services: {e}", exc_info=True)
    sys.exit(1)

# Initialize core components
db = Neo4jConnector(
    CONFIG["NEO4J_URI"],
    CONFIG["NEO4J_USER"],
    CONFIG["NEO4J_PASSWORD"]
)

memory_engine = MemoryEngine(db)
nlp_engine = NLP()
emotion_fusion_engine = EmotionFusionEngine(memory_engine, nlp_engine, db)

# Cache for Gallery Images
gallery_cache = {}

@lru_cache(maxsize=300)
def cached_get_gallery_images():
    logger.info("[GALLERY] Fetching gallery images.")
    image_folder = os.path.join(app.static_folder, "images")
    all_files = os.listdir(image_folder)
    image_files = [f for f in all_files if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
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
    """
    Check if the user's question matches any standard phrase and return the corresponding response.

    :param question: The user's input to check against standard phrases.
    :return: A tuple (response, intent) if a match is found, otherwise (None, None).
    """
    question = question.lower()
    for category, responses in standard_responses.items():
        for response_item in responses:
            if any(phrase in question for phrase in response_item['input']):
                return response_item['response'], category
    return None, None

def should_trigger_search(question):
    """
    Determine if the user's question should trigger a search.

    :param question: The user's input to check.
    :return: Boolean indicating if a search should be performed.
    """
    search_keywords = ['find', 'search', 'look up', 'what is', 'who is', 'where is', 'when is', 'how to']
    return any(keyword in question.lower() for keyword in search_keywords)

def perform_search(query):
    """
    Perform a search operation. This is a placeholder and should be replaced with actual search logic.

    :param query: The query to search for.
    :return: A list of search results (placeholder).
    """
    # Example using a hypothetical search function
    # In practice, you would implement real search logic here
    results = ["This is a placeholder for search results"]
    return results

def summarize_search_results(results):
    """
    Summarize search results. This is a placeholder and should be replaced with actual summarization logic.

    :param results: List of search results.
    :return: A string summarizing the results.
    """    # Example summarization
    return f"Here are some results: {', '.join(results)}"

def update_model_with_text(text):
    """
    Function to update the model with new text data in a background thread.
    
    This function simulates the process of continuous learning by initiating
    a background task to train the model with new text. Actual implementation
    would involve proper data preparation, model training, and integration with
    your learning framework.

    :param text: Text to be used for updating the model.
    """
    def train_task(text):
        try:
            # Placeholder for data preparation
            tokenized_input = model.tokenize(text)
            input_ids = tokenized_input["input_ids"]
            
            # Placeholder for model training
            # Here you would implement the logic to update or retrain your model
            # with new data, possibly using something like model.fit() or similar
            # depending on your framework
            logger.info(f"[MODEL TRAIN] Training on text: {text[:50]}...")
        except Exception as e:
            logger.error(f"[MODEL TRAIN ERROR] Error during training: {e}", exc_info=True)

    # Start the training in a background thread
    Thread(target=train_task, args=(text,)).start()

# Routes
@app.route('/')
def index():
    logger.info("[ROUTE] Index accessed.")
    return render_template('index.html')

@app.route('/embed', methods=['POST'])
def embed():
    """
    Endpoint to generate text embeddings and optionally update the model.
    
    :return: JSON response with the text embedding or an error message.
    """
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        embedding = model.encode(text).tolist()
        
        # Continuous learning: Optionally update model here with new text
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
    """
    Handle file uploads with optional advanced processing.
    """
    if 'file' not in request.files:
        return jsonify({"message": "No file uploaded", "status": "error"}), 400

    file = request.files['file']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"message": "Invalid file type", "status": "error"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        # Save the file temporarily
        file.save(filepath)

        # Parse the file
        result = file_parser.parse(filepath)
        if result is None:
            raise ValueError("File parsing returned None, possibly unsupported file type or parsing error.")
        if not isinstance(result, str) or not result.strip():
            raise ValueError("Parsed content is empty or not a string.")

        # Advanced processing (e.g., metadata extraction) based on request parameter
        is_advanced = request.args.get("advanced", "false").lower() == "true"
        if is_advanced:
            logger.info("[ADVANCED PROCESSING] Additional processing steps applied.")
            # Add advanced-specific logic here (e.g., semantic analysis, metadata extraction)

        # Store the parsed content in memory
        memory_engine.store_memory(
            text=result,
            emotions=["neutral"],  # Placeholder emotion
            extra_properties={"source": filename, "type": "advanced_upload" if is_advanced else "upload"}
        )

        # Continuous Learning: Train model on new file content
        train_model_on_file_content(result)

        # Clean up the temporary file
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
    # This would be more complex in practice, involving data preparation and batching
    sentences = content.split('. ')
    for sentence in sentences:
        update_model_with_text(sentence)  # Use our earlier function for individual sentences

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    try:
        audio_file = request.files['audio']
        save_path = os.path.join('uploads', audio_file.filename)
        audio_file.save(save_path)

        # Pass the file to your parser (replace with your implementation)
        transcription = file_parser.parse_audio(save_path, language="en")
        return jsonify({"text": transcription})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask_maia', methods=['POST'])
def ask_maia():
    """
    Endpoint to handle user questions, using standard responses for simple phrases.

    :return: JSON response with MAIA's response, intent, or an error message.
    """
    try:
        data = request.get_json()
        validate_request_data(data, ['question'])
        question = data['question'].lower()  # Convert to lower case for easier comparison
        
        # Check for standard responses first
        response, intent = get_standard_response(question)
        if response:
            # Log the interaction for potential future learning, but only if it's not a simple greeting
            if intent != "greeting":
                memory_engine.store_memory(text=f"Q: {question}\nA: {response}", emotions=["neutral"], extra_properties={"type": "conversation"})
            return jsonify({"response": response, "intent": intent, "status": "success"}), 200
        
        # If no standard response, proceed with normal NLP processing
        if should_trigger_search(question):
            search_results = perform_search(question)
            response = summarize_search_results(search_results)
            intent = "search"
        else:
            intent = nlp_engine.intent_detector.detect_intent(question)
            response, _ = nlp_engine.process(question)

        # Log the interaction for potential future learning
        memory_engine.store_memory(text=f"Q: {question}\nA: {response}", emotions=["neutral"], extra_properties={"type": "conversation"})

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

if __name__ == "__main__":
    logger.info("[START] MAIA Server is starting...")
    self_initiated_conversation.start_scheduler()
    socketio.run(app, host="0.0.0.0", port=5000, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")