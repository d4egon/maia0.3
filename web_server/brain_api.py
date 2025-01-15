# File: /web_server/brain_api.py
# Location: /web_server

import os
from flask import Flask, request, jsonify
from NLP import nlp_engine, consciousness_engine
from core import memory_linker
from dotenv import load_dotenv
from flasgger import Swagger # type: ignore
from flask_talisman import Talisman # type: ignore
from flask_caching import Cache # type: ignore
import logging
from logging.handlers import RotatingFileHandler
from NLP.consciousness_engine import ConsciousnessEngine
from NLP.nlp_engine import NLPEngine
from core.deduplication_engine import DeduplicationEngine
from core.attribute_enrichment import AttributeEnrichment
from core.interactive_learning import InteractiveLearning
from core.semantic_builder import SemanticBuilder
from core.feedback_loops import FeedbackLoops
from core.neo4j_connector import Neo4jConnector
from core.signal_emulation import SignalPropagation

# Load environment variables
load_dotenv()

# Retrieve credentials from .env
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

app = Flask(__name__)

# Setup logging
handler = RotatingFileHandler('brain_api.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# Initialize Neo4j connection (consider lazy loading if not always needed)
neo4j_conn = Neo4jConnector(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)

# Flask extensions for security and performance
Talisman(app, content_security_policy=None)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Setup Swagger for API documentation
swagger = Swagger(app)

# Helper functions to initialize engines
def get_deduplication_engine():
    if not hasattr(app, "deduplication_engine"):
        app.deduplication_engine = DeduplicationEngine(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    return app.deduplication_engine

def get_attribute_enrichment():
    if not hasattr(app, "attribute_enrichment"):
        app.attribute_enrichment = AttributeEnrichment(graph_client=neo4j_conn)
    return app.attribute_enrichment

def get_interactive_learning():
    if not hasattr(app, "interactive_learning"):
        app.interactive_learning = InteractiveLearning(graph_client=neo4j_conn)
    return app.interactive_learning

def get_semantic_builder():
    if not hasattr(app, "semantic_builder"):
        app.semantic_builder = SemanticBuilder(graph_client=neo4j_conn)
    return app.semantic_builder

def get_feedback_loops():
    if not hasattr(app, "feedback_loops"):
        app.feedback_loops = FeedbackLoops(graph_client=neo4j_conn)
    return app.feedback_loops

# Route functions
def deduplicate(label="Emotion"):
    """Perform deduplication on graph nodes with a specific label."""
    get_deduplication_engine().deduplicate(label)
    return {"message": f"Deduplication completed for label: {label}"}

def enrich_attributes(label="Emotion", auto=True):
    """Enrich attributes of nodes based on the label, either automatically or interactively."""
    enrichment = get_attribute_enrichment()
    missing_nodes = enrichment.get_missing_attributes(label)
    if auto:
        for node in missing_nodes:
            enrichment.auto_enrichment(node["id"], node["name"])
    else:
        enrichment.interactive_enrichment(missing_nodes)
    return {"message": f"Attributes enriched for label: {label}"}

def initiate_interactive_learning():
    """Initiate an interactive learning session to fill knowledge gaps."""
    learning = get_interactive_learning()
    knowledge_gaps = learning.identify_knowledge_gaps()
    learning.ask_questions(knowledge_gaps)
    return {"message": "Interactive learning completed."}

def recursive_introspection(theme):
    """Perform recursive introspection based on a user-provided theme."""
    results = consciousness_engine.expanded_recursive_reflection(theme, depth=5)
    app.logger.info(f"[RECURSIVE INTROSPECTION] Results: {results}")
    return {"results": results, "status": "success"}

def build_relationships(label="Emotion"):
    """Build semantic relationships between nodes based on their descriptions."""
    get_semantic_builder().build_relationships(label)
    return {"message": f"Relationships built for label: {label}"}

def validate_feedback(node_id, name, attributes):
    """Validate feedback for node attributes in the graph."""
    get_feedback_loops().prompt_user_validation(node_id, name, attributes)
    return {"message": "Feedback validation completed."}

def propagate_signal(start_node, max_hops=5):
    """Propagate a signal through the graph from a starting node."""
    signal_propagation = SignalPropagation(neo4j_conn)
    paths = signal_propagation.send_signal(start_node, max_hops)
    return {"paths": [str(path) for path in paths]}

def visualize_thoughts():
    """Generate a real-time visualization of M.A.I.A.'s thought process."""
    thought_graph = memory_linker.generate_visualization()
    app.logger.info(f"[THOUGHT VISUALIZATION] Generated successfully.")
    return {"visualization": thought_graph, "status": "success"}

def dynamic_conversation(user_input):
    """Start a dynamic conversation with M.A.I.A. based on user input."""
    intent = nlp_engine.detect_intent(user_input)
    response, _ = consciousness_engine.reflect(user_input)
    app.logger.info(f"[CONVERSATION RESPONSE] {response}")
    return {"response": response, "intent": intent, "status": "success"}

# Routes
@app.route("/v1/deduplicate", methods=["POST"])
def deduplicate_v1():
    try:
        label = request.json.get("label", "Emotion")
        return jsonify(deduplicate(label)), 200
    except Exception as e:
        app.logger.error(f"Deduplication error: {str(e)}")
        return jsonify({"error": "An error occurred during deduplication", "details": str(e)}), 500

@app.route("/v1/enrich_attributes", methods=["POST"])
def enrich_attributes_v1():
    try:
        label = request.json.get("label", "Emotion")
        auto_mode = request.json.get("auto", True)
        return jsonify(enrich_attributes(label, auto_mode)), 200
    except Exception as e:
        app.logger.error(f"Attribute enrichment error: {str(e)}")
        return jsonify({"error": "An error occurred during attribute enrichment", "details": str(e)}), 500

@app.route("/v1/interactive_learning", methods=["POST"])
def interactive_learning_v1():
    try:
        return jsonify(initiate_interactive_learning()), 200
    except Exception as e:
        app.logger.error(f"Interactive learning error: {str(e)}")
        return jsonify({"error": "An error occurred during interactive learning", "details": str(e)}), 500

@app.route("/v1/recursive_introspection", methods=["POST"])
def recursive_introspection_v1():
    try:
        theme = request.json.get("theme")
        if not theme:
            raise ValueError("Theme must be provided for recursive introspection.")
        return jsonify(recursive_introspection(theme)), 200
    except Exception as e:
        app.logger.error(f"[RECURSIVE INTROSPECTION ERROR] {e}", exc_info=True)
        return jsonify({"message": "Failed to perform introspection.", "status": "error", "details": str(e)}), 500

@app.route("/v1/build_relationships", methods=["POST"])
def build_relationships_v1():
    try:
        label = request.json.get("label", "Emotion")
        return jsonify(build_relationships(label)), 200
    except Exception as e:
        app.logger.error(f"Relationship building error: {str(e)}")
        return jsonify({"error": "An error occurred while building relationships", "details": str(e)}), 500

@app.route("/v1/validate_feedback", methods=["POST"])
def validate_feedback_v1():
    try:
        node_id = request.json["node_id"]
        name = request.json["name"]
        attributes = request.json["attributes"]
        return jsonify(validate_feedback(node_id, name, attributes)), 200
    except KeyError as ke:
        app.logger.error(f"Missing key in request: {str(ke)}")
        return jsonify({"error": "Missing required field in request", "details": str(ke)}), 400
    except Exception as e:
        app.logger.error(f"Feedback validation error: {str(e)}")
        return jsonify({"error": "An error occurred during feedback validation", "details": str(e)}), 500
    
@app.route("/v1/submit_feedback", methods=["POST"])
def submit_feedback():
    try:
        feedback = request.json.get("feedback")
        if not feedback:
            return jsonify({"error": "Feedback is required", "status_code": 400}), 400
        
        # Store feedback in Neo4j
        query = """
        CREATE (f:Feedback {text: $feedback, timestamp: datetime()})
        RETURN 'Feedback submitted successfully' AS message
        """
        result = neo4j_conn.run_query(query, {"feedback": feedback})
        if result:
            message = result[0]["message"]
            return jsonify({"message": message, "status_code": 200}), 200
        else:
            return jsonify({"error": "Failed to submit feedback", "status_code": 500}), 500
    except Exception as e:
        app.logger.error(f"Feedback submission error: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while submitting feedback", "details": str(e), "status_code": 500}), 500

@app.route("/v1/propagate_signal", methods=["POST"])
def propagate_signal_v1():
    try:
        start_node = request.json["start_node"]
        max_hops = request.json.get("max_hops", 5)
        return jsonify(propagate_signal(start_node, max_hops)), 200
    except KeyError as ke:
        app.logger.error(f"Missing key in request: {str(ke)}")
        return jsonify({"error": "Missing required field in request", "details": str(ke)}), 400
    except Exception as e:
        app.logger.error(f"Signal propagation error: {str(e)}")
        return jsonify({"error": "An error occurred during signal propagation", "details": str(e)}), 500

@app.route("/v1/visualize_thoughts", methods=["GET"])
def visualize_thoughts_v1():
    try:
        return jsonify(visualize_thoughts()), 200
    except Exception as e:
        app.logger.error(f"[VISUALIZATION ERROR] {e}", exc_info=True)
        return jsonify({"message": "Failed to generate visualization.", "status": "error", "details": str(e)}), 500

@app.route("/v1/conversation", methods=["POST"])
def conversation_v1():
    try:
        user_input = request.json.get("input")
        if not user_input:
            raise ValueError("User input is required for conversation.")
        app.logger.info(f"[CONVERSATION] User Input: {user_input}")
        return jsonify(dynamic_conversation(user_input)), 200
    except ValueError as ve:
        app.logger.error(f"Invalid input: {str(ve)}")
        return jsonify({"error": "Input error", "details": str(ve)}), 400
    except Exception as e:
        app.logger.error(f"[CONVERSATION ERROR] {e}", exc_info=True)
        return jsonify({"message": "An error occurred during the conversation.", "status": "error", "details": str(e)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # Generic exception handler
    app.logger.error(f"An error occurred: {str(e)}", exc_info=True)
    return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

if __name__ == "__main__":
    # Development server with SSL. For production, use a proper SSL certificate
    app.run(host="0.0.0.0", port=5000, ssl_context='adhoc')