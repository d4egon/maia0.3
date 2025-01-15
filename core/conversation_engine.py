from email.headerregistry import DateHeader
import logging
from typing import Dict, Any, Optional, List
import spacy
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, SentenceTransformer, losses
from core.memory_engine import MemoryEngine
from NLP.response_generator import ResponseGenerator
from core.context_search import ContextSearchEngine
from core.collaborative_learning import CollaborativeLearning
from core.semantic_builder import SemanticBuilder

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConversationEngine:
    def __init__(self, memory_engine: MemoryEngine, response_generator: ResponseGenerator,
                 context_search: ContextSearchEngine):
        """Initialize conversation components."""
        self.memory_engine = memory_engine
        self.response_generator = response_generator
        self.context_search = context_search
        self.collaborative_learning = CollaborativeLearning(self)
        self.semantic_builder = SemanticBuilder(memory_engine.db)
        
        # Initialize NLP components
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        self.nlp = spacy.load("en_core_web_sm")
        
        # Save model state
        self._save_initial_model()

    def _save_initial_model(self):
        """Save initial model state."""
        model_directory = os.path.join(os.getcwd(), "model_checkpoint")
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        self.model.save(model_directory)
        logger.info(f"[MODEL INIT] Saved to {model_directory}")

    def analyze(self, content: str) -> Dict[str, Any]:
        """Analyze conversation content."""
        try:
            # Parse and encode
            doc = self.nlp(content)
            embeddings = self.model.encode(content)
            
            # Build analysis
            analysis = {
                'parsed': {
                    'entities': [ent.text for ent in doc.ents],
                    'key_phrases': [chunk.text for chunk in doc.noun_chunks],
                    'structure': {
                        'subjects': [token.text for token in doc if token.dep_ == 'nsubj'],
                        'verbs': [token.text for token in doc if token.pos_ == 'VERB'],
                        'objects': [token.text for token in doc if token.dep_ == 'dobj']
                    }
                },
                'embeddings': embeddings.tolist(),
                'context': self.context_search.get_relevant_context(content),
                'timestamp': time.time()
            }

            # Store in memory
            self.memory_engine.store_memory(
                content=content,
                types=["conversation"],
                extra_properties=analysis
            )
            
            logger.info(f"[CONVERSATION] Analyzed content: {len(doc)} tokens")
            return analysis
            
        except Exception as e:
            logger.error(f"[CONVERSATION ERROR] {str(e)}")
            return {
                "error": str(e),
                "status": "failed",
                "timestamp": time.time()
            }

    def process_user_input(self, user_input: str) -> str:
        """Process user input and generate response."""
        logger.info(f"[INPUT] {user_input}")
        
        try:
            # Analysis pipeline
            analysis = self.analyze(user_input)
            if "error" in analysis:
                return "I encountered an error processing that input."
            
            # Memory search
            memory = self.memory_engine.search_memory(user_input)
            if memory:
                return self.response_generator.generate_response(memory, user_input)
                
            # Context search
            context = self.context_search.search_related(user_input)
            if context:
                return self.response_generator.generate_response(context, user_input)
                
            # Semantic analysis
            semantic_links = self.semantic_builder.infer_relationships(user_input, "past_conversations")
            if semantic_links:
                self.memory_engine.store_memory(
                    user_input, 
                    ["semantic"], 
                    {"semantic_links": semantic_links}
                )
                related = self.memory_engine.search_by_semantic_links(semantic_links)
                if related:
                    return self.response_generator.generate_response(related, user_input)
            
            # Fallback response
            return self.response_generator.generate_fallback_response()
            
        except Exception as e:
            logger.error(f"[PROCESSING ERROR] {str(e)}")
            return "I'm having trouble processing that right now."

    def process_feedback(self, feedback: str):
        """
        Process user feedback to refine M.A.I.A.'s knowledge and responses.
        """
        logger.info(f"[FEEDBACK PROCESSING] Received feedback: {feedback}")
        self.memory_engine.store_feedback(feedback)
        
        # Use feedback for semantic analysis
        feedback_analysis = self.semantic_builder.infer_relationships(feedback, "past feedback")
        if feedback_analysis:
            logger.info(f"[FEEDBACK ANALYSIS] {feedback_analysis}")
            self.memory_engine.update_with_feedback(feedback, feedback_analysis)

        # Update collaborative learning with feedback
        self.collaborative_learning.process_feedback(feedback)

        logger.info("[FEEDBACK PROCESSING] Feedback stored, analyzed, and used for learning successfully.")

    def update_conversation_model(self, new_conversations: List[str]):
        """
        Update or fine-tune the NLP model with new conversation data.

        :param new_conversations: List of new conversations to learn from.
        """
        try:
            # Prepare training examples
            train_examples = []
            for i in range(len(new_conversations) - 1):  # Create pairs for similarity training
                # Adjust similarity based on context or use a default value
                similarity = 0.8 if i % 2 == 0 else 0.5  # Example: alternating similarity
                train_examples.append(InputExample(texts=[new_conversations[i], new_conversations[i+1]], label=similarity))
            
            # DataLoader for batching
            train_dataloader = DateHeader(train_examples, shuffle=True, batch_size=16)
            
            # Loss function
            train_loss = losses.CosineSimilarityLoss(self.model)

            # Training loop
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=3,  # You might want to adjust this based on how much you want to update per session
                warmup_steps=500,
                output_path="model_checkpoint",  # Save after each update
                show_progress_bar=True
            )

            logger.info(f"[MODEL UPDATE] Model updated with {len(new_conversations)} new inputs")
            self.memory_engine.store_bulk_memories(new_conversations, ["learning"])
            
            # Trigger collaborative learning after model update
            self.collaborative_learning.update_from_new_data(new_conversations)

        except Exception as e:
            logger.error(f"[MODEL UPDATE ERROR] {e}", exc_info=True)

    def train_model(self, new_conversations: List[str]):
        """Train model on new conversations."""
        try:
            # Create training examples
            train_examples = [
                InputExample(texts=[conv]) 
                for conv in new_conversations
            ]
            
            # Create proper DataLoader
            train_dataloader = DataLoader(
                train_examples, 
                shuffle=True, 
                batch_size=16
            )
            
            # Training setup
            train_loss = losses.CosineSimilarityLoss(self.model)

            # Training execution
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=3,
                warmup_steps=500,
                output_path="model_checkpoint",
                show_progress_bar=True
            )

            # Store training results
            self.memory_engine.store_bulk_memories(
                new_conversations, 
                ["learning"],
                {"training_date": datetime.now().isoformat()}
            )
            
            # Update collaborative learning
            self.collaborative_learning.update_from_new_data(new_conversations)
            
            logger.info(f"[MODEL UPDATE] Trained on {len(new_conversations)} conversations")
            
        except Exception as e:
            logger.error(f"[MODEL UPDATE ERROR] {e}", exc_info=True)
            raise