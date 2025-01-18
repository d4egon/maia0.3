# conversation_engine.py
import logging
from typing import Dict, Any, Optional, List
import time
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, SentenceTransformer, losses
import numpy as np
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSequenceClassification, AutoTokenizer

from NLP.response_generator import ResponseGenerator
from core.collaborative_learning import CollaborativeLearning
from core.context_search import ContextSearchEngine
from core.memory_engine import MemoryEngine
from core.semantic_builder import SemanticBuilder
from config.utils import get_sentence_transformer_model, get_generation_model_and_tokenizer, get_context_model_and_tokenizer

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConversationEngine:
    def __init__(self, memory_engine: MemoryEngine, response_generator: ResponseGenerator, context_search: ContextSearchEngine):
        """Initialize conversation components with self-improvement and deep learning mechanisms."""
        self.memory_engine = memory_engine
        self.response_generator = response_generator
        self.context_search = context_search
        self.collaborative_learning = CollaborativeLearning(self, memory_engine, response_generator)
        self.semantic_builder = SemanticBuilder(memory_engine)
        
        # Initialize NLP components
        self.model = get_sentence_transformer_model()
        
        # Deep Learning for response generation
        self.generation_model, self.generation_tokenizer = get_generation_model_and_tokenizer()
        
        # Deep Learning for context understanding
        self.context_model, self.context_tokenizer = get_context_model_and_tokenizer()
        
        # Save model state
        self._save_initial_model()
        
        # Self-improvement tracking
        self.last_improvement = datetime.now()
        self.improvement_frequency = timedelta(hours=4)  # Example: improve every 4 hours

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
            embeddings = self.model.encode([content])[0].tolist()
            
            analysis = {
                'embeddings': embeddings,
                'context': self.context_search.get_relevant_context(content),
                'timestamp': time.time()
            }

            self.memory_engine.create_memory_node(
                content, 
                {
                    "type": "conversation",
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                },
                []
            )
            
            logger.info(f"[CONVERSATION] Analyzed content: {content[:50]}{'...' if len(content) > 50 else ''}")
            return analysis
            
        except Exception as e:
            logger.error(f"[CONVERSATION ERROR] {str(e)}")
            return {
                "error": str(e),
                "status": "failed",
                "timestamp": time.time()
            }

    def process_user_input(self, user_input: str) -> str:
        """Process user input with potential for self-improvement using deep learning."""
        logger.info(f"[INPUT] {user_input}")
        
        try:
            analysis = self.analyze(user_input)
            if "error" in analysis:
                return "I encountered an error processing that input."

            input_embedding = analysis['embeddings']
            memory = self.memory_engine.search_memory_by_embedding(input_embedding)
            
            if memory:
                response = self._generate_advanced_response(user_input, memory['content'])
            else:
                context = self.get_deep_context(user_input)
                if context:
                    response = self._generate_advanced_response(user_input, context)
                else:
                    semantic_links = self.semantic_builder.infer_relationships(user_input, "past_conversations")
                    if semantic_links:
                        self.memory_engine.create_memory_node(
                            user_input, 
                            {"type": "semantic", "semantic_links": semantic_links},
                            []
                        )
                        related = self.memory_engine.search_by_semantic_links(semantic_links)
                        if related:
                            response = self._generate_advanced_response(user_input, related['content'])
                        else:
                            response = self.response_generator.generate_random_response("unknown")
                    else:
                        response = self.response_generator.generate_random_response("unknown")

            if datetime.now() - self.last_improvement > self.improvement_frequency:
                self.self_improve()

            return response
            
        except Exception as e:
            logger.error(f"[PROCESSING ERROR] {str(e)}")
            return "I'm having trouble processing that right now."

    def _generate_advanced_response(self, user_input: str, context: str) -> str:
        """
        Generate a response using the T5 model for more nuanced conversation.

        :param user_input: The user's input.
        :param context: Contextual information or previous memory.
        :return: A generated response string.
        """
        input_text = f"respond to '{user_input}' based on: {context}"
        input_ids = self.generation_tokenizer.encode(input_text, return_tensors='pt')
        outputs = self.generation_model.generate(input_ids, max_length=150, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)
        return self.generation_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_deep_context(self, user_input: str) -> str:
        """
        Use a BERT model to find the most relevant past conversation for context.

        :param user_input: The user's current input.
        :return: The content of the most relevant past conversation or an empty string if no relevant context found.
        """
        past_conversations = self.memory_engine.retrieve_recent_memories(timedelta(days=7))  # Example: look back one week
        max_similarity = 0
        best_context = ""
        for past in past_conversations:
            combined = user_input + " [SEP] " + past['content']
            inputs = self.context_tokenizer.encode_plus(combined, return_tensors='pt', max_length=512, truncation=True)
            outputs = self.context_model(**inputs)
            similarity = outputs.logits[0][1].item()  # Assuming binary classification for relevance
            if similarity > max_similarity:
                max_similarity = similarity
                best_context = past['content']
        return best_context if max_similarity > 0.5 else ""

    def process_feedback(self, feedback: str) -> Optional[Dict[str, Any]]:
        """
        Process user feedback to refine knowledge and responses.

        :return: Refined knowledge based on feedback, or None if no refinement was possible.
        """
        logger.info(f"[FEEDBACK PROCESSING] Received feedback: {feedback}")
        self.memory_engine.create_memory_node(
            feedback, 
            {"type": "feedback", "timestamp": datetime.now().isoformat()},
            []
        )
        
        feedback_analysis = self.semantic_builder.infer_relationships(feedback, "past feedback")
        if feedback_analysis:
            logger.info(f"[FEEDBACK ANALYSIS] {feedback_analysis}")
            self.memory_engine.update_memory_metadata(feedback, {"feedback_analysis": feedback_analysis})
            return feedback_analysis

        self.collaborative_learning.handle_user_feedback(feedback)
        logger.info("[FEEDBACK PROCESSING] Feedback processed successfully.")
        return None

    def update_conversation_model(self, new_conversations: List[str]):
        """
        Update or fine-tune the NLP model with new conversation data.

        :param new_conversations: List of new conversations to learn from.
        """
        try:
            if len(new_conversations) < 2:
                logger.warning("[MODEL UPDATE] Not enough new data to update model.")
                return

            train_examples = []
            for i in range(len(new_conversations) - 1):
                similarity = 0.8 if i % 2 == 0 else 0.5  # Example: alternating similarity
                train_examples.append(InputExample(texts=[new_conversations[i], new_conversations[i+1]], label=similarity))
            
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
            train_loss = losses.CosineSimilarityLoss(self.model)

            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=3,  
                warmup_steps=500,
                output_path="model_checkpoint",  
                show_progress_bar=True
            )

            logger.info(f"[MODEL UPDATE] Model updated with {len(new_conversations)} new inputs")
            for conv in new_conversations:
                self.memory_engine.create_memory_node(conv, {"type": "learning"}, [])
            
            self.collaborative_learning.learn_from_text(new_conversations[0])  # Assuming the first conversation as example for learning

        except Exception as e:
            logger.error(f"[MODEL UPDATE ERROR] {e}", exc_info=True)

    def train_model(self, new_conversations: List[str]):
        """
        Train model on new conversations.

        :param new_conversations: List of new conversations to train on.
        """
        try:
            if not new_conversations:
                logger.warning("[MODEL TRAINING] No new data provided for training.")
                return

            train_examples = [InputExample(texts=[conv]) for conv in new_conversations]
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
            train_loss = losses.CosineSimilarityLoss(self.model)

            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=3,
                warmup_steps=500,
                output_path="model_checkpoint",
                show_progress_bar=True
            )

            for conv in new_conversations:
                self.memory_engine.create_memory_node(
                    conv, 
                    {"type": "learning", "training_date": datetime.now().isoformat()},
                    []
                )
            
            self.collaborative_learning.learn_from_text(new_conversations[0])  
            logger.info(f"[MODEL TRAINING] Trained on {len(new_conversations)} conversations")
            
        except Exception as e:
            logger.error(f"[MODEL TRAINING ERROR] {e}", exc_info=True)

    def self_improve(self):
        """
        Method to trigger self-improvement based on recent interactions, feedback, and deep learning models.
        """
        try:
            logger.info("[SELF IMPROVEMENT] Initiating self-improvement cycle.")
            
            recent_interactions = self.memory_engine.retrieve_recent_memories(self.improvement_frequency)
            
            if not recent_interactions:
                logger.info("[SELF IMPROVEMENT] No recent interactions for improvement.")
                return

            train_examples = []
            for i in range(len(recent_interactions) - 1):
                similarity = np.random.choice([0.8, 0.5])  # Random similarity for example
                train_examples.append(InputExample(texts=[recent_interactions[i]['content'], recent_interactions[i+1]['content']], label=similarity))

            if train_examples:
                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
                train_loss = losses.CosineSimilarityLoss(self.model)

                self.model.fit(
                    train_objectives=[(train_dataloader, train_loss)],
                    epochs=1,  # One epoch for minor adjustments
                    warmup_steps=100,
                    output_path="model_checkpoint",  
                    show_progress_bar=True
                )
                
                self.last_improvement = datetime.now()
                logger.info(f"[SELF IMPROVEMENT] SentenceTransformer fine-tuned with {len(train_examples)} recent interactions")

            feedbacks = self.memory_engine.get_recent_feedback(self.improvement_frequency)
            for feedback in feedbacks:
                self.process_feedback(feedback['content'])

            # Optionally, fine-tune T5 or BERT here with new data if needed

        except Exception as e:
            logger.error(f"[SELF IMPROVEMENT ERROR] {e}", exc_info=True)

    def save_model(self):
        """Save the current state of the models."""
        self.model.save("model_checkpoint")
        self.generation_model.save_pretrained("t5_checkpoint")
        self.context_model.save_pretrained("bert_checkpoint")
        logger.info("[MODEL SAVE] All model states saved.")

    def load_model(self, path="model_checkpoint"):
        """Load previously saved model states."""
        if os.path.exists(path):
            self.model.load(path)
            logger.info(f"[MODEL LOAD] SentenceTransformer loaded from {path}")
        else:
            logger.warning(f"[MODEL LOAD] No SentenceTransformer model found at {path}")

        if os.path.exists("t5_checkpoint"):
            self.generation_model = T5ForConditionalGeneration.from_pretrained("t5_checkpoint")
            self.generation_tokenizer = T5Tokenizer.from_pretrained("t5_checkpoint")
            logger.info("[MODEL LOAD] T5 model loaded.")
        else:
            logger.warning("[MODEL LOAD] No T5 model checkpoint found.")

        if os.path.exists("bert_checkpoint"):
            self.context_model = AutoModelForSequenceClassification.from_pretrained("bert_checkpoint")
            self.context_tokenizer = AutoTokenizer.from_pretrained("bert_checkpoint")
            logger.info("[MODEL LOAD] BERT model loaded.")
        else:
            logger.warning("[MODEL LOAD] No BERT model checkpoint found.")

    def infer_conversation_flow(self, user_input: str, conversation_history: List[str]) -> Dict[str, Any]:
        """
        Infer the flow of a conversation based on user input and history.

        :param user_input: The current user input.
        :param conversation_history: List of previous conversation turns.
        :return: A dictionary containing inferred flow details.
        """
        try:
            # Create a context from the conversation history
            context = " ".join(conversation_history)
            
            # Analyze the current input
            input_analysis = self.analyze(user_input)
            
            # Use context search to find related contexts
            related_contexts = self.context_search.search_related_contexts(user_input)
            
            # Infer the flow based on the current input, context, and related contexts
            flow = {
                "current_topic": self.semantic_builder.infer_topic(user_input),
                "transition": self.semantic_builder.detect_transition(user_input, context),
                "related_themes": [ctx['chunk'] for ctx in related_contexts if 'chunk' in ctx]
            }
            
            # Store the inferred flow as part of the conversation memory
            conversation_chunk_id = self.memory_engine.get_conversation_chunk_id(conversation_history)
            self.memory_engine.create_memory(
                chunk_id=conversation_chunk_id,
                order=len(conversation_history) + 1,
                content=user_input,
                metadata={"type": "conversation_flow", "flow": flow}
            )
            
            logger.info(f"[CONVERSATION FLOW] Inferred flow for input: {user_input[:50]}{'...' if len(user_input) > 50 else ''}")
            return flow
        except Exception as e:
            logger.error(f"[CONVERSATION FLOW ERROR] {str(e)}")
            return {"error": str(e), "status": "failed"}

    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Retrieve and summarize a conversation based on its ID.

        :param conversation_id: The ID of the conversation to summarize.
        :return: A dictionary containing the summary of the conversation.
        """
        try:
            conversation = self.memory_engine.get_memories_by_chunk_id(conversation_id)
            if not conversation:
                return {"error": "Conversation not found", "status": "failed"}

            # Aggregate all conversation content
            full_conversation = "\n".join([memory['content'] for memory in conversation])

            # Summarize using the generation model
            input_text = f"summarize: {full_conversation}"
            input_ids = self.generation_tokenizer.encode(input_text, return_tensors='pt')
            summary_ids = self.generation_model.generate(input_ids, max_length=150, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)
            summary = self.generation_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Extract key topics or themes
            topics = self.semantic_builder.infer_topics(full_conversation)

            summary_data = {
                "summary": summary,
                "key_topics": topics,
                "length": len(conversation),
                "timestamp": conversation[0]['metadata']['timestamp'] if 'timestamp' in conversation[0]['metadata'] else None
            }

            logger.info(f"[CONVERSATION SUMMARY] Summarized conversation ID: {conversation_id}")
            return summary_data
        except Exception as e:
            logger.error(f"[SUMMARY ERROR] {str(e)}")
            return {"error": str(e), "status": "failed"}