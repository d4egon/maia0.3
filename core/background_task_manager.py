# core/background_task_manager.py
import threading
import time
import psutil  # For system monitoring
from core.neo4j_connector import Neo4jConnection
from core.memory_engine import MemoryReinterpretationEngine, MemoryEngine
from core.reflective_journaling import ReflectiveJournaling
from core.self_initiated_conversation import SelfInitiatedConversation
from NLP.nlp_engine import NLP
import logging
from dotenv import load_dotenv # type: ignore
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Neo4j Connection using environment variables
db = Neo4jConnection(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

class BackgroundTaskManager:
    def __init__(self, nlp_engine: NLP, memory_engine: MemoryEngine):
        self.active = True
        self.memory_engine = memory_engine
        self.nlp_engine = nlp_engine
        self.memory_reinterpretation_engine = MemoryReinterpretationEngine(db)
        self.journal_engine = ReflectiveJournaling(db)
        self.conversation_engine = SelfInitiatedConversation(db)

    def resource_check(self):
        """
        Monitors CPU and memory usage to prevent system overload.
        
        :return: Boolean indicating if resources are within safe thresholds.
        """
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        logger.debug(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")
        return cpu_usage < 75 and memory_usage < 75  # Safe thresholds

    def introspection_cycle(self):
        """
        Periodically triggers memory reinterpretation.
        """
        while self.active:
            try:
                if self.resource_check():
                    memory = self.memory_reinterpretation_engine.retrieve_memory_for_reflection()
                    if memory:
                        reinterpretation = self.memory_reinterpretation_engine.reinterpret_memory(memory)
                        logger.info(f"Introspection Result: {reinterpretation}")
            except Exception as e:
                logger.error(f"Error in introspection cycle: {e}", exc_info=True)
            finally:
                time.sleep(1800)  # Every 30 minutes

    def journaling_cycle(self):
        """
        Periodically triggers reflective journaling.
        """
        while self.active:
            try:
                if self.resource_check():
                    self.journal_engine.log_event(
                        "Background Thought",
                        "I took a moment to reflect on recent events.",
                        "reflective"
                    )
                    logger.info("Journal Entry Created.")
            except Exception as e:
                logger.error(f"Error in journaling cycle: {e}", exc_info=True)
            finally:
                time.sleep(3600)  # Every hour

    def conversation_cycle(self):
        """
        Triggers self-initiated conversations on unresolved queries.
        """
        while self.active:
            try:
                if self.resource_check():
                    self.conversation_engine.check_emotional_triggers()
            except Exception as e:
                logger.error(f"Error in conversation cycle: {e}", exc_info=True)
            finally:
                time.sleep(900)  # Every 15 minutes

    def monitor_and_retrain(self):
        """
        Monitor conditions to trigger model retraining in the background.
        """
        while self.active:
            try:
                if self.resource_check() and self._should_retrain():
                    self._retrain_model()
            except Exception as e:
                logger.error(f"Error in retraining process: {e}", exc_info=True)
            finally:
                time.sleep(3600)  # Check hourly or adjust as needed

    def _should_retrain(self) -> bool:
        """
        Determine if retraining should be initiated based on certain conditions.

        :return: True if retraining is necessary, False otherwise.
        """
        new_data_count = len(self.memory_engine.retrieve_recent_memories())
        return new_data_count > 100  # Example condition: Retrain if more than 100 new memories

    def _retrain_model(self):
        """
        Initiate the retraining of the model.
        """
        logger.info("Initiating model retraining...")
        # Pseudo-code for model retraining
        # new_data = self.memory_engine.retrieve_recent_memories()
        # self.nlp_engine.retrain_model(new_data)

    def start(self):
        """
        Starts all background task threads.
        """
        threading.Thread(target=self.introspection_cycle, daemon=True, name="Introspection").start()
        threading.Thread(target=self.journaling_cycle, daemon=True, name="Journaling").start()
        threading.Thread(target=self.conversation_cycle, daemon=True, name="Conversation").start()
        threading.Thread(target=self.monitor_and_retrain, daemon=True, name="ModelRetraining").start()
        logger.info("All background tasks have been started.")

    def stop(self):
        """
        Stops all background tasks by setting active to False.
        """
        self.active = False
        logger.info("Background tasks stopping...")

# Test Function (Comment out in production)
if __name__ == "__main__":
    # Assuming these are initialized elsewhere or for testing purposes
    nlp_engine = NLP(None, None, None)  # Placeholder initialization
    memory_engine = MemoryEngine(db)  # Placeholder initialization
    manager = BackgroundTaskManager(nlp_engine, memory_engine)
    manager.start()

    try:
        logger.info("Background tasks started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()