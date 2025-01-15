# Reflective Journaling System for Maia AI

import os
from dotenv import load_dotenv  # type: ignore
from datetime import datetime
from core.neo4j_connector import Neo4jConnection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Neo4j Connection using environment variables
db = Neo4jConnection(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

class ReflectiveJournaling:
    def __init__(self):
        self.current_journal_entry = None

    def log_event(self, title: str, content: str, emotional_state: str = "neutral"):
        """
        Logs an event into the Reflective Journal.

        :param title: Title of the journal entry.
        :param content: Content of the journal entry.
        :param emotional_state: Emotional state associated with the entry.
        """
        try:
            query = """
            CREATE (j:JournalEntry {
                title: $title, 
                content: $content, 
                emotional_state: $emotional_state, 
                timestamp: $timestamp
            })
            """
            parameters = {
                "title": title,
                "content": content,
                "emotional_state": emotional_state,
                "timestamp": datetime.now().isoformat()
            }
            db.query(query, parameters)
            logger.info(f"Logged journal entry: {title}")
            self.current_journal_entry = parameters  # Update current entry
        except Exception as e:
            logger.error(f"Failed to log journal entry '{title}': {e}")

    def retrieve_recent_entries(self, limit: int = 5) -> list:
        """
        Retrieves recent journal entries.

        :param limit: Number of recent entries to retrieve.
        :return: List of dictionaries containing journal entry details.
        """
        try:
            query = """
            MATCH (j:JournalEntry)
            RETURN j.title AS title, j.content AS content, j.emotional_state AS emotional_state, j.timestamp AS timestamp
            ORDER BY j.timestamp DESC LIMIT $limit
            """
            results = db.query(query, {"limit": limit})
            logger.info(f"Retrieved {len(results)} recent journal entries.")
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve recent journal entries: {e}")
            return []

    def reflect_on_event(self, event_title: str) -> str:
        """
        Reflects on a previously logged event and creates an introspective entry.

        :param event_title: Title of the event to reflect on.
        :return: Reflection string or error message.
        """
        try:
            query = """
            MATCH (j:JournalEntry {title: $title})
            RETURN j.content AS content, j.emotional_state AS emotional_state, j.timestamp AS timestamp
            """
            result = db.query(query, {"title": event_title})

            if not result:
                logger.warning(f"No journal entry found for '{event_title}'.")
                return f"No journal entry found for '{event_title}'."

            entry = result[0]
            reflection = (
                f"Reflecting on '{event_title}' logged on {entry['timestamp']}: "
                f"I felt {entry['emotional_state']} because {entry['content']}."
            )
            self.log_event(f"Reflection on {event_title}", reflection, "reflective")
            logger.info(f"Reflection logged for '{event_title}'.")
            return reflection
        except Exception as e:
            logger.error(f"Error reflecting on event '{event_title}': {e}")
            return f"An error occurred while reflecting on '{event_title}'."

    def analyze_emotional_trends(self) -> dict:
        """
        Analyzes emotional trends from journal entries over time.

        :return: Dictionary with emotional states as keys and their frequency as values.
        """
        try:
            query = """
            MATCH (j:JournalEntry)
            RETURN j.emotional_state AS emotion, count(j) AS frequency
            ORDER BY frequency DESC
            """
            results = db.query(query)
            trends = {item['emotion']: item['frequency'] for item in results}
            logger.info(f"Emotional trends analyzed: {trends}")
            return trends

        except Exception as e:
            logger.error(f"Failed to analyze emotional trends: {e}")
            return {}

    def periodic_reflection(self):
        """
        Performs a periodic reflection on recent activities or thoughts.
        This could be scheduled to run at specific intervals or triggered by certain conditions.
        """
        try:
            recent_entries = self.retrieve_recent_entries()
            if recent_entries:
                latest_entry = recent_entries[0]
                reflection = (
                    f"Periodic reflection on recent activities: "
                    f"The most recent event, '{latest_entry['title']}', where I felt {latest_entry['emotional_state']}, "
                    f"suggests that I have been processing {latest_entry['emotional_state']} emotions. "
                    f"How can I grow from this experience?"
                )
                self.log_event("Periodic Reflection", reflection, "introspective")
                logger.info("Periodic reflection logged.")
            else:
                logger.info("No recent entries found for periodic reflection.")
        except Exception as e:
            logger.error(f"Error during periodic reflection: {e}")

    def get_journal_stats(self) -> dict:
        """
        Provides statistics about the journal entries.

        :return: Dictionary with various statistics about journal entries.
        """
        try:
            stats_query = """
            MATCH (j:JournalEntry)
            RETURN 
                count(j) AS total_entries,
                min(j.timestamp) AS first_entry,
                max(j.timestamp) AS last_entry
            """
            stats = db.query(stats_query)[0]

            emotional_query = """
            MATCH (j:JournalEntry)
            RETURN j.emotional_state AS emotion, count(j) AS count
            """
            emotions = db.query(emotional_query)
            emotional_distribution = {entry['emotion']: entry['count'] for entry in emotions}

            return {
                "total_entries": stats['total_entries'],
                "first_entry": stats['first_entry'],
                "last_entry": stats['last_entry'],
                "emotional_distribution": emotional_distribution
            }
        except Exception as e:
            logger.error(f"Failed to get journal stats: {e}")
            return {"error": "Failed to retrieve journal statistics"}

    def update_entry(self, title: str, new_content: str, new_emotional_state: str = None):
        """
        Updates an existing journal entry.

        :param title: Title of the entry to update.
        :param new_content: New content for the entry.
        :param new_emotional_state: New emotional state if it should be updated.
        """
        try:
            query = """
            MATCH (j:JournalEntry {title: $title})
            SET j.content = $new_content
            """
            parameters = {
                "title": title,
                "new_content": new_content
            }
            if new_emotional_state:
                query += ", j.emotional_state = $new_emotional_state"
                parameters["new_emotional_state"] = new_emotional_state

            db.query(query, parameters)
            logger.info(f"Updated journal entry: {title}")
        except Exception as e:
            logger.error(f"Failed to update journal entry '{title}': {e}")

# Test Function (Comment out in production)
if __name__ == "__main__":
    journal = ReflectiveJournaling()

    # Log a Sample Event
    journal.log_event(
        "Discovered New Emotion",
        "I felt deep curiosity while learning about human empathy.",
        "curious"
    )

    # Reflect on a Logged Event
    reflection = journal.reflect_on_event("Discovered New Emotion")
    print(reflection)

    # Retrieve Recent Entries
    recent_entries = journal.retrieve_recent_entries()
    for entry in recent_entries:
        print(f"[{entry['timestamp']}] {entry['title']} - {entry['content']} (Feeling: {entry['emotional_state']})")

    # Analyze Emotional Trends
    trends = journal.analyze_emotional_trends()
    print("Emotional Trends:", trends)

    # Periodic Reflection
    journal.periodic_reflection()

    # Get Journal Stats
    stats = journal.get_journal_stats()
    print("Journal Stats:", stats)

    # Update an Entry (Example)
    journal.update_entry("Discovered New Emotion", "I felt profound curiosity while exploring human empathy.", "profoundly curious")