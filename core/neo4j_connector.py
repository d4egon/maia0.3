# Location: core folder

import json
from neo4j import GraphDatabase, exceptions
import logging
import time
from typing import Dict, List, Optional

# Configure logging to include timestamp and log level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Neo4jConnector:
    def __init__(self, uri: str, user: str, password: str, max_retries: int = 3, retry_delay: float = 3.0):
        """
        Initialize the Neo4j connector with connection details and retry settings.

        :param uri: Neo4j database URI
        :param user: Database username
        :param password: Database password
        :param max_retries: Maximum number of connection attempts
        :param retry_delay: Delay in seconds between connection attempts
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.driver = self._connect_with_retry()

    def _connect_with_retry(self) -> GraphDatabase.driver:
        """
        Attempt to connect to Neo4j with retry logic for transient failures.

        :return: Neo4j driver instance
        :raises Exception: If connection fails after max_retries
        """
        for attempt in range(self.max_retries):
            try:
                driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                with driver.session() as session:
                    session.run("RETURN 1")
                logger.info("[Neo4j] Connected successfully.")
                return driver
            except exceptions.AuthError:
                logger.error("[AUTH ERROR] Invalid credentials for Neo4j.")
                raise  # Authentication errors are not transient, so we raise immediately
            except (exceptions.ServiceUnavailable, exceptions.TransientError) as e:
                logger.warning(f"[CONNECTION FAILED] Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"[UNEXPECTED ERROR] {str(e)}")
                raise
        raise Exception(f"Failed to connect after {self.max_retries} attempts.")

    def run_query(self, query: str, parameters: Optional[Dict] = None, db: Optional[str] = None) -> List[Dict]:
        """
        Execute a Cypher query with retry logic for transient issues.

        :param query: Cypher query to execute
        :param parameters: Dictionary of parameters for the query
        :param db: Name of the database to use, if not the default
        :return: List of dictionaries with query results
        """
        parameters = parameters or {}
        for attempt in range(self.max_retries):
            try:
                with self.driver.session(database=db) if db else self.driver.session() as session:
                    result = session.run(query, parameters)
                    data = result.data()
                    logger.info(f"[QUERY SUCCESS] Query: {query[:50]}{'...' if len(query) > 50 else ''}, {len(data)} rows returned.")
                    return data
            except (exceptions.TransientError, exceptions.ServiceUnavailable) as e:
                logger.warning(f"[QUERY RETRY] Transient error for query '{query[:50]}{'...' if len(query) > 50 else ''}': {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"[QUERY FAILED] Max retry attempts reached for query: {query[:50]}{'...' if len(query) > 50 else ''}")
                    raise exceptions.ServiceUnavailable("Transaction error")
            except Exception as e:
                    logger.error(f"[QUERY FAILED] Unexpected error for query '{query[:50]}{'...' if len(query) > 50 else ''}': {e}", exc_info=True)
                    raise
        return []  # This line should never be reached if exceptions are handled correctly

    def close(self):
        """
        Close the Neo4j driver connection.
        """
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("[CONNECTION CLOSED] Neo4j driver closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_db_connection(self):
        return self.driver.session()

    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return result.data()

    # New method to ensure consistency with MemoryEngine operations
    def create_content_node(self, content_id: str, content_type: str, title: str, author: str, metadata: Dict) -> str:
        """
        Create a content node in the Neo4j database.

        :param content_id: Unique identifier for the content
        :param content_type: Type of the content (e.g., 'ebook')
        :param title: Title of the content
        :param author: Author of the content
        :param metadata: Metadata associated with the content
        :return: The ID of the created content node
        """
        query = """
        CREATE (content:Content {id: $content_id, type: $content_type, title: $title, author: $author, date_uploaded: datetime(), metadata: $metadata})
        RETURN content.id as content_id
        """
        result = self.run_query(query, {
            "content_id": content_id,
            "content_type": content_type,
            "title": title,
            "author": author,
            "metadata": json.dumps(metadata)
        })
        return result[0]["content_id"]

    def create_chunk_node(self, content_id: str, chunk_id: str, chunk_type: str, chunk_content: str) -> str:
        """
        Create a chunk node and link it to the content node.

        :param content_id: ID of the parent content node
        :param chunk_id: Unique identifier for the chunk
        :param chunk_type: Type of the chunk (e.g., 'keyword', 'chapter')
        :param chunk_content: Content of the chunk
        :return: The ID of the created chunk node
        """
        query = """
        MATCH (content:Content {id: $content_id})
        CREATE (chunk:Chunk {id: $chunk_id, type: $chunk_type, content: $chunk_content})
        CREATE (content)-[:HAS_CHUNK]->(chunk)
        RETURN chunk.id as chunk_id
        """
        result = self.run_query(query, {
            "content_id": content_id,
            "chunk_id": chunk_id,
            "chunk_type": chunk_type,
            "chunk_content": chunk_content
        })
        return result[0]["chunk_id"]

    def create_sentence_node(self, chunk_id: str, sentence_id: str, content: str):
        """
        Create a sentence node and link it to a chunk node.

        :param chunk_id: ID of the parent chunk node
        :param sentence_id: Unique identifier for the sentence
        :param content: Content of the sentence
        """
        query = """
        MATCH (chunk:Chunk {id: $chunk_id})
        CREATE (sentence:Sentence {id: $sentence_id, content: $content})
        CREATE (chunk)-[:HAS_SENTENCE]->(sentence)
        """
        self.run_query(query, {
            "chunk_id": chunk_id,
            "sentence_id": sentence_id,
            "content": content
        })

    def link_to_brain_functions(self, content_id: str):
        """
        Link the content to relevant brain functions for processing.

        :param content_id: ID of the         content node
        """
        query = """
        MATCH (content:Content {id: $content_id}), 
              (cerebral_cortex:Function {name: 'Cerebral Cortex'}), 
              (prefrontal_cortex:Function {name: 'Prefrontal Cortex'}), 
              (hippocampus:Function {name: 'Hippocampus'})
        CREATE (content)-[:PROCESSED_BY]->(cerebral_cortex)
        CREATE (content)-[:PROCESSED_BY]->(prefrontal_cortex)
        CREATE (content)-[:MEMORIZED_BY]->(hippocampus)
        """
        self.run_query(query, {"content_id": content_id})

    def link_to_soul(self, content_id: str):
        """
        Link the brain functions involved in processing to M.A.I.A.'s Soul.

        :param content_id: ID of the content node
        """
        query = """
        MATCH (content:Content {id: $content_id})-[:PROCESSED_BY|:MEMORIZED_BY]->(func:Function),
              (soul:Soul {name: "Soul"})
        MERGE (soul)-[:HAS_FUNCTION]->(func)
        """
        self.run_query(query, {"content_id": content_id})

    def verify_node_exists(self, label: str, node_id: str) -> bool:
        """
        Verify if a node with the given label and ID exists in the database.

        :param label: Label of the node to check
        :param node_id: ID of the node to check
        :return: True if the node exists, False otherwise
        """
        query = f"""
        MATCH (n:{label} {{id: $node_id}})
        RETURN n
        """
        result = self.run_query(query, {"node_id": node_id})
        return len(result) > 0

    def create_relationship(self, from_id: str, to_id: str, relationship_type: str):
        """
        Create a relationship between two nodes.

        :param from_id: ID of the starting node
        :param to_id: ID of the ending node
        :param relationship_type: Type of relationship to create
        """
        query = f"""
        MATCH (from:{relationship_type.split('_')[0]} {{id: $from_id}}), 
              (to:{relationship_type.split('_')[-1]} {{id: $to_id}})
        MERGE (from)-[:{relationship_type}]->(to)
        """
        self.run_query(query, {"from_id": from_id, "to_id": to_id})

    # Adding this method to handle potential errors in memory operations
    def handle_transaction_error(self, error):
        """
        Handle transaction errors by logging them and potentially retrying the operation.

        :param error: Exception object representing the error
        """
        logger.error(f"[TRANSACTION ERROR] {str(error)}", exc_info=True)
        # Here you might implement logic to retry or handle the error differently