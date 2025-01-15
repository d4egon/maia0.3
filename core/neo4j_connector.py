# Filename: neo4j_connector.py
# Location: core folder

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
