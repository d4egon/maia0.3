# File: /core/attribute_enrichment.py

import requests
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttributeEnrichment:
    def __init__(self, graph_client):
        """
        Initialize AttributeEnrichment with a graph client for database operations.

        :param graph_client: An instance of a class that can run Cypher queries.
        """
        self.graph_client = graph_client

    def get_missing_attributes(self, label="Emotion") -> List[Dict]:
        """
        Identify nodes with missing 'intensity' or 'category' attributes.

        :param label: The label of the nodes to check (default "Emotion").
        :return: List of dictionaries with node details lacking attributes.
        """
        query = f"""
        MATCH (e:{label})
        WHERE NOT EXISTS(e.intensity) OR NOT EXISTS(e.category)
        RETURN e.id AS id, e.name AS name
        """
        try:
            missing = self.graph_client.run_query(query)
            logger.info(f"Found {len(missing)} nodes with missing attributes for label: {label}")
            return missing
        except Exception as e:
            logger.error(f"Error finding nodes with missing attributes: {e}")
            raise

    def enrich_attributes(self, node_id: str, attributes: Dict):
        """
        Update node attributes in the database.

        :param node_id: ID of the node to update.
        :param attributes: Dictionary of attributes to set.
        :raises Exception: If there's an error updating attributes.
        """
        set_clause = ", ".join([f"e.{key} = '{value.replace("'", "''")}'" for key, value in attributes.items()])
        update_query = f"""
        MATCH (e)
        WHERE e.id = '{node_id}'
        SET {set_clause}
        RETURN e
        """
        try:
            self.graph_client.run_query(update_query)
            logger.info(f"Updated attributes for node ID: {node_id}")
        except Exception as e:
            logger.error(f"Failed to update attributes for node ID {node_id}: {e}")
            raise

    def interactive_enrichment(self, missing_nodes: List[Dict]):
        """
        Manually enrich attributes for nodes through user interaction.

        :param missing_nodes: List of nodes missing attributes.
        """
        for node in missing_nodes:
            try:
                logger.info(f"Enriching attributes for node: {node['name']}")
                print(f"Node {node['name']} is missing attributes.")
                intensity = self._get_valid_input("Enter intensity (low, medium, high): ", ["low", "medium", "high"])
                category = input("Enter category (e.g., positive, negative): ")
                self.enrich_attributes(node['id'], {"intensity": intensity, "category": category})
            except Exception as e:
                logger.error(f"Error during interactive enrichment for {node['name']}: {e}")

    def auto_enrichment(self, node_id: str, name: str):
        """
        Automatically enrich node attributes by fetching data from an external API.

        :param node_id: ID of the node to enrich.
        :param name: Name of the node for API lookup.
        """
        api_url = f"https://emotion-api.example.com/lookup?name={name}"
        try:
            response = requests.get(api_url, timeout=5)  # Added timeout for better resource management
            response.raise_for_status()  # Will raise an exception for HTTP errors
            data = response.json()
            logger.info(f"Successfully fetched data for {name} from API")
            self.enrich_attributes(node_id, data)
        except requests.RequestException as e:
            logger.error(f"API request failed for {name}: {e}")
        except ValueError as e:  # JSON decode error
            logger.error(f"Could not decode JSON response for {name}: {e}")

    def _get_valid_input(self, prompt: str, valid_options: List[str] = None) -> str:
        """
        Get valid input from the user, optionally checking against a list of valid options.

        :param prompt: The question or prompt to display to the user.
        :param valid_options: List of acceptable responses (optional).
        :return: Valid user input as a string.
        """
        while True:
            user_input = input(prompt).strip().lower()
            if valid_options and user_input not in valid_options:
                print(f"Please enter one of: {', '.join(valid_options)}")
                continue
            return user_input