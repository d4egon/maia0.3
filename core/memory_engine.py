from typing import Any, List, Dict, Optional
import logging
from core.neo4j_connector import Neo4jConnector
from uuid import uuid4
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MemoryEngine:
    def __init__(self, neo4j_connector: Neo4jConnector):
        """
        Initialize the memory engine with a Neo4j connector and a model for semantic analysis.

        :param neo4j_connector: An instance of Neo4jConnector for database operations.
        """
        self.neo4j = neo4j_connector
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

    def create_memory_node(self, text: str, metadata: Dict[str, Any], keywords: List[str]) -> str:
        """
        Create a high-level memory node with an embedding for semantic search.

        :param text: Textual content of the node.
        :param metadata: Additional metadata.
        :param keywords: Keywords for search.
        :return: ID of the created node.
        """
        node_id = str(uuid4())
        embedding = self.model.encode([text])[0].tolist()  # Convert embedding to list for JSON serialization
        query = """
        CREATE (node:MemoryNode {id: $node_id, text: $text, metadata: $metadata, keywords: $keywords, embedding: $embedding})
        RETURN node.id as node_id
        """
        result = self.neo4j.run_query(query, {
            "node_id": node_id,
            "text": text,
            "metadata": metadata,
            "keywords": keywords,
            "embedding": embedding
        })
        return result[0]["node_id"]

    def create_memory_chunk(self, memory_node_id: str, order: int, chunk_type: str, keywords: List[str]) -> str:
        """
        Create a memory chunk and link it to a memory node.

        :param memory_node_id: ID of the parent memory node.
        :param order: Sequential order of the chunk.
        :param chunk_type: Type of the chunk (e.g., "text", "image").
        :param keywords: Aggregated keywords for the chunk.
        :return: ID of the created chunk.
        """
        chunk_id = str(uuid4())
        query = """
        MATCH (node:MemoryNode {id: $memory_node_id})
        CREATE (chunk:MemoryChunk {id: $chunk_id, order: $order, chunk_type: $chunk_type, keywords: $keywords})
        MERGE (node)-[:HAS_CHUNK]->(chunk)
        RETURN chunk.id as chunk_id
        """
        result = self.neo4j.run_query(query, {
            "chunk_id": chunk_id,
            "memory_node_id": memory_node_id,
            "order": order,
            "chunk_type": chunk_type,
            "keywords": keywords
        })
        return result[0]["chunk_id"]

    def create_memory(self, chunk_id: str, order: int, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a granular memory and link it to a memory chunk.

        :param chunk_id: ID of the parent chunk.
        :param order: Sequential order of the memory.
        :param content: The content of the memory.
        :param metadata: Additional properties specific to the memory.
        :return: ID of the created memory.
        """
        query = """
        MATCH (chunk:MemoryChunk)
        WHERE ID(chunk) = $chunk_id
        CREATE (memory:Memory {order: $order, content: $content, metadata: $metadata})
        CREATE (chunk)-[:HAS_MEMORY]->(memory)
        RETURN ID(memory) AS memory_id
        """
        result = self.neo4j.run_query(query, {
            "chunk_id": chunk_id,
            "order": order,
            "content": content,
            "metadata": metadata or {},
        })
        memory_id = result[0]["memory_id"]
        logger.info(f"Created Memory: Order {order} (ID: {memory_id})")
        return memory_id

    def search_memories(self, keywords: List[str]) -> List[Dict[str, str]]:
        """
        Search for memories using keywords.

        :param keywords: List of keywords to search for.
        :return: List of matching memories with their metadata.
        """
        query = """
        MATCH (memory:MemoryNode)
        WHERE ANY(keyword IN $keywords WHERE keyword IN memory.keywords)
        RETURN memory.id AS id, memory.text AS text, memory.metadata AS metadata
        """
        result = self.neo4j.run_query(query, {"keywords": keywords})
        logger.info(f"Found {len(result)} matching memories for keywords: {keywords}")
        return result

    def search_memory_by_embedding(self, embedding: List[float]) -> Dict:
        """
        Search for the most similar memory based on the given embedding.

        :param embedding: The embedding vector to compare against.
        :return: The memory with the highest similarity or an empty dict if no match found.
        """
        query = """
        MATCH (n:MemoryNode)
        RETURN n.id AS id, n.text AS text, n.metadata AS metadata, n.embedding AS embedding
        """
        memories = self.neo4j.run_query(query)
        best_match = None
        highest_similarity = 0
        for memory in memories:
            memory_embedding = np.array(memory['embedding'])
            similarity = util.cos_sim(embedding, memory_embedding).item()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = memory
        
        if best_match:
            logger.info(f"Best semantic match found with similarity {highest_similarity}")
            return best_match
        return {}

    def retrieve_all_memories(self) -> List[Dict]:
        """
        Retrieve all memories from the database.

        :return: List of all memory nodes.
        """
        query = """
        MATCH (n:MemoryNode)
        RETURN n.id AS id, n.text AS text, n.metadata AS metadata, n.embedding AS embedding
        """
        return self.neo4j.run_query(query)

    def retrieve_memories_by_time(self, period: str) -> List[Dict]:
        """
        Retrieve memories based on a time period.

        :param period: String representing the time period ('recent', 'last_week', 'all_time').
        :return: List of memories within the specified period.
        """
        if period == "recent":
            query = """
            MATCH (n:MemoryNode)
            WHERE datetime(n.metadata.created_at) > datetime() - duration('P1D')
            RETURN n.id AS id, n.text AS text, n.metadata AS metadata, n.embedding AS embedding
            """
        elif period == "last_week":
            query = """
            MATCH (n:MemoryNode)
            WHERE datetime(n.metadata.created_at) > datetime() - duration('P7D')
            RETURN n.id AS id, n.text AS text, n.metadata AS metadata, n.embedding AS embedding
            """
        else:  # all_time
            query = """
            MATCH (n:MemoryNode)
            RETURN n.id AS id, n.text AS text, n.metadata AS metadata, n.embedding AS embedding
            """
        return self.neo4j.run_query(query)

    def retrieve_memories_by_time_range(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Retrieve memories based on a time range.

        :param start_date: Start date in 'YYYY-MM-DD' format.
        :param end_date: End date in 'YYYY-MM-DD' format.
        :return: List of memories within the specified time range.
        """
        query = """
        MATCH (n:MemoryNode)
        WHERE datetime(n.metadata.created_at) >= datetime($start_date) 
          AND datetime(n.metadata.created_at) <= datetime($end_date)
        RETURN n.id AS id, n.text AS text, n.metadata AS metadata, n.embedding AS embedding
        """
        return self.neo4j.run_query(query, {"start_date": start_date, "end_date": end_date})

    def update_memory(self, memory_id: str, updates: Dict) -> None:
        """
        Update a memory node with new information.

        :param memory_id: ID of the memory node to update.
        :param updates: Dictionary of updates to apply to the memory node.
        """
        set_clause = ", ".join([f"n.{key} = ${key}" for key in updates])
        query = f"""
        MATCH (n:MemoryNode {{id: '{memory_id}'}})
        SET {set_clause}
        """
        self.neo4j.run_query(query, updates)
        logger.info(f"Updated memory node {memory_id}")

    def create_relationship(self, from_id: str, to_id: str, relationship_type: str) -> None:
        """
        Create a relationship between two memory nodes.

        :param from_id: ID of the starting node.
        :param to_id: ID of the ending node.
        :param relationship_type: Type of relationship to create.
        """
        query = """
        MATCH (from:MemoryNode {id: $from_id}), (to:MemoryNode {id: $to_id})
        MERGE (from)-[:{relationship_type}]->(to)
        """.format(relationship_type=relationship_type)
        self.neo4j.run_query(query, {"from_id": from_id, "to_id": to_id})
        logger.info(f"Created relationship {relationship_type} from {from_id} to {to_id}")

    def get_error_logs(self) -> List[Dict]:
        """
        Retrieve error logs stored as memories.

        :return: List of error logs where each log is a dictionary.
        """
        query = """
        MATCH (n:MemoryNode)
        WHERE n.metadata.type = 'error'
        RETURN n.id AS id, n.text AS error_message, n.metadata AS metadata
        """
        return self.neo4j.run_query(query)

    def link_memories_sequentially(self, memory_ids: List[str]):
        """
        Link memories sequentially with NEXT relationships.

        :param memory_ids: List of memory node IDs in order.
        """
        validate_query = """
        MATCH (n:MemoryNode)
        WHERE n.id IN $memory_ids
        WITH collect(n.id) as found_ids, $memory_ids as expected_ids
        RETURN 
            size(found_ids) = size(expected_ids) as all_nodes_exist,
            [x IN expected_ids WHERE NOT x IN found_ids] as missing_ids
        """
        result = self.neo4j.run_query(validate_query, {"memory_ids": memory_ids})
        
        if not result[0]["all_nodes_exist"]:
            raise ValueError(f"Missing nodes: {result[0]['missing_ids']}")
            
        link_query = """
        UNWIND range(0, size($ids)-2) as i
        MATCH (m1:MemoryNode {id: $ids[i]}), (m2:MemoryNode {id: $ids[i+1]})
        MERGE (m1)-[:NEXT]->(m2)
        """
        self.neo4j.run_query(link_query, {"ids": memory_ids})
        logger.info(f"Linked {len(memory_ids) - 1} memories sequentially.")