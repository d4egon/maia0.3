# File: /core/memory_engine.py

from typing import List, Dict, Optional
import logging
from core.neo4j_connector import Neo4jConnector
from uuid import uuid4

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MemoryEngine:
    def __init__(self, neo4j_connector: Neo4jConnector):
        """
        Initialize the memory engine.

        :param neo4j_connector: An instance of Neo4jConnector for database operations.
        """
        self.neo4j = neo4j_connector

    def create_memory_node(self, name: str, metadata: Dict[str, str], keywords: List[str]) -> str:
        """
        Create a high-level memory node.

        :param name: Name of the memory node (e.g., "Algebra Basics").
        :param metadata: Additional metadata (e.g., author, source).
        :param keywords: Keywords for profiling and search.
        :return: ID of the created node.
        """
        node_id = str(uuid4())
        query = """
        CREATE (node:MemoryNode {id: $node_id, name: $name, metadata: $metadata, keywords: $keywords})
        RETURN node.id as node_id
        """
        result = self.neo4j.run_query(query, {
            "node_id": node_id,
            "name": name,
            "metadata": metadata,
            "keywords": keywords
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

    def create_memory(self, chunk_id: str, order: int, content: str, metadata: Optional[Dict[str, str]] = None) -> str:
        """
        Create a granular memory and link it to a memory chunk.

        :param chunk_id: ID of the parent chunk.
        :param order: Sequential order of the memory.
        :param content: The content of the memory (e.g., sentence, image binary data).
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
        MATCH (memory:Memory)
        WHERE ANY(keyword IN $keywords WHERE keyword IN memory.keywords)
        RETURN memory.content AS content, memory.metadata AS metadata
        """
        result = self.neo4j.run_query(query, {"keywords": keywords})
        logger.info(f"Found {len(result)} matching memories for keywords: {keywords}")
        return result

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

# Usage Example
# memory_engine = MemoryEngine(neo4j_connector)
# memory_node_id = memory_engine.create_memory_node("Sample Book", {"author": "Author Name"}, ["education", "learning"])
# chunk_id = memory_engine.create_memory_chunk(memory_node_id, 1, "text", ["chapter 1", "introduction"])
# memory_id = memory_engine.create_memory(chunk_id, 1, "This is the first sentence.")
