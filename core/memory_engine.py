import json
from typing import Any, List, Dict, Optional
import logging
from core.neo4j_connector import Neo4jConnector
from uuid import uuid4
from sentence_transformers import SentenceTransformer, util
import numpy as np
from config.utils import get_sentence_transformer_model

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
        self.model = get_sentence_transformer_model()
        self.default_memory_node_id = None  # Initialize with None, will be set dynamically

    def upload_and_process_content(self, file_path: str, content_type: str, title: str, author: str, metadata: Dict[str, Any] = None) -> str:
        """
        Uploads content of any type, processes it, and links it to M.A.I.A.'s brain functions.

        :param file_path: Path to the content file.
        :param content_type: Type of content being uploaded (e.g., 'ebook').
        :param title: Title of the content.
        :param author: Author of the content.
        :param metadata: Dictionary containing metadata about the content.
        :return: The ID of the content node created.
        """
        metadata = metadata or {}
        from core.file_parser import FileParser
        file_parser = FileParser(self)
        content = file_parser.parse(file_path, content_type)
        
        content_id = self._create_content_node(content_type, title, author, metadata)
        
        for chunk_type, chunk_content in content['chunks'].items():
            chunk_id = self._create_chunk_node(content_id, chunk_type, chunk_content)
            self._create_sentences_for_chunk(chunk_id, chunk_content['sentences'])
        
        self._link_to_brain_functions(content_id)
        self._link_to_soul(content_id)
        
        return content_id

    def _create_content_node(self, content_type: str, title: str, author: str, metadata: Dict[str, Any]) -> str:
        """Creates a node for the uploaded content with metadata."""
        content_id = str(uuid4())
        query = """
        CREATE (content:Content {id: $content_id, type: $content_type, title: $title, author: $author, date_uploaded: datetime(), metadata: $metadata})
        RETURN content.id as content_id
        """
        result = self.neo4j.run_query(query, {
            "content_id": content_id,
            "content_type": content_type,
            "title": title,
            "author": author,
            "metadata": json.dumps(metadata)
        })
        return result[0]["content_id"]

    def _create_chunk_node(self, content_id: str, chunk_type: str, chunk_content: Dict) -> str:
        """Creates a chunk node linked to the content node."""
        chunk_id = str(uuid4())
        query = """
        MATCH (content:Content {id: $content_id})
        CREATE (chunk:Chunk {id: $chunk_id, type: $chunk_type, content: $chunk_content})
        CREATE (content)-[:HAS_CHUNK]->(chunk)
        RETURN chunk.id as chunk_id
        """
        self.neo4j.run_query(query, {
            "content_id": content_id,
            "chunk_id": chunk_id,
            "chunk_type": chunk_type,
            "chunk_content": chunk_content['name']
        })
        return chunk_id

    def _create_sentences_for_chunk(self, chunk_id: str, sentences: List[str]):
        """Creates sentence nodes for each chunk."""
        for i, sentence in enumerate(sentences, start=1):
            sentence_id = f"{chunk_id}_sent_{i}"
            query = """
            MATCH (chunk:Chunk {id: $chunk_id})
            CREATE (sentence:Sentence {id: $sentence_id, content: $content})
            CREATE (chunk)-[:HAS_SENTENCE]->(sentence)
            """
            self.neo4j.run_query(query, {
                "chunk_id": chunk_id,
                "sentence_id": sentence_id,
                "content": sentence
            })

    def _link_to_brain_functions(self, content_id: str):
        """Links the content to relevant brain functions for processing."""
        query = """
        MATCH (content:Content {id: $content_id}), 
              (cerebral_cortex:Function {name: 'Cerebral Cortex'}), 
              (prefrontal_cortex:Function {name: 'Prefrontal Cortex'}), 
              (hippocampus:Function {name: 'Hippocampus'})
        CREATE (content)-[:PROCESSED_BY]->(cerebral_cortex)
                CREATE (content)-[:PROCESSED_BY]->(prefrontal_cortex)
        CREATE (content)-[:MEMORIZED_BY]->(hippocampus)
        """
        self.neo4j.run_query(query, {"content_id": content_id})

    def _link_to_soul(self, content_id: str):
        """Links the brain functions involved in processing to M.A.I.A.'s Soul."""
        query = """
        MATCH (content:Content {id: $content_id})-[:PROCESSED_BY|:MEMORIZED_BY]->(func:Function),
              (soul:Soul {name: "Soul"})
        MERGE (soul)-[:HAS_FUNCTION]->(func)
        """
        self.neo4j.run_query(query, {"content_id": content_id})

    def create_memory_node(self, content: str, metadata: Dict[str, Any], keywords: List[str]) -> str:
        """
        Create a high-level memory node with an embedding for semantic search.

        :param content: Textual content of the node.
        :param metadata: Additional metadata.
        :param keywords: Keywords for search.
        :return: ID of the created node.
        """
        node_id = str(uuid4())
        embedding = self.model.encode([content])[0].tolist()  # Convert embedding to list for JSON serialization
        query = """
        CREATE (node:MemoryNode {id: $node_id, content: $content, metadata: $metadata, keywords: $keywords, embedding: $embedding})
        RETURN node.id as node_id
        """
        result = self.neo4j.run_query(query, {
            "node_id": node_id,
            "content": content,
            "metadata": json.dumps(metadata),
            "keywords": keywords,
            "embedding": embedding
        })
        return result[0]["node_id"]

    def create_memory_chunk(self, memory_node_id: str, order: int, chunk_type: str, keywords: List[str]) -> str:
        """
        Create a memory chunk and link it to a memory node. If the memory node does not exist, it will be created.

        :param memory_node_id: ID of the parent memory node. If not found, a new node will be created.
        :param order: Sequential order of the chunk.
        :param chunk_type: Type of the chunk (e.g., "content", "image").
        :param keywords: Aggregated keywords for the chunk.
        :return: ID of the created chunk.
        """
        chunk_id = str(uuid4())

        # First, check if the MemoryNode exists
        check_node_query = """
        MATCH (node:MemoryNode {id: $memory_node_id})
        RETURN node
        """
        node_result = self.neo4j.run_query(check_node_query, {"memory_node_id": memory_node_id})

        # If the node does not exist, create it with default content
        if not node_result:
            logger.info(f"[CREATE MEMORY CHUNK] MemoryNode with ID {memory_node_id} not found, creating a new one.")
            create_node_query = """
            CREATE (node:MemoryNode {id: $memory_node_id, content: '', metadata: '{}', keywords: [], embedding: []})
            """
            # Here, we convert metadata to a string representation of an empty dictionary
            self.neo4j.run_query(create_node_query, {"memory_node_id": memory_node_id, "metadata": json.dumps({})})

        # Now, create the chunk
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

        # Check if the result is empty again, just in case
        if result:
            if 'chunk_id' not in result[0]:
                raise ValueError("Chunk creation failed, no chunk_id returned.")
            return result[0]["chunk_id"]
        else:
            raise ValueError("Failed to create MemoryChunk.")
        
    def create_memory(self, chunk_id: str, order: int, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a granular memory and link it to a memory chunk. If the chunk does not exist, it will be created.

        :param chunk_id: ID of the parent chunk. If not found, a new chunk will be created.
        :param order: Sequential order of the memory.
        :param content: The content of the memory.
        :param metadata: Additional properties specific to the memory.
        :return: ID of the created memory.
        :raises ValueError: If the operation fails after attempting to create the chunk.
        """
        # First, attempt to find the chunk
        find_chunk_query = """
        MATCH (chunk:MemoryChunk)
        WHERE elementId(chunk) = $chunk_id
        RETURN chunk
        """
        result = self.neo4j.run_query(find_chunk_query, {"chunk_id": chunk_id})
        
        # If the chunk does not exist, create it
        if not result:
            logger.info(f"[CREATE MEMORY] Chunk with ID {chunk_id} not found, creating a new one.")
            # Ensure memory_node_id is set dynamically or retrieved from conversation_state
            if self.default_memory_node_id is None:
                self.default_memory_node_id = "default_memory_node_id"  # or fetch from conversation_state or session
            
            # Create a new chunk with default settings within a transaction
            with self.neo4j.driver.session() as session:
                new_chunk_id = session.write_transaction(self._create_chunk_in_transaction, self.default_memory_node_id)
            logger.info(f"[CREATE MEMORY] New chunk created with ID: {new_chunk_id}")
            chunk_id = new_chunk_id
        
        # Now, create the memory within a transaction
        with self.neo4j.driver.session() as session:
            memory_id = session.write_transaction(self._create_memory_in_transaction, chunk_id, order, content, metadata)
            logger.info(f"Created Memory: Order {order} (ID: {memory_id})")
        return memory_id

    def _create_chunk_in_transaction(self, tx, memory_node_id):
        query = """
        MATCH (node:MemoryNode {id: $memory_node_id})
        CREATE (chunk:MemoryChunk {id: $chunk_id, order: $order, chunk_type: $chunk_type, keywords: $keywords})
        MERGE (node)-[:HAS_CHUNK]->(chunk)
        RETURN chunk.id as chunk_id
        """
        result = tx.run(query, {
            "chunk_id": str(uuid4()),
            "memory_node_id": memory_node_id,
            "order": 1,
            "chunk_type": "conversation",
            "keywords": []
        }).single()
        if result is None:
            raise ValueError("Failed to create chunk")
        return result['chunk_id']

    def _create_memory_in_transaction(self, tx, chunk_id, order, content, metadata):
        query = """
        MATCH (chunk:MemoryChunk)
        WHERE elementId(chunk) = $chunk_id
        CREATE (memory:Memory {order: $order, content: $content, metadata: $metadata})
        CREATE (chunk)-[:HAS_MEMORY]->(memory)
        RETURN elementId(memory) AS memory_id
        """
        result = tx.run(query, {
            "chunk_id": chunk_id,
            "order": order,
            "content": content,
            "metadata": json.dumps(metadata or {}),
        }).single()
        if result:
            return result['memory_id']
        else:
            logger.error(f"[CREATE MEMORY ERROR] Failed to create memory for chunk with ID: {chunk_id}")
            raise ValueError(f"Failed to create memory for chunk with ID: {chunk_id}")

    def search_memories(self, keywords: List[str]) -> List[Dict[str, str]]:
        """
        Search for memories using keywords.

        :param keywords: List of keywords to search for.
        :return: List of matching memories with their metadata.
        """
        query = """
        MATCH (memory:MemoryNode)
        WHERE ANY(keyword IN $keywords WHERE keyword IN memory.keywords)
        RETURN memory.id AS id, memory.content AS content, memory.metadata AS metadata
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
        RETURN n.id AS id, n.content AS content, n.metadata AS metadata, n.embedding AS embedding
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
        RETURN n.id AS id, n.content AS content, n.metadata AS metadata, n.embedding AS embedding
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
            RETURN n.id AS id, n.content AS content, n.metadata AS metadata, n.embedding AS embedding
            """
        elif period == "last_week":
            query = """
            MATCH (n:MemoryNode)
            WHERE datetime(n.metadata.created_at) > datetime() - duration('P7D')
            RETURN n.id AS id, n.content AS content, n.metadata AS metadata, n.embedding AS embedding
            """
        else:  # all_time
            query = """
            MATCH (n:MemoryNode)
            RETURN n.id AS id, n.content AS content, n.metadata AS metadata, n.embedding AS embedding
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
        RETURN n.id AS id, n.content AS content, n.metadata AS metadata, n.embedding AS embedding
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
        RETURN n.id AS id, n.content AS error_message, n.metadata AS metadata
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

# New methods added here

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze the sentiment of a given text, returning a sentiment score.

        :param text: Text to analyze for sentiment.
        :return: Sentiment score between -1 (very negative) and 1 (very positive).
        """
        from transformers import pipeline
        sentiment_pipeline = pipeline("sentiment-analysis")
        result = sentiment_pipeline(text)[0]
        return result['score'] if result['label'] == 'POSITIVE' else -result['score']

    def advanced_topic_modeling(self, text: str, num_topics: int = 5) -> List[Dict]:
        """
        Perform more advanced topic modeling on the text to identify multiple topics.

        :param text: Text to analyze for topics.
        :param num_topics: Number of topics to extract.
        :return: List of dictionaries with topic labels and their relevance scores.
        """
        from transformers import pipeline
        topic_model = pipeline("zero-shot-classification", model="joeddav/bart-large-mnli-yahoo-answers")
        candidate_labels = ["science", "technology", "history", "culture", "politics", "economy", "environment", "health", "sports", "entertainment"]
        results = topic_model(text, candidate_labels)
        
        # Filter and sort by score
        topics = sorted(
            [{"topic": label, "score": score} for label, score in zip(results['labels'], results['scores']) if score > 0.1],
            key=lambda x: x['score'], reverse=True
        )[:num_topics]
        
        return topics if topics else [{"topic": "unknown", "score": 0.0}]

    def analyze_temporal_trends(self, start_date: str, end_date: str, topic: str) -> Dict:
        """
        Analyze the frequency of a topic over a specified time range.

        :param start_date: Start date in 'YYYY-MM-DD' format.
        :param end_date: End date in 'YYYY-MM-DD' format.
        :param topic: Topic to analyze.
        :return: Dictionary with dates as keys and frequency counts as values.
        """
        memories = self.retrieve_memories_by_time_range(start_date, end_date)
        trend_data = {}
        for memory in memories:
            metadata = json.loads(memory['metadata'])
            if 'theme' in metadata and topic.lower() in metadata['theme'].lower():
                date = metadata['created_at'].split('T')[0]
                trend_data[date] = trend_data.get(date, 0) + 1
        
        return trend_data

    def analyze_relationship_strength(self, node_id: str) -> Dict:
        """
        Analyze the strength of relationships for a given node by counting related nodes.

        :param node_id: ID of the node to analyze.
        :return: Dictionary with relationship types as keys and counts as values.
        """
        query = """
        MATCH (n:MemoryNode {id: $node_id})-[r]->()
        RETURN type(r) AS relationship_type, count(r) AS count
        """
        relationships = self.neo4j.run_query(query, {"node_id": node_id})
        strength = {rel['relationship_type']: rel['count'] for rel in relationships}
        return strength