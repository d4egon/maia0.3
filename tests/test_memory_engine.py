import os
import pytest
from unittest.mock import MagicMock, patch
from core.memory_engine import MemoryEngine
from core.neo4j_connector import Neo4jConnector

@pytest.fixture
def neo4j_connector():
    connector = MagicMock(spec=Neo4jConnector)
    connector.driver = MagicMock()
    return connector

@pytest.fixture
def memory_engine(neo4j_connector: MagicMock):
    return MemoryEngine(neo4j_connector)

def test_create_memory_node(memory_engine: MemoryEngine, neo4j_connector: MagicMock):
    content = "Test content"
    metadata = {"author": "test"}
    keywords = ["test", "content"]
    expected_node_id = "1234"
    
    neo4j_connector.run_query.return_value = [{"node_id": expected_node_id}]
    
    node_id = memory_engine.create_memory_node(content, metadata, keywords)
    
    assert node_id == expected_node_id
    neo4j_connector.run_query.assert_called_once()

def test_create_memory_chunk(memory_engine: MemoryEngine, neo4j_connector: MagicMock):
    memory_node_id = "1234"
    order = 1
    chunk_type = "content"
    keywords = ["test"]
    expected_chunk_id = "5678"
    
    neo4j_connector.run_query.side_effect = [
        [],  # Memory node does not exist
        None,  # Create memory node
        [{"chunk_id": expected_chunk_id}]  # Create memory chunk
    ]
    
    chunk_id = memory_engine.create_memory_chunk(memory_node_id, order, chunk_type, keywords)
    
    assert chunk_id == expected_chunk_id
    neo4j_connector.run_query.assert_called()

def test_create_memory(memory_engine: MemoryEngine, neo4j_connector: MagicMock):
    chunk_id = "5678"
    order = 1
    content = "Test memory content"
    metadata = {"type": "test"}
    expected_memory_id = "91011"

    neo4j_connector.run_query.side_effect = [
        [],  # Memory chunk does not exist
        None,  # Create memory chunk
        [{"memory_id": expected_memory_id}]  # Create memory
    ]

    memory_id = memory_engine.create_memory(chunk_id, order, content, metadata)

    assert memory_id == expected_memory_id
    neo4j_connector.run_query.assert_called()

def test_search_memories(memory_engine: MemoryEngine, neo4j_connector: MagicMock):
    keywords = ["test"]
    expected_result = [{"id": "1234", "content": "Test content", "metadata": {"author": "test"}}]
    
    neo4j_connector.run_query.return_value = expected_result
    
    result = memory_engine.search_memories(keywords)
    
    assert result == expected_result
    neo4j_connector.run_query.assert_called_once()

def test_search_memory_by_embedding(memory_engine: MemoryEngine, neo4j_connector: MagicMock):
    embedding = [0.1, 0.2, 0.3]
    expected_memory = {"id": "1234", "content": "Test content", "metadata": {"author": "test"}, "embedding": [0.1, 0.2, 0.3]}

    neo4j_connector.run_query.return_value = [expected_memory]

    result = memory_engine.search_memory_by_embedding(embedding)

    assert result == expected_memory
    neo4j_connector.run_query.assert_called_once()

def test_retrieve_all_memories(memory_engine: MemoryEngine, neo4j_connector: MagicMock):
    expected_result = [{"id": "1234", "content": "Test content", "metadata": {"author": "test"}, "embedding": [0.1, 0.2, 0.3]}]
    
    neo4j_connector.run_query.return_value = expected_result
    
    result = memory_engine.retrieve_all_memories()
    
    assert result == expected_result
    neo4j_connector.run_query.assert_called_once()

def test_retrieve_memories_by_time(memory_engine: MemoryEngine, neo4j_connector: MagicMock):
    period = "recent"
    expected_result = [{"id": "1234", "content": "Test content", "metadata": {"author": "test"}, "embedding": [0.1, 0.2, 0.3]}]
    
    neo4j_connector.run_query.return_value = expected_result
    
    result = memory_engine.retrieve_memories_by_time(period)
    
    assert result == expected_result
    neo4j_connector.run_query.assert_called_once()

def test_retrieve_memories_by_time_range(memory_engine: MemoryEngine, neo4j_connector: MagicMock):
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    expected_result = [{"id": "1234", "content": "Test content", "metadata": {"author": "test"}, "embedding": [0.1, 0.2, 0.3]}]
    
    neo4j_connector.run_query.return_value = expected_result
    
    result = memory_engine.retrieve_memories_by_time_range(start_date, end_date)
    
    assert result == expected_result
    neo4j_connector.run_query.assert_called_once()

def test_update_memory(memory_engine: MemoryEngine, neo4j_connector: MagicMock):
    memory_id = "1234"
    updates = {"content": "Updated content"}
    
    memory_engine.update_memory(memory_id, updates)
    
    neo4j_connector.run_query.assert_called_once()

def test_create_relationship(memory_engine: MemoryEngine, neo4j_connector: MagicMock):
    from_id = "1234"
    to_id = "5678"
    relationship_type = "RELATED_TO"

    memory_engine.create_relationship(from_id, to_id, relationship_type)

    neo4j_connector.run_query.assert_called_once_with(
        """
        MATCH (from:MemoryNode {id: $from_id}), (to:MemoryNode {id: $to_id})
        MERGE (from)-[r:RELATED_TO]->(to)
        """,
        {"from_id": from_id, "to_id": to_id}
    )

def test_get_error_logs(memory_engine: MemoryEngine, neo4j_connector: MagicMock):
    expected_result = [{"id": "1234", "error_message": "Test error", "metadata": {"type": "error"}}]
    
    neo4j_connector.run_query.return_value = expected_result
    
    result = memory_engine.get_error_logs()
    
    assert result == expected_result
    neo4j_connector.run_query.assert_called_once()

def test_link_memories_sequentially(memory_engine: MemoryEngine, neo4j_connector: MagicMock):
    memory_ids = ["1234", "5678", "91011"]
    
    neo4j_connector.run_query.side_effect = [
        [{"all_nodes_exist": True, "missing_ids": []}],  # Validate nodes
        None  # Link nodes
    ]
    
    memory_engine.link_memories_sequentially(memory_ids)
    
    assert neo4j_connector.run_query.call_count == 2