from unittest import result
import pytest
from core.memory_engine import MemoryEngine
from core.neo4j_connector import Neo4jConnector
from uuid import uuid4

# Mocking Neo4jConnector for testing
class MockNeo4jConnector:
    def run_query(self, query, parameters):
        if "CREATE (memory:Memory" in query:
            return [{"memory_id": str(uuid4())}]
        elif "CREATE (chunk:MemoryChunk" in query:
            return [{"chunk_id": str(uuid4())}]
        elif "MATCH (memory:Memory)" in query:
            return [{"content": "Math content", "metadata": {"source": "Math Book"}}]
        return [{"node_id": str(uuid4())}]  # Default for other queries

@pytest.fixture
def memory_engine():
    mock_neo4j = MockNeo4jConnector()
    return MemoryEngine(mock_neo4j)

def test_create_memory_node(memory_engine):
    node_id = memory_engine.create_memory_node("Math 101", {"author": "John Doe"}, ["math", "education"])
    assert isinstance(node_id, str)
    assert len(node_id) > 0  # UUID should be non-empty

def test_create_memory_chunk(memory_engine):
    # Assuming we have a memory node created for this test
    memory_node_id = str(uuid4())
    chunk_id = memory_engine.create_memory_chunk(memory_node_id, 1, "text", ["introduction", "basics"])
    assert isinstance(chunk_id, str)
    assert len(chunk_id) > 0

def test_create_memory(memory_engine):
    # Assuming we have a memory chunk created for this test
    chunk_id = str(uuid4())
    memory_id = memory_engine.create_memory(chunk_id, 1, "Sample content", {"source": "Book A"})
    assert isinstance(memory_id, str)
    assert len(memory_id) > 0

def test_search_memories(memory_engine):
    # This test assumes some memories with keywords exist
    results = memory_engine.search_memories(["math"])
    assert isinstance(results, list)
    # Check if the result contains expected fields
    if results:
        assert all('content' in item and 'metadata' in item for item in results)

def test_link_memories_sequentially(memory_engine):
    # Create some memory node IDs
    memory_ids = [str(uuid4()) for _ in range(3)]
    
    # Mocking that all nodes exist in the database
    memory_engine.neo4j.run_query = lambda query, params: [{"all_nodes_exist": True, "missing_ids": []}]
    
    memory_engine.link_memories_sequentially(memory_ids)
    # Here, you might want to check if the NEXT relationships were created, 
    # but since this is mocked, we'll check if no exception was raised for existing nodes

def test_link_memories_sequentially_missing_nodes(memory_engine):
    # Test for error when linking non-existent nodes
    memory_ids = [str(uuid4()), str(uuid4())]  # Assuming one of these doesn't exist
    
    # Mocking that one node is missing
    memory_engine.neo4j.run_query = lambda query, params: [{"all_nodes_exist": False, "missing_ids": [memory_ids[1]]}]
    
    with pytest.raises(ValueError):
        memory_engine.link_memories_sequentially(memory_ids)

if __name__ == "__main__":
    pytest.main([__file__])
    # Check if the result contains expected fields
    if result:
        assert all('content' in item and 'metadata' in item for item in result)

def test_link_memories_sequentially(memory_engine):
    # Create some memory node IDs
    memory_ids = [str(uuid4()) for _ in range(3)]
    
    # Mocking that all nodes exist in the database
    memory_engine.neo4j.run_query = lambda query, params: [{"all_nodes_exist": True, "missing_ids": []}]
    
    memory_engine.link_memories_sequentially(memory_ids)
    # Here, you might want to check if the NEXT relationships were created, 
    # but since this is mocked, we'll check if no exception was raised for existing nodes

def test_link_memories_sequentially_missing_nodes(memory_engine):
    # Test for error when linking non-existent nodes
    memory_ids = [str(uuid4()), str(uuid4())]  # Assuming one of these doesn't exist
    
    # Mocking that one node is missing
    memory_engine.neo4j.run_query = lambda query, params: [{"all_nodes_exist": False, "missing_ids": [memory_ids[1]]}]
    
    with pytest.raises(ValueError):
        memory_engine.link_memories_sequentially(memory_ids)

if __name__ == "__main__":
    pytest.main([__file__])