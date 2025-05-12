
from pydantic import BaseModel

from skllm.llm.gpt.clients.openai.completion import get_parsed_completion

# Add __test__ = False to prevent pytest collection
class TestEvent(BaseModel):
    __test__ = False
    event_name: str
    date: str
    attendees: list[str]

def test_openai_structured_output():
    """Test that structured outputs are properly parsed into Pydantic models."""
    messages = [
        {"role": "system", "content": "Extract event information in JSON format"},
        {"role": "user", "content": "Alice and Bob are attending the science fair on Friday"}
    ]
    
    # Test successful parsing
    result = get_parsed_completion(
        messages=messages,
        output_model=TestEvent,
        key="dummy_value",  # Replace with actual key
        org="dummy_value",   # Replace with actual org
        model="gpt-4o-mini"
    )
    
    # Validate the result structure
    assert isinstance(result, TestEvent)
    assert isinstance(result.event_name, str)
    assert len(result.event_name) > 0
    assert isinstance(result.date, str)
    assert len(result.date) > 0
    assert isinstance(result.attendees, list)
    assert len(result.attendees) >= 2  # Should have at least Alice and Bob
    assert all(isinstance(name, str) for name in result.attendees)

