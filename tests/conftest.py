# tests/conftest.py
import pytest
from types import SimpleNamespace
import skllm.llm.gpt.clients.openai.completion as completion_mod
from test_structured_outputs import TestEvent  

class DummyCompletions:
    def __init__(self, model_cls):
        self._model_cls = model_cls

    def parse(self, *, model, messages, response_format, temperature):
        # response_format is the Pydantic model class (TestEvent)
        fake = self._model_cls(
            event_name="science fair",
            date="Friday",
            attendees=["Alice", "Bob"],
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=fake))]
        )

    def create(self, *, temperature, messages, **kwargs):
        # if you ever test get_chat_completion
        return {"id": "dummy", "choices": []}

class DummyClient:
    def __init__(self, model_cls):
        self.chat = SimpleNamespace(completions=DummyCompletions(model_cls))
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=DummyCompletions(model_cls)))


@pytest.fixture(autouse=True)
def patch_openai(monkeypatch):
    monkeypatch.setattr(
        completion_mod,
        "set_credentials",
        lambda key, org: DummyClient(TestEvent),
    )
    monkeypatch.setattr(
        completion_mod,
        "set_azure_credentials",
        lambda key, org: DummyClient(TestEvent),
    )
