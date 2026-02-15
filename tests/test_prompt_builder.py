import sys
import types
import importlib


def test_img_prompt_builder_formats_context_and_images(monkeypatch):
    messages_mod = types.ModuleType("langchain_core.messages")

    class _Message:
        def __init__(self, content):
            self.content = content

    messages_mod.HumanMessage = _Message
    messages_mod.SystemMessage = _Message

    langchain_core_mod = types.ModuleType("langchain_core")
    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core_mod)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages_mod)

    pipeline = importlib.import_module("src.llm_output.pipeline")
    LLMOutputGenerator = pipeline.LLMOutputGenerator

    data = {
        "context": {
            "texts": ["A" * 5000],
            "images": [{"path": "img/a.png", "summary": "chart"}],
        },
        "question": "What is shown?",
    }

    messages = LLMOutputGenerator.img_prompt_func(data, max_context_chars=200)
    text = "\n".join(getattr(m, "content", str(m)) for m in messages)

    assert "Images:" in text
    assert "path=img/a.png" in text
    assert "[TRUNCATED CONTEXT]" in text
    assert "Question:" in text
