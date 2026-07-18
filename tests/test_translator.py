import json

from manga_localizer import translator
from manga_localizer.ocr import PageOCR, TextUnit
from manga_localizer.translator import (
    OllamaTranslator,
    OpenAICompatibleTranslator,
    PromptTranslator,
    available_remote_models,
    pytorch_device,
)


def test_shared_gpu_name_is_normalized_for_pytorch():
    assert pytorch_device("gpu:0") == "cuda:0"
    assert pytorch_device("gpu:2") == "cuda:2"
    assert pytorch_device("cpu") == "cpu"
    assert pytorch_device("auto") == "auto"


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


def test_ollama_uses_native_chat_api(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse({"message": {"content": "[p1-u1] 你好"}})

    monkeypatch.setattr(translator, "urlopen", fake_urlopen)
    result = OllamaTranslator("http://localhost:11434", "qwen-test")._generate("翻译")
    assert result == "[p1-u1] 你好"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["body"]["model"] == "qwen-test"
    assert captured["body"]["stream"] is False


def test_openai_compatible_api_keeps_key_in_header(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.data)
        return FakeResponse({"choices": [{"message": {"content": "[p1-u1] 你好"}}]})

    monkeypatch.setattr(translator, "urlopen", fake_urlopen)
    client = OpenAICompatibleTranslator("https://example.test/v1", "model-a", "secret")
    assert client._generate("翻译") == "[p1-u1] 你好"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret"
    assert captured["body"]["model"] == "model-a"


def test_remote_model_discovery_supports_ollama(monkeypatch):
    monkeypatch.setattr(
        translator,
        "urlopen",
        lambda _request, timeout: FakeResponse({"models": [{"name": "qwen:7b"}]}),
    )
    assert available_remote_models("ollama", "http://localhost:11434") == ["qwen:7b"]


def test_chunk_translation_reports_incremental_page_progress():
    class EchoTranslator(PromptTranslator):
        def _generate(self, prompt, max_new_tokens=1600):
            del max_new_tokens
            ids = translator.re.findall(r"\[(p\d+u\d+)\]", prompt)
            return "\n".join(f"[{unit_id}] 译文" for unit_id in ids)

    pages = [
        PageOCR(
            page,
            f"{page}.png",
            100,
            100,
            [TextUnit(f"p{page:03d}u01", [0, 0, 20, 20], [0, 0, 20, 20], "日本", 1.0)],
        )
        for page in range(1, 6)
    ]
    updates = []
    EchoTranslator().translate_pages(
        pages, chunk_pages=2, progress=lambda current, total: updates.append((current, total))
    )
    assert updates == [(2, 5), (4, 5), (5, 5)]


def test_translation_quality_gate_rejects_kana_and_context_leakage():
    unit = TextUnit("p001u01", [0, 0, 20, 20], [0, 0, 20, 20], "ガク君", 1.0)
    assert PromptTranslator._valid_translation(unit, "岳君") is True
    assert PromptTranslator._valid_translation(unit, "ガク君") is False
    assert PromptTranslator._valid_translation(unit, "第1页：这是被错误复述的一整段上下文") is False


def test_translation_quality_gate_allows_natural_chinese_expansion():
    unit = TextUnit("p001u01", [0, 0, 20, 20], [0, 0, 20, 20], "みんな引っ越した", 1.0)
    assert PromptTranslator._valid_translation(
        unit, "大家不是搬走了，就是不再来了。"
    ) is True
