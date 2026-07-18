import json

from manga_localizer import translator
from manga_localizer.ocr import PageOCR, TextUnit
from manga_localizer.translator import (
    OllamaTranslator,
    OpenAICompatibleTranslator,
    PromptTranslator,
    available_remote_models,
    interjection_fallback,
    kana_bad_words_ids,
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
    assert PromptTranslator._valid_translation(unit, "郭君啊 / 这样不行 / 不要啊") is False
    assert PromptTranslator._valid_translation(unit, "呀ń咻") is False

    latin_source = TextUnit(
        "p001u02", [0, 0, 20, 20], [0, 0, 20, 20], "Ｗ〒ッ", 1.0
    )
    assert PromptTranslator._valid_translation(latin_source, "W〒！") is True


def test_translation_quality_gate_allows_natural_chinese_expansion():
    unit = TextUnit("p001u01", [0, 0, 20, 20], [0, 0, 20, 20], "みんな引っ越した", 1.0)
    assert PromptTranslator._valid_translation(
        unit, "大家不是搬走了，就是不再来了。"
    ) is True


def test_short_interjection_cannot_absorb_another_dialogue_line():
    unit = TextUnit("p001u01", [0, 0, 20, 20], [0, 0, 20, 20], "にゃ", 1.0)
    assert PromptTranslator._valid_translation(
        unit, "格克君别动怎么了很疼吗身体发麻但是"
    ) is False


def test_reviewed_valid_translation_is_not_sent_through_model_again():
    valid = TextUnit(
        "p001u01", [0, 0, 20, 20], [0, 0, 20, 20], "秘密", 1.0, zh="秘密"
    )
    invalid = TextUnit(
        "p001u02", [20, 0, 40, 20], [20, 0, 40, 20], "ニュプ", 1.0, zh="ニュプ"
    )
    assert PromptTranslator._needs_translation(valid, preserve_sfx=False) is False
    assert PromptTranslator._needs_translation(invalid, preserve_sfx=False) is True


def test_invalid_kana_output_gets_a_distinct_repair_pass():
    class RepairingTranslator(PromptTranslator):
        def __init__(self):
            super().__init__()
            self.responses = iter(["ガク君", "还是ガク君"])

        def _generate(self, prompt, max_new_tokens=1600):
            del prompt, max_new_tokens
            return next(self.responses)

        def _generate_repair(self, prompt, max_new_tokens):
            del prompt, max_new_tokens
            return "岳君"

    unit = TextUnit("p001u01", [0, 0, 20, 20], [0, 0, 20, 20], "ガク君", 1.0)
    result = RepairingTranslator()._translate_one(unit, [], "无")
    assert result == "岳君"
    assert unit.translation_attempts == ["ガク君", "还是ガク君", "岳君"]


def test_kana_generation_constraints_cover_vocab_tokens_and_encoded_characters():
    class FakeTokenizer:
        def get_vocab(self):
            return {"你": 1, "ガク": 2, "byte-piece": 3}

        def decode(self, token_ids, skip_special_tokens=True):
            del skip_special_tokens
            return {1: "你", 2: "ガク", 3: ""}[token_ids[0]]

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [3, ord(text)]

    blocked = kana_bad_words_ids(FakeTokenizer())
    assert [2] in blocked
    assert [3, ord("ン")] in blocked
    assert [1] not in blocked


def test_recurring_names_become_a_glossary_and_inconsistent_candidates_are_repaired():
    class NameAwareTranslator(PromptTranslator):
        def _generate(self, prompt, max_new_tokens=1600):
            del max_new_tokens
            ids = translator.re.findall(r"\[(p\d+u\d+)\]", prompt)
            if ids:
                return "\n".join(f"[{unit_id}] 小刚君" for unit_id in ids)
            return "加克君"

        def _generate_repair(self, prompt, max_new_tokens):
            del max_new_tokens
            return "加克" if "人物名字" in prompt else "加克君"

    pages = [
        PageOCR(
            1,
            "1.png",
            100,
            100,
            [TextUnit("p001u01", [0, 0, 20, 20], [0, 0, 20, 20], "ガク君？", 1.0)],
        ),
        PageOCR(
            2,
            "2.png",
            100,
            100,
            [
                TextUnit(
                    "p002u01",
                    [0, 0, 20, 20],
                    [0, 0, 20, 20],
                    "ガク．．．君！",
                    1.0,
                )
            ],
        ),
    ]
    service = NameAwareTranslator()
    service.translate_pages(pages, preserve_sfx=False)
    assert service.resolved_glossary == {"ガク": "加克"}
    assert [page.units[0].zh for page in pages] == ["加克君", "加克君"]
    assert all(page.units[0].translation_attempts[0] == "小刚君" for page in pages)


def test_inferred_name_is_rejected_when_it_leaks_into_an_unrelated_line():
    service = PromptTranslator()
    service.inferred_glossary = {"ガク": "格克"}
    unit = TextUnit("p001u01", [0, 0, 20, 20], [0, 0, 20, 20], "勇気", 1.0)
    assert service._respects_glossary(unit, "格克勇气", {"ガク": "格克"}) is False


def test_same_batch_candidate_for_different_sources_is_retranslated_individually():
    class DuplicateRepairTranslator(PromptTranslator):
        def _generate(self, prompt, max_new_tokens=1600):
            del max_new_tokens
            ids = translator.re.findall(r"\[(p\d+u\d+)\]", prompt)
            if ids:
                return "\n".join(f"[{unit_id}] 请稍等一下" for unit_id in ids)
            return "啊！" if "【原文】あ！" in prompt else "等等！"

    page = PageOCR(
        1,
        "1.png",
        100,
        100,
        [
            TextUnit("p001u01", [0, 0, 20, 20], [0, 0, 20, 20], "あ！", 1.0),
            TextUnit("p001u02", [20, 0, 40, 20], [20, 0, 40, 20], "待って！", 1.0),
        ],
    )
    DuplicateRepairTranslator().translate_pages([page], preserve_sfx=False)
    assert [unit.zh for unit in page.units] == ["啊！", "等等！"]


def test_individual_retry_only_receives_glossary_terms_present_in_its_source():
    class CapturingTranslator(PromptTranslator):
        def __init__(self):
            super().__init__()
            self.prompt = ""

        def _generate(self, prompt, max_new_tokens=1600):
            del max_new_tokens
            self.prompt = prompt
            return "勇气"

    service = CapturingTranslator()
    unit = TextUnit("p001u01", [0, 0, 20, 20], [0, 0, 20, 20], "勇気", 1.0)
    result = service._translate_one(
        unit, [], "ガク译为格克", {"ガク": "格克"}
    )
    assert result == "勇气"
    assert "固定译名：无" in service.prompt
    assert "格克" not in service.prompt


def test_failed_short_vocalization_has_a_local_target_script_fallback():
    assert interjection_fallback("．ッ♡♥ゃンっ♡") == "♡♥呀嗯♡"
    assert interjection_fallback("サナッマンコ締めすぎっ") == ""
