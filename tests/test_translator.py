import json

import pytest

from manga_localizer import translator
from manga_localizer.name_dictionary import NameCandidate
from manga_localizer.ocr import PageOCR, TextUnit
from manga_localizer.translator import (
    HyMTTranslator,
    LocalQualityTranslator,
    OllamaTranslator,
    OpenAICompatibleTranslator,
    PromptTranslator,
    calm_preference_fallback,
    available_remote_models,
    ensure_ollama_model,
    hy_mt_generation_options,
    interjection_fallback,
    kana_bad_words_ids,
    pytorch_device,
    validate_local_model_size,
)


def test_shared_gpu_name_is_normalized_for_pytorch():
    assert pytorch_device("gpu:0") == "cuda:0"
    assert pytorch_device("gpu:2") == "cuda:2"
    assert pytorch_device("cpu") == "cpu"


def test_local_model_size_rejects_explicit_models_above_9b():
    validate_local_model_size("huihui_ai/qwen3.5-abliterated:9b")
    validate_local_model_size("custom-local-model")
    with pytest.raises(ValueError, match="最多支持 9B"):
        validate_local_model_size("qwen3.5:14b")
    assert pytorch_device("auto") == "auto"


def test_hymt_weights_are_lazy_until_the_candidate_stage(tmp_path):
    service = HyMTTranslator(tmp_path / "hy-mt2")
    assert service.model is None
    assert service.tokenizer is None


def test_hy_mt_uses_documented_sampling_for_translation_but_not_format_repair():
    assert hy_mt_generation_options() == {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.6,
        "top_k": 20,
        "repetition_penalty": 1.05,
    }
    assert hy_mt_generation_options(repair=True) == {
        "do_sample": False,
        "repetition_penalty": 1.05,
    }


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
    assert captured["body"]["think"] is False
    assert captured["body"]["options"]["temperature"] == 0.2


def test_ollama_model_is_pulled_automatically_when_missing(monkeypatch):
    responses = iter(
        [
            {"models": []},
            {"status": "success"},
            {"models": [{"name": "quality-local"}]},
        ]
    )
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(
            (request.full_url, json.loads(request.data) if request.data else None)
        )
        return FakeResponse(next(responses))

    monkeypatch.setattr(translator, "urlopen", fake_urlopen)
    assert ensure_ollama_model("http://localhost:11434", "quality-local") is True
    assert requests[1] == (
        "http://localhost:11434/api/pull",
        {"model": "quality-local", "stream": False},
    )


def test_existing_ollama_model_is_not_pulled(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return FakeResponse({"models": [{"name": "quality-local"}]})

    monkeypatch.setattr(translator, "urlopen", fake_urlopen)
    assert ensure_ollama_model("http://localhost:11434", "quality-local") is False
    assert calls == ["http://localhost:11434/api/tags"]


def test_context_review_repairs_split_sentence_without_changing_unit_mapping():
    class ReviewingTranslator(OllamaTranslator):
        def __init__(self):
            PromptTranslator.__init__(self)

        def _generate(self, prompt, max_new_tokens=1600):
            del max_new_tokens
            assert "相邻 OCR 单元可能把同一句话拆开" in prompt
            return "\n".join(
                [
                    "[p001u01] 听说他比我们大两岁，连高中都没上……",
                    "[p001u02] 好像直接去工作了……",
                ]
            )

    page = PageOCR(
        1,
        "1.png",
        100,
        100,
        [
            TextUnit(
                "p001u01",
                [0, 0, 20, 20],
                [0, 0, 20, 20],
                "確か僕たちの２つ上で、高校にも行かないで…",
                1.0,
                zh="听说我们俩跳级了……",
            ),
            TextUnit(
                "p001u02",
                [20, 0, 40, 20],
                [20, 0, 40, 20],
                "働きに行ったとか聞いてたけど…",
                1.0,
                zh="结果你倒先跑来了……",
            ),
        ],
    )
    ReviewingTranslator().review_pages([page], {}, preserve_sfx=False)
    assert [unit.zh for unit in page.units] == [
        "听说他比我们大两岁，连高中都没上……",
        "好像直接去工作了……",
    ]


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
        pages,
        chunk_pages=2,
        progress=lambda current, total: updates.append((current, total)),
    )
    assert updates == [(2, 5), (4, 5), (5, 5)]


def test_context_batch_parser_accepts_hymt_split_id_lines():
    service = PromptTranslator()
    output = """[id]p001u01
：这里是

[id]p001u02
我们的秘密基地
[p001u03] 很好哦"""

    assert service._parse_indexed_translations(
        output, ["p001u01", "p001u02", "p001u03"]
    ) == {
        "p001u01": "这里是",
        "p001u02": "我们的秘密基地",
        "p001u03": "很好哦",
    }


def test_translation_quality_gate_rejects_kana_and_context_leakage():
    unit = TextUnit("p001u01", [0, 0, 20, 20], [0, 0, 20, 20], "ガク君", 1.0)
    assert PromptTranslator._valid_translation(unit, "岳君") is True
    assert PromptTranslator._valid_translation(unit, "ガク君") is False
    assert (
        PromptTranslator._valid_translation(unit, "第1页：这是被错误复述的一整段上下文")
        is False
    )
    assert (
        PromptTranslator._valid_translation(unit, "郭君啊 / 这样不行 / 不要啊") is False
    )
    assert PromptTranslator._valid_translation(unit, "呀ń咻") is False

    latin_source = TextUnit("p001u02", [0, 0, 20, 20], [0, 0, 20, 20], "Ｗ〒ッ", 1.0)
    assert PromptTranslator._valid_translation(latin_source, "W〒！") is True


def test_translation_quality_gate_allows_natural_chinese_expansion():
    unit = TextUnit("p001u01", [0, 0, 20, 20], [0, 0, 20, 20], "みんな引っ越した", 1.0)
    assert (
        PromptTranslator._valid_translation(unit, "大家不是搬走了，就是不再来了。")
        is True
    )


def test_short_interjection_cannot_absorb_another_dialogue_line():
    unit = TextUnit("p001u01", [0, 0, 20, 20], [0, 0, 20, 20], "にゃ", 1.0)
    assert (
        PromptTranslator._valid_translation(unit, "格克君别动怎么了很疼吗身体发麻但是")
        is False
    )


def test_reviewed_valid_translation_is_not_sent_through_model_again():
    valid = TextUnit("p001u01", [0, 0, 20, 20], [0, 0, 20, 20], "秘密", 1.0, zh="秘密")
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


def test_confirmed_recurring_names_become_a_glossary_and_inconsistent_candidates_are_repaired():
    class NameAwareTranslator(PromptTranslator):
        def _generate(self, prompt, max_new_tokens=1600):
            del max_new_tokens
            ids = translator.re.findall(r"\[(p\d+u\d+)\]", prompt)
            if ids:
                return "\n".join(f"[{unit_id}] 小刚君" for unit_id in ids)
            return "岳君？" if "【原文】ガク君？" in prompt else "岳君"

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
    service.translate_pages(pages, preserve_sfx=False, glossary={"ガク": "岳"})
    assert service.resolved_glossary == {"ガク": "岳"}
    assert [page.units[0].zh for page in pages] == ["岳君？", "岳君"]
    assert all(page.units[0].translation_attempts[0] == "小刚君" for page in pages)


def test_recurring_name_is_resolved_automatically_from_dictionary_candidates():
    class FakeNameDictionary:
        def lookup(self, names):
            assert names == ["ガク"]
            return {
                "ガク": [
                    NameCandidate("岳", ("male given name or forename",)),
                    NameCandidate("賀句", ("male given name or forename",)),
                ]
            }

    class AutoNameTranslator(PromptTranslator):
        def _generate(self, prompt, max_new_tokens=1600):
            del max_new_tokens
            ids = translator.re.findall(r"\[(p\d+u\d+)\]", prompt)
            return "\n".join(f"[{unit_id}] 岳君？" for unit_id in ids)

        def _generate_repair(self, prompt, max_new_tokens):
            del max_new_tokens
            return "1" if "候选" in prompt else "岳君？"

    pages = [
        PageOCR(
            page,
            f"{page}.png",
            100,
            100,
            [
                TextUnit(
                    f"p{page:03d}u01",
                    [0, 0, 20, 20],
                    [0, 0, 20, 20],
                    "ガク君？",
                    1.0,
                )
            ],
        )
        for page in range(1, 3)
    ]
    service = AutoNameTranslator(name_dictionary=FakeNameDictionary())
    service.translate_pages(pages, preserve_sfx=False)
    assert service.resolved_glossary == {"ガク": "岳"}
    assert [page.units[0].zh for page in pages] == ["岳君？", "岳君？"]


def test_inferred_name_is_rejected_when_it_leaks_into_an_unrelated_line():
    service = PromptTranslator()
    service.inferred_glossary = {"ガク": "格克"}
    unit = TextUnit("p001u01", [0, 0, 20, 20], [0, 0, 20, 20], "勇気", 1.0)
    assert service._respects_glossary(unit, "格克勇气", {"ガク": "格克"}) is False


def test_glossary_matches_a_word_split_by_manga_ellipses():
    service = PromptTranslator()
    unit = TextUnit(
        "p080u01",
        [0, 0, 20, 20],
        [0, 0, 20, 20],
        "え？コン．．．ドーム？．．．",
        1.0,
    )
    glossary = {"コンドーム": "避孕套"}
    assert service._source_term_present("コンドーム", unit.ja) is True
    assert service._respects_glossary(unit, "诶？避……孕套？", glossary) is True
    assert service._respects_glossary(unit, "诶？康……多姆？", glossary) is False


def test_plain_recurring_katakana_name_is_detected_across_pages_but_sfx_is_not():
    pages = [
        PageOCR(
            index,
            f"{index}.png",
            100,
            100,
            [
                TextUnit(
                    f"p{index:03d}u01",
                    [0, 0, 20, 20],
                    [0, 0, 20, 20],
                    source,
                    1.0,
                ),
                TextUnit(
                    f"p{index:03d}u02",
                    [20, 0, 40, 20],
                    [20, 0, 40, 20],
                    "カチ",
                    1.0,
                    is_sfx=True,
                ),
            ],
        )
        for index, source in enumerate(
            ["サナはいつも", "サナに聞いた", "サナ、待って"], start=1
        )
    ]
    assert PromptTranslator._recurring_katakana_names(pages) == ["サナ"]


def test_inferred_recurring_name_recovers_speech_mislabeled_as_sfx():
    service = PromptTranslator()
    service.inferred_glossary = {"サナ": "沙那"}
    unit = TextUnit(
        "p001u01",
        [0, 0, 100, 200],
        [0, 0, 100, 200],
        "ああサナッ\nイクイク…",
        1.0,
        is_sfx=True,
    )
    page = PageOCR(1, "001.png", 100, 200, [unit])
    assert service._restore_named_dialogue_roles([page]) == {unit.id}
    assert unit.is_sfx is False

    duplicate = TextUnit(
        "p001u02",
        [0, 0, 100, 200],
        [0, 0, 100, 200],
        "ああサナッ\nイクイク…",
        1.0,
        is_sfx=True,
        special="ocr_duplicate",
    )
    page.units.append(duplicate)
    assert service._restore_named_dialogue_roles([page]) == set()
    assert duplicate.is_sfx is True


def test_name_selector_receives_full_simplified_gender_labeled_candidates():
    class NameChoiceTranslator(PromptTranslator):
        def _generate(self, prompt, max_new_tokens=1600):
            del max_new_tokens
            options = dict(
                (name, number)
                for number, name in translator.re.findall(
                    r"(\d+)\.([\u3400-\u9fff]+)（", prompt
                )
            )
            return options["纱奈"]

    candidates = [
        NameCandidate(f"佐{chr(0x4E00 + index)}", ("female given name or forename",))
        for index in range(40)
    ] + [NameCandidate("紗奈", ("female given name or forename",))]
    service = NameChoiceTranslator()
    assert (
        service._choose_name_candidate("サナ", candidates, ["サナはいつも"]) == "纱奈"
    )


def test_name_selector_excludes_demonstrative_looking_spelling_when_natural_exists():
    class FirstChoiceTranslator(PromptTranslator):
        def _generate(self, prompt, max_new_tokens=1600):
            del prompt, max_new_tokens
            return "1"

    candidates = [
        NameCandidate("沙那", ("female given name or forename",)),
        NameCandidate("紗奈", ("female given name or forename",)),
    ]
    service = FirstChoiceTranslator()

    assert (
        service._choose_name_candidate("サナ", candidates, ["サナはいつも"]) == "纱奈"
    )


def test_confirmed_sfx_accepts_compact_natural_translation():
    unit = TextUnit(
        "p001u01",
        [0, 0, 100, 100],
        [0, 0, 100, 100],
        "ドドドドドドドド",
        1.0,
        is_sfx=True,
    )
    assert PromptTranslator._preserves_audit_information(unit, "轰隆隆") is True
    assert PromptTranslator()._candidate_is_acceptable(unit, "轰隆隆", {}) is True
    assert (
        PromptTranslator()._candidate_is_acceptable(unit, "莫非你也对我感兴趣？", {})
        is False
    )


def test_sfx_retry_uses_context_free_effect_prompt():
    class EffectTranslator(PromptTranslator):
        def _generate(self, prompt, max_new_tokens=1600):
            del max_new_tokens
            return "巴肯！" if "把下面" in prompt else "怦咚"

    unit = TextUnit(
        "p001u01",
        [0, 0, 100, 100],
        [0, 0, 100, 100],
        "ドキン",
        1.0,
        is_sfx=True,
    )
    service = EffectTranslator()
    assert service._translate_one(unit, [], "无", {}) == "怦咚"
    assert unit.translation_attempts == ["怦咚"]


def test_common_sfx_families_use_deterministic_manga_wording():
    service = PromptTranslator()
    cases = {
        "ドキン": "怦咚",
        "バクン\nバクン\nバクンッ": "怦咚怦咚怦咚！",
        "キャッ": "呀！",
        "ドドドドド": "轰隆隆",
        "ゴッ": "咚！",
    }
    for source, expected in cases.items():
        unit = TextUnit(
            "p001u01",
            [0, 0, 100, 100],
            [0, 0, 100, 100],
            source,
            1.0,
            is_sfx=True,
        )
        assert service._translate_one(unit, [], "无", {}) == expected


def test_adult_register_prevents_euphemism_from_becoming_generic_action():
    pages = [
        PageOCR(
            1,
            "1.png",
            100,
            100,
            [
                TextUnit("p001u01", [0, 0, 20, 20], [0, 0, 20, 20], "中出し", 1.0),
                TextUnit(
                    "p001u02", [20, 0, 40, 20], [20, 0, 40, 20], "またイキそう", 1.0
                ),
            ],
        )
    ]
    register = PromptTranslator._detect_translation_register(pages)
    assert "成人漫画" in register
    assert "误译成崩溃" in register


def test_adult_domain_terms_are_added_without_overriding_user_glossary():
    pages = [
        PageOCR(
            1,
            "1.png",
            100,
            100,
            [
                TextUnit("p001u01", [0, 0, 20, 20], [0, 0, 20, 20], "中出し", 1.0),
                TextUnit(
                    "p001u02", [20, 0, 40, 20], [20, 0, 40, 20], "またイキそう", 1.0
                ),
            ],
        )
    ]
    service = PromptTranslator()
    service.translation_register = service._detect_translation_register(pages)
    assert service._resolve_glossary(pages, {"中出し": "射在里面"}) == {
        "中出し": "射在里面",
        "イキ": "高潮",
    }


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
    result = service._translate_one(unit, [], "ガク译为格克", {"ガク": "格克"})
    assert result == "勇气"
    assert "固定译名：无" in service.prompt
    assert "格克" not in service.prompt


def test_failed_short_vocalization_has_a_local_target_script_fallback():
    assert interjection_fallback("．ッ♡♥ゃンっ♡") == "♡♥呀嗯♡"
    assert interjection_fallback("サナッマンコ締めすぎっ") == ""


def test_calm_preference_fallback_does_not_add_emotional_intensity():
    assert calm_preference_fallback("僕は、秘密基地が好きだ。") == "我喜欢秘密基地。"
    assert (
        calm_preference_fallback("僕は、ここ秘密基地が好きだ。")
        == "我喜欢这个秘密基地。"
    )
    assert calm_preference_fallback("秘密基地が好きか？") == ""

    class NoGeneration(PromptTranslator):
        def _generate(self, prompt, max_new_tokens=1600):
            raise AssertionError("deterministic statement must not call a model")

    unit = TextUnit(
        "p001u01",
        [0, 0, 100, 200],
        [0, 0, 100, 200],
        "僕は、ここ秘密基地が好きだ。",
        1.0,
    )
    assert NoGeneration()._translate_one(unit, [], "", {}) == "我喜欢这个秘密基地。"


def test_semantic_audit_cannot_collapse_a_complete_title():
    unit = TextUnit(
        "p001u01",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "僕に勇気があったなら",
        1.0,
        zh="要是我有勇气的话",
    )
    assert OllamaTranslator._preserves_audit_information(unit, "我") is False
    assert (
        OllamaTranslator._preserves_audit_information(unit, "如果我有勇气的话") is True
    )


def test_semantic_audit_cannot_drop_source_negation():
    unit = TextUnit(
        "p001u01",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "奥まで入れちゃだめ",
        1.0,
        zh="不可以插到底",
    )
    assert OllamaTranslator._preserves_audit_information(unit, "插到底") is False
    assert OllamaTranslator._preserves_audit_information(unit, "别插到底") is True


def test_semantic_audit_accepts_chinese_restriction_for_shika_nai():
    unit = TextUnit(
        "p080u02",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "僕画像でしか見たことないけど．．",
        1.0,
    )
    assert (
        OllamaTranslator._preserves_audit_information(unit, "我只在图片里见过……")
        is True
    )
    assert OllamaTranslator._preserves_audit_information(unit, "我见过图片……") is False


def test_semantic_audit_accepts_chinese_rhetorical_negation():
    unit = TextUnit(
        "p114u03",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "コンドームなんて持ってるわけないかっ",
        1.0,
    )
    assert (
        OllamaTranslator._preserves_audit_information(unit, "怎么可能会带着避孕套啊！")
        is True
    )
    assert (
        OllamaTranslator._preserves_audit_information(unit, "带着避孕套啊！") is False
    )


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("少なくとも土日は絶対安静だ", "至少周末要绝对休息"),
        ("そこっ！？きっきたないですっ！！", "那里啊！？真恶心！！"),
        ("うち貧乏だからお金貯めないとだし", "我家穷，得攒钱呢"),
        ("一緒に観ない？", "一起看吗？"),
        ("もしよかったら一緒に行かない．．．？東京．．．", "方便的话一起去吧？东京……"),
    ],
)
def test_semantic_gate_distinguishes_lexical_nai_obligation_and_invitations(
    source, target
):
    unit = TextUnit(
        "p001u01",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        source,
        1.0,
    )
    assert OllamaTranslator._preserves_audit_information(unit, target) is True


def test_semantic_gate_allows_natural_kana_to_chinese_compression():
    unit = TextUnit(
        "p001u01",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "あ：はいありがとうございます",
        1.0,
    )
    assert OllamaTranslator._preserves_audit_information(unit, "啊，谢谢您") is True


def test_semantic_gate_still_rejects_a_lost_real_negation():
    unit = TextUnit(
        "p001u01",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "深夜なのに生放送でしかもめっちゃ賑やかでさ．．．"
        "それ聴いてたらひとりぼっちじゃない",
        1.0,
    )
    assert (
        OllamaTranslator._preserves_audit_information(
            unit, "深夜还在直播，听着就觉得很孤独"
        )
        is False
    )
    assert (
        OllamaTranslator._preserves_audit_information(
            unit, "深夜还在直播，听着就觉得自己不是孤单一人"
        )
        is True
    )


def test_translation_normalization_localizes_ok_loanword():
    assert PromptTranslator()._normalize_translation("不，不是不行！OK！") == (
        "不，不是不行！好！"
    )


def test_semantic_gate_preserves_implicit_questions_and_relationship_possession():
    challenge = TextUnit(
        "p001u01",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "気持ちいいのかよ!!",
        1.0,
    )
    assert (
        OllamaTranslator._preserves_audit_information(challenge, "真舒服啊！") is False
    )
    assert OllamaTranslator._preserves_audit_information(challenge, "爽吗？！") is True
    assert (
        OllamaTranslator._preserves_audit_information(challenge, "你觉得很爽吗？！")
        is True
    )

    relationship = TextUnit(
        "p001u02",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "流石に彼氏とかいんだろ？",
        1.0,
    )
    assert (
        OllamaTranslator._preserves_audit_information(
            relationship, "他好歹也是男朋友吧？"
        )
        is False
    )
    assert (
        OllamaTranslator._preserves_audit_information(
            relationship, "你应该有男朋友了吧？"
        )
        is True
    )
    why = TextUnit(
        "p001u03",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "何で撮ってるの!?",
        1.0,
    )
    assert OllamaTranslator._preserves_audit_information(why, "为何要拍！") is True


def test_semantic_gate_keeps_explicit_actions_from_short_source_lines():
    laughing = TextUnit("p001u01", [0, 0, 20, 80], [0, 0, 20, 80], "もう笑ってる…", 1.0)
    gaming = TextUnit(
        "p001u02",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "誘えばゲームするんだ…",
        1.0,
    )
    assert OllamaTranslator._preserves_audit_information(laughing, "已经") is False
    assert OllamaTranslator._preserves_audit_information(laughing, "已经笑了……") is True
    assert OllamaTranslator._preserves_audit_information(gaming, "只要邀请她") is False
    assert (
        OllamaTranslator._preserves_audit_information(gaming, "邀请她就会一起玩游戏")
        is True
    )
    named_invitation = TextUnit(
        "p001u03",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "ミナを誘えばゲームするんだ",
        1.0,
    )
    assert (
        OllamaTranslator._preserves_audit_information(
            named_invitation, "米娜要诱我就玩吧"
        )
        is False
    )
    assert (
        OllamaTranslator._preserves_audit_information(
            named_invitation, "只要邀请米娜，她就会来玩游戏"
        )
        is True
    )
    emphasis = TextUnit(
        "p001u04",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "ちょっと待ってミナッズルイって！！",
        1.0,
    )
    assert (
        OllamaTranslator._preserves_audit_information(
            emphasis, "等等，米娜说你狡猾啊！！"
        )
        is False
    )
    assert (
        OllamaTranslator._preserves_audit_information(
            emphasis, "等等，米娜！太狡猾了！！"
        )
        is True
    )
    assert (
        OllamaTranslator._preserves_audit_information(
            emphasis, "等等，米娜要加塞儿！！"
        )
        is False
    )


def test_all_translation_stages_reject_truncated_negation():
    service = OllamaTranslator("http://localhost:11434", "local-9b")
    unit = TextUnit(
        "p001u01",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "奥まで入れちゃだめっ♡",
        1.0,
    )
    assert service._candidate_is_acceptable(unit, "都插到", {}) is False
    assert service._candidate_is_acceptable(unit, "不可以插到最里面♡", {}) is True


def test_hard_gate_rejects_context_expansion_and_wrong_age_unit():
    service = OllamaTranslator("http://localhost:11434", "local-9b")
    expansion = TextUnit(
        "p001u01", [0, 0, 20, 80], [0, 0, 20, 80], "見れないんだよな", 1.0
    )
    age = TextUnit("p001u02", [0, 0, 20, 80], [0, 0, 20, 80], "僕たちの2つ上", 1.0)
    assert (
        service._candidate_is_acceptable(
            expansion, "真拿你没办法最近连正眼都瞧不上的样子", {}
        )
        is False
    )
    assert service._candidate_is_acceptable(age, "比我们高两级", {}) is False
    assert service._candidate_is_acceptable(age, "比我们大两岁", {}) is True


def test_hard_gate_rejects_added_emotional_intensity():
    service = OllamaTranslator("http://localhost:11434", "local-9b")
    calm = TextUnit(
        "p001u01",
        [0, 0, 100, 200],
        [0, 0, 100, 200],
        "僕は、秘密基地が好きだ。",
        1.0,
    )
    assert service._candidate_is_acceptable(calm, "我最爱这个秘密基地！", {}) is False
    assert service._candidate_is_acceptable(calm, "我喜欢这个秘密基地。", {}) is True


def test_residual_gate_finds_repeated_long_phrase_and_double_negation():
    service = OllamaTranslator("http://localhost:11434", "local-9b")
    units = [
        TextUnit(
            f"p001u0{index}",
            [0, 0, 20, 80],
            [0, 0, 20, 80],
            source,
            1.0,
            zh=target,
        )
        for index, (source, target) in enumerate(
            [
                ("見えない", "最近真拿你没办法"),
                ("君が", "你啊真拿你没办法"),
                ("あっ", "啊真拿你没办法"),
                ("へち", "嘿真拿你没办法"),
                ("奥まで入れちゃだめ", "最里面\n别插进去\n不行"),
            ],
            1,
        )
    ]
    page = PageOCR(1, "001.png", 100, 100, units)
    assert service.residual_quality_ids([page], {}) == {unit.id for unit in units}

    natural = TextUnit(
        "p002u01",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "大丈夫、怖くないから",
        1.0,
        zh="没关系，没什么可怕的",
    )
    assert (
        service.residual_quality_ids([PageOCR(2, "002.png", 100, 100, [natural])], {})
        == set()
    )


def test_rejected_short_vocalization_uses_deterministic_residual_fallback():
    service = OllamaTranslator("http://localhost:11434", "local-9b")
    unit = TextUnit(
        "p001u01",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "うんっ♡\nんンっ\nんふぅ♡",
        1.0,
        zh="嗯♡啊♡唔呲♡（注：若需更精准双关可微调，但受字数限制）",
    )
    page = PageOCR(1, "001.png", 100, 100, [unit])
    assert service.apply_deterministic_fallbacks([page], {}) == {unit.id}
    assert unit.zh == "呜嗯♡嗯呼呜♡"
    assert service.residual_quality_ids([page], {}) == set()


def test_residual_gate_never_hides_empty_dialogue():
    service = OllamaTranslator("http://localhost:11434", "local-9b")
    unit = TextUnit("p001u01", [0, 0, 100, 200], [0, 0, 100, 200], "楽しい時間", 1.0)
    page = PageOCR(1, "001.png", 100, 200, [unit])
    assert service.residual_quality_ids([page], {}) == {unit.id}


def test_deterministic_residual_fallback_handles_calm_preference():
    service = OllamaTranslator("http://localhost:11434", "local-9b")
    unit = TextUnit(
        "p001u01",
        [0, 0, 100, 200],
        [0, 0, 100, 200],
        "僕は、\nここ\n秘密基地が\n好きだ。",
        1.0,
    )
    page = PageOCR(1, "001.png", 100, 200, [unit])
    assert service.apply_deterministic_fallbacks([page], {}) == {unit.id}
    assert unit.zh == "我喜欢这个秘密基地。"


def test_candidate_gate_ignores_ascii_ellipsis_in_source_density():
    service = OllamaTranslator("http://localhost:11434", "local-9b")
    unit = TextUnit(
        "p001u01",
        [0, 0, 100, 200],
        [0, 0, 100, 200],
        "...\n彼氏\nかれし...",
        1.0,
    )
    assert service._candidate_is_acceptable(unit, "男友……", {}) is True


def test_candidate_gate_rejects_review_protocol_leaks():
    service = OllamaTranslator("http://localhost:11434", "local-9b")
    unit = TextUnit("p001u01", [0, 0, 100, 200], [0, 0, 100, 200], "何？", 1.0)
    assert service._candidate_is_acceptable(unit, "原文：何？", {}) is False
    assert service._candidate_is_acceptable(unit, "什么？ || 怎么了？", {}) is False


def test_source_tone_normalization_removes_invented_terminal_emphasis():
    calm = TextUnit("p001u01", [0, 0, 100, 200], [0, 0, 100, 200], "抜いてよ", 1.0)
    emphatic = TextUnit(
        "p001u02", [0, 0, 100, 200], [0, 0, 100, 200], "抜いてよ！", 1.0
    )
    assert OllamaTranslator._restore_source_tone(calm, "拔出来呀！") == "拔出来呀"
    assert OllamaTranslator._restore_source_tone(emphatic, "快拔出来！") == "快拔出来！"


def test_simplified_target_normalizes_every_model_candidate():
    service = OllamaTranslator("http://localhost:11434", "local-9b")
    assert service._normalize_translation("早點讓兩人回來") == "早点让两人回来"


def test_semantic_audit_hints_are_language_rules_not_book_specific_answers():
    unit = TextUnit(
        "p001u01",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "無理して買った",
        1.0,
    )
    hints = OllamaTranslator._semantic_audit_hints(unit, "バイクで来た")
    assert "硬撑" in hints
    assert "特意" in hints
    assert "买下来的" not in hints


def test_semantic_audit_hints_use_page_context_for_polysemy():
    unit = TextUnit(
        "p001u01",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "後で乗せてやるよ",
        1.0,
    )
    assert "载人" in OllamaTranslator._semantic_audit_hints(unit, "バイクで来た")
    assert OllamaTranslator._semantic_audit_hints(unit, "教室で話した") == ""


def test_layout_budget_scales_from_source_density_not_page_identity():
    short = TextUnit(
        "anything", [0, 0, 20, 80], [0, 0, 20, 80], "私っ妊娠とか…無理だって！！", 1.0
    )
    long = TextUnit(
        "anything-else",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "確か僕たちの2つ上で、高校にも行かないで",
        1.0,
    )
    assert OllamaTranslator._layout_char_budget(short) == 12
    assert OllamaTranslator._layout_char_budget(long) > 12


def test_reasoning_json_retries_with_bounded_output_budget_when_truncated(monkeypatch):
    budgets = []
    timeouts = []

    def fake_request(_url, payload, timeout=180):
        budgets.append(payload["options"]["num_predict"])
        timeouts.append(timeout)
        if len(budgets) < 3:
            return {"done_reason": "length"}
        return {"done_reason": "stop", "message": {"content": '{"ok": true}'}}

    monkeypatch.setattr(translator, "_json_request", fake_request)
    service = OllamaTranslator("http://localhost:11434", "local-9b")
    result = service._generate_json(
        "audit",
        {"type": "object", "properties": {"ok": {"type": "boolean"}}},
        max_new_tokens=8_192,
        think=True,
    )
    assert result == {"ok": True}
    assert budgets == [8_192, 16_384, 32_768]
    assert timeouts == [600, 600, 600]


def test_local_quality_pipeline_uses_independent_candidate_before_staged_9b_audit():
    events = []

    class Editor(PromptTranslator):
        def unload(self):
            events.append("unload")

        def translate_pages(self, pages, *args, **kwargs):
            del args, kwargs
            events.append("draft")
            self.resolved_glossary = {"サナ": "纱那"}
            self.inferred_glossary = dict(self.resolved_glossary)
            self.translation_register = "成人漫画"
            pages[0].units[0].zh = "草稿"

    class Candidate(PromptTranslator):
        def translate_pages(self, pages, *args, **kwargs):
            del args, kwargs
            events.append("candidate")
            assert pages[0].units[0].zh == ""
            pages[0].units[0].zh = "候选"

        def unload(self):
            events.append("candidate-unload")

    class Auditor(PromptTranslator):
        def review_pages(self, pages, glossary, *args, **kwargs):
            del glossary, args
            events.append("review")
            assert pages[0].units[0].zh == "草稿"
            assert kwargs == {}

        def judge_candidates(self, pages, alternatives, *args, **kwargs):
            del args
            events.append("judge")
            assert pages[0].units[0].zh == "草稿"
            assert alternatives == {}
            assert kwargs["deep_reasoning"] is False

        def retranslate_risk_units(self, pages, target_ids, *args, **kwargs):
            del pages, args, kwargs
            events.append("risk")
            assert target_ids == {"p001u01"}

        def residual_quality_ids(self, pages, glossary):
            del pages, glossary
            return set()

        def unload(self):
            events.append("auditor-unload")

    page = PageOCR(
        1,
        "001.png",
        100,
        100,
        [TextUnit("p001u01", [0, 0, 20, 80], [0, 0, 20, 80], "サナ", 1.0)],
    )
    service = LocalQualityTranslator(Editor(), Candidate(), Auditor())
    service.translate_pages([page], preserve_sfx=True)
    assert events == [
        "draft",
        "unload",
        "candidate",
        "candidate-unload",
        "review",
        "judge",
        "risk",
        "auditor-unload",
    ]
    assert service.resolved_glossary == {"サナ": "纱那"}


def test_local_quality_promotes_a_valid_candidate_when_editor_is_empty():
    class Editor(PromptTranslator):
        def translate_pages(self, pages, *args, **kwargs):
            del args, kwargs
            pages[0].units[0].zh = ""

    class Candidate(PromptTranslator):
        def translate_pages(self, pages, *args, **kwargs):
            del args, kwargs
            pages[0].units[0].zh = "快乐时光"

    class Auditor(PromptTranslator):
        def review_pages(self, *args, **kwargs):
            del args, kwargs

        def judge_candidates(self, pages, alternatives, *args, **kwargs):
            del args, kwargs
            assert pages[0].units[0].zh == "快乐时光"
            assert alternatives == {}

        def retranslate_risk_units(self, *args, **kwargs):
            del args, kwargs

    page = PageOCR(
        1,
        "001.png",
        100,
        100,
        [TextUnit("p001u01", [0, 0, 20, 80], [0, 0, 20, 80], "楽しい時間", 1.0)],
    )
    LocalQualityTranslator(Editor(), Candidate(), Auditor()).translate_pages([page])
    assert page.units[0].zh == "快乐时光"


def test_semantic_retranslation_recovers_unresolved_non_risk_units():
    class Auditor(OllamaTranslator):
        def _generate_json(self, prompt, schema, max_new_tokens=4096, think=False):
            del prompt, schema, max_new_tokens
            assert think is False
            return {"items": [{"id": "p001u01", "zh": "快乐时光"}]}

    unit = TextUnit("p001u01", [0, 0, 20, 80], [0, 0, 20, 80], "楽しい時間", 1.0)
    page = PageOCR(1, "001.png", 100, 100, [unit])
    Auditor("http://localhost:11434", "local-9b").retranslate_risk_units(
        [page], {unit.id}, {}, preserve_sfx=True
    )
    assert unit.zh == "快乐时光"


def test_semantic_retranslation_uses_bounded_non_thinking_json_for_risk_units():
    class Auditor(OllamaTranslator):
        def _generate_json(self, prompt, schema, max_new_tokens=4096, think=False):
            del prompt, schema
            assert think is False
            assert 2048 <= max_new_tokens <= 4096
            return {"items": [{"id": "p001u01", "zh": "别插到最里面"}]}

    unit = TextUnit(
        "p001u01",
        [0, 0, 20, 80],
        [0, 0, 20, 80],
        "奥まで入れちゃだめ",
        1.0,
    )
    page = PageOCR(1, "001.png", 100, 100, [unit])
    Auditor("http://localhost:11434", "local-9b").retranslate_risk_units(
        [page], {unit.id}, {}, preserve_sfx=True
    )
    assert unit.zh == "别插到最里面"
