from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .model_manager import ModelDependencyError
from .ocr import PageOCR, TextUnit


LINE_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(.+?)\s*$")
# Japanese letters and prolonged sound mark, excluding shared punctuation such
# as the middle dot (・), which is valid in otherwise Chinese typesetting.
KANA_RE = re.compile(r"[ぁ-ゖァ-ヺーヽヾ]")
KATAKANA_NAME_RE = re.compile(
    r"([ァ-ヺー]{2,12})(?=(?:[．.・…\s]*)?(?:君|さん|ちゃん|様|先輩|先生))"
)
CHINESE_NAME_RE = re.compile(r"[\u3400-\u9fff·]{1,12}")
LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")
SOURCE_LATIN_RE = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]")
INTERJECTION_HINT_RE = re.compile(r"[♡♥ッっぁぃぅぇぉゃゅょァィゥェォャュョ]")
KANJI_RE = re.compile(r"[\u3400-\u9fff]")
INTERJECTION_VOICE_MAP = {
    **dict.fromkeys("あぁアァ", "啊"),
    **dict.fromkeys("いぃイィひヒ", "咿"),
    **dict.fromkeys("うぅウゥ", "呜"),
    **dict.fromkeys("えぇエェへヘ", "诶"),
    **dict.fromkeys("おぉオォほホ", "哦"),
    **dict.fromkeys("んン", "嗯"),
    **dict.fromkeys("やゃヤャ", "呀"),
    **dict.fromkeys("はハ", "哈"),
    **dict.fromkeys("ふフ", "呼"),
    "ッ": "",
    "っ": "",
}


def interjection_fallback(source: str) -> str:
    """Convert a failed short vocalization without inventing dialogue meaning."""
    if (
        len(source) > 18
        or KANJI_RE.search(source)
        or not INTERJECTION_HINT_RE.search(source)
    ):
        return ""
    output: list[str] = []
    punctuation = set("♡♥！？!?,，、…〜～〰：:〒〃＞>・")
    for character in source:
        if character in INTERJECTION_VOICE_MAP:
            replacement = INTERJECTION_VOICE_MAP[character]
        elif KANA_RE.fullmatch(character):
            replacement = "嗯"
        elif character in punctuation:
            replacement = character
        elif character in "．。":
            replacement = "…"
        else:
            replacement = character if SOURCE_LATIN_RE.fullmatch(character) else ""
        if replacement and (not output or replacement != output[-1] or replacement in "♡♥"):
            output.append(replacement)
    candidate = "".join(output).strip("…，、")
    return candidate if KANJI_RE.search(candidate) else ""


def kana_bad_words_ids(tokenizer) -> list[list[int]]:
    """Build tokenizer-native sequences that can produce Japanese kana."""
    sequences: set[tuple[int, ...]] = set()
    for token, token_id in tokenizer.get_vocab().items():
        decoded = tokenizer.decode([token_id], skip_special_tokens=True)
        if KANA_RE.search(token) or KANA_RE.search(decoded):
            sequences.add((int(token_id),))
    kana_characters = (
        [chr(codepoint) for codepoint in range(ord("ぁ"), ord("ゖ") + 1)]
        + [chr(codepoint) for codepoint in range(ord("ァ"), ord("ヺ") + 1)]
        + list("ーヽヾ")
    )
    for character in kana_characters:
        encoded = tokenizer.encode(character, add_special_tokens=False)
        if encoded:
            sequences.add(tuple(int(token_id) for token_id in encoded))
    return [list(sequence) for sequence in sorted(sequences)]


def pytorch_device(device: str) -> str:
    """Translate the shared UI device vocabulary to a PyTorch device name."""
    if device.startswith("gpu"):
        return "cuda" + device[3:]
    return device


class InferenceConnectionError(RuntimeError):
    pass


def _endpoint(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _json_request(
    url: str,
    payload: dict | None = None,
    api_key: str = "",
    timeout: int = 180,
) -> dict:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise InferenceConnectionError(
            f"推理服务返回 HTTP {exc.code}: {detail or exc.reason}"
        ) from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise InferenceConnectionError(f"无法连接推理服务 {url}: {exc}") from exc


class PromptTranslator:
    def __init__(self, target_language: str = "简体中文"):
        self.target_language = target_language
        self.resolved_glossary: dict[str, str] = {}
        self.inferred_glossary: dict[str, str] = {}

    def _generate(self, prompt: str, max_new_tokens: int = 1600) -> str:
        raise NotImplementedError

    def _generate_repair(self, prompt: str, max_new_tokens: int) -> str:
        return self._generate(prompt, max_new_tokens=max_new_tokens)

    @staticmethod
    def _page_text(pages: Iterable[PageOCR]) -> str:
        rows = []
        for page in pages:
            texts = " / ".join(unit.ja for unit in page.units)
            if texts:
                rows.append(f"第{page.page}页：{texts}")
        return "\n".join(rows)

    @staticmethod
    def _valid_translation(unit: TextUnit, text: str) -> bool:
        clean = text.strip()
        context_leak = (
            "固定译名" in clean
            or "/" in clean
            or bool(re.search(r"第\d+页[：:]", clean))
        )
        latin_leak = bool(LATIN_RE.search(clean)) and not SOURCE_LATIN_RE.search(unit.ja)
        return (
            bool(clean)
            and not KANA_RE.search(clean)
            and not context_leak
            and not latin_leak
            and len(clean) <= max(12, min(120, len(unit.ja) * 4))
        )

    @staticmethod
    def _first_line(text: str) -> str:
        line = next((item.strip() for item in text.splitlines() if item.strip()), "")
        match = LINE_RE.match(line)
        return (match.group(2) if match else line).strip(" \"“”")

    @classmethod
    def _needs_translation(cls, unit: TextUnit, preserve_sfx: bool) -> bool:
        return (
            not unit.skip
            and not (preserve_sfx and unit.is_sfx)
            and not cls._valid_translation(unit, unit.zh)
        )

    def _respects_glossary(
        self, unit: TextUnit, text: str, glossary: dict[str, str]
    ) -> bool:
        for source, target in glossary.items():
            if source in unit.ja and target not in text:
                return False
            if source not in unit.ja and source in self.inferred_glossary and target in text:
                return False
        return True

    @staticmethod
    def _recurring_katakana_names(pages: Iterable[PageOCR]) -> list[str]:
        names = Counter(
            match.group(1)
            for page in pages
            for unit in page.units
            for match in KATAKANA_NAME_RE.finditer(unit.ja)
        )
        return sorted(name for name, count in names.items() if count >= 2)

    def _translate_name(self, name: str) -> str:
        prompt = f"""把日语人物名字【{name}】转换成一个简短、自然且固定的中文名字。
不要附带君、先生等称谓，不得含日文假名、拉丁字母、解释或标点；只输出中文名字。"""
        candidate = self._first_line(self._generate_repair(prompt, max_new_tokens=24))
        return candidate if CHINESE_NAME_RE.fullmatch(candidate) else ""

    def _resolve_glossary(
        self, pages: list[PageOCR], glossary: dict[str, str]
    ) -> dict[str, str]:
        inferred: dict[str, str] = {}
        for name in self._recurring_katakana_names(pages):
            if any(name in source for source in glossary):
                continue
            translated = self._translate_name(name)
            if translated:
                inferred[name] = translated
        self.inferred_glossary = dict(inferred)
        return {**inferred, **glossary}

    def translate_pages(
        self,
        pages: list[PageOCR],
        context_pages: int = 3,
        story_context: bool = True,
        preserve_sfx: bool = True,
        glossary: dict[str, str] | None = None,
        chunk_pages: int = 4,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        glossary = self._resolve_glossary(pages, glossary or {})
        self.resolved_glossary = dict(glossary)
        terms = "；".join(f"{source}译为{target}" for source, target in glossary.items()) or "无"
        for start in range(0, len(pages), chunk_pages):
            chunk = pages[start : start + chunk_pages]
            units = [
                unit
                for page in chunk
                for unit in page.units
                if self._needs_translation(unit, preserve_sfx)
            ]
            if not units:
                if progress:
                    progress(min(start + len(chunk), len(pages)), len(pages))
                continue
            source_rows = []
            for page in chunk:
                source_rows.append(f"--- 第{page.page}页 ---")
                source_rows.extend(
                    f"[{unit.id}] {unit.ja}"
                    for unit in page.units
                    if self._needs_translation(unit, preserve_sfx)
                )
            previous = pages[max(0, start - context_pages) : start] if story_context else []
            prompt = f"""【背景信息】
这是按页连续阅读的日语漫画。结合前文修正明显 OCR 错字，人物称呼和语气必须前后一致。
固定译名：{terms}
前文：
{self._page_text(previous) or '无'}

【任务】
逐条翻译为自然、简洁的{self.target_language}，适合直接放进漫画气泡。
以自然中文和故事连贯为先；可以适度扩写省略的日语主语，但不要把上下文补写进当前行，相邻单元不要重复同一信息。
严格保留每行开头的[id]，一条输入对应一条输出，不合并、不遗漏。只输出[id]和译文。

{chr(10).join(source_rows)}"""
            translated: dict[str, str] = {}
            token_budget = min(1600, max(256, sum(len(unit.ja) for unit in units) * 3))
            for line in self._generate(prompt, max_new_tokens=token_budget).splitlines():
                match = LINE_RE.match(line)
                if match:
                    translated[match.group(1)] = match.group(2).strip()
            candidate_sources: dict[str, set[str]] = {}
            for unit in units:
                candidate = translated.get(unit.id, "").strip()
                candidate_sources.setdefault(candidate, set()).add(unit.ja)
            for unit in units:
                candidate = translated.get(unit.id, "")
                unit.translation_attempts.append(candidate)
                duplicate_context_leak = (
                    len(candidate) >= 4 and len(candidate_sources.get(candidate.strip(), set())) > 1
                )
                if (
                    self._valid_translation(unit, candidate)
                    and not duplicate_context_leak
                    and self._respects_glossary(unit, candidate, glossary)
                ):
                    unit.zh = candidate
                else:
                    unit.zh = self._translate_one(unit, previous + chunk, terms, glossary)
            if progress:
                progress(min(start + len(chunk), len(pages)), len(pages))

    def _translate_one(
        self,
        unit: TextUnit,
        context: list[PageOCR],
        terms: str,
        glossary: dict[str, str] | None = None,
    ) -> str:
        glossary = glossary or {}
        unit_terms = "；".join(
            f"{source}译为{target}"
            for source, target in glossary.items()
            if source in unit.ja
        ) or "无"
        limit = max(12, min(120, len(unit.ja) * 4))
        context_text = self._page_text(context)[-1200:]
        prompt = f"""只翻译【原文】为自然简洁的{self.target_language}，只输出一行译文。
上下文仅供理解，绝对不要复述上下文。译文不得含日文假名，姓名也译成中文；最多 {limit} 个字符。
固定译名：{unit_terms}
上下文：{context_text}
【原文】{unit.ja}"""
        candidate = self._first_line(
            self._generate(prompt, max_new_tokens=max(32, min(128, limit * 3)))
        )
        unit.translation_attempts.append(candidate)
        if self._valid_translation(unit, candidate) and self._respects_glossary(
            unit, candidate, glossary
        ):
            return candidate
        retry = f"""把【原文】翻译为{self.target_language}。不得解释、不得复述上下文、不得含日文假名，姓名译成中文；只输出不超过 {limit} 个字符的一行译文。
固定译名：{unit_terms}
【原文】{unit.ja}"""
        retry_candidate = self._first_line(
            self._generate(retry, max_new_tokens=max(32, min(96, limit * 3)))
        )
        unit.translation_attempts.append(retry_candidate)
        if self._valid_translation(unit, retry_candidate) and self._respects_glossary(
            unit, retry_candidate, glossary
        ):
            return retry_candidate
        repair = f"""下面的翻译仍含日文假名、为空或格式不合格，请彻底改写。
只使用自然简洁的{self.target_language}、数字和必要标点；即使是人名、语气词、喘息声或拟声词，也不能原样保留任何日文假名。
不得解释，只输出不超过 {limit} 个字符的一行最终译文。
固定译名：{unit_terms}
【原文】{unit.ja}
【不合格输出】{retry_candidate or candidate or '空'}"""
        repaired = self._first_line(
            self._generate_repair(repair, max_new_tokens=max(32, min(96, limit * 3)))
        )
        unit.translation_attempts.append(repaired)
        if self._valid_translation(unit, repaired) and self._respects_glossary(
            unit, repaired, glossary
        ):
            return repaired
        fallback = interjection_fallback(unit.ja)
        if fallback:
            unit.translation_attempts.append(fallback)
        if self._valid_translation(unit, fallback) and self._respects_glossary(
            unit, fallback, glossary
        ):
            return fallback
        return ""


class HyMTTranslator(PromptTranslator):
    def __init__(self, model_path: Path, target_language: str = "简体中文", device: str = "auto"):
        super().__init__(target_language)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ModelDependencyError(
                "Translation dependencies are missing. Run scripts/bootstrap with an ML profile."
            ) from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        kwargs = {"trust_remote_code": True}
        torch_device = pytorch_device(device)
        if torch_device == "auto":
            kwargs.update(dtype=torch.bfloat16, device_map="auto")
        else:
            kwargs.update(dtype=torch.float32 if torch_device == "cpu" else torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if torch_device not in {"auto", "cpu"}:
            self.model.to(torch_device)
        self.model.eval()
        self._kana_bad_words_ids: list[list[int]] | None = None

    def _generate(self, prompt: str, max_new_tokens: int = 1600) -> str:
        return self._generate_with_constraints(prompt, max_new_tokens)

    def _generate_repair(self, prompt: str, max_new_tokens: int) -> str:
        if self._kana_bad_words_ids is None:
            self._kana_bad_words_ids = kana_bad_words_ids(self.tokenizer)
        return self._generate_with_constraints(
            prompt,
            max_new_tokens,
            bad_words_ids=self._kana_bad_words_ids,
        )

    def _generate_with_constraints(
        self,
        prompt: str,
        max_new_tokens: int,
        bad_words_ids: list[list[int]] | None = None,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=bad_words_ids,
            )
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()



class OllamaTranslator(PromptTranslator):
    def __init__(self, base_url: str, model: str, target_language: str = "简体中文"):
        super().__init__(target_language)
        self.base_url = base_url
        self.model = model

    def _generate(self, prompt: str, max_new_tokens: int = 1600) -> str:
        payload = _json_request(
            _endpoint(self.base_url, "/api/chat"),
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0, "num_predict": max_new_tokens},
            },
        )
        try:
            return payload["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise InferenceConnectionError("Ollama 返回格式中缺少 message.content") from exc


class OpenAICompatibleTranslator(PromptTranslator):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        target_language: str = "简体中文",
    ):
        super().__init__(target_language)
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

    def _generate(self, prompt: str, max_new_tokens: int = 1600) -> str:
        payload = _json_request(
            _endpoint(self.base_url, "/chat/completions"),
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_new_tokens,
            },
            api_key=self.api_key,
        )
        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceConnectionError("兼容 API 返回格式中缺少 choices[0].message.content") from exc


def available_remote_models(backend: str, base_url: str, api_key: str = "") -> list[str]:
    if backend == "ollama":
        payload = _json_request(_endpoint(base_url, "/api/tags"), timeout=15)
        return [item["name"] for item in payload.get("models", []) if item.get("name")]
    payload = _json_request(_endpoint(base_url, "/models"), api_key=api_key, timeout=15)
    return [item["id"] for item in payload.get("data", []) if item.get("id")]
