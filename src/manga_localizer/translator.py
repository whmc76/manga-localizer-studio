from __future__ import annotations

import json
import re
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

    def _generate(self, prompt: str, max_new_tokens: int = 1600) -> str:
        raise NotImplementedError

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
        context_leak = "固定译名" in clean or bool(re.search(r"第\d+页[：:]", clean))
        return (
            bool(clean)
            and not KANA_RE.search(clean)
            and not context_leak
            and len(clean) <= max(60, len(unit.ja) * 6)
        )

    @staticmethod
    def _first_line(text: str) -> str:
        line = next((item.strip() for item in text.splitlines() if item.strip()), "")
        match = LINE_RE.match(line)
        return (match.group(2) if match else line).strip(" \"“”")

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
        glossary = glossary or {}
        terms = "；".join(f"{source}译为{target}" for source, target in glossary.items()) or "无"
        for start in range(0, len(pages), chunk_pages):
            chunk = pages[start : start + chunk_pages]
            units = [
                unit
                for page in chunk
                for unit in page.units
                if not unit.skip and not (preserve_sfx and unit.is_sfx)
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
                    if not unit.skip and not (preserve_sfx and unit.is_sfx)
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
            for unit in units:
                candidate = translated.get(unit.id, "")
                if self._valid_translation(unit, candidate):
                    unit.zh = candidate
                else:
                    unit.zh = self._translate_one(unit, previous + chunk, terms)
            if progress:
                progress(min(start + len(chunk), len(pages)), len(pages))

    def _translate_one(self, unit: TextUnit, context: list[PageOCR], terms: str) -> str:
        limit = max(12, min(120, len(unit.ja) * 4))
        context_text = self._page_text(context)[-1200:]
        prompt = f"""只翻译【原文】为自然简洁的{self.target_language}，只输出一行译文。
上下文仅供理解，绝对不要复述上下文。译文不得含日文假名，姓名也译成中文；最多 {limit} 个字符。
固定译名：{terms}
上下文：{context_text}
【原文】{unit.ja}"""
        candidate = self._first_line(
            self._generate(prompt, max_new_tokens=max(32, min(128, limit * 3)))
        )
        if self._valid_translation(unit, candidate):
            return candidate
        retry = f"""把【原文】翻译为{self.target_language}。不得解释、不得复述上下文、不得含日文假名，姓名译成中文；只输出不超过 {limit} 个字符的一行译文。
固定译名：{terms}
【原文】{unit.ja}"""
        return self._first_line(
            self._generate(retry, max_new_tokens=max(32, min(96, limit * 3)))
        )


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

    def _generate(self, prompt: str, max_new_tokens: int = 1600) -> str:
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
