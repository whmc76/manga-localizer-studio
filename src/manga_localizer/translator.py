from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from .model_manager import ModelDependencyError
from .ocr import PageOCR, TextUnit


LINE_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(.+?)\s*$")


def pytorch_device(device: str) -> str:
    """Translate the shared UI device vocabulary to a PyTorch device name."""
    if device.startswith("gpu"):
        return "cuda" + device[3:]
    return device


class HyMTTranslator:
    def __init__(self, model_path: Path, target_language: str = "简体中文", device: str = "auto"):
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
        self.target_language = target_language

    @staticmethod
    def _page_text(pages: Iterable[PageOCR]) -> str:
        rows = []
        for page in pages:
            texts = " / ".join(unit.ja for unit in page.units)
            if texts:
                rows.append(f"第{page.page}页：{texts}")
        return "\n".join(rows)

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

    def translate_pages(
        self,
        pages: list[PageOCR],
        context_pages: int = 3,
        story_context: bool = True,
        preserve_sfx: bool = True,
        glossary: dict[str, str] | None = None,
        chunk_pages: int = 4,
    ) -> None:
        glossary = glossary or {}
        terms = "；".join(f"{source}译为{target}" for source, target in glossary.items()) or "无"
        for start in range(0, len(pages), chunk_pages):
            chunk = pages[start : start + chunk_pages]
            units = [
                unit for page in chunk for unit in page.units if not (preserve_sfx and unit.is_sfx)
            ]
            if not units:
                continue
            source_rows = []
            for page in chunk:
                source_rows.append(f"--- 第{page.page}页 ---")
                source_rows.extend(
                    f"[{unit.id}] {unit.ja}"
                    for unit in page.units
                    if not (preserve_sfx and unit.is_sfx)
                )
            previous = pages[max(0, start - context_pages) : start] if story_context else []
            prompt = f"""【背景信息】
这是按页连续阅读的日语漫画。结合前文修正明显 OCR 错字，人物称呼和语气必须前后一致。
固定译名：{terms}
前文：
{self._page_text(previous) or '无'}

【任务】
逐条翻译为自然、简洁的{self.target_language}，适合直接放进漫画气泡。
严格保留每行开头的[id]，一条输入对应一条输出，不合并、不遗漏。只输出[id]和译文。

{chr(10).join(source_rows)}"""
            translated: dict[str, str] = {}
            for line in self._generate(prompt).splitlines():
                match = LINE_RE.match(line)
                if match:
                    translated[match.group(1)] = match.group(2).strip()
            for unit in units:
                if unit.id in translated:
                    unit.zh = translated[unit.id]
                else:
                    unit.zh = self._translate_one(unit, previous + chunk, terms)

    def _translate_one(self, unit: TextUnit, context: list[PageOCR], terms: str) -> str:
        prompt = f"""把下列漫画文字翻译为自然简洁的{self.target_language}，只输出译文。
固定译名：{terms}
上下文：{self._page_text(context)}
原文：{unit.ja}"""
        return self._generate(prompt, max_new_tokens=128).splitlines()[0].strip(" \"“”")
