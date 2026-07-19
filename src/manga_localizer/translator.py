from __future__ import annotations

import json
import hashlib
import gc
import re
from collections import Counter
from collections.abc import Callable, Iterable
from copy import deepcopy
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from opencc import OpenCC

from .model_manager import ModelDependencyError
from .name_dictionary import JapaneseNameDictionary, NameCandidate
from .ocr import PageOCR, TextUnit


LINE_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(.+?)\s*$")
# Japanese letters and prolonged sound mark, excluding shared punctuation such
# as the middle dot (・), which is valid in otherwise Chinese typesetting.
KANA_RE = re.compile(r"[ぁ-ゖァ-ヺーヽヾ]")
KATAKANA_NAME_RE = re.compile(
    r"([ァ-ヺー]{2,12})(?=(?:[．.・…\s]*)?(?:君|さん|ちゃん|様|先輩|先生))"
)
KATAKANA_TOKEN_RE = re.compile(r"[ァ-ヺー]{2,12}")
LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")
SOURCE_LATIN_RE = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]")
INTERJECTION_HINT_RE = re.compile(r"[♡♥ッっぁぃぅぇぉゃゅょァィゥェォャュョ]")
KANJI_RE = re.compile(r"[\u3400-\u9fff]")
ADULT_REGISTER_RE = re.compile(
    r"中出し|マンコ|ちんこ|チンコ|射精|挿入|処女|絶頂|イ(?:キ|ク|ッ)|イク|孕"
)
SEMANTIC_RISK_RE = re.compile(
    r"ない|なく|ません|無理|だめ|駄目|上|下|より|ほど|前|後|まで|だけ|しか|"
    r"から|けど|のに|ても|させ|られ|れる|れば|なら|つもり|はず|乗せ|"
    r"処女マン|マンコ|：|のかよ|なのか|いんだろ|いるんだろ|笑|ゲーム"
)
TARGET_NEGATION_RE = re.compile(r"不|没|别|无|勿|莫|不能|不要|不可|难以|算了|拒绝")
GENERAL_TERMS = {"キス": "亲吻"}
ADULT_TERMS = {"中出し": "内射", "イキ": "高潮", "絶頂": "高潮"}
DEFAULT_LOCAL_TRANSLATION_MODEL = "huihui_ai/qwen3.5-abliterated:9b"
LOCAL_MODEL_MAX_BILLIONS = 9.0
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
        if replacement and (
            not output or replacement != output[-1] or replacement in "♡♥"
        ):
            output.append(replacement)
    candidate = "".join(output).strip("…，、")
    return candidate if KANJI_RE.search(candidate) else ""


def calm_preference_fallback(source: str) -> str:
    """Translate a narrow, unambiguous calm preference without embellishment."""
    clean = re.sub(r"\s+", "", source)
    match = re.fullmatch(
        r"(?:僕|私|俺)は[、,]?(ここ)?([\u3400-\u9fff]{1,16})が好きだ[。．]?",
        clean,
    )
    if not match:
        return ""
    target_object = OpenCC("t2s").convert(match.group(2))
    demonstrative = "这个" if match.group(1) else ""
    return f"我喜欢{demonstrative}{target_object}。"


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


def hy_mt_generation_options(repair: bool = False) -> dict:
    """Use Hy-MT2's documented decoding recipe for translation candidates.

    Repair passes stay deterministic because they are format recovery rather
    than a second creative translation attempt.
    """
    if repair:
        return {"do_sample": False, "repetition_penalty": 1.05}
    return {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.6,
        "top_k": 20,
        "repetition_penalty": 1.05,
    }


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
    def __init__(
        self,
        target_language: str = "简体中文",
        name_dictionary: JapaneseNameDictionary | None = None,
    ):
        self.target_language = target_language
        self.name_dictionary = name_dictionary
        self.resolved_glossary: dict[str, str] = {}
        self.inferred_glossary: dict[str, str] = {}
        self.translation_target_ids: set[str] = set()
        self.translation_register = "自然、简洁的漫画口语"

    def _generate(self, prompt: str, max_new_tokens: int = 1600) -> str:
        raise NotImplementedError

    def _generate_repair(self, prompt: str, max_new_tokens: int) -> str:
        return self._generate(prompt, max_new_tokens=max_new_tokens)

    def unload(self) -> None:
        """Release model resources when a staged translator no longer needs them."""

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
            or "原文：" in clean
            or "原文:" in clean
            or "初译：" in clean
            or "初译:" in clean
            or "||" in clean
            or "/" in clean
            or bool(re.search(r"第\d+页[：:]", clean))
        )
        latin_leak = bool(LATIN_RE.search(clean)) and not SOURCE_LATIN_RE.search(
            unit.ja
        )
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
        return (match.group(2) if match else line).strip(' "“”')

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
            if (
                source not in unit.ja
                and source in self.inferred_glossary
                and target in text
            ):
                return False
        return True

    def _normalize_translation(self, text: str) -> str:
        """Normalize model output to the requested Chinese writing system."""
        clean = text.strip()
        if "简体" in self.target_language:
            clean = OpenCC("t2s").convert(clean)
        return clean

    @staticmethod
    def _restore_source_tone(unit: TextUnit, candidate: str) -> str:
        """Remove terminal emphasis invented for a calm source line."""
        source_is_emphatic = bool(
            re.search(r"[!！っッ]|(?:すごく|とても|めっちゃ|超|大好き|一番)", unit.ja)
        )
        if not source_is_emphatic:
            return re.sub(r"[!！]+$", "", candidate).rstrip()
        return candidate

    def _candidate_is_acceptable(
        self, unit: TextUnit, candidate: str, glossary: dict[str, str]
    ) -> bool:
        """Apply one semantic acceptance contract at every model boundary."""
        return (
            self._valid_translation(unit, candidate)
            and self._respects_glossary(unit, candidate, glossary)
            and self._preserves_audit_information(unit, candidate)
        )

    @staticmethod
    def _preserves_audit_information(unit: TextUnit, candidate: str) -> bool:
        """Reject summaries, truncated clauses, and lost source negation."""
        source = re.sub(r"[\s.．。…、，,!?！？:：♡♥〰〜～]", "", unit.ja)
        proposed = re.sub(r"[\s。…、，,!?！？:：♡♥〰〜～]", "", candidate)
        source_is_question = bool(
            re.search(r"[?？]", unit.ja)
            or re.search(r"(?:のかよ|なのか)[!！…．.]*$", unit.ja.strip())
        )
        if len(source) >= 5 and len(proposed) < 2:
            return False
        length_ratio = 0.25 if source_is_question else 0.4
        minimum_length = 2 if source_is_question else 3
        if len(source) >= 7 and len(proposed) < max(
            minimum_length, round(len(source) * length_ratio)
        ):
            return False
        if len(source) >= 4 and len(proposed) > max(12, round(len(source) * 2.0)):
            return False
        source_has_negation = bool(
            re.search(r"ない|なく|ません|無理|だめ|駄目|じゃない|ではない", unit.ja)
        )
        if source_has_negation and not TARGET_NEGATION_RE.search(candidate):
            return False
        target_is_question = bool(
            re.search(
                r"[?？]|吗|么|是不是|有没有|难道|怎么|什么|谁|哪|为何|为啥|何必|可否",
                candidate,
            )
        )
        if source_is_question and not target_is_question:
            return False
        relationship_question = re.search(
            r"(彼氏|彼女).{0,10}(?:いる|いんだろ|いるんだろ)", unit.ja, re.S
        )
        if relationship_question and not re.search(
            r"(?:有|有没有|没|交|对象).{0,8}(?:男朋友|女朋友|男友|女友)|"
            r"(?:男朋友|女朋友|男友|女友|对象).{0,8}(?:有|有没有|没|交)",
            candidate,
        ):
            return False
        if "笑" in unit.ja and "笑" not in candidate:
            return False
        if "ゲーム" in unit.ja and not re.search(r"游戏|玩", candidate):
            return False
        if "说你" in candidate and not re.search(r"言|話|喋", unit.ja):
            return False
        if "誘えば" in unit.ja and (
            re.search(r"诱我|誘惑我|邀请我", candidate)
            or not re.search(r"邀请|约|叫上", candidate)
        ):
            return False
        if re.search(r"ズルイ|ずるい", unit.ja) and not re.search(
            r"不公平|狡猾|耍赖|赖皮|作弊", candidate
        ):
            return False
        if re.search(r"\d+つ上", unit.ja) and not re.search(
            r"岁|年长|年龄|大(?:一|二|两|三|四|五|六|七|八|九|十|\d)", candidate
        ):
            return False
        source_is_emphatic = bool(
            re.search(
                r"[!！っッ]|(?:すごく|とても|めっちゃ|超|大好き|一番)",
                unit.ja,
            )
        )
        if not source_is_emphatic and re.search(r"超爱|最爱|爱死|！|!", candidate):
            return False
        if re.match(r"^[你我他她它]，", candidate.strip()):
            return False
        return True

    def residual_quality_ids(
        self, pages: list[PageOCR], glossary: dict[str, str]
    ) -> set[str]:
        """Find cross-unit hallucinations and candidates rejected by hard gates."""
        units = [
            unit
            for page in pages
            for unit in page.units
            if not unit.skip and not unit.is_sfx
        ]
        flagged = {
            unit.id
            for unit in units
            if not self._candidate_is_acceptable(unit, unit.zh, glossary)
        }
        phrase_units: dict[str, set[str]] = {}
        for unit in units:
            compact = re.sub(r"[^\u3400-\u9fff]", "", unit.zh or "")
            for size in range(6, min(10, len(compact)) + 1):
                for start in range(0, len(compact) - size + 1):
                    phrase_units.setdefault(compact[start : start + size], set()).add(
                        unit.id
                    )
        for ids in phrase_units.values():
            if len(ids) >= 4:
                flagged.update(ids)
        for unit in units:
            if re.search(r"(?:别|不要|不能|不可).{0,12}不行", unit.zh, re.S):
                flagged.add(unit.id)
        return flagged

    def apply_deterministic_fallbacks(
        self, pages: list[PageOCR], glossary: dict[str, str]
    ) -> set[str]:
        """Repair rejected short vocalizations without another model call."""
        changed: set[str] = set()
        for page in pages:
            for unit in page.units:
                if (
                    unit.skip
                    or unit.is_sfx
                    or self._candidate_is_acceptable(unit, unit.zh, glossary)
                ):
                    continue
                fallback = self._normalize_translation(
                    calm_preference_fallback(unit.ja) or interjection_fallback(unit.ja)
                )
                if self._candidate_is_acceptable(unit, fallback, glossary):
                    unit.translation_attempts.append(fallback)
                    unit.zh = fallback
                    changed.add(unit.id)
        return changed

    @staticmethod
    def _detect_translation_register(pages: Iterable[PageOCR]) -> str:
        adult_hits = sum(
            1
            for page in pages
            for unit in page.units
            if ADULT_REGISTER_RE.search(unit.ja)
        )
        if adult_hits >= 2:
            return (
                "自然、地道、简洁的成人漫画口语；忠实处理性语境中的双关和委婉语，"
                "不把高潮、射精等语义误译成崩溃、结束或普通动作"
            )
        return "自然、地道、简洁的漫画口语"

    @staticmethod
    def _recurring_katakana_names(pages: Iterable[PageOCR]) -> list[str]:
        page_hits: dict[str, set[int]] = {}
        honorific_hits: Counter[str] = Counter()
        contextual_hits: Counter[str] = Counter()
        particle_pattern = re.compile(
            r"^(?:は|が|を|に|の|と|も|へ|から|って|、|。|！|？)"
        )
        for page in pages:
            for unit in page.units:
                if unit.is_sfx:
                    continue
                honorifics = {
                    match.group(1) for match in KATAKANA_NAME_RE.finditer(unit.ja)
                }
                honorific_hits.update(honorifics)
                for match in KATAKANA_TOKEN_RE.finditer(unit.ja):
                    name = match.group(0)
                    suffix = unit.ja[match.end() :]
                    if name in honorifics or particle_pattern.match(suffix):
                        page_hits.setdefault(name, set()).add(page.page)
                        contextual_hits[name] += 1
        return sorted(
            name
            for name, hits in page_hits.items()
            if honorific_hits[name] >= 2
            or (len(hits) >= 3 and contextual_hits[name] >= 3)
        )

    def _resolve_glossary(
        self, pages: list[PageOCR], glossary: dict[str, str]
    ) -> dict[str, str]:
        names = self._recurring_katakana_names(pages)
        missing_names = [
            name for name in names if not any(name in source for source in glossary)
        ]
        inferred = {
            source: target for source, target in glossary.items() if source in names
        }
        inferred.update(self._auto_resolve_names(pages, missing_names))
        builtins = {
            source: target
            for source, target in GENERAL_TERMS.items()
            if any(source in unit.ja for page in pages for unit in page.units)
        }
        if "成人漫画" in self.translation_register:
            builtins.update(
                {
                    source: target
                    for source, target in ADULT_TERMS.items()
                    if any(source in unit.ja for page in pages for unit in page.units)
                }
            )
        self.inferred_glossary = dict(inferred)
        return {**builtins, **inferred, **glossary}

    def _restore_named_dialogue_roles(self, pages: list[PageOCR]) -> set[str]:
        """Recover speech that an SFX classifier swallowed despite a known name."""
        names = set(self.inferred_glossary)
        restored: set[str] = set()
        if not names:
            return restored
        for page in pages:
            for unit in page.units:
                if (
                    unit.is_sfx
                    and unit.special != "ocr_duplicate"
                    and any(name in unit.ja for name in names)
                ):
                    unit.is_sfx = False
                    restored.add(unit.id)
        return restored

    @staticmethod
    def _name_context(pages: list[PageOCR], name: str) -> list[str]:
        return [
            unit.ja
            for page in pages
            for unit in page.units
            if name in unit.ja and not unit.is_sfx
        ][:8]

    @staticmethod
    def _rank_name_candidates(
        candidates: list[NameCandidate], context: list[str]
    ) -> list[NameCandidate]:
        male_hint = any(
            re.search(rf"{re.escape(name)}(?:[．.・…\s]*)君", text)
            for text in context
            for name in KATAKANA_TOKEN_RE.findall(text)
        )

        def score(candidate: NameCandidate) -> tuple[int, int, int]:
            gender = (
                2
                if male_hint and candidate.is_male
                else 1
                if candidate.is_given_name
                else 0
            )
            preferred_length = 1 if male_hint else 2
            return (
                gender,
                -abs(len(candidate.written) - preferred_length),
                -sum(ord(char) > 0x9FA5 for char in candidate.written),
            )

        return sorted(
            (candidate for candidate in candidates if candidate.is_given_name),
            key=score,
            reverse=True,
        )[:32]

    def _choose_name_candidate(
        self, name: str, candidates: list[NameCandidate], context: list[str]
    ) -> str:
        ranked = self._rank_name_candidates(candidates, context)
        if not ranked:
            return ""
        options = " ".join(
            f"{index}.{candidate.written}"
            for index, candidate in enumerate(ranked, start=1)
        )
        prompt = f"""判断片假名【{name}】在上下文中是否是人物名；若不是人物名，只输出0。
若是人物名，为简体中文漫画选择最自然的日文汉字写法。优先让中文读者一眼看出是姓名，避免普通词歧义、罕见字、地名或生硬音译。
只能输出候选编号或0。候选：{options}
上下文：{"；".join(context)}
只输出候选前的一个数字，不要解释。"""
        raw = self._first_line(self._generate_repair(prompt, max_new_tokens=8))
        match = re.search(r"\d+", raw)
        selected = int(match.group()) if match else 0
        if selected == 0:
            return ""
        candidate = ranked[selected - 1] if 1 <= selected <= len(ranked) else ranked[0]
        return OpenCC("t2s").convert(candidate.written)

    def _fallback_name(self, name: str, context: list[str]) -> str:
        prompt = f"""判断片假名【{name}】在上下文中是否确实是人物名。普通名词、拟声词、品牌或动作只输出NONE。
若是人物名，自动规范为自然、固定的简体中文日漫姓名；不要机械外文音译，不要称谓、解释或标点，只输出 1 到 3 个汉字。
上下文：{"；".join(context)}"""
        candidate = self._first_line(self._generate_repair(prompt, max_new_tokens=24))
        candidate = re.sub(r"(?:君|先生|同学|小姐|女士)$", "", candidate)
        return candidate if re.fullmatch(r"[\u3400-\u9fff]{1,3}", candidate) else ""

    def _auto_resolve_names(
        self, pages: list[PageOCR], names: list[str]
    ) -> dict[str, str]:
        if not names:
            return {}
        candidates = self.name_dictionary.lookup(names) if self.name_dictionary else {}
        resolved: dict[str, str] = {}
        for name in names:
            context = self._name_context(pages, name)
            dictionary_candidates = candidates.get(name, [])
            has_name_address = any(
                re.search(
                    rf"{re.escape(name)}(?:[．.・…\s]*)"
                    r"(?:君|さん|ちゃん|様|先輩|先生)",
                    text,
                )
                for text in context
            )
            # Repetition followed by a Japanese particle also describes common
            # nouns (for example ゲームは/ゲームに). Without either a name
            # dictionary entry or direct address evidence, inventing a canon
            # name is less safe than leaving the contextual translator to read it.
            if not dictionary_candidates and not has_name_address:
                continue
            selected = (
                self._choose_name_candidate(name, dictionary_candidates, context)
                if dictionary_candidates
                else self._fallback_name(name, context)
            )
            if selected:
                resolved[name] = selected
        return resolved

    def translate_pages(
        self,
        pages: list[PageOCR],
        context_pages: int = 6,
        story_context: bool = True,
        preserve_sfx: bool = True,
        glossary: dict[str, str] | None = None,
        chunk_pages: int = 4,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        self.translation_register = self._detect_translation_register(pages)
        glossary = self._resolve_glossary(pages, glossary or {})
        self._restore_named_dialogue_roles(pages)
        self.resolved_glossary = dict(glossary)
        self.translation_target_ids = {
            unit.id
            for page in pages
            for unit in page.units
            if self._needs_translation(unit, preserve_sfx)
        }
        terms = (
            "；".join(f"{source}译为{target}" for source, target in glossary.items())
            or "无"
        )
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
            previous = (
                pages[max(0, start - context_pages) : start] if story_context else []
            )
            prompt = f"""【背景信息】
这是按页连续阅读的日语漫画。结合前文修正明显 OCR 错字，人物称呼和语气必须前后一致。
固定译名：{terms}
前文：
{self._page_text(previous) or "无"}

【任务】
逐条翻译为{self.target_language}，风格必须是：{self.translation_register}，适合直接放进漫画气泡。
以自然中文和故事连贯为先；只补足当前原文语法成立所必需的省略主语，不得添加原文没有的情绪、评价、动作或口头禅。上下文不得补写进当前行，相邻单元不要重复同一信息。
严格保留每行开头的[id]，一条输入对应一条输出，不合并、不遗漏。只输出[id]和译文。

{chr(10).join(source_rows)}"""
            translated: dict[str, str] = {}
            token_budget = min(1600, max(256, sum(len(unit.ja) for unit in units) * 3))
            for line in self._generate(
                prompt, max_new_tokens=token_budget
            ).splitlines():
                match = LINE_RE.match(line)
                if match:
                    translated[match.group(1)] = self._normalize_translation(
                        match.group(2)
                    )
            candidate_sources: dict[str, set[str]] = {}
            for unit in units:
                candidate = translated.get(unit.id, "").strip()
                candidate_sources.setdefault(candidate, set()).add(unit.ja)
            for unit in units:
                candidate = translated.get(unit.id, "")
                unit.translation_attempts.append(candidate)
                duplicate_context_leak = (
                    len(candidate) >= 4
                    and len(candidate_sources.get(candidate.strip(), set())) > 1
                )
                if (
                    self._candidate_is_acceptable(unit, candidate, glossary)
                    and not duplicate_context_leak
                ):
                    unit.zh = candidate
                else:
                    unit.zh = self._translate_one(
                        unit, previous + chunk, terms, glossary
                    )
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
        deterministic = self._normalize_translation(calm_preference_fallback(unit.ja))
        if self._candidate_is_acceptable(unit, deterministic, glossary):
            unit.translation_attempts.append(deterministic)
            return deterministic
        unit_terms = (
            "；".join(
                f"{source}译为{target}"
                for source, target in glossary.items()
                if source in unit.ja
            )
            or "无"
        )
        limit = max(12, min(120, len(unit.ja) * 4))
        context_text = self._page_text(context)[-1200:]
        calm_statement = unit.ja.rstrip().endswith(("。", "．")) and not re.search(
            r"[!！っッ]|すごく|とても|めっちゃ|超|大好き|一番", unit.ja
        )
        tone_constraint = (
            "原文是平静陈述；不得加入感叹号，也不得改成超爱、最爱、爱死等强化表达。"
            if calm_statement
            else "保持原文语气强度，不得凭上下文额外煽情。"
        )
        prompt = f"""只翻译【原文】为{self.target_language}，只输出一行译文。
译文风格：{self.translation_register}。
语气约束：{tone_constraint}
上下文仅供理解，绝对不要复述上下文。译文不得含日文假名，姓名也译成中文；最多 {limit} 个字符。
固定译名：{unit_terms}
上下文：{context_text}
【原文】{unit.ja}"""
        candidate = self._normalize_translation(
            self._first_line(
                self._generate(prompt, max_new_tokens=max(32, min(128, limit * 3)))
            )
        )
        unit.translation_attempts.append(candidate)
        if self._candidate_is_acceptable(unit, candidate, glossary):
            return candidate
        retry = f"""把【原文】翻译为{self.target_language}，风格必须是：{self.translation_register}。不得解释、不得复述上下文、不得含日文假名，姓名译成中文；只输出不超过 {limit} 个字符的一行译文。
语气约束：{tone_constraint}
固定译名：{unit_terms}
【原文】{unit.ja}"""
        retry_candidate = self._normalize_translation(
            self._first_line(
                self._generate(retry, max_new_tokens=max(32, min(96, limit * 3)))
            )
        )
        unit.translation_attempts.append(retry_candidate)
        if self._candidate_is_acceptable(unit, retry_candidate, glossary):
            return retry_candidate
        repair = f"""下面的翻译仍含日文假名、为空或格式不合格，请彻底改写。
只使用自然简洁的{self.target_language}、数字和必要标点；即使是人名、语气词、喘息声或拟声词，也不能原样保留任何日文假名。
语气约束：{tone_constraint}
不得解释，只输出不超过 {limit} 个字符的一行最终译文。
固定译名：{unit_terms}
【原文】{unit.ja}
【不合格输出】{retry_candidate or candidate or "空"}"""
        repaired = self._normalize_translation(
            self._first_line(
                self._generate_repair(
                    repair, max_new_tokens=max(32, min(96, limit * 3))
                )
            )
        )
        unit.translation_attempts.append(repaired)
        if self._candidate_is_acceptable(unit, repaired, glossary):
            return repaired
        fallback = interjection_fallback(unit.ja)
        if fallback:
            unit.translation_attempts.append(fallback)
        if self._candidate_is_acceptable(unit, fallback, glossary):
            return fallback
        return ""


class HyMTTranslator(PromptTranslator):
    def __init__(
        self,
        model_path: Path,
        target_language: str = "简体中文",
        device: str = "auto",
        name_dictionary: JapaneseNameDictionary | None = None,
    ):
        super().__init__(target_language, name_dictionary)
        self.model_path = model_path
        self.device = device
        self.torch = None
        self.tokenizer = None
        self.model = None
        self._kana_bad_words_ids: list[list[int]] | None = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ModelDependencyError(
                "Translation dependencies are missing. Run scripts/bootstrap with an ML profile."
            ) from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        kwargs = {"trust_remote_code": True}
        torch_device = pytorch_device(self.device)
        if torch_device == "auto":
            kwargs.update(dtype=torch.bfloat16, device_map="auto")
        else:
            kwargs.update(
                dtype=torch.float32 if torch_device == "cpu" else torch.bfloat16
            )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        if torch_device not in {"auto", "cpu"}:
            self.model.to(torch_device)
        self.model.eval()

    def _generate(self, prompt: str, max_new_tokens: int = 1600) -> str:
        self._ensure_loaded()
        return self._generate_with_constraints(prompt, max_new_tokens, sample=True)

    def _generate_repair(self, prompt: str, max_new_tokens: int) -> str:
        self._ensure_loaded()
        if self._kana_bad_words_ids is None:
            self._kana_bad_words_ids = kana_bad_words_ids(self.tokenizer)
        return self._generate_with_constraints(
            prompt,
            max_new_tokens,
            bad_words_ids=self._kana_bad_words_ids,
            sample=False,
        )

    def _generate_with_constraints(
        self,
        prompt: str,
        max_new_tokens: int,
        bad_words_ids: list[list[int]] | None = None,
        sample: bool = True,
    ) -> str:
        self._ensure_loaded()
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        if sample:
            seed = int.from_bytes(
                hashlib.sha256(prompt.encode("utf-8")).digest()[:8], "big"
            )
            self.torch.manual_seed(seed)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=bad_words_ids,
                **hy_mt_generation_options(repair=not sample),
            )
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self._kana_bad_words_ids = None
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class OllamaTranslator(PromptTranslator):
    def __init__(
        self,
        base_url: str,
        model: str,
        target_language: str = "简体中文",
        name_dictionary: JapaneseNameDictionary | None = None,
    ):
        super().__init__(target_language, name_dictionary)
        self.base_url = base_url
        self.model = model
        self._loaded = False

    def _generate(self, prompt: str, max_new_tokens: int = 1600) -> str:
        self._loaded = True
        payload = _json_request(
            _endpoint(self.base_url, "/api/chat"),
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                # Qwen3.5 exposes a reasoning channel by default.  Translation
                # needs the final lines only, and the abliterated build is used
                # specifically so adult or otherwise sensitive manga dialogue
                # is translated faithfully instead of softened or refused.
                "think": False,
                "options": {
                    "temperature": 0.2,
                    "num_ctx": 32_768,
                    "num_predict": max_new_tokens,
                },
            },
        )
        try:
            return payload["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise InferenceConnectionError(
                "Ollama 返回格式中缺少 message.content"
            ) from exc

    def unload(self) -> None:
        """Release this Ollama model between local pipeline stages."""
        if not self._loaded:
            return
        try:
            _json_request(
                _endpoint(self.base_url, "/api/chat"),
                {
                    "model": self.model,
                    "messages": [],
                    "stream": False,
                    "keep_alive": 0,
                },
            )
        except InferenceConnectionError:
            # Unloading is an optimization. A failed best-effort release must
            # not discard an otherwise valid translation draft.
            return
        finally:
            self._loaded = False

    def _generate_json(
        self,
        prompt: str,
        schema: dict,
        max_new_tokens: int = 4096,
        think: bool = False,
    ) -> dict:
        budgets = [max_new_tokens]
        # Reasoning tokens share ``num_predict`` with the visible JSON. A
        # single 2x retry still truncated real 9B reviews before any JSON was
        # emitted, so grow adaptively while keeping ordinary non-thinking
        # calls on their small fixed budget. This is independent from num_ctx.
        while think and budgets[-1] < 32_768:
            budgets.append(min(32_768, budgets[-1] * 2))
        payload = {}
        for token_budget in budgets:
            self._loaded = True
            payload = _json_request(
                _endpoint(self.base_url, "/api/chat"),
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": think,
                    "format": schema,
                    "options": {
                        "temperature": 0,
                        "num_ctx": 32_768,
                        "num_predict": token_budget,
                    },
                },
                timeout=600 if think else 180,
            )
            if payload.get("done_reason") != "length":
                break
        else:
            raise InferenceConnectionError(
                f"Ollama 结构化输出在 {budgets[-1]} token 上限下仍被截断"
            )
        try:
            return json.loads(payload["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise InferenceConnectionError("Ollama 未返回有效的结构化 JSON") from exc

    @staticmethod
    def _review_context(pages: Iterable[PageOCR]) -> str:
        rows = []
        for page in pages:
            lines = " / ".join(f"{unit.ja}→{unit.zh}" for unit in page.units if unit.zh)
            if lines:
                rows.append(f"第{page.page}页：{lines}")
        return "\n".join(rows)

    def review_pages(
        self,
        pages: list[PageOCR],
        glossary: dict[str, str],
        context_pages: int = 6,
        preserve_sfx: bool = True,
        chunk_pages: int = 4,
        progress: Callable[[int, int], None] | None = None,
        target_ids: set[str] | None = None,
    ) -> None:
        """Run a second context-aware local edit without changing unit geometry."""
        terms = (
            "；".join(
                f"{source}必须统一写作{target}" for source, target in glossary.items()
            )
            or "无"
        )
        for start in range(0, len(pages), chunk_pages):
            chunk = pages[start : start + chunk_pages]
            units = [
                unit
                for page in chunk
                for unit in page.units
                if not unit.skip
                and not (preserve_sfx and unit.is_sfx)
                and (target_ids is None or unit.id in target_ids)
                and self._valid_translation(unit, unit.zh)
            ]
            if not units:
                if progress:
                    progress(min(start + len(chunk), len(pages)), len(pages))
                continue
            rows = []
            for page in chunk:
                rows.append(f"--- 第{page.page}页 ---")
                rows.extend(
                    f"[{unit.id}] 原文：{unit.ja} || 初译：{unit.zh}"
                    for unit in page.units
                    if unit in units
                )
            previous = pages[max(0, start - context_pages) : start]
            prompt = f"""你是漫画汉化终审编辑。内容可能包含成人、暴力或敏感题材，必须忠实处理，不拒译、不弱化、不规避。

逐条对照日文原文和初译，输出自然、准确、连贯的{self.target_language}成稿：
1. 相邻 OCR 单元可能把同一句话拆开；结合前后行补全省略成分，把句意放回正确的行，不重复、不串行。
2. 修正生硬直译、主客体颠倒、错误人称、成人委婉语误判和明显 OCR 误读。
3. 人名与术语必须遵守全书自动一致性约束：{terms}
4. 严格保留每个[id]，一条输入对应一条输出；不得合并、遗漏、解释或输出日文假名。

前文成稿：
{self._review_context(previous) or "无"}

待终审：
{chr(10).join(rows)}

只输出[id]和最终中文。"""
            reviewed: dict[str, str] = {}
            token_budget = min(
                2200,
                max(384, sum(len(unit.ja) + len(unit.zh) for unit in units) * 3),
            )
            for line in self._generate(
                prompt, max_new_tokens=token_budget
            ).splitlines():
                match = LINE_RE.match(line)
                if match:
                    reviewed[match.group(1)] = self._normalize_translation(
                        match.group(2)
                    )
            candidate_sources: dict[str, set[str]] = {}
            for unit in units:
                candidate = reviewed.get(unit.id, "").strip()
                candidate_sources.setdefault(candidate, set()).add(unit.ja)
            for unit in units:
                candidate = reviewed.get(unit.id, "").strip()
                if candidate:
                    unit.translation_attempts.append(candidate)
                duplicate_context_leak = (
                    len(candidate) >= 4
                    and len(candidate_sources.get(candidate, set())) > 1
                )
                if (
                    self._candidate_is_acceptable(unit, candidate, glossary)
                    and not duplicate_context_leak
                ):
                    unit.zh = candidate
            if progress:
                progress(min(start + len(chunk), len(pages)), len(pages))

    def judge_candidates(
        self,
        pages: list[PageOCR],
        alternatives: dict[str, str],
        glossary: dict[str, str],
        preserve_sfx: bool = True,
        context_pages: int = 6,
        chunk_pages: int = 4,
        progress: Callable[[int, int], None] | None = None,
        deep_reasoning: bool = False,
    ) -> None:
        """Resolve independent translation disagreements with constrained JSON.

        This is deliberately contrastive instead of a free-form self-review:
        the model sees the Japanese source and two independently produced
        candidates, then chooses one or supplies a bounded correction.
        """
        del context_pages
        eligible = [
            unit
            for page in pages
            for unit in page.units
            if not unit.skip
            and not (preserve_sfx and unit.is_sfx)
            and self._candidate_is_acceptable(unit, unit.zh, glossary)
            and self._candidate_is_acceptable(
                unit, alternatives.get(unit.id, ""), glossary
            )
            and unit.zh.strip() != alternatives.get(unit.id, "").strip()
        ]
        if not eligible:
            if progress:
                progress(len(pages), len(pages))
            return
        # Judging no longer uses page context, so sparse risk lines are packed
        # together instead of paying one model call per four source pages.
        batch_size = max(8, min(16, chunk_pages * 3))
        for start in range(0, len(eligible), batch_size):
            units = eligible[start : start + batch_size]
            rows = "\n".join(
                f"[{unit.id}] 日文：{unit.ja} || A：{unit.zh} || B：{alternatives[unit.id]}"
                for unit in units
            )
            prompt = f"""你是日中漫画翻译裁决器。逐条对照日文，在A、B中选语义准确的一个；两者都错才重写。必须显式检查：比较关系方向、否定作用域、动作是禁止发生还是已经发生、主客体、因果转折。只返回JSON。

{rows}"""
            ids = [unit.id for unit in units]
            schema = {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "enum": ids},
                                "choice": {
                                    "type": "string",
                                    "enum": ["A", "B", "rewrite"],
                                },
                                "final": {"type": "string"},
                            },
                            "required": ["id", "choice", "final"],
                        },
                    }
                },
                "required": ["items"],
            }
            token_budget = (
                8_192
                if deep_reasoning
                else min(
                    2048,
                    max(
                        384,
                        256
                        + sum(
                            len(unit.ja) + len(unit.zh) + len(alternatives[unit.id])
                            for unit in units
                        )
                        * 8,
                    ),
                )
            )
            payload = self._generate_json(
                prompt, schema, token_budget, think=deep_reasoning
            )
            decisions = {
                str(item.get("id")): item
                for item in payload.get("items", [])
                if isinstance(item, dict) and item.get("id") in ids
            }
            for unit in units:
                decision = decisions.get(unit.id, {})
                choice = decision.get("choice")
                if choice == "A":
                    candidate = unit.zh
                elif choice == "B":
                    candidate = alternatives[unit.id]
                else:
                    candidate = str(decision.get("final", "")).strip()
                candidate = self._normalize_translation(candidate)
                if candidate:
                    unit.translation_attempts.append(candidate)
                if self._candidate_is_acceptable(unit, candidate, glossary):
                    unit.zh = candidate
            if progress:
                judged = min(start + len(units), len(eligible))
                progress(round(judged / len(eligible) * len(pages)), len(pages))

    def retranslate_risk_units(
        self,
        pages: list[PageOCR],
        target_ids: set[str],
        glossary: dict[str, str],
        preserve_sfx: bool = True,
        progress: Callable[[int, int], None] | None = None,
        forced_ids: set[str] | None = None,
    ) -> None:
        """Independently retranslate semantic-risk lines with deep reasoning."""
        forced_ids = forced_ids or set()
        eligible = [
            (page, unit)
            for page in pages
            for unit in page.units
            if unit.id in target_ids
            and not unit.skip
            and not (preserve_sfx and unit.is_sfx)
            and (
                unit.id in forced_ids
                or not self._candidate_is_acceptable(unit, unit.zh, glossary)
                or (len(unit.ja) >= 7 and SEMANTIC_RISK_RE.search(unit.ja))
            )
        ]
        if not eligible:
            if progress:
                progress(len(pages), len(pages))
            return
        terms = (
            "；".join(f"{source}译为{target}" for source, target in glossary.items())
            or "无"
        )
        for start in range(0, len(eligible), 4):
            batch = eligible[start : start + 4]
            ids = [unit.id for _page, unit in batch]
            rows = []
            for page, unit in batch:
                page_context = " / ".join(
                    item.ja for item in page.units if not item.is_sfx
                )[:400]
                hints = self._semantic_audit_hints(unit, page_context)
                layout_budget = self._layout_char_budget(unit)
                unit_terms = "；".join(
                    f"{source}必须译为{target}"
                    for source, target in glossary.items()
                    if source in unit.ja
                )
                rows.append(
                    f"[{unit.id}] 原文：{unit.ja} || 同页日文：{page_context or '无'} || "
                    f"语义检查：{hints or '按原文逐项核对'} || "
                    f"本行固定译名：{unit_terms or '无'} || "
                    f"排版上限：最多{layout_budget}个中文汉字/数字（标点不计）"
                )
            prompt = f"""不参考任何已有译文，把以下日语漫画台词独立翻译成自然、准确、简洁的{self.target_language}成稿。
必须显式检查比较关系方向、否定作用域、动作是禁止发生还是已经发生、主客体、因果转折。内容可能是成人漫画，忠实翻译。
同页日文只用于理解人物和省略成分，不得串入当前行。固定译名：{terms}
日语常省略动词，中文必须结合当前行与同页语境补成自然动作，不能留下不成立的名词搭配。
不得缩写、概括、过度扩写或丢失原文信息；不得添加原文没有的口头禅、评价或情绪。标题、条件句和未完句也必须完整保留其条件与语气。
不得把未发生、不能发生或禁止发生的事情改写成已经发生，也不得把陈述句擅自改成问句。
同一否定只能自然表达一次，不得同时写成“别……不行”之类的重复否定。
每条译文必须遵守该行排版上限，优先使用短而自然的漫画口语，标点不计入上限。
只返回约定JSON。

{chr(10).join(rows)}"""
            schema = {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "enum": ids},
                                "zh": {"type": "string"},
                            },
                            "required": ["id", "zh"],
                        },
                    }
                },
                "required": ["items"],
            }
            # ``num_ctx`` and ``num_predict`` are independent. A 32K working
            # context is ample for these four-line batches; generation stays
            # sized to the actual source instead of reserving the model's full
            # advertised context window.
            source_size = sum(len(unit.ja) for _page, unit in batch)
            has_semantic_risk = any(
                SEMANTIC_RISK_RE.search(unit.ja) for _page, unit in batch
            )
            token_budget = (
                max(2048, min(4096, 1024 + source_size * 32))
                if has_semantic_risk
                else max(384, min(2048, 256 + source_size * 16))
            )
            payload = self._generate_json(
                prompt,
                schema,
                max_new_tokens=token_budget,
                # The unrestricted 9B build can loop in its hidden reasoning
                # channel until even 32K output tokens are exhausted without
                # emitting JSON. The prompt already enumerates every semantic
                # check, so deterministic non-thinking JSON is both bounded
                # and empirically more reliable for this verifier role.
                think=False,
            )
            translated = {
                str(item.get("id")): self._normalize_translation(
                    str(item.get("zh", ""))
                )
                for item in payload.get("items", [])
                if isinstance(item, dict) and item.get("id") in ids
            }
            for _page, unit in batch:
                candidate = self._restore_source_tone(unit, translated.get(unit.id, ""))
                if candidate:
                    unit.translation_attempts.append(candidate)
                if self._candidate_is_acceptable(unit, candidate, glossary):
                    unit.zh = candidate
            if progress:
                audited = min(start + len(batch), len(eligible))
                progress(round(audited / len(eligible) * len(pages)), len(pages))

    @staticmethod
    def _semantic_audit_hints(unit: TextUnit, page_context: str) -> str:
        """Return general Japanese translation hazards present in one source line."""
        source = unit.ja
        hints: list[str] = []
        if re.search(r"\d+つ(?:上|下)", source):
            hints.append("人物的数字+つ上/下是年龄差，译为大/小几岁，不是高/低几级")
        if "無理して" in source:
            hints.append("無理して表示勉强自己、硬撑或咬牙，不是“特意”")
        if re.search(r"まで.*(?:だめ|駄目)", source, re.S):
            hints.append("まで与だめ共同限定禁止达到的边界，不表示动作已经完成")
        if "入り口だけ" in source:
            hints.append("だけ限制动作深度；中文要补出动作，不能直译成“留下入口”")
        if "力抜いて" in source:
            hints.append("力を抜く在此表示放松身体，不是收回力气")
        if "乗せ" in source and re.search(r"バイク|自転車|車", page_context):
            hints.append("交通工具语境的乗せる表示载人、让人搭乘或兜风")
        if re.search(r"こんな.+に.+され", source, re.S):
            hints.append("被动句必须同时保留施事者和所受动作，不得缩成只剩人物")
        if re.search(r"(?:のかよ|なのか)[!！…．.]*$", source.strip()):
            hints.append("句末のかよ/なのか是反问或质问，中文必须保留疑问语气")
        if re.search(r"(彼氏|彼女).{0,10}(?:いる|いんだろ|いるんだろ)", source, re.S):
            hints.append("彼氏/彼女＋いる是在问对方有没有恋人，不是说第三者本身是恋人")
        if "笑" in source:
            hints.append("笑う/笑ってる的笑这一动作必须保留")
        if "ゲーム" in source:
            hints.append("ゲーム必须译出玩游戏这一动作，不能只保留条件从句")
        if "って" in source and not re.search(r"言|話|喋", source):
            hints.append("句末って可能是强调语气；没有说话动词时不得凭空译成‘说你’")
        if "誘えば" in source:
            hints.append(
                "人名＋誘えば表示邀请/约/叫上该人，不是中文的诱惑，也不是邀请我"
            )
        if re.search(r"ズルイ|ずるい", source):
            hints.append("ズルイ按语境译为不公平、狡猾、耍赖、赖皮或作弊，不是加塞")
        if re.search(r"処女マン|マンコ", source):
            hints.append("成人语境的マン/マンコ指女性下体或小穴，不是处女膜")
        if re.search(r"妊娠.*(?:無理|できない)", source, re.S):
            hints.append("保留不能/不可能怀孕的情态，不得改成已经怀孕或疑问")
        if source.rstrip().endswith("："):
            hints.append("竖排OCR常把省略号识别为冒号；中文不得保留句尾孤立冒号")
        if source.rstrip().endswith(("。", "．")) and not re.search(
            r"[!！っッ]|すごく|とても|めっちゃ|超|大好き|一番", source
        ):
            hints.append("原文是平静陈述，不得擅自加入感叹号、超爱/最爱等强化语气")
        return "；".join(hints)

    @staticmethod
    def _layout_char_budget(unit: TextUnit) -> int:
        """Estimate a Chinese glyph budget from source density, without page rules."""
        source = re.sub(r"[\s．。…、，,!?！？:：♡♥〰〜～]", "", unit.ja)
        return max(6, round(len(source) * 1.1))


class LocalQualityTranslator(PromptTranslator):
    """Local ensemble using one unrestricted 9B model in staged roles."""

    def __init__(
        self,
        editor: OllamaTranslator,
        candidate: PromptTranslator,
        auditor: OllamaTranslator | None = None,
    ):
        super().__init__(editor.target_language, editor.name_dictionary)
        self.editor = editor
        self.candidate = candidate
        # Reusing the editor after its explicit unload keeps the local model
        # ceiling at 9B. Hy-MT2 still supplies an independent translation
        # candidate, while the second 9B pass receives a different comparison
        # and semantic-risk prompt.
        self.auditor = auditor or editor

    def translate_pages(
        self,
        pages: list[PageOCR],
        context_pages: int = 6,
        story_context: bool = True,
        preserve_sfx: bool = True,
        glossary: dict[str, str] | None = None,
        chunk_pages: int = 4,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        initial_target_ids = {
            unit.id
            for page in pages
            for unit in page.units
            if self._needs_translation(unit, preserve_sfx)
        }
        page_total = max(1, len(pages))

        def staged(start: float, end: float):
            if progress is None:
                return None

            def callback(current: int, total: int) -> None:
                ratio = current / max(1, total)
                progress(
                    round((start + (end - start) * ratio) * page_total), page_total
                )

            return callback

        try:
            self.editor.translate_pages(
                pages,
                context_pages,
                story_context,
                preserve_sfx,
                glossary,
                chunk_pages,
                staged(0.0, 0.35),
            )
        finally:
            self.editor.unload()
        target_ids = initial_target_ids | self.editor.translation_target_ids
        self.resolved_glossary = dict(self.editor.resolved_glossary)
        self.inferred_glossary = dict(self.editor.inferred_glossary)
        self.translation_register = self.editor.translation_register

        candidate_pages = deepcopy(pages)
        for page in candidate_pages:
            for unit in page.units:
                if unit.id in target_ids:
                    unit.zh = ""
                    unit.translation_attempts = []
        try:
            self.candidate.translate_pages(
                candidate_pages,
                context_pages,
                story_context,
                preserve_sfx,
                self.resolved_glossary,
                chunk_pages,
                staged(0.35, 0.55),
            )
            alternatives = {
                unit.id: unit.zh for page in candidate_pages for unit in page.units
            }
        finally:
            self.candidate.unload()
        # A valid independent candidate must be allowed to recover an editor
        # omission before contrastive judging. Previously the judge required
        # two valid inputs, so a good B candidate was stranded when A was empty.
        for page in pages:
            for unit in page.units:
                alternative = alternatives.get(unit.id, "")
                if (
                    unit.id in target_ids
                    and not self._candidate_is_acceptable(
                        unit, unit.zh, self.resolved_glossary
                    )
                    and self._candidate_is_acceptable(
                        unit, alternative, self.resolved_glossary
                    )
                ):
                    unit.translation_attempts.append(alternative)
                    unit.zh = alternative
        self.auditor.inferred_glossary = dict(self.inferred_glossary)
        try:
            self.auditor.review_pages(
                pages,
                self.resolved_glossary,
                context_pages,
                preserve_sfx,
                chunk_pages,
                staged(0.55, 0.75),
                target_ids,
            )
            # Low-risk lines do not benefit from another generation after either
            # the 9B editor or independent MT candidate passes every hard gate.
            # Reserve the staged 9B review pass for true semantic hazards or for
            # lines where both local candidates remain invalid.
            judge_ids = {
                unit.id
                for page in pages
                for unit in page.units
                if unit.id in target_ids
                and SEMANTIC_RISK_RE.search(unit.ja)
                and self._candidate_is_acceptable(unit, unit.zh, self.resolved_glossary)
                and self._candidate_is_acceptable(
                    unit, alternatives.get(unit.id, ""), self.resolved_glossary
                )
            }
            self.auditor.judge_candidates(
                pages,
                {
                    unit_id: candidate
                    for unit_id, candidate in alternatives.items()
                    if unit_id in judge_ids
                },
                self.resolved_glossary,
                preserve_sfx,
                context_pages,
                chunk_pages,
                staged(0.75, 0.85),
                # A/B selection must stay practical. The following risk pass
                # retains deeper reasoning only for negation, comparison,
                # action scope, and unresolved units.
                deep_reasoning=False,
            )
            self.auditor.retranslate_risk_units(
                pages,
                target_ids,
                self.resolved_glossary,
                preserve_sfx,
                staged(0.85, 1.0),
            )
            for _attempt in range(3):
                self.auditor.apply_deterministic_fallbacks(
                    pages, self.resolved_glossary
                )
                residual_ids = (
                    self.auditor.residual_quality_ids(pages, self.resolved_glossary)
                    & target_ids
                )
                if not residual_ids:
                    break
                self.auditor.retranslate_risk_units(
                    pages,
                    residual_ids,
                    self.resolved_glossary,
                    preserve_sfx,
                    forced_ids=residual_ids,
                )
            if progress:
                progress(page_total, page_total)
        finally:
            self.auditor.unload()


class OpenAICompatibleTranslator(PromptTranslator):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        target_language: str = "简体中文",
        name_dictionary: JapaneseNameDictionary | None = None,
    ):
        super().__init__(target_language, name_dictionary)
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
            raise InferenceConnectionError(
                "兼容 API 返回格式中缺少 choices[0].message.content"
            ) from exc


def available_remote_models(
    backend: str, base_url: str, api_key: str = ""
) -> list[str]:
    if backend == "ollama":
        payload = _json_request(_endpoint(base_url, "/api/tags"), timeout=15)
        return [item["name"] for item in payload.get("models", []) if item.get("name")]
    payload = _json_request(_endpoint(base_url, "/models"), api_key=api_key, timeout=15)
    return [item["id"] for item in payload.get("data", []) if item.get("id")]


def validate_local_model_size(model: str) -> None:
    """Reject explicitly sized local models above the project's 9B ceiling."""
    sizes = [
        float(value)
        for value in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[bB](?!\w)", model)
    ]
    if sizes and max(sizes) > LOCAL_MODEL_MAX_BILLIONS:
        raise ValueError(
            f"本地模式最多支持 9B 模型，当前配置为 {model}；"
            "请改用 9B 或更小的解除限制模型"
        )


def ensure_ollama_model(base_url: str, model: str) -> bool:
    """Ensure a selected local Ollama model exists, pulling it when absent.

    The Ollama service remains the common local runtime; this function only
    automates weight acquisition and never falls back to a cloud API.
    """
    validate_local_model_size(model)
    if model in available_remote_models("ollama", base_url):
        return False
    _json_request(
        _endpoint(base_url, "/api/pull"),
        {"model": model, "stream": False},
        timeout=7200,
    )
    if model not in available_remote_models("ollama", base_url):
        raise InferenceConnectionError(f"Ollama 已完成下载请求，但仍未找到模型 {model}")
    return True
