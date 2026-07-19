from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SfxMeaning:
    kind: str
    chinese: str


_PUNCTUATION_RE = re.compile(r"[\s．。…、，,!?！？:：♡♥〰〜～・]+")


def _compact(source: str) -> str:
    return _PUNCTUATION_RE.sub("", source).replace("ッ", "ッ")


def _emphasis(source: str) -> str:
    if re.search(r"[!！]|ッ\s*$", source.strip()):
        return "！"
    if re.search(r"[…．。]{2,}", source):
        return "……"
    return ""


def lookup_sfx(source: str) -> SfxMeaning | None:
    """Return a conservative, language-level manga SFX interpretation.

    These rules cover productive Japanese sound families instead of book- or
    character-specific strings.  They provide a semantic prior for the local
    model and a deterministic answer only where the Chinese convention is
    unambiguous enough to beat unconstrained phonetic generation.
    """

    compact = _compact(source)
    if not compact:
        return None

    heartbeat_source = compact.rstrip("ッ")
    heartbeat_hits = re.findall(r"(?:ドキン?|ドクン?|バクン?)", heartbeat_source)
    if heartbeat_hits and "".join(heartbeat_hits) == heartbeat_source:
        repeated = "".join("怦咚" for _ in range(min(4, len(heartbeat_hits))))
        return SfxMeaning("heartbeat", repeated + _emphasis(source))
    if compact.count("バ") >= 2 and "バクン" in compact:
        repeated = "".join("怦咚" for _ in range(min(4, compact.count("バ"))))
        return SfxMeaning("heartbeat", repeated + _emphasis(source))

    if re.fullmatch(r"[ドトッ]{4,}", compact):
        return SfxMeaning("rumble", "轰隆隆" + _emphasis(source))
    if re.fullmatch(r"ゴ{3,}", compact):
        return SfxMeaning("rumble", "隆隆隆" + _emphasis(source))
    if re.fullmatch(r"(?:ブロ+ン?|ブル+ン?)+", compact):
        return SfxMeaning("engine", "轰隆" + _emphasis(source))

    if re.fullmatch(r"キャッ+", compact):
        return SfxMeaning("vocalization", "呀！")
    if re.fullmatch(r"ギャッ+", compact):
        return SfxMeaning("vocalization", "哇啊！")
    if re.fullmatch(r"ヒッ+", compact):
        return SfxMeaning("vocalization", "咿！")

    exact = {
        "ゴッ": SfxMeaning("impact", "咚" + _emphasis(source)),
        "ドン": SfxMeaning("impact", "咚" + _emphasis(source)),
        "ドーン": SfxMeaning("impact", "轰" + _emphasis(source)),
        "バン": SfxMeaning("impact", "砰" + _emphasis(source)),
        "バタン": SfxMeaning("impact", "砰" + _emphasis(source)),
        "ガタン": SfxMeaning("impact", "哐当" + _emphasis(source)),
        "ガン": SfxMeaning("impact", "铛" + _emphasis(source)),
        "カチャ": SfxMeaning("mechanical", "咔哒" + _emphasis(source)),
        "カチ": SfxMeaning("mechanical", "咔" + _emphasis(source)),
        "カチッ": SfxMeaning("mechanical", "咔哒！"),
        "パチ": SfxMeaning("snap", "啪" + _emphasis(source)),
        "パン": SfxMeaning("snap", "啪" + _emphasis(source)),
        "ズル": SfxMeaning("movement", "滋溜" + _emphasis(source)),
        "ズズ": SfxMeaning("movement", "簌簌" + _emphasis(source)),
        "ゴクン": SfxMeaning("swallow", "咕咚" + _emphasis(source)),
        "ゴクリ": SfxMeaning("swallow", "咕咚" + _emphasis(source)),
        "ペロ": SfxMeaning("lick", "舔" + _emphasis(source)),
        "クチュ": SfxMeaning("wet", "咕啾" + _emphasis(source)),
        "グチュ": SfxMeaning("wet", "咕啾" + _emphasis(source)),
    }
    if compact in exact:
        return exact[compact]

    if re.fullmatch(r"(?:パン){2,}", compact):
        return SfxMeaning("repeated_impact", "啪啪" + _emphasis(source))
    if re.fullmatch(r"(?:カチ){2,}", compact):
        return SfxMeaning("mechanical", "咔哒咔哒" + _emphasis(source))
    return None
