from __future__ import annotations

import gzip
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


JMNEDICT_URL = "https://www.edrdg.org/pub/Nihongo/JMnedict.xml.gz"


@dataclass(frozen=True)
class NameCandidate:
    written: str
    types: tuple[str, ...]

    @property
    def is_given_name(self) -> bool:
        return any("given name" in item or "forename" in item for item in self.types)

    @property
    def is_male(self) -> bool:
        return any("male given" in item for item in self.types)

    @property
    def is_female(self) -> bool:
        return any("female given" in item for item in self.types)


def katakana_to_hiragana(text: str) -> str:
    return "".join(
        chr(ord(character) - 0x60) if "ァ" <= character <= "ヶ" else character
        for character in text
    )


class JapaneseNameDictionary:
    """On-demand JMnedict reader used for automatic manga name canonization.

    The dictionary is data, not a model weight.  It is downloaded once to the
    application cache, then all lookups are fully local.  JMnedict is kept out
    of the MIT wheel and retains its own CC BY-SA attribution.
    """

    def __init__(self, cache_dir: Path, url: str = JMNEDICT_URL):
        self.cache_dir = cache_dir
        self.url = url
        self.path = cache_dir / "JMnedict.xml.gz"

    def ensure(self) -> Path:
        if self.path.is_file() and self.path.stat().st_size > 1_000_000:
            return self.path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        partial = self.cache_dir / "JMnedict.xml.gz.part"
        request = Request(
            self.url, headers={"User-Agent": "Manga-Localizer-Studio/0.5"}
        )
        try:
            with (
                urlopen(request, timeout=120) as response,
                partial.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
        except (OSError, URLError) as exc:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"无法自动下载日文姓名词典：{exc}") from exc
        if partial.stat().st_size <= 1_000_000:
            partial.unlink(missing_ok=True)
            raise RuntimeError("日文姓名词典下载不完整")
        partial.replace(self.path)
        return self.path

    def lookup(self, names: list[str]) -> dict[str, list[NameCandidate]]:
        if not names:
            return {}
        source_by_reading = {katakana_to_hiragana(name): name for name in names}
        found: dict[str, list[NameCandidate]] = {name: [] for name in names}
        seen: dict[str, set[str]] = {name: set() for name in names}
        with gzip.open(self.ensure(), "rb") as stream:
            for _event, entry in ET.iterparse(stream, events=("end",)):
                if entry.tag != "entry":
                    continue
                readings = {
                    item.text for item in entry.findall("./r_ele/reb") if item.text
                }
                matched = readings.intersection(source_by_reading)
                if matched:
                    written = [
                        item.text for item in entry.findall("./k_ele/keb") if item.text
                    ]
                    types = tuple(
                        item.text
                        for item in entry.findall("./trans/name_type")
                        if item.text
                    )
                    for reading in matched:
                        source = source_by_reading[reading]
                        for candidate in written:
                            item = NameCandidate(candidate, types)
                            if (
                                candidate not in seen[source]
                                and item.is_given_name
                                and 1 <= len(candidate) <= 3
                                and all(
                                    "\u3400" <= char <= "\u9fff" for char in candidate
                                )
                            ):
                                seen[source].add(candidate)
                                found[source].append(item)
                entry.clear()
        return found
