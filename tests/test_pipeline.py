import json

import pytest
from PIL import Image

from manga_localizer.pipeline import (
    OCR_CACHE_VERSION,
    LocalizerPipeline,
    completion_summary,
    page_from_dict,
    unsafe_semantic_missing,
)


def test_legacy_skip_is_imported_as_unresolved_instead_of_completed():
    page = page_from_dict(
        {
            "page": 1,
            "file": "001.jpg",
            "width": 100,
            "height": 200,
            "units": [
                {
                    "id": "p001u01",
                    "bbox": [1, 2, 30, 40],
                    "ja": "秘密基地",
                    "zh": "秘密基地",
                    "paddle_scores": [0.9],
                },
                {
                    "id": "p001u02",
                    "bbox": [40, 2, 70, 40],
                    "ja": "ノイズ",
                    "zh": "噪点",
                    "skip": True,
                },
            ],
        }
    )
    assert page.units[0].score == 0.9
    assert page.units[0].crop_bbox == page.units[0].bbox
    assert page.units[1].zh == "噪点"
    assert page.units[1].skip is True
    assert page.units[1].skip_reason == "unresolved"
    assert completion_summary([page])["unresolved_ids"] == ["p001u02"]


def test_explicit_skip_reason_is_counted_as_reviewed():
    page = page_from_dict(
        {
            "page": 1,
            "file": "001.jpg",
            "width": 100,
            "height": 200,
            "units": [
                {
                    "id": "p001u01",
                    "bbox": [1, 2, 30, 40],
                    "ja": "ノイズ",
                    "skip": True,
                    "skip_reason": "noise",
                }
            ],
        }
    )
    summary = completion_summary([page])
    assert summary["explicitly_skipped_units"] == 1
    assert summary["unresolved_units"] == 0


def test_reviewed_transcript_rejects_ambiguous_legacy_skip(tmp_path):
    source = tmp_path / "001.jpg"
    Image.new("RGB", (100, 200), "white").save(source)
    page = page_from_dict(
        {
            "page": 1,
            "file": source.name,
            "width": 100,
            "height": 200,
            "units": [
                {
                    "id": "p001u01",
                    "bbox": [1, 2, 30, 40],
                    "ja": "台詞",
                    "skip": True,
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="ambiguous skipped units"):
        LocalizerPipeline._validate_reviewed_pages([source], [page])


def test_japanese_left_in_translation_is_unresolved():
    page = page_from_dict(
        {
            "page": 1,
            "file": "001.jpg",
            "width": 100,
            "height": 200,
            "units": [
                {
                    "id": "p001u01",
                    "bbox": [1, 2, 30, 40],
                    "ja": "ニュプ",
                    "zh": "ニュプ",
                }
            ],
        }
    )
    summary = completion_summary([page])
    assert summary["unresolved_ids"] == ["p001u01"]
    assert summary["invalid_translation_ids"] == ["p001u01"]


def test_valid_cached_ocr_page_is_reused(tmp_path):
    image_path = tmp_path / "001.jpg"
    Image.new("RGB", (100, 200), "white").save(image_path)
    work = tmp_path / "work"
    work.mkdir()
    cached = {
        "page": 1,
        "file": image_path.name,
        "width": 100,
        "height": 200,
        "units": [{"id": "p001u01", "bbox": [1, 2, 30, 40], "ja": "秘密"}],
        "_cache": {
            "version": OCR_CACHE_VERSION,
            "ocr_backend": "builtin",
            "quality_profile": "quality",
            "source_size": image_path.stat().st_size,
            "source_mtime_ns": image_path.stat().st_mtime_ns,
        },
    }
    (work / "001.ocr.json").write_text(
        json.dumps(cached, ensure_ascii=False), encoding="utf-8"
    )
    page = LocalizerPipeline._load_cached_ocr(work, image_path, 1)
    assert page is not None
    assert page.units[0].ja == "秘密"


def test_cached_mixed_kana_sound_effect_gets_current_classification(tmp_path):
    image_path = tmp_path / "001.jpg"
    Image.new("RGB", (100, 200), "white").save(image_path)
    work = tmp_path / "work"
    work.mkdir()
    cached = {
        "page": 1,
        "file": image_path.name,
        "width": 100,
        "height": 200,
        "units": [
            {
                "id": "p001u01",
                "bbox": [1, 2, 30, 40],
                "ja": "そのカキノカキ",
                "is_sfx": False,
            }
        ],
        "_cache": {
            "version": OCR_CACHE_VERSION,
            "ocr_backend": "builtin",
            "quality_profile": "quality",
            "source_size": image_path.stat().st_size,
            "source_mtime_ns": image_path.stat().st_mtime_ns,
        },
    }
    (work / "001.ocr.json").write_text(
        json.dumps(cached, ensure_ascii=False), encoding="utf-8"
    )
    page = LocalizerPipeline._load_cached_ocr(work, image_path, 1)
    assert page is not None
    assert page.units[0].is_sfx is True


def test_stale_cached_ocr_page_is_rejected(tmp_path):
    image_path = tmp_path / "001.jpg"
    Image.new("RGB", (100, 200), "white").save(image_path)
    work = tmp_path / "work"
    work.mkdir()
    cached = {
        "page": 2,
        "file": image_path.name,
        "width": 100,
        "height": 200,
        "units": [],
    }
    (work / "001.ocr.json").write_text(json.dumps(cached), encoding="utf-8")
    assert LocalizerPipeline._load_cached_ocr(work, image_path, 1) is None


def test_legacy_cache_is_not_reused_for_a_different_ocr_profile(tmp_path):
    image_path = tmp_path / "001.jpg"
    Image.new("RGB", (100, 200), "white").save(image_path)
    work = tmp_path / "work"
    work.mkdir()
    cached = {
        "page": 1,
        "file": image_path.name,
        "width": 100,
        "height": 200,
        "units": [],
    }
    (work / "001.ocr.json").write_text(json.dumps(cached), encoding="utf-8")
    assert (
        LocalizerPipeline._load_cached_ocr(work, image_path, 1, "ollama", "quality")
        is None
    )


def test_fingerprinted_ocr_cache_matches_the_source_and_backend(tmp_path):
    image_path = tmp_path / "001.jpg"
    Image.new("RGB", (100, 200), "white").save(image_path)
    source_stat = image_path.stat()
    work = tmp_path / "work"
    work.mkdir()
    cached = {
        "page": 1,
        "file": image_path.name,
        "width": 100,
        "height": 200,
        "units": [],
        "_cache": {
            "version": OCR_CACHE_VERSION,
            "ocr_backend": "builtin",
            "quality_profile": "quality",
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
        },
    }
    cache_path = work / "001.ocr.json"
    cache_path.write_text(json.dumps(cached), encoding="utf-8")
    assert LocalizerPipeline._load_cached_ocr(work, image_path, 1) is not None

    cached["_cache"]["source_size"] += 1
    cache_path.write_text(json.dumps(cached), encoding="utf-8")
    assert LocalizerPipeline._load_cached_ocr(work, image_path, 1) is None


def test_old_fingerprinted_ocr_cache_version_is_rejected(tmp_path):
    image_path = tmp_path / "001.jpg"
    Image.new("RGB", (100, 200), "white").save(image_path)
    source_stat = image_path.stat()
    work = tmp_path / "work"
    work.mkdir()
    cached = {
        "page": 1,
        "file": image_path.name,
        "width": 100,
        "height": 200,
        "units": [],
        "_cache": {
            "version": OCR_CACHE_VERSION - 1,
            "ocr_backend": "builtin",
            "quality_profile": "quality",
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
        },
    }
    (work / "001.ocr.json").write_text(json.dumps(cached), encoding="utf-8")
    assert LocalizerPipeline._load_cached_ocr(work, image_path, 1) is None


def test_hybrid_cache_is_bound_to_the_exact_vision_model(tmp_path):
    image_path = tmp_path / "001.jpg"
    Image.new("RGB", (100, 200), "white").save(image_path)
    work = tmp_path / "work"
    work.mkdir()
    cached = {
        "page": 1,
        "file": image_path.name,
        "width": 100,
        "height": 200,
        "units": [],
        "_cache": {
            "version": OCR_CACHE_VERSION,
            "ocr_backend": "hybrid",
            "quality_profile": "quality",
            "ocr_model": "vision-a",
            "source_size": image_path.stat().st_size,
            "source_mtime_ns": image_path.stat().st_mtime_ns,
        },
    }
    (work / "001.ocr.json").write_text(json.dumps(cached), encoding="utf-8")
    assert (
        LocalizerPipeline._load_cached_ocr(
            work, image_path, 1, "hybrid", "quality", "vision-a"
        )
        is not None
    )
    assert (
        LocalizerPipeline._load_cached_ocr(
            work, image_path, 1, "hybrid", "quality", "vision-b"
        )
        is None
    )


def test_unboxed_dialogue_fails_but_preserved_sfx_is_safe():
    page = page_from_dict(
        {
            "page": 1,
            "file": "001.jpg",
            "width": 100,
            "height": 200,
            "units": [],
            "semantic_missing": [
                {"text": "台詞", "is_sfx": False},
                {"text": "ドン", "is_sfx": True},
                {"text": "と", "is_sfx": False, "score": 0.9},
                {"text": "き…", "is_sfx": False, "score": 0.9},
                {"text": "あっ♡", "is_sfx": False, "score": 0.9},
                {"text": "私", "is_sfx": False, "score": 0.9},
                {"text": "その", "is_sfx": False, "score": 0.9},
            ],
        }
    )
    assert [item["text"] for item in unsafe_semantic_missing(page, True)] == [
        "台詞",
        "私",
        "その",
    ]
    assert [item["text"] for item in unsafe_semantic_missing(page, False)] == [
        "台詞",
        "ドン",
        "と",
        "き…",
        "あっ♡",
        "私",
        "その",
    ]
