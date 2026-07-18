import pytest
from PIL import Image

from manga_localizer.pipeline import LocalizerPipeline, completion_summary, page_from_dict


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
