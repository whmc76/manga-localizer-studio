from manga_localizer.pipeline import page_from_dict


def test_reviewed_transcript_accepts_legacy_units_and_skips_marked_items():
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
