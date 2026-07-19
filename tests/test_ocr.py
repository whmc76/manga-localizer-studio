import io
import json
import warnings
from pathlib import Path

from PIL import Image, ImageDraw

from manga_localizer.ocr import (
    HybridMangaOCR,
    OllamaVisionOCR,
    PageOCR,
    PaddleMangaOCR,
    TextUnit,
    _light_on_dark_regions,
    _suppress_optional_ccache_warning,
    duplicate_tiny_fragment,
    likely_sfx_text,
    list_images,
    malformed_tiny_ocr,
    manga_force_cpu,
    merge_region_candidates,
    prefer_semantic_ocr,
    semantic_sfx_classification,
    tiny_low_confidence_nontext,
)


def test_tiny_zero_confidence_fragment_is_preserved_as_nontext():
    unit = TextUnit(
        "p001u01",
        [1099, 2439, 1186, 2468],
        [1089, 2429, 1196, 2478],
        "『インターネットの",
        0.0,
    )
    page = PageOCR(1, "001.png", 2126, 3661, [unit])
    assert tiny_low_confidence_nontext(page, unit) is True

    dialogue = TextUnit(
        "p001u02", [100, 100, 220, 180], [90, 90, 230, 190], "ん？", 0.0
    )
    assert tiny_low_confidence_nontext(page, dialogue) is False
    tiny_narration = TextUnit(
        "p001u03",
        [1546, 99, 1612, 125],
        [1536, 89, 1622, 135],
        "そして最近では、",
        0.0,
    )
    assert tiny_low_confidence_nontext(page, tiny_narration) is False


def test_low_confidence_tiny_repeated_kanji_is_malformed_ocr():
    malformed = TextUnit(
        "p001u01",
        [1766, 1442, 1806, 1571],
        [1756, 1432, 1816, 1581],
        "目目まで、",
        0.76,
    )
    normal = TextUnit(
        "p001u02", [1766, 1442, 1806, 1571], [1756, 1432, 1816, 1581], "目まで、", 0.76
    )
    page = PageOCR(1, "001.png", 2126, 3661, [malformed, normal])
    assert malformed_tiny_ocr(page, malformed) is True
    assert malformed_tiny_ocr(page, normal) is False


def test_tiny_ruby_or_ocr_fragment_repeated_in_larger_unit_is_preserved():
    small = TextUnit(
        "p001u01",
        [1712, 661, 1758, 729],
        [1700, 650, 1770, 740],
        "彼氏\nかれし",
        0.92,
    )
    large = TextUnit(
        "p001u02",
        [1602, 0, 2095, 672],
        [1580, 0, 2110, 690],
        "でももう\n流石に彼氏とか\nいんだろ？",
        0.95,
    )
    page = PageOCR(1, "001.png", 2126, 3661, [small, large])
    assert duplicate_tiny_fragment(page, small) is True
    assert duplicate_tiny_fragment(page, large) is False

    repeated_dialogue = TextUnit(
        "p001u03",
        [1098, 1869, 1266, 2147],
        [1088, 1859, 1276, 2157],
        "彼氏",
        0.69,
    )
    page.units.append(repeated_dialogue)
    assert duplicate_tiny_fragment(page, repeated_dialogue) is False


def test_horizontal_vlm_echo_yields_to_detailed_vertical_detector_boxes():
    echo = TextUnit(
        "p001u01",
        [739, 1617, 1290, 2063],
        [717, 1604, 1312, 2076],
        "ああミナッ\nイクイク…",
        0.95,
        erase_boxes=[[739, 1617, 1290, 2063]],
    )
    dialogue = TextUnit(
        "p001u02",
        [27, 1495, 293, 2303],
        [16, 1471, 304, 2327],
        "あぁミナッ\nイクイクイク…",
        0.92,
        erase_boxes=[[27, 1503, 164, 2303], [145, 1495, 293, 2124]],
    )
    page = PageOCR(1, "001.png", 2126, 3661, [echo, dialogue])
    assert duplicate_tiny_fragment(page, echo) is True
    assert duplicate_tiny_fragment(page, dialogue) is False


def test_only_the_optional_ccache_notice_is_suppressed():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _suppress_optional_ccache_warning():
            warnings.warn(
                "No ccache found. Compilation cache is optional.", UserWarning
            )
            warnings.warn("actionable inference warning", RuntimeWarning)
    assert [str(item.message) for item in caught] == ["actionable inference warning"]


def test_images_are_naturally_sorted(tmp_path):
    for name in ("10.png", "2.jpg", "1.webp", "notes.txt"):
        path = tmp_path / name
        if path.suffix == ".txt":
            path.write_text("ignore", encoding="utf-8")
        else:
            Image.new("RGB", (8, 8), "white").save(path)
    assert [item.name for item in list_images(tmp_path)] == [
        "1.webp",
        "2.jpg",
        "10.png",
    ]


def test_second_pass_missing_drops_duplicate_vlm_hallucination():
    page = PageOCR(
        6,
        "006.png",
        1000,
        2000,
        [
            TextUnit(
                "p006u01",
                [700, 50, 800, 120],
                [690, 40, 810, 130],
                "そして最近では、",
                0.9,
                erase_boxes=[[700, 50, 800, 120]],
            )
        ],
    )
    missing = [
        {
            "text": "そして最近では、",
            "bbox": [10, 400, 200, 500],
            "score": 0.9,
            "is_sfx": False,
        }
    ]
    assert OllamaVisionOCR.filter_covered_missing(page, missing) == []


def test_second_pass_missing_drops_geometry_already_owned_by_detector():
    page = PageOCR(
        2,
        "002.png",
        1000,
        2000,
        [
            TextUnit(
                "p002u01",
                [100, 800, 300, 1200],
                [90, 790, 310, 1210],
                "違う文字",
                0.9,
                erase_boxes=[[100, 800, 300, 1200]],
            )
        ],
    )
    missing = [
        {
            "text": "別の候補",
            "bbox": [120, 420, 280, 580],
            "score": 0.9,
            "is_sfx": False,
        }
    ]
    assert OllamaVisionOCR.filter_covered_missing(page, missing) == []


def test_punctuated_short_hiragana_is_dialogue_even_without_detector_score():
    assert semantic_sfx_classification("いや、", 0.0, True) is False
    assert semantic_sfx_classification("しかし、", 0.0, True) is False
    assert semantic_sfx_classification("ん？", 0.85, True) is False
    assert semantic_sfx_classification("ここは", 0.95, True) is False
    assert semantic_sfx_classification("いいですよ．．．", 0.96, True) is False
    assert semantic_sfx_classification("サナは", 0.82, True) is False
    assert semantic_sfx_classification("いつまでも", 0.86, True) is False
    assert semantic_sfx_classification("その", 0.0, True) is False
    assert semantic_sfx_classification("いい", 0.87, True) is False
    assert semantic_sfx_classification("カナ、", 0.94, True) is False
    assert semantic_sfx_classification("彼氏：", 0.68, True) is False
    assert semantic_sfx_classification("し", 0.2, False) is True


def test_missing_recovery_accepts_only_enlarged_detector_owned_geometry(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (1000, 1000), "white").save(source)

    class Detector:
        def predict(self, _image):
            return [
                {
                    "res": {
                        "dt_polys": [[[100, 100], [300, 100], [300, 300], [100, 300]]],
                        "dt_scores": [0.91],
                    }
                }
            ]

    service = PaddleMangaOCR.__new__(PaddleMangaOCR)
    service.detector = Detector()
    service.reader = lambda _crop: "見落とし"
    page = PageOCR(4, source.name, 1000, 1000, [])
    recovered = service.recover_missing(
        source,
        page,
        [
            {
                "text": "見落とし",
                "bbox": [400, 400, 600, 600],
                "score": 0.9,
                "is_sfx": False,
            }
        ],
    )
    assert len(recovered.units) == 1
    assert recovered.units[0].ja == "見落とし"
    assert recovered.units[0].erase_boxes == [recovered.units[0].bbox]


def test_hybrid_ocr_does_not_silently_accept_an_empty_detector_page(tmp_path):
    source = tmp_path / "empty-detector.png"
    Image.new("RGB", (1000, 2000), "white").save(source)
    recovered_unit = TextUnit(
        "p003u01",
        [100, 200, 300, 600],
        [90, 190, 310, 610],
        "見落とし",
        0.9,
        erase_boxes=[[100, 200, 300, 600]],
    )

    class Geometry:
        def analyze(self, _path, page_number):
            return PageOCR(page_number, source.name, 1000, 2000, [])

        def recover_missing(self, _path, page, missing):
            assert missing[0]["text"] == "見落とし"
            page.units = [recovered_unit]
            return page

    class Semantics:
        def analyze(self, _path, page_number):
            return PageOCR(page_number, source.name, 1000, 2000, [recovered_unit])

        def refine(self, _path, page):
            return page, []

        def filter_covered_missing(self, page, missing):
            return OllamaVisionOCR.filter_covered_missing(page, missing)

    service = HybridMangaOCR.__new__(HybridMangaOCR)
    service.geometry = Geometry()
    service.semantics = Semantics()
    service.last_missing = []
    page = service.analyze(source, 3)
    assert [unit.ja for unit in page.units] == ["見落とし"]
    assert page.semantic_missing == []


def test_manga_ocr_gpu_choice_is_independent_from_paddle():
    assert manga_force_cpu("auto", cuda_available=True) is False
    assert manga_force_cpu("gpu:0", cuda_available=True) is False
    assert manga_force_cpu("cpu", cuda_available=True) is True
    assert manga_force_cpu("auto", cuda_available=False) is True


def test_repeated_mixed_kana_ocr_noise_is_preserved_as_sfx():
    assert likely_sfx_text("カチ") is True
    assert likely_sfx_text("そのカキノカキ") is True
    assert likely_sfx_text("このゲーム") is False
    assert likely_sfx_text("サナはいつも") is False


def test_windows_cpu_detector_disables_mkldnn_regression():
    source = (
        Path(__file__).parents[1] / "src" / "manga_localizer" / "ocr.py"
    ).read_text(encoding="utf-8")
    assert "enable_mkldnn=False" in source


def test_detection_only_payload_becomes_tight_regions():
    payload = {
        "dt_polys": [
            [[10, 20], [70, 18], [72, 90], [12, 92]],
            [[0, 0], [5, 0], [5, 5], [0, 5]],
        ],
        "dt_scores": [0.91, 0.99],
    }
    assert PaddleMangaOCR._regions(payload) == [
        {
            "box": [10, 18, 72, 92],
            "text": "",
            "score": 0.91,
        }
    ]


def test_rotation_alone_does_not_skip_dialogue_as_sfx():
    payload = {
        "dt_polys": [[[10, 20], [70, 5], [90, 75], [30, 90]]],
        "dt_scores": [0.8],
    }
    assert "sfx_hint" not in PaddleMangaOCR._regions(payload)[0]


def test_light_on_dark_title_becomes_generic_ocr_candidate():
    image = Image.new("RGB", (1000, 1400), "black")
    draw = ImageDraw.Draw(image)
    for index in range(7):
        left = 100 + index * 115
        draw.rectangle((left, 1120, left + 68, 1240), fill="white")
        draw.rectangle((left + 20, 1085, left + 48, 1270), fill="white")
    regions = _light_on_dark_regions(image)
    assert len(regions) == 1
    assert regions[0]["reverse"] is True
    assert regions[0]["box"][0] <= 100
    assert regions[0]["box"][2] >= 858


def test_light_background_does_not_trigger_reverse_title_fallback():
    image = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(image)
    for index in range(7):
        left = 100 + index * 115
        draw.rectangle((left, 1120, left + 68, 1240), fill="black")
    assert _light_on_dark_regions(image) == []


def test_sparse_full_page_display_text_is_reread_as_one_cover_title(tmp_path):
    source = tmp_path / "cover.png"
    Image.new("RGB", (1000, 1400), "white").save(source)
    units = [
        TextUnit("p001u01", [580, 160, 720, 330], [570, 150, 730, 340], "さて", 0.9),
        TextUnit("p001u02", [560, 380, 720, 560], [550, 370, 730, 570], "にし", 0.9),
        TextUnit("p001u03", [300, 590, 520, 930], [290, 580, 530, 940], "勇気", 0.9),
        TextUnit(
            "p001u04",
            [320, 950, 500, 1320],
            [310, 940, 510, 1330],
            "があったなら",
            0.9,
        ),
    ]
    page = PageOCR(1, source.name, 1000, 1400, units)
    assert PaddleMangaOCR.is_cover_title_candidate(page) is True

    service = object.__new__(PaddleMangaOCR)
    service.reader = lambda _crop: "僕に勇気があったなら"
    refined = service.refine_cover_title(source, page)
    assert len(refined.units) == 1
    assert refined.units[0].ja == "僕に勇気があったなら"
    assert refined.units[0].special == "cover_title"
    assert len(refined.units[0].erase_boxes) == 4


def test_recognition_candidates_can_add_but_never_delete_detector_regions():
    primary = [
        {"box": [10, 10, 80, 120], "text": "", "score": 0.9},
        {"box": [200, 10, 280, 120], "text": "", "score": 0.8},
    ]
    secondary = [
        {"box": [12, 12, 78, 118], "text": "日本", "score": 0.7},
        {"box": [400, 10, 480, 120], "text": "追加", "score": 0.7},
    ]
    merged = merge_region_candidates(primary, secondary)
    assert [item["box"] for item in merged] == [
        [10, 10, 80, 120],
        [200, 10, 280, 120],
        [400, 10, 480, 120],
    ]


def test_semantic_ocr_cannot_replace_a_complete_line_with_its_prefix():
    assert (
        prefer_semantic_ocr("私っ妊娠とか．．．無理だって！！", "私っ妊娠とか…", 0.92)
        is False
    )
    assert prefer_semantic_ocr(
        "行ったとか聞いてたけど", "東京に行ったとか聞いてたけど", 0.9
    )


def test_semantic_sfx_classification_preserves_uncertain_fragments():
    assert semantic_sfx_classification("い！！", 0.62, False) is True
    assert semantic_sfx_classification("し", 0.0, False) is True
    assert semantic_sfx_classification("いい", 0.0, False) is False


def test_semantic_sfx_classification_recovers_confident_narration():
    assert semantic_sfx_classification("ついに、", 0.92, True) is False


def test_semantic_refiner_applies_fragment_gate_when_vlm_omits_a_region(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (400, 600), "white").save(source)
    page = PageOCR(
        1,
        source.name,
        400,
        600,
        [
            TextUnit("p001u01", [10, 10, 50, 50], [5, 5, 55, 55], "い！！", 0.62),
            TextUnit("p001u02", [80, 10, 180, 60], [75, 5, 185, 65], "ついに、", 0.92),
        ],
    )
    service = OllamaVisionOCR("http://127.0.0.1:11434", "vision-local")
    service._chat = lambda _images, _prompt, _schema: {
        "regions": [
            {
                "id": "p001u02",
                "text": "ついに、",
                "score": 0.92,
                "is_sfx": True,
            }
        ],
        "missing": [],
    }
    refined, _missing = service.refine(source, page)
    assert refined.units[0].is_sfx is True
    assert refined.units[1].is_sfx is False


def test_ollama_vision_ocr_uses_local_image_endpoint(monkeypatch, tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (400, 600), "white").save(source)
    response_payload = {
        "message": {
            "content": json.dumps(
                {
                    "regions": [
                        {
                            "bbox": [0.1, 0.2, 0.4, 0.5],
                            "text": "こんにちは",
                            "score": 0.9,
                            "is_sfx": False,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        }
    }
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return io.BytesIO(
            json.dumps(response_payload, ensure_ascii=False).encode("utf-8")
        )

    monkeypatch.setattr("manga_localizer.ocr.urlopen", fake_urlopen)
    page = OllamaVisionOCR("http://127.0.0.1:11434", "vision-local").analyze(source, 2)
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["body"]["model"] == "vision-local"
    assert captured["body"]["messages"][0]["images"]
    assert page.units[0].bbox == [40, 120, 160, 300]
    assert page.units[0].id == "p002u01"


def test_tight_crop_audit_repairs_swapped_dialogue_and_sfx_roles(monkeypatch, tmp_path):
    source = tmp_path / "crop-audit.png"
    Image.new("RGB", (600, 800), "white").save(source)
    page = PageOCR(
        1,
        source.name,
        600,
        800,
        [
            TextUnit(
                "p001u01",
                [20, 20, 45, 85],
                [20, 20, 45, 85],
                "お客様から何でしょうか",
                0.0,
                False,
                "错误译文",
            ),
            TextUnit(
                "p001u02",
                [320, 250, 560, 720],
                [320, 250, 560, 720],
                "カチャ",
                0.86,
                True,
                "",
            ),
            TextUnit(
                "p001u03",
                [100, 300, 260, 520],
                [100, 300, 260, 520],
                "１０月２９日",
                0.0,
                False,
                "10月29日",
            ),
        ],
    )
    service = OllamaVisionOCR("http://127.0.0.1:11434", "vision")
    monkeypatch.setattr(
        service,
        "_chat",
        lambda images, prompt, schema: {
            "items": [
                {
                    "id": "p001u01",
                    "text": "",
                    "role": "nontext",
                    "confidence": 0.74,
                },
                {
                    "id": "p001u02",
                    "text": "僕は、秘密基地が好きだ。",
                    "role": "dialogue",
                    "confidence": 0.92,
                },
                {
                    "id": "p001u03",
                    "text": "ピロッ♪",
                    "role": "sfx",
                    "confidence": 0.9,
                },
            ]
        },
    )
    result = service.refine_local_crops(source, page)
    assert result.units[0].is_sfx is True
    assert result.units[1].ja == "僕は、秘密基地が好きだ。"
    assert result.units[1].is_sfx is False
    assert result.units[2].is_sfx is True
