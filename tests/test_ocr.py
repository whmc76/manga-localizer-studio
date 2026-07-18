import io
import json
import warnings
from pathlib import Path

from PIL import Image, ImageDraw

from manga_localizer.ocr import (
    OllamaVisionOCR,
    PageOCR,
    PaddleMangaOCR,
    TextUnit,
    _light_on_dark_regions,
    _suppress_optional_ccache_warning,
    list_images,
    manga_force_cpu,
)


def test_only_the_optional_ccache_notice_is_suppressed():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _suppress_optional_ccache_warning():
            warnings.warn("No ccache found. Compilation cache is optional.", UserWarning)
            warnings.warn("actionable inference warning", RuntimeWarning)
    assert [str(item.message) for item in caught] == ["actionable inference warning"]


def test_images_are_naturally_sorted(tmp_path):
    for name in ("10.png", "2.jpg", "1.webp", "notes.txt"):
        path = tmp_path / name
        if path.suffix == ".txt":
            path.write_text("ignore", encoding="utf-8")
        else:
            Image.new("RGB", (8, 8), "white").save(path)
    assert [item.name for item in list_images(tmp_path)] == ["1.webp", "2.jpg", "10.png"]


def test_manga_ocr_gpu_choice_is_independent_from_paddle():
    assert manga_force_cpu("auto", cuda_available=True) is False
    assert manga_force_cpu("gpu:0", cuda_available=True) is False
    assert manga_force_cpu("cpu", cuda_available=True) is True
    assert manga_force_cpu("auto", cuda_available=False) is True


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
        return io.BytesIO(json.dumps(response_payload, ensure_ascii=False).encode("utf-8"))

    monkeypatch.setattr("manga_localizer.ocr.urlopen", fake_urlopen)
    page = OllamaVisionOCR("http://127.0.0.1:11434", "vision-local").analyze(source, 2)
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["body"]["model"] == "vision-local"
    assert captured["body"]["messages"][0]["images"]
    assert page.units[0].bbox == [40, 120, 160, 300]
    assert page.units[0].id == "p002u01"
