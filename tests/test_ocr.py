from PIL import Image

from manga_localizer.ocr import list_images


def test_images_are_naturally_sorted(tmp_path):
    for name in ("10.png", "2.jpg", "1.webp", "notes.txt"):
        path = tmp_path / name
        if path.suffix == ".txt":
            path.write_text("ignore", encoding="utf-8")
        else:
            Image.new("RGB", (8, 8), "white").save(path)
    assert [item.name for item in list_images(tmp_path)] == ["1.webp", "2.jpg", "10.png"]
