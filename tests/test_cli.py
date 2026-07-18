from manga_localizer.cli import _font_status, configure_cli_encoding


class ReconfigurableStream:
    def __init__(self):
        self.calls = []

    def reconfigure(self, **kwargs):
        self.calls.append(kwargs)


def test_cli_streams_are_normalized_to_utf8():
    first = ReconfigurableStream()
    second = ReconfigurableStream()
    configure_cli_encoding(first, second)
    assert first.calls == [{"encoding": "utf-8", "errors": "backslashreplace"}]
    assert second.calls == first.calls


def test_doctor_font_status_is_safe_before_assets_are_downloaded(tmp_path):
    def missing(_paths):
        raise FileNotFoundError("no system CJK font")

    assert _font_status(missing, tmp_path).startswith("missing;")
