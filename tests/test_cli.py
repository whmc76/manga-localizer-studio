from manga_localizer.cli import configure_cli_encoding


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
