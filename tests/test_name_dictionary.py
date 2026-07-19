import gzip

from manga_localizer.name_dictionary import JapaneseNameDictionary, katakana_to_hiragana


def test_katakana_reading_is_normalized_for_jmnedict():
    assert katakana_to_hiragana("ガク") == "がく"
    assert katakana_to_hiragana("サナ") == "さな"


def test_name_dictionary_returns_only_short_given_name_candidates(tmp_path):
    payload = """<?xml version="1.0" encoding="UTF-8"?>
<JMnedict>
  <entry><k_ele><keb>岳</keb></k_ele><r_ele><reb>がく</reb></r_ele><trans><name_type>male given name or forename</name_type><trans_det>Gaku</trans_det></trans></entry>
  <entry><k_ele><keb>学園前</keb></k_ele><r_ele><reb>がく</reb></r_ele><trans><name_type>place name</name_type><trans_det>Gaku</trans_det></trans></entry>
  <entry><k_ele><keb>紗奈</keb></k_ele><r_ele><reb>さな</reb></r_ele><trans><name_type>female given name or forename</name_type><trans_det>Sana</trans_det></trans></entry>
</JMnedict>"""
    cache = tmp_path / "cache"
    cache.mkdir()
    with gzip.open(cache / "JMnedict.xml.gz", "wb") as stream:
        stream.write(payload.encode("utf-8"))
    # The production integrity threshold avoids accepting truncated downloads;
    # the fixture is trusted and bypasses only that download check.
    service = JapaneseNameDictionary(cache)
    service.ensure = lambda: cache / "JMnedict.xml.gz"
    result = service.lookup(["ガク", "サナ"])
    assert [item.written for item in result["ガク"]] == ["岳"]
    assert result["ガク"][0].is_male is True
    assert [item.written for item in result["サナ"]] == ["紗奈"]
    assert result["サナ"][0].is_female is True
    assert result["サナ"][0].is_male is False
