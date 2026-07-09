from app.services.preprocessing import extract_url_features, preprocess_text


def test_preprocess_text_removes_noise():
    text = "Breaking!!! This is a TEST. Visit https://example.com now"
    cleaned = preprocess_text(text)
    assert "https" not in cleaned
    assert "test" in cleaned


def test_url_features_https_flag():
    meta = extract_url_features("https://example.com/news")
    assert meta["vector"][0] == 1
