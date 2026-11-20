import pytest

from spam_detector.features import BinaryTextFeatures
from spam_detector.models.naive_bayes import MultinomialNaiveBayes
from spam_detector.nb_release import NBTextPredictor, save_nb_release

TEXTS = [
    "win free money now", "free prize claim money", "cheap pills free offer",
    "meeting schedule for monday", "please review the gas contract", "monday meeting notes attached",
]
LABELS = [1, 1, 1, 0, 0, 0]


@pytest.fixture
def release(tmp_path):
    features = BinaryTextFeatures(k=20, min_df=1).fit(TEXTS, LABELS)
    model = MultinomialNaiveBayes().fit(features.transform(TEXTS), LABELS)
    save_nb_release(tmp_path / "nb", model, features, {"note": "test"})
    return tmp_path / "nb", model, features


def test_release_roundtrip(release):
    path, model, features = release
    emails = ["free money offer", "monday contract review"]
    result = NBTextPredictor(path).predict_texts(emails)
    expected = model.predict_proba(features.transform(emails))[:, 1]
    assert [r["label"] for r in result] == [1, 0]
    assert [r["score_value"] for r in result] == pytest.approx(expected.tolist())
    assert result[0]["score_kind"] == "probability_uncalibrated"
    assert result[0]["top_words"][0]["word"] in {"free", "money", "offer"}
    assert len(result[0]["top_words"]) <= 5


def test_checksum_mismatch_rejected(release):
    path = release[0]
    target = path / "nb.json"
    target.write_bytes(target.read_bytes().replace(b"alpha", b"alphA", 1))
    with pytest.raises(ValueError):
        NBTextPredictor(path)


def test_no_vocabulary_warning(release):
    result = NBTextPredictor(release[0]).predict_texts(["zzzz qqqq"])[0]
    assert result["matched_features"] == 0 and result["warning"] is not None
