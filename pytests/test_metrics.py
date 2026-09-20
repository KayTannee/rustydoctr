import pytest
from pybaseline.metrics import score_words, detection_summary, recognition_score


def word(x0=0, x1=1, text='Lorem'):
    return {'polygon': [[x0,0],[x1,0],[x1,1],[x0,1]], 'text': text}


def test_duplicate_prediction_only_matches_once():
    result = detection_summary([score_words([word()], [word(), word()])])
    assert result['matched'] == 1
    assert result['precision'] == .5
    assert result['recall'] == 1


def test_merged_words_and_missing():
    result = score_words([word(0,.4),word(.6,1)], [word()])
    assert result['merge_candidates'] == 1
    assert result['matched'] == 0


def test_empty_and_wrong_text():
    assert score_words([word()], [])['matched'] == 0
    assert score_words([], [word()])['predicted'] == 1
    assert score_words([word()], [word(text='wrong')])['exact'] == 0


def test_cer_counts_insertions_and_case():
    score = recognition_score(['abc', 'Def'], ['abcd', 'def'])
    assert score['cer'] == pytest.approx(2/6)
    assert score['casefold_accuracy'] == .5


def test_polygon_rotation_not_axis_envelope():
    a = {'polygon': [[0,0],[1,1],[.9,1.1],[-.1,.1]], 'text':'a'}
    b = {'polygon': [[0,1],[1,0],[1.1,.1],[.1,1.1]], 'text':'a'}
    assert score_words([a],[b])['matched'] == 0
