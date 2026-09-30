import numpy as np
from scripts.test_thin_recovery import candidates


def fixture():
    p=np.zeros((100,100),np.float32)
    p[40:50,10:30]=.9
    p[40:50,60:80]=.9
    p[41:49,44:46]=.8
    return p


def test_discarded_thin_component_requires_two_line_anchors():
    p=fixture()
    found=candidates(p)
    assert len(found)==1
    assert found[0]['raw']==[44,41,2,8]
    p[:,60:]=0
    assert not candidates(p)


def test_horizontal_marks_and_surviving_components_are_not_added():
    p=fixture()
    p[41:49,44:46]=0
    p[60:62,40:50]=.8
    assert not candidates(p)
    p=fixture()
    p[38:52,44:48]=.8
    assert not candidates(p)


def test_candidate_count_is_bounded():
    # Separate lines with local anchors; each line offers one discarded component.
    p=np.zeros((1024,1024),np.float32)
    for y in range(10,1000,20):
        p[y:y+10,10:30]=.9
        p[y:y+10,60:80]=.9
        p[y+1:y+9,44:46]=.8
    assert len(candidates(p))==32
