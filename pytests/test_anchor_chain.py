import numpy as np
from scripts.experiment_anchor_chain import chain_candidates


def fixture(second_x=85):
    p=np.zeros((150,150),np.float32)
    p[40:50,32:52]=.9
    p[40:50,second_x:second_x+20]=.9
    p[41:49,20:22]=.8
    return p


def test_one_near_anchor_can_borrow_one_aligned_outward_anchor():
    proposals=chain_candidates(fixture())
    assert len(proposals)==1
    assert proposals[0]['raw']==[20,41,2,8]


def test_no_bridge_or_wrong_line_does_not_relax_to_one_anchor():
    p=fixture();p[:,85:]=0
    assert not chain_candidates(p)
    p[75:85,85:105]=.9
    assert not chain_candidates(p)


def test_existing_two_direct_anchors_are_not_proposed_again():
    assert not chain_candidates(fixture(second_x=60))


def test_still_requires_detector_signal_and_does_not_chain_across_large_gaps():
    p=fixture();p[41:49,20:22]=.2
    assert not chain_candidates(p)
    p=fixture(second_x=120)
    assert not chain_candidates(p)
