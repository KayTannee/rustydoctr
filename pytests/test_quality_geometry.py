"""CPU geometry checks for quality experiments; no inference or model loading."""
import numpy as np
from pybaseline.quality_study import rotate,remap
from pybaseline.quality_experiments import dense_band
from pybaseline.quality_candidates import candidates


def test_rotation_mapping_returns_original_coordinates():
    image=np.full((240,160,3),255,np.uint8)
    points=np.array([[.2,.25],[.6,.25],[.6,.5],[.2,.5]])
    for angle in [-1.5,-.5,.5,45,90,180,270]:
        fixed,matrix=rotate(image,angle)
        transformed=np.c_[points*[160,240],np.ones(4)]@matrix.T
        word={'polygon':(transformed/[fixed.shape[1],fixed.shape[0]]).tolist(),'text':'I'}
        restored=remap([word],matrix,image.shape,fixed.shape)[0]
        np.testing.assert_allclose(restored['polygon'],points,atol=1e-10)


def test_blank_page_does_not_request_refinement():
    image=np.full((600,400,3),255,np.uint8)
    assert dense_band(image) is None
    assert candidates(image,[])==[]
