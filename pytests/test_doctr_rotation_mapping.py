"""CPU checks for the comparison harness's default docTR coordinate observer."""
import numpy as np
from pybaseline.compare_doctr_orientation import default_straightening_inverse


def test_default_doc_tr_inverse_maps_visible_landmark():
    page=np.full((180,120,3),255,np.uint8)
    page[40:45,30:35]=[255,0,0]
    for angle in [0,90,180,-90,-1,2]:
        fixed,inverse=default_straightening_inverse(page,angle)
        marker=(fixed[:,:,0]>200)&(fixed[:,:,1]<100)&(fixed[:,:,2]<100)
        ys,xs=np.nonzero(marker)
        assert len(xs)>0
        point=inverse@np.array([xs.mean(),ys.mean(),1.])
        np.testing.assert_allclose(point[:2],[32.,42.],atol=.4,rtol=0)
