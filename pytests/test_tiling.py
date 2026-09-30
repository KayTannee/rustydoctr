from scripts.benchmark_tiling import windows,mapped,merge

def test_overlap_and_shift_cover_page():
    for overlap in [0,128,256]:
        for shift in [0,.5]:
            tiles=windows(800,1200,2048,overlap,shift)
            for y in range(0,1200,13):
                for x in range(0,800,13):assert any(a<=x<c and b<=y<d for a,b,c,d in tiles)

def test_complete_observation_wins_without_duplicate():
    word=dict(polygon=[[.01,.2],[.21,.3]],text='hello',objectness=.9)
    a=mapped(word,(500,0,1500,1000),2000,2000)
    b=mapped(dict(word,polygon=[[.51,.2],[.71,.3]]),(0,0,1000,1000),2000,2000)
    assert merge([a,b])==[b]

def test_mapping_roundtrip():
    w=mapped(dict(polygon=[[.1,.2],[.3,.4]],text='I'),(100,200,1100,1200),2000,2000)
    assert w['polygon']==[[.1,.2],[.2,.2],[.2,.3],[.1,.3]]

def test_boundary_diagnostics_distinguish_overlap_protection():
    from scripts.report_tiling import diagnose
    page=dict(width=200,height=100,words=[dict(text='word',polygon=[[.45,.2],[.55,.2],[.55,.4],[.45,.4]])])
    record=dict(words=[],tiles=[(0,0,100,100),(100,0,200,100)])
    rows,groups=diagnose(page,record)
    assert groups['cut_in_every_tile']['truth']==1
    record['tiles']=[(0,0,120,100),(80,0,200,100)]
    rows,groups=diagnose(page,record)
    assert groups['cut_in_every_tile']['truth']==0
    assert rows[0]['intact_tile']

def test_duplicate_diagnostic_counts_extra_observations():
    from scripts.report_tiling import diagnose
    word=dict(text='I',polygon=[[.4,.2],[.5,.2],[.5,.4],[.4,.4]])
    _,groups=diagnose(dict(width=100,height=100,words=[word]),dict(words=[word,word],tiles=[]))
    assert groups['all']==dict(truth=1,detected=1,exact=1,no_overlap=0,duplicates=1)

def test_guard_removes_edge_fragment_only_with_alternative_context():
    from scripts.remerge_tiling import guarded
    word=mapped(dict(polygon=[[.001,.2],[.1,.3]],text='fragment',objectness=.8),(500,0,1500,1000),2000,2000)
    assert guarded([word],[(500,0,1500,1000)],2000,2000)==[word]
    assert guarded([word],[(0,0,1000,1000),(500,0,1500,1000)],2000,2000)==[]
