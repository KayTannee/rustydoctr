"""Checks public scoring adapters against upstream behavior, including ignored text."""
import ast, contextlib, gzip, io, json, time
import numpy as np
import pytest
from scripts.score_public import HIER_SOURCE,hier_evaluator,hier_input,points,strata

def word(text,box,ignore=False):return dict(text=text,polygon=box,ignore=ignore)
def test_box_expansion_and_single_character_diagnostics():
    w=word('I',[[.1,.1],[.2,.2]])
    assert points(w).shape==(4,2)
    groups,_=strata(dict(words=[w]),[dict(w),dict(w)])
    assert groups['single']['truth']==groups['single']['detected']==groups['single']['exact']==1
    groups,_=strata(dict(words=[w]),[])
    assert groups['single']['no_overlap']==1

@pytest.mark.skipif(not HIER_SOURCE.exists(),reason='Download public evaluator first')
def test_official_ignore_regions_and_wrong_text():
    a=word('I',[[.1,.1],[.2,.2]]);b=word('',[[.6,.6],[.8,.8]],True);page=dict(width=100,height=100,words=[a,b])
    e=hier_evaluator();r=e.evaluate_one_image(hier_input(page,[a,word('noise',b['polygon'])]))
    assert r['num_groundtruth']==r['num_prediction']==r['tp_det_cnt']==r['tp_e2e_cnt']==1
    r=e.evaluate_one_image(hier_input(page,[word('-',a['polygon'])]))
    assert r['tp_det_cnt']==1 and r['tp_e2e_cnt']==0

@pytest.mark.skipif(not HIER_SOURCE.exists(),reason='Download public evaluator first')
def test_adapter_matches_official_parser_on_real_annotations():
    # Load only the upstream parser function so the Apache Beam CLI dependency is
    # unnecessary. The parser body is unmodified and used solely as a test oracle.
    tree=ast.parse((HIER_SOURCE/'eval.py').read_text(encoding='utf-8'))
    selected=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_annotation_dict'],type_ignores=[])
    env=dict(np=np,time=time);exec(compile(selected,str(HIER_SOURCE/'eval.py'),'exec'),env)
    with gzip.open(HIER_SOURCE/'gt/test.jsonl.gz','rt',encoding='utf-8') as f:data=json.load(f)
    for anno in data['annotations'][:3]:
        W,H=anno['image_width'],anno['image_height'];gt=[w for p in anno['paragraphs'] for l in p['lines'] for w in l['words']];pred=gt[:4]
        sample=dict(anno,output_paragraphs=[dict(lines=[dict(words=pred)])])
        with contextlib.redirect_stdout(io.StringIO()):_,expected,_,_=env['parse_annotation_dict'](sample,False,False,1)
        page=dict(width=W,height=H,words=[word(w['text'],(np.array(w['vertices'])/[W,H]).tolist(),not w['legible']) for w in gt])
        actual=hier_input(page,[word(w['text'],(np.array(w['vertices'])/[W,H]).tolist()) for w in pred])
        for key in ['gt_boxes','detection_boxes']:
            for a,b in zip(actual[key],expected[key]):np.testing.assert_allclose(a,b,atol=1e-10)
        for key in ['gt_weights','gt_texts','pred_texts']:np.testing.assert_array_equal(actual[key],expected[key])
        assert hier_evaluator().evaluate_one_image(actual)==pytest.approx(hier_evaluator().evaluate_one_image(expected))
