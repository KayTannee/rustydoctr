"""Accuracy-first shortlist and rotation-policy comparison before long streaming runs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from pybaseline.metrics import score_words


def diagnostics(folder, manifest):
    predictions=json.loads((folder/'predictions.json').read_text())
    pages=[]
    for gt,pred in zip(manifest['pages'],predictions):
        matched=score_words(gt['words'],pred['words'],.25,return_pairs=True)
        pairs=dict(matched['pairs'])
        confusion=Counter(); rotated_total=rotated_exact=single_total=single_exact=0
        details=[]
        for i,word in enumerate(gt['words']):
            j=pairs.get(i)
            value=pred['words'][j]['text'] if j is not None else '<unmatched>'
            if word['single_character']:
                single_total+=1; single_exact+=value==word['text']
                confusion[(word['text'],value)]+=1
                details.append({'truth':word['text'],'predicted':value,'sentence':word['sentence'],'role':word['role'],'rotation':word['rotation']})
            if word['role']=='rotated':
                rotated_total+=1; rotated_exact+=value==word['text']
        pages.append({'page':gt['id'],'single_total':single_total,'single_exact':single_exact,
                      'rotated_total':rotated_total,'rotated_exact':rotated_exact,
                      'confusions':[{'truth':a,'predicted':b,'count':n} for (a,b),n in confusion.items()],
                      'single_character_details':details})
    output={'diagnostic_iou':.25,'note':'Font-metric boxes, not tight ink. A dash/underscore unmatched is not necessarily an orientation error. Main accuracy remains IoU 0.5.',
            'pages':pages}
    (folder/'character_diagnostics.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('pybaseline/results/rotation_screen_v2'))
    p.add_argument('--manifest',type=Path,default=Path('testdata/rotation/manifest.json'))
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(a.manifest.read_text())
    pairs=[('db_resnet34','crnn_mobilenet_v3_large'),('db_resnet34','parseq'),
           ('fast_base','crnn_mobilenet_v3_large'),('fast_base','parseq'),
           ('linknet_resnet50','crnn_mobilenet_v3_large')]
    candidates=[]
    def run(det,reco,rotation):
        name=f'{det}_{reco}_{rotation}'; folder=a.output/name; folder.mkdir(exist_ok=True)
        if not (folder/'result.json').exists():
            cmd=[sys.executable,'-m','pybaseline.run','--kind','ocr','--det',det,'--reco',reco,
                 '--rotation',rotation,'--size','1536','--manifest',str(a.manifest),
                 '--output',str(folder),'--min-seconds','5','--repeats','2']
            print('START',name,flush=True)
            with (folder/'run.log').open('w',encoding='utf-8') as log:
                completed=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            if completed.returncode:
                raise RuntimeError(f'{name} failed; inspect {folder}/run.log')
        result=json.loads((folder/'result.json').read_text())
        diag=diagnostics(folder,manifest)
        print('DONE',name,result['accuracy']['end_to_end_exact_recall'],flush=True)
        return {'name':name,'det':det,'reco':reco,'rotation':rotation,'result':result,'diagnostics':diag}
    for det,reco in pairs:
        candidates.append(run(det,reco,'crops'))
    # Primary ranking is main IoU-0.5 exact recall, then singleton exact count, then throughput.
    best=max(candidates,key=lambda c:(c['result']['accuracy']['end_to_end_exact_recall'],
                                     sum(p['single_exact'] for p in c['diagnostics']['pages']),
                                     c['result']['throughput_per_second']))
    variants=[best]+[run(best['det'],best['reco'],mode) for mode in ['straight','straighten','straighten_no_crop']]
    # Keep local orientation enabled for the selected mixed-document streaming policy.
    # Disabling it is an ablation, not a policy that satisfies mixed-orientation input.
    eligible=[c for c in variants if c['rotation'] in ['crops','straighten']]
    winner=max(eligible,key=lambda c:(c['result']['accuracy']['end_to_end_exact_recall'],
                                   sum(p['single_exact'] for p in c['diagnostics']['pages']),
                                   c['result']['throughput_per_second']))
    selection={'det':winner['det'],'reco':winner['reco'],'rotation':winner['rotation'],'size':1536,
               'selected_case':winner['name'],'reason':'Highest IoU-0.5 exact recall on the targeted corpus among crop-orientation-enabled policies; singleton accuracy and speed break ties. Disabling crop orientation is an ablation because input includes genuinely rotated words.',
               'limits':'Small synthetic development set, not an independent held-out accuracy result. Selection may change on real documents.'}
    (a.output/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print(json.dumps(selection),flush=True)


if __name__=='__main__':
    main()
