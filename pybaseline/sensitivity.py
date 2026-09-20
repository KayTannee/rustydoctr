"""Rescore saved boxes to distinguish missed text from strict boundary mismatch."""
import argparse
import json
from pathlib import Path
from pybaseline.metrics import score_words, detection_summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('pybaseline/results/baseline'))
    p.add_argument('--cases',nargs='+',default=['ocr_fast_base_parseq','ocr_fast_base_parseq_size1536'])
    args=p.parse_args()
    output=[]
    for case in args.cases:
        folder=args.root/case
        result=json.loads((folder/'result.json').read_text())
        manifest=json.loads(Path(result['config']['manifest']).read_text())
        predictions=json.loads((folder/'predictions.json').read_text())
        for threshold in [.25,.5,.75]:
            by_page=[dict(page=gt['id'],**score_words(gt['words'],pred['words'],threshold))
                     for gt,pred in zip(manifest['pages'],predictions)]
            output.append({'case':case,'iou_threshold':threshold,'accuracy':detection_summary(by_page),'by_page':by_page})
    (args.root/'iou_sensitivity.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print('Saved IoU sensitivity for',args.cases)


if __name__=='__main__':
    main()
