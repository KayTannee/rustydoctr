"""Installed-wheel RGB streaming and original-coordinate parity with native OCR."""
import argparse
import json
from pathlib import Path
from threading import Thread
import numpy as np
from PIL import Image
from rustydoctr import Stream

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=ROOT/'pybaseline/results/native_orientation_v1/refined/summary.json')
    args=parser.parse_args()
    reference=json.loads(args.reference.read_text())
    expected={p['id']:p for p in reference['first_pages'].values()}
    fixtures=ROOT/'output/pdf/quality'
    pages=[p for p in json.loads((fixtures/'manifest.json').read_text())['pages'] if p['id'] in ['statement_0_cw0','statement_0_cw90','statement_0_cw180','statement_0_cw90.5']]
    config=dict(reference['config'],inflight=1)
    errors=[];results=[]
    with Stream(models=ROOT/'models',config=config) as stream:
        def feed():
            try:
                for page in pages:
                    with Image.open(fixtures/page['image']) as source:
                        rgb=source.convert('RGB')
                        stream.submit_rgb(page['id'],rgb.width,rgb.height,rgb.tobytes())
            except BaseException as exc:errors.append(exc)
            finally:stream.finish_input()
        producer=Thread(target=feed,daemon=True);producer.start()
        try:
            for i,record in enumerate(stream):
                assert record['id']==pages[i]['id'] and record['sequence']==i
                native=expected[record['id']]
                actual_geometry=dict(record['geometry']);expected_geometry=dict(native['geometry'])
                # Summary Value conversion widens f32 to f64; streaming JSON prints f32 directly.
                assert np.float32(actual_geometry.pop('class_confidence'))==np.float32(expected_geometry.pop('class_confidence'))
                assert actual_geometry==expected_geometry
                assert record['refinement_tiles']==native['refinement_tiles']
                assert [w['text'] for w in record['words']]==[w['text'] for w in native['words']]
                np.testing.assert_allclose([w['quadrilateral'] for w in record['words']],[w['quadrilateral'] for w in native['words']],atol=1e-6,rtol=0)
                results.append(record)
            assert len(results)==len(pages) and stream.stats['max_inflight']==1
        finally:
            stream.close();producer.join(10)
        assert not producer.is_alive() and not errors,errors
    (args.reference.parent.parent/'python_stream.json').write_text(json.dumps(results,indent=2))
    # Invalid option combinations must fail before launching workers.
    try:
        Stream(models=ROOT/'models',config=dict(config,page_orientation=False,deskew=True))
    except ValueError:pass
    else:raise AssertionError('deskew without orientation accepted')
    print('PASS: installed wheel, independent producer/consumer, one-slot drain, native text/geometry/quad parity, invalid-option rejection')


if __name__=='__main__':main()
