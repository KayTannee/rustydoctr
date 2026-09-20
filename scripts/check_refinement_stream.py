"""Check installed Python wheel refinement against native smoke results."""
import json
from pathlib import Path
from threading import Thread
from PIL import Image
from rustydoctr import Stream


def main():
    root=Path(__file__).resolve().parents[1]
    reference=json.loads((root/'pybaseline/results/native_quality_smoke/summary.json').read_text())
    config=dict(reference['config'],inflight=1)
    inputs=json.loads((root/'.cache/quality-workload.json').read_text())['pages']
    errors=[]
    with Stream(models=root/'models',config=config) as stream:
        def feed():
            try:
                for page in inputs:
                    with Image.open(page['image']) as image:
                        rgb=image.convert('RGB')
                        stream.submit_rgb(page['id'],rgb.width,rgb.height,rgb.tobytes())
            except BaseException as exc:errors.append(exc)
            finally:stream.finish_input()
        producer=Thread(target=feed,daemon=True)
        producer.start()
        results=[]
        try:
            for i,result in enumerate(stream):
                expected=reference['first_pages'][str(i)]
                assert result['id']==expected['id']
                assert result['refinement_tiles']==expected['refinement_tiles']
                assert [w['text'] for w in result['words']]==[w['text'] for w in expected['words']]
                results.append(result)
            assert len(results)==len(inputs) and stream.stats['max_inflight']==1
        finally:
            stream.close();producer.join(10)
        assert not errors and not producer.is_alive(), errors
    (root/'pybaseline/results/native_quality_smoke/python_stream.json').write_text(json.dumps(results,indent=2))
    print('PASS: installed wheel, RGB producer/consumer, one admission slot, native text and tile parity, clean drain')


if __name__=='__main__':main()
