"""Stream image files as RGB into native OCR; consume results independently."""
import argparse
import json
from pathlib import Path
from threading import Thread
from PIL import Image
from rustydoctr import Stream, default_config


def process(images, output, models='models', config=None, repeats=1):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    errors = []
    with Stream(models=models, config=config or default_config('low-vram')) as stream:
        def feed():
            try:
                for repeat in range(repeats):
                    for index, filename in enumerate(images):
                        with Image.open(filename) as im:
                            rgb = im.convert('RGB')
                        stream.submit_rgb(f'{repeat}:{index}:{Path(filename).name}',
                                          rgb.width, rgb.height, rgb.tobytes())
            except BaseException as exc:
                errors.append(exc)
            finally:
                stream.finish_input()
        producer = Thread(target=feed, name='image-producer')
        producer.start()
        try:
            with Path(output).open('w', encoding='utf-8') as file:
                for record in stream:
                    # Replace with downstream processing. Feeding proceeds independently.
                    file.write(json.dumps(record) + '\n')
                    file.flush()
        finally:
            stream.close()
            producer.join()
        if errors:
            raise errors[0]
        return stream.stats

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('images', nargs='+')
    p.add_argument('--models', default='models')
    p.add_argument('--config')
    p.add_argument('--output', default='results.jsonl')
    p.add_argument('--repeats', type=int, default=1)
    a = p.parse_args()
    print(json.dumps(process(a.images, a.output, a.models, a.config, a.repeats), indent=2))
