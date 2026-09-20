"""Render PDFs on a producer thread; independently consume ordered OCR results."""
import argparse
import json
from pathlib import Path
from threading import Thread

import pymupdf
from rustydoctr import Stream


def process(pdfs, config, output, models="models", dpi=200):
    errors = []
    with Stream(models=models, config=config) as stream:
        def feed():
            try:
                for filename in pdfs:
                    with pymupdf.open(filename) as document:
                        for index, page in enumerate(document):
                            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
                            stream.submit_rgb(f"{filename}:{index + 1}", pix.width, pix.height, pix.samples)
            except BaseException as exc:
                errors.append(exc)
            finally:
                stream.finish_input()

        producer = Thread(target=feed, name="pdf-producer")
        producer.start()
        try:
            with Path(output).open("w", encoding="utf-8") as result_file:
                for result in stream:
                    # Replace with database writes, extraction, etc. Feeding continues.
                    result_file.write(json.dumps(result) + "\n")
                    result_file.flush()
                    print(f"{result['id']}: {len(result['words'])} words", flush=True)
        finally:
            # Unblock a producer even when rendering/consuming fails.
            stream.close()
            producer.join()
        if errors:
            raise errors[0]
        return stream.stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+")
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", default="models")
    parser.add_argument("--output", default="results.jsonl")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(process(args.pdfs, args.config, args.output, args.models, args.dpi), indent=2))
