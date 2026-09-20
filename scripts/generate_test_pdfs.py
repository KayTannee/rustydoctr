"""Generate deterministic OCR stress pages and word polygons from drawing geometry.

Run: python scripts/generate_test_pdfs.py --output testdata/generated
Coordinates are normalized, top-left origin, ordered TL/TR/BR/BL in word space.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageOps
from reportlab.lib.pagesizes import A3, A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

LOREM = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur".split()


def generate(output: Path, seed=1729, dpi=200):
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pdf = output / "ocr_stress.pdf"
    c = canvas.Canvas(str(pdf), pagesize=A4, invariant=1)
    pages = []
    specs = [
        ("a4_control", A4, 11, 15, 1, 0, 0, False),
        ("a4_small_dense", A4, 6, 7.5, 3, 0, 0, False),
        ("a4_large", A4, 22, 31, 1, 0, 0, False),
        ("a4_tight_tracking", A4, 10, 13, 2, -0.25, 0, False),
        ("a4_wide_tracking", A4, 10, 16, 2, 1.5, 0, False),
        ("a4_tight_lines", A4, 9, 10, 2, 0, 0, False),
        ("a4_loose_lines", A4, 11, 26, 2, 0, 0, False),
        ("a4_white_on_black", A4, 11, 16, 2, 0, 0, True),
        ("a4_skew_7", A4, 11, 17, 1, 0, 7, False),
        ("a4_mixed_rotation", A4, 11, 17, 1, 0, 0, False),
        ("a4_fonts_blocks", A4, 11, 17, 2, 0, 0, False),
        ("a3_very_dense", A3, 5.5, 7, 4, -0.05, 0, False),
    ]
    for index, (name, size, fs, leading, columns, tracking, angle, inverse) in enumerate(specs):
        width, height = size
        c.setPageSize(size)
        if inverse:
            c.setFillColorRGB(0, 0, 0)
            c.rect(0, 0, width, height, fill=1, stroke=0)
        words = []

        def block(x, y, bw, bh, font, fontsize, spacing, charspace=0, rotation=0):
            # Baseline uses bottom-left PDF coordinates; boxes use font ascent/descent.
            c.saveState()
            c.translate(x, y)
            c.rotate(rotation)
            c.setFillColorRGB(*((1, 1, 1) if inverse else (0, 0, 0)))
            asc, desc = pdfmetrics.getAscentDescent(font, fontsize)
            a = math.radians(rotation)
            def point(px, py):
                return [(x + px * math.cos(a) - py * math.sin(a)) / width,
                        1 - (y + px * math.sin(a) + py * math.cos(a)) / height]
            baseline = -asc
            while baseline + desc >= -bh:
                cursor = 0.0
                while True:
                    word = rng.choice(LOREM)
                    ww = pdfmetrics.stringWidth(word, font, fontsize) + charspace * (len(word) - 1)
                    if cursor + ww > bw:
                        break
                    t = c.beginText(cursor, baseline)
                    t.setFont(font, fontsize)
                    t.setCharSpace(charspace)
                    t.textOut(word)
                    c.drawText(t)
                    poly = [point(cursor, baseline + asc), point(cursor + ww, baseline + asc),
                            point(cursor + ww, baseline + desc), point(cursor, baseline + desc)]
                    assert all(0 <= v <= 1 for pt in poly for v in pt), (name, poly)
                    words.append({"text": word, "polygon": poly, "font": font, "size_pt": fontsize,
                                  "rotation": rotation, "tracking_pt": charspace})
                    cursor += ww + pdfmetrics.stringWidth(" ", font, fontsize) + max(charspace, 0)
                baseline -= spacing
            c.restoreState()

        fonts = ["Helvetica", "Times-Roman", "Courier", "Helvetica-Bold", "Times-Italic"]
        if name == "a4_mixed_rotation":
            block(35, height-35, 235, 170, fonts[0], fs, leading)
            block(320, height-45, 210, 140, fonts[1], fs, leading, rotation=-12)
            block(60, 280, 220, 150, fonts[2], fs, leading, rotation=90)
            block(width-40, 50, 210, 130, fonts[3], fs, leading, rotation=180)
            block(325, 420, 170, 150, fonts[0], fs, leading, rotation=-90)
        elif angle:
            block(45, height-110, width-190, height-180, fonts[0], fs, leading, rotation=angle)
        else:
            gap, margin = 20, 32
            bw = (width-2*margin-gap*(columns-1))/columns
            for col in range(columns):
                if name == "a4_fonts_blocks":
                    for row, font in enumerate(fonts):
                        block(margin+col*(bw+gap), height-margin-row*150, bw, 115, font, fs+col*2, leading+col*2)
                else:
                    block(margin+col*(bw+gap), height-margin, bw, height-2*margin,
                          fonts[col % 3] if columns > 1 else fonts[0], fs, leading, tracking)
        pages.append({"id": name, "page_index": index, "size_pt": list(size), "words": words})
        c.showPage()
    c.save()
    doc = fitz.open(pdf)
    for p, gt in zip(doc, pages):
        image_path = output / f"{gt['id']}.png"
        pix = p.get_pixmap(dpi=dpi, alpha=False)
        pix.save(image_path)
        gt.update(image=image_path.name, width=pix.width, height=pix.height)
    manifest = {"schema_version": 1, "seed": seed, "dpi": dpi, "pdf": pdf.name,
                "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "pages": pages,
                "box_definition": "font advance width and ascent/descent, not tight ink; normalized polygon"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sheet = Image.new('RGB', (1200, 1100), '#e0e4e8')
    labels = ImageDraw.Draw(sheet)
    for i, page in enumerate(pages):
        x, y = (i % 4)*300, (i // 4)*365
        with Image.open(output/page['image']) as img:
            sheet.paste(ImageOps.contain(img, (280, 320)), (x, y+25))
        labels.text((x+5,y+5), page['id'], fill='black')
    sheet.save(output/'contact.png')
    print(f"Created {len(pages)} pages, {sum(len(p['words']) for p in pages)} words: {pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("testdata/generated"))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    generate(args.output, args.seed, args.dpi)
