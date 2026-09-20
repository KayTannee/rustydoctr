"""Targeted single-character/rotation regression corpus with sentence context."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import pymupdf
from PIL import Image, ImageDraw, ImageOps
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

SENTENCES = [
    'It was cold outside I put my coat on',
    'I am here and I can read a clear sentence',
    'A B C I a i l 1 x y z are separate characters',
    'The symbols _ and - must stay as printed',
    'I saw a sign - I stopped and took a photo',
    'Fill the blank _ with a letter I if instructed',
    'I paid 1 dollar for a book and a pen',
    'This line contains I I I and _ _ _ and - - -',
]


def generate(output, dpi=250):
    output.mkdir(parents=True,exist_ok=True)
    width,height=A4
    pdf=output/'rotation_regression.pdf'
    c=canvas.Canvas(str(pdf),pagesize=A4,invariant=1)
    pages=[]
    for name, skew, mixed in [('upright_characters',0,False),('skew7_characters',7,False),('skew7_mixed_characters',7,True)]:
        words=[]
        global_angle=math.radians(skew)
        c.saveState(); c.translate(width/2,height/2); c.rotate(skew); c.translate(-width/2,-height/2)
        def line(text,x,y,font='Helvetica',size=12,angle=0,role='body'):
            c.saveState(); c.translate(x,y); c.rotate(angle); c.setFont(font,size)
            asc,desc=pdfmetrics.getAscentDescent(font,size)
            local=math.radians(angle)
            def point(px,py):
                ax=x+px*math.cos(local)-py*math.sin(local)-width/2
                ay=y+px*math.sin(local)+py*math.cos(local)-height/2
                gx=width/2+ax*math.cos(global_angle)-ay*math.sin(global_angle)
                gy=height/2+ax*math.sin(global_angle)+ay*math.cos(global_angle)
                return [gx/width,1-gy/height]
            cursor=0
            for token in text.split():
                ww=pdfmetrics.stringWidth(token,font,size)
                c.drawString(cursor,0,token)
                poly=[point(cursor,asc),point(cursor+ww,asc),point(cursor+ww,desc),point(cursor,desc)]
                assert all(0<=v<=1 for pt in poly for v in pt),(name,token,poly)
                words.append({'text':token,'polygon':poly,'font':font,'size_pt':size,'rotation':skew+angle,
                              'local_rotation':angle,'role':role,'sentence':text,'single_character':len(token)==1})
                cursor+=ww+pdfmetrics.stringWidth(' ',font,size)
            c.restoreState()
        for section,font in enumerate(['Helvetica','Times-Roman','Courier']):
            for row,text in enumerate(SENTENCES):
                line(text,68,height-85-(section*8+row)*21,font,size=11)
        if mixed:
            line('I read a rotated line',95,90,'Helvetica',12,90,'rotated')
            line('I read a rotated line',230,245,'Times-Roman',12,-90,'rotated')
            line('I see _ and - here',510,115,'Courier',12,180,'rotated')
            line('I am a tilted word',290,245,'Helvetica',12,-18,'rotated')
        else:
            line('I read a rotated line',95,180,'Helvetica',12,0,'control')
            line('I read a rotated line',95,150,'Times-Roman',12,0,'control')
            line('I see _ and - here',95,120,'Courier',12,0,'control')
            line('I am a tilted word',95,90,'Helvetica',12,0,'control')
        c.restoreState(); c.showPage()
        pages.append({'id':name,'page_index':len(pages),'size_pt':[width,height],'page_skew':skew,'words':words})
    c.save()
    doc=pymupdf.open(pdf)
    sheet=Image.new('RGB',(1200,600),'#e0e4e8'); draw=ImageDraw.Draw(sheet)
    for i,(page,gt) in enumerate(zip(doc,pages)):
        pix=page.get_pixmap(dpi=dpi,alpha=False)
        name=gt['id']+'.png'; pix.save(output/name)
        gt.update(image=name,width=pix.width,height=pix.height)
        with Image.open(output/name) as image:
            sheet.paste(ImageOps.contain(image,(380,550)),(i*400,30))
        draw.text((i*400+10,10),gt['id'],fill='black')
    sheet.save(output/'contact.png')
    manifest={'schema_version':1,'suite':'single_character_rotation','dpi':dpi,'pdf':pdf.name,
              'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'pages':pages,
              'box_definition':'font advance/ascent/descent; underscore and dash ink occupy only part of this box'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Created {len(pages)} pages, {sum(len(p["words"]) for p in pages)} words')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('testdata/rotation'))
    p.add_argument('--dpi',type=int,default=250)
    a=p.parse_args(); generate(a.output,a.dpi)
