"""Realistic, labelled statement fixtures and a controlled text-scale ladder.
Page angles are clockwise in image coordinates; polygons retain reading order.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import cv2
import pymupdf
from PIL import Image, ImageDraw, ImageOps
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import A4
from reportlab.graphics.barcode.code128 import Code128

W,H=A4
class Page:
    def __init__(self,c): self.c=c;self.words=[];self.regions=[]
    def line(self,text,x,y,size=10,font='Helvetica',angle=0,region='body'):
        c=self.c;c.saveState();c.translate(x,y);c.rotate(angle);c.setFont(font,size)
        asc,desc=pdfmetrics.getAscentDescent(font,size);cursor=0;a=math.radians(angle)
        def point(px,py):return [(x+px*math.cos(a)-py*math.sin(a))/W,1-(y+px*math.sin(a)+py*math.cos(a))/H]
        for word in text.split():
            width=pdfmetrics.stringWidth(word,font,size)
            c.drawString(cursor,0,word)
            poly=[point(cursor,asc),point(cursor+width,asc),point(cursor+width,desc),point(cursor,desc)]
            assert all(0<=v<=1 for pt in poly for v in pt),(word,poly)
            self.words.append(dict(text=word,polygon=poly,size_pt=size,font=font,local_rotation_deg=-angle,region=region))
            cursor+=width+pdfmetrics.stringWidth(' ',font,size)
        c.restoreState()
    def paragraph(self,text,x,y,width,size,leading,region):
        line=[]
        for word in text.split():
            if line and pdfmetrics.stringWidth(' '.join(line+[word]),'Times-Roman',size)>width:
                self.line(' '.join(line),x,y,size,'Times-Roman',region=region);y-=leading;line=[]
            line.append(word)
        if line:self.line(' '.join(line),x,y,size,'Times-Roman',region=region)
        return y-leading

def statement(c,index):
    p=Page(c);left=42;right=W-42
    c.setFillColorRGB(.10,.19,.28);c.rect(0,H-112,W,112,fill=1,stroke=0)
    c.setFillColorRGB(1,1,1)
    p.line('LOREM BANK',left,H-53,29+index*3,'Helvetica-Bold',region='title')
    p.line('Account statement',left,H-82,15,region='title')
    c.setFillColorRGB(.08,.08,.08)
    for j,line in enumerate(['A Ipsum','18 Lorem Avenue','Dolor Sit 6000']):p.line(line,left,H-146-j*14,11,region='address')
    for j,(label,value) in enumerate([('Account:','001-234 — 567890'),('Period:','01 Jan - 31 Jan 2026'),('Reference:','I / a / O / 1')]):
        p.line(label,325,H-144-j*20,10,'Helvetica-Bold',region='labels')
        p.line(value,390,H-144-j*20,9,region='labels')
    barcode=Code128('00123456789012345678',barWidth=.65,barHeight=13,humanReadable=False)
    barcode.drawOn(c,left,H-220)
    p.regions.append(dict(kind='barcode',polygon=[[left/W,1-(H-207)/H],[(left+barcode.width)/W,1-(H-207)/H],[(left+barcode.width)/W,220/H],[left/W,220/H]]))
    p.line('I was cold outside — I put a coat on.',left,H-242,10,region='characters')
    p.line('I a A i l 1 O 0 — - _ ; : (a) I.',left,H-258,10,'Times-Roman',region='characters')
    p.line('Opening balance',left,H-287,11,'Helvetica-Bold',region='labels');p.line('1,234.56',right-62,H-287,11,region='labels')
    top=H-315;columns=[left,left+57,left+120,right-115,right-55,right]
    c.setFillColorRGB(.89,.92,.95);c.rect(left,top-21,right-left,25,fill=1,stroke=0);c.setFillColorRGB(0,0,0)
    for label,x in zip(['Date','Code','Description','Debit','Credit'],columns):p.line(label,x+3,top-12,9,'Helvetica-Bold',region='table')
    count=[12,17,20][index];leading=[16,12,10][index];fs=[9,8,7.5][index]
    for row in range(count):
        y=top-34-row*leading
        if row%2==0:
            c.setFillColorRGB(.97,.97,.97);c.rect(left,y-4,right-left,leading,fill=1,stroke=0);c.setFillColorRGB(0,0,0)
        values=[f'{row+1:02d}/01',('I','a','DR','CR')[row%4],['Lorem ipsum — a','I amet, dolor.','Adipiscing - elit','A sed (ipsum)'][row%4],f'{12+row*3}.45',f'{100+row}.00']
        for value,x in zip(values,columns):p.line(value,x+3,y,fs,region='table')
    bottom=top-24-count*leading
    c.setLineWidth(.25);c.setStrokeColorRGB(.6,.65,.7)
    for x in columns:c.line(x,top+4,x,bottom)
    c.line(left,bottom,right,bottom)
    # Peripheral genuine rotated content, including punctuation and single characters.
    p.line('REF I A — X9-004 / 2026',W-17,260,8,'Helvetica',angle=90,region='local90')
    p.line('Amount / I',W-145,230,9,'Helvetica-Bold',angle=45,region='local45')
    p.line('Closing balance: 1,456.78',left,bottom-24,12,'Helvetica-Bold',region='labels')
    legal=('Lorem ipsum dolor sit amet, consectetur adipiscing elit. I agree — a statement is provided; '
           'I retain a copy. A - B; (I), [a], 1:0. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ')
    p.line('Terms and information',left,192,9,'Helvetica-Bold',region='legal_heading')
    fs=[7,6,5.5][index]
    p.paragraph(legal*([5,7,8][index]),left,178,right-left,fs,fs*1.25,'legal')
    p.line(f'Synthetic OCR fixture — page {index+1}',left,26,8,region='footer')
    return p

def transform(image,words,regions,angle):
    h,w=image.shape[:2]
    if angle % 90 == 0:
        quarter=int(angle//90)%4
        # Exact pixel permutations: right-angle cases must not add resampling blur.
        out=np.rot90(image,-quarter).copy();nh,nw=out.shape[:2]
        r=np.array([[[1,0,0],[0,1,0]],[[0,-1,h],[1,0,0]],
                    [[-1,0,w],[0,-1,h]],[[0,1,0],[-1,0,w]]][quarter],dtype=float)
    else:
        r=cv2.getRotationMatrix2D((w/2,h/2),-angle,1)
        nw=int(math.ceil(abs(r[0,0])*w+abs(r[0,1])*h));nh=int(math.ceil(abs(r[0,1])*w+abs(r[0,0])*h))
        r[:,2]+=[(nw-w)/2,(nh-h)/2]
        out=cv2.warpAffine(image,r,(nw,nh),flags=cv2.INTER_CUBIC,borderValue=(255,255,255))
    def mapped(items):
        result=copy.deepcopy(items)
        for item in result:
            xy=np.array(item['polygon'])*[w,h];xy=np.c_[xy,np.ones(4)]@r.T
            item['polygon']=(xy/[nw,nh]).tolist()
        return result
    return out,mapped(words),mapped(regions),r.tolist()

def generate(output,dpi):
    output.mkdir(parents=True,exist_ok=True);base=output/'statements.pdf';c=canvas.Canvas(str(base),pagesize=A4,invariant=1)
    specs=[]
    for i in range(3):
        page=statement(c,i);specs.append(page);c.showPage()
    c.save();pages=[]
    with pymupdf.open(base) as doc:
        for i,spec in enumerate(specs):
            pix=doc[i].get_pixmap(dpi=dpi,colorspace=pymupdf.csRGB,alpha=False)
            image=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)
            for angle in [0,-1.5,-.5,.5,1.5,90,180,270,90.5]:
                raster,words,regions,matrix=transform(image,spec.words,spec.regions,angle)
                id=f'statement_{i}_cw{angle:g}';Image.fromarray(raster).save(output/f'{id}.png')
                pages.append(dict(id=id,family=f'statement_{i}',image=f'{id}.png',width=raster.shape[1],height=raster.shape[0],page_rotation_deg=angle,words=words,regions=regions,transform=matrix))
    ladder=output/'text_scale.pdf';c=canvas.Canvas(str(ladder),pagesize=A4,invariant=1);page=Page(c)
    page.line('TEXT SCALE — identical words, changing size',42,H-25,10,region='heading')
    y=H-150
    for size in [144,96,72,48,36,24,18,12,10,8,6]:
        page.line('Sit I a',42,y,size,'Helvetica',region=f'pt_{size}');y-=size*.85+15
    c.showPage();c.save()
    with pymupdf.open(ladder) as doc:
        pix=doc[0].get_pixmap(dpi=dpi,alpha=False);pix.save(output/'text_scale.png')
        pages.append(dict(id='text_scale',family='text_scale',image='text_scale.png',width=pix.width,height=pix.height,page_rotation_deg=0,words=page.words,regions=[]))
    manifest=dict(schema_version=2,dpi=dpi,angle_convention='clockwise in image coordinates; local rotations relative to unrotated page',box_definition='font advance and ascent/descent, not tight ink',pdfs=[base.name,ladder.name],pages=pages)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    selected=[pages[j] for j in [0,3,5,6,9,18,27]];sheet=Image.new('RGB',(1400,800),'#d9dfe6');draw=ImageDraw.Draw(sheet)
    for j,p in enumerate(selected):
        x=(j%4)*350;y=(j//4)*400;draw.text((x+8,y+8),p['id'],fill='black')
        with Image.open(output/p['image']) as image:sheet.paste(ImageOps.contain(image,(330,365)),(x+8,y+28))
    sheet.save(output/'contact.png');print(f'{len(pages)} labelled cases in {output}')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('output/pdf/quality'));parser.add_argument('--dpi',type=int,default=300);a=parser.parse_args();generate(a.output,a.dpi)

