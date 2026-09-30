"""Independent fixtures frozen after selecting the one-hop anchor rule."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from scripts.generate_document_fixtures import transform
from scripts.tune_dense_detection import cache,read,write,BASE,call,recognize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False);fixtures=out/'fixtures';fixtures.mkdir()
    pages=[]
    for variant,(face,size) in enumerate([('calibri.ttf',25),('verdana.ttf',22)]):
        image=Image.new('RGB',(2481,3508),'white');draw=ImageDraw.Draw(image);words=[]
        def line(text,x,y,size,region,fill='black'):
            font=ImageFont.truetype('C:/Windows/Fonts/'+face,size)
            for word in text.split():
                l,t,r,b=font.getbbox(word,anchor='ls');draw.text((x,y),word,font=font,fill=fill,anchor='ls')
                words.append(dict(text=word,region=region,polygon=[[(x+l)/2481,(y+t)/3508],[(x+r)/2481,(y+t)/3508],[(x+r)/2481,(y+b)/3508],[(x+l)/2481,(y+b)/3508]]))
                x+=font.getlength(word)+size*.42
        line('DELIVERY AND ACCOUNT SUMMARY',155,190,60,'title')
        line('Customer reference QZ-7008 Date 23/09/2026',155,285,34,'labels')
        for i in range(10):
            y=435+i*135
            line(f'Batch {i} Boxes {9-i} Status dispatched',180,y,31,'numeric_fields')
            line('I received a parcel and I retained the delivery note.',180,y+48,27,'body')
            draw.line((160,y+70,2300,y+70),fill='#999999',width=2)
        for x in [145,1250,2320]:draw.line((x,350,x,1765),fill='black',width=3)
        draw.rectangle((150,1880,2320,2060),fill='black')
        line('I confirm a replacement is required. Case number 7 remains open.',185,1980,31,'reversed_text','white')
        # Negative controls: bars and long rules, no unlabelled letter-shaped pseudo-text.
        for j in range(90):
            x=180+j*10;draw.rectangle((x,2170,x+(2 if j%4 else 5),2230),fill='black')
        line('Reference copy enclosed',180,2290,28,'labels')
        draw.line((1300,2160,2300,2160),fill='black',width=2)
        for y in [2200,2250]:
            for x in range(1320,2300,30):draw.line((x,y,x+15,y),fill='black',width=2)
        text='I understand a delivery charge applies. If I dispute an item, a revised invoice will follow; I retain this record.'
        for row in range(17):line(text,155,2530+row*45,size,'legal')
        if variant: image=image.filter(ImageFilter.GaussianBlur(.45))
        for angle in [0,-.8,180]:
            raster,truth,_,_=transform(np.array(image),words,[],angle);name=f'anchor_holdout_{variant}_cw{angle:g}';path=fixtures/f'{name}.png';Image.fromarray(raster).save(path)
            pages.append(dict(id=name,image=str(path.resolve()),width=raster.shape[1],height=raster.shape[0],page_rotation_deg=angle,words=truth,font=face,blur_radius=.45 if variant else 0))
    write(out/'frozen_manifest.json',pages)
    source=out/'source';cache(source,pages);write(source/'grid.json',[BASE])
    call(['--cache',source/'cache','--grid',source/'grid.json','--output',source/'boxes.json','--thin-recovery'])
    rows=recognize(source,read(source/'boxes.json'),[0])
    for row in rows:
        row['words']=[w for w in row['words'] if not w.get('thin_recovery') or (w['confidence']>=.9 and 1<=len(w['text'])<=2 and w['text'].isalnum())]
    write(source/'recognition.json',rows)


if __name__=='__main__':main()
