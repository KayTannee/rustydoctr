"""Fresh ordinary-character and negative-control fixtures for thin recovery."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts.generate_document_fixtures import transform
from scripts.tune_dense_detection import cache,postprocess,recognize,write,BASE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    out=args.output or ROOT/'pybaseline/results'/('thin_fresh_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False);fixtures=out/'fixtures';fixtures.mkdir();pages=[]
    for face in ['arial.ttf','times.ttf']:
        image=Image.new('RGB',(2481,3508),'white');draw=ImageDraw.Draw(image);words=[]
        def line(text,x,y,size,region):
            font=ImageFont.truetype('C:/Windows/Fonts/'+face,size)
            for word in text.split():
                left,top,right,bottom=font.getbbox(word,anchor='ls')
                draw.text((x,y),word,font=font,fill='black',anchor='ls')
                words.append(dict(text=word,region=region,polygon=[[(x+left)/2481,(y+top)/3508],[(x+right)/2481,(y+top)/3508],[(x+right)/2481,(y+bottom)/3508],[(x+left)/2481,(y+bottom)/3508]]))
                x+=float(font.getlength(word))+size*.4
        line('SERVICE AND PAYMENT RECORD',160,180,60,'title')
        line('Customer reference AB-2048 Statement September 2026',160,280,32,'labels')
        for i in range(10):
            y=460+i*170
            line(f'Item {i} Quantity {i} Status confirmed',170,y,32,'numeric_fields')
            line('I confirm receipt and a copy is retained. I request a statement.',170,y+57,28,'body')
            draw.line((155,y+85,2300,y+85),fill='#aaa',width=2)
        for x in [140,1250,2320]:draw.line((x,365,x,2150),fill='black',width=2)
        # Unlabelled non-text: deterministic long, short barcode-like stripes.
        for i in range(85):
            x=180+i*9
            draw.rectangle((x,2300,x+(2 if i%3 else 5),2360),fill='black')
        for i in range(9):
            line('I agree a payment is due. Please retain a copy. If I request a change, send a revised record.',160,2600+i*65,24,'legal')
        for angle in [0,.6,90]:
            raster,truth,_,_=transform(np.array(image),words,[],angle)
            name=f'thin_{Path(face).stem}_cw{angle:g}';path=fixtures/f'{name}.png';Image.fromarray(raster).save(path)
            pages.append(dict(id=name,image=str(path.resolve()),width=raster.shape[1],height=raster.shape[0],page_rotation_deg=angle,words=truth))
    write(out/'frozen_manifest.json',pages)
    source=out/'baseline';cache(source,pages);rows=postprocess(source,[BASE]);recognize(source,rows,[0])


if __name__=='__main__':main()
