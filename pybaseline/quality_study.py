"""Exploratory quality study using identical ONNX weights; no changes to baseline defaults."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('DOCTR_CACHE_DIR',str(ROOT/'.cache/doctr'))
import argparse,json,time,math
import cv2,numpy as np,torch,onnxruntime as ort
from PIL import Image
from doctr.models.preprocessor import PreProcessor
from doctr.models.detection.differentiable_binarization.base import DBPostProcessor
from doctr.models.detection._utils import _remove_padding
from doctr.models.recognition.parseq.pytorch import PARSeqPostProcessor
from doctr.models.recognition.predictor._utils import split_crops,remap_preds
from doctr.utils.geometry import extract_crops
from pybaseline.metrics import score_words,detection_summary

class Engine:
    def __init__(self):
        torch.set_num_threads(1);cv2.setNumThreads(1)
        ort.set_default_logger_severity(3)
        ort.preload_dlls(directory=str(ROOT/'.venv-baseline/Lib/site-packages/torch/lib'))
        meta=json.loads((ROOT/'models/metadata.json').read_text());options=ort.SessionOptions();options.intra_op_num_threads=1;options.log_severity_level=3
        self.det=ort.InferenceSession(str(ROOT/'models/db_resnet34.onnx'),sess_options=options,providers=[('CUDAExecutionProvider',{'use_tf32':'0','cudnn_conv_algo_search':'HEURISTIC','gpu_mem_limit':str(4*2**30)})])
        self.rec=ort.InferenceSession(str(ROOT/'models/parseq.onnx'),sess_options=options,providers=[('CUDAExecutionProvider',{'use_tf32':'0','cudnn_conv_algo_search':'HEURISTIC','gpu_mem_limit':str(3*2**30)})])
        assert self.det.get_providers()[0]==self.rec.get_providers()[0]=='CUDAExecutionProvider'
        self.meta=meta;self.post=DBPostProcessor(assume_straight_pages=True)
        self.rpre=PreProcessor((32,128),256,preserve_aspect_ratio=True,symmetric_pad=False,mean=meta['parseq']['mean'],std=meta['parseq']['std'])
        self.rpost=PARSeqPostProcessor(vocab=meta['parseq']['vocab'])
    def detect(self,image,size):
        pre=PreProcessor((size,size),1,preserve_aspect_ratio=True,symmetric_pad=True,mean=self.meta['db_resnet34']['mean'],std=self.meta['db_resnet34']['std'])
        tensor=pre([image])[0].numpy();logits=self.det.run(None,{'input':tensor})[0];prob=1/(1+np.exp(-np.clip(logits,-80,80)))
        raw=self.post(prob.transpose(0,2,3,1))[0][0]
        boxes=_remove_padding([image],[{'words':raw}],True,True,True)[0]['words']
        return boxes,prob[0,0]
    def recognize(self,image,boxes):
        crops=extract_crops(image,boxes[:,:4]);crops,mapping,remap=split_crops(crops,8,6,.5);text=[]
        for batch in self.rpre(crops) if crops else []:text.extend(self.rpost(torch.from_numpy(self.rec.run(None,{'input':batch.numpy()})[0])))
        if remap:text=remap_preds(text,mapping,.5)
        return [dict(polygon=[[float(b[0]),float(b[1])],[float(b[2]),float(b[3])]],objectness=float(b[4]),text=t,confidence=float(c)) for b,(t,c) in zip(boxes,text)]
    def run(self,image,size):
        boxes,_=self.detect(image,size);return self.recognize(image,boxes)

def rotate(image,angle):
    h,w=image.shape[:2];m=cv2.getRotationMatrix2D((w/2,h/2),angle,1);nw=int(math.ceil(abs(m[0,0])*w+abs(m[0,1])*h));nh=int(math.ceil(abs(m[0,1])*w+abs(m[0,0])*h));m[:,2]+=[(nw-w)/2,(nh-h)/2]
    return cv2.warpAffine(image,m,(nw,nh),flags=cv2.INTER_CUBIC,borderValue=(255,255,255)),m

def skew(image):
    scale=min(1,1500/max(image.shape[:2]));gray=cv2.cvtColor(cv2.resize(image,None,fx=scale,fy=scale),cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(gray,60,160);lines=cv2.HoughLinesP(edges,1,np.pi/3600,threshold=60,minLineLength=gray.shape[1]*.12,maxLineGap=15)
    pairs=[]
    for x0,y0,x1,y1 in np.asarray(lines).reshape(-1,4) if lines is not None else []:
        a=math.degrees(math.atan2(y1-y0,x1-x0));a=(a+45)%90-45
        if abs(a)<8:pairs.append((a,math.hypot(x1-x0,y1-y0)))
    if not pairs:return 0.,0
    pairs.sort();total=sum(w for a,w in pairs);acc=0
    for a,w in pairs:
        acc+=w
        if acc>=total/2:return a,len(pairs)

def remap(words,m,source_shape,target_shape):
    inv=cv2.invertAffineTransform(m);h,w=source_shape[:2];th,tw=target_shape[:2];out=[]
    for word in words:
        points=np.array(word['polygon'])
        if len(points)==2:
            (x0,y0),(x1,y1)=points;points=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
        xy=np.c_[points*[tw,th],np.ones(4)]@inv.T
        out.append(dict(word,polygon=(xy/[w,h]).tolist()))
    return out

def offsets(words,box,shape):
    x,y,w,h=box;ih,iw=shape[:2];out=[]
    for word in words:
        points=np.array(word['polygon']);points=(points*[w,h]+[x,y])/[iw,ih];out.append(dict(word,polygon=points.tolist()))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-inference',action='store_true',help='Explicitly opt into GPU inference after reviewing the plan');p.add_argument('--fixtures',type=Path,default=Path('output/pdf/quality'));p.add_argument('--output',type=Path,default=Path('pybaseline/results/quality'));a=p.parse_args()
    if not a.run_inference:p.error('GPU study is opt-in: pass --run-inference only when the GPU is available')
    a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.fixtures/'manifest.json').read_text());engine=Engine();rows=[]
    cases=[q for q in manifest['pages'] if q['id']=='text_scale' or q['id'] in ['statement_0_cw0','statement_1_cw0','statement_2_cw0']]
    for page in cases:
        image=np.array(Image.open(a.fixtures/page['image']).convert('RGB'))
        for size in [1024,1536,2048,2560]:
            path=a.output/f"{page['id']}_size{size}.json"
            if path.exists():rows.append(json.loads(path.read_text()));continue
            t=time.perf_counter();words=engine.run(image,size);elapsed=time.perf_counter()-t
            pairs=score_words(page['words'],words,return_pairs=True)['pairs']
            scores={}
            for region in sorted({w['region'] for w in page['words']}):
                ids={i for i,w in enumerate(page['words']) if w['region']==region}
                matched=[(i,j) for i,j in pairs if i in ids]
                scores[region]=dict(truth=len(ids),matched=len(matched),exact=sum(page['words'][i]['text']==words[j]['text'] for i,j in matched))
            row=dict(page=page['id'],size=size,seconds_including_first_shape_warmup=elapsed,words=words,score=detection_summary([score_words(page['words'],words)]),by_region=scores)
            path.write_text(json.dumps(row,indent=2));rows.append(row);print(page['id'],size,row['score']['matched'],row['score']['exact'],round(elapsed,2),flush=True)
    (a.output/'scale_results.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
