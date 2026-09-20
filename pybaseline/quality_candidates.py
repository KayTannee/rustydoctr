"""Find previously uncovered, non-horizontal text groups from image components."""
import numpy as np,cv2,math

def candidates(image,words):
    h,w=image.shape[:2];scale=min(1,1800/max(h,w));gray=cv2.cvtColor(cv2.resize(image,None,fx=scale,fy=scale),cv2.COLOR_RGB2GRAY);sh,sw=gray.shape
    known=np.zeros_like(gray)
    for word in words:
        pts=np.array(word['polygon'])*[sw,sh]
        if len(pts)==2:
            (x0,y0),(x1,y1)=pts;pts=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
        cv2.fillPoly(known,[pts.astype(np.int32)],255)
    _,labels,stats,centers=cv2.connectedComponentsWithStats((gray<160).astype(np.uint8),8)
    valid=[];mask=np.zeros_like(gray)
    for idx,(x,y,cw,ch,area) in enumerate(stats[1:],1):
        cx,cy=centers[idx];cx=min(sw-1,int(cx));cy=min(sh-1,int(cy))
        if 3<=area and 2<=cw<=80 and 2<=ch<=80 and max(cw,ch)/min(cw,ch)<5 and known[cy,cx]==0:
            valid.append(idx);mask[labels==idx]=255
    results=[]
    for angle in [-90,-45,45]:
        a=math.radians(angle);kernel=np.zeros((19,19),np.uint8);dx,dy=round(math.cos(a)*8),round(math.sin(a)*8);cv2.line(kernel,(9-dx,9-dy),(9+dx,9+dy),1,3)
        grouped=cv2.dilate(mask,kernel);contours,_=cv2.findContours(grouped,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            rect=cv2.minAreaRect(contour);(cx,cy),(rw,rh),ra=rect;long=max(rw,rh);short=min(rw,rh)
            if short<=0 or long<35 or long/short<2.3:continue
            box=cv2.boxPoints(rect);d=np.roll(box,-1,axis=0)-box;edge=d[np.linalg.norm(d,axis=1).argmax()];actual=(math.degrees(math.atan2(edge[1],edge[0]))+90)%180-90
            difference=abs((actual-angle+90)%180-90)
            if difference>15:continue
            # Require multiple distinct character components, rejecting isolated lines.
            ids={int(labels[min(sh-1,int(y)),min(sw-1,int(x))]) for x,y in centers[valid] if cv2.pointPolygonTest(contour,(float(x),float(y)),False)>=0}
            if len(ids)<2:continue
            # Recover neighbouring letters lost to a partial detector box.
            basis=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]]);points=box@basis;lo=points.min(axis=0)-[24,6];hi=points.max(axis=0)+[24,6]
            quad=np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]])@basis.T
            quad[:,0]=np.clip(quad[:,0],0,sw-1);quad[:,1]=np.clip(quad[:,1],0,sh-1)
            results.append(dict(angle=angle,polygon=(quad/[sw,sh]).tolist(),components=len(ids),score=float(long/short*len(ids))))
    results.sort(key=lambda r:r['score'],reverse=True)
    selected=[]
    for r in results:
        p=np.array(r['polygon']);center=p.mean(axis=0)
        if any(np.linalg.norm(center-np.array(s['polygon']).mean(axis=0))<.025 for s in selected):continue
        selected.append(r)
        if len(selected)==6:break
    merged=[]
    for r in selected:
        joined=False
        for old in merged:
            if old['angle']!=r['angle']:continue
            a=math.radians(r['angle']);basis=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
            p=np.array(r['polygon'])*[w,h]@basis;q=np.array(old['polygon'])*[w,h]@basis
            if min(p[:,0].max(),q[:,0].max())>=max(p[:,0].min(),q[:,0].min()) and min(p[:,1].max(),q[:,1].max())>=max(p[:,1].min(),q[:,1].min()):
                both=np.r_[p,q];lo=both.min(axis=0);hi=both.max(axis=0)
                quad=np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]])@basis.T
                old['polygon']=(quad/[w,h]).tolist();old['components']+=r['components'];joined=True;break
        if not joined:merged.append(r)
    return merged
