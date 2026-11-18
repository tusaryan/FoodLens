import sys, glob, os, re, datetime as dt
import numpy as np, pandas as pd
from PIL import Image, ExifTags
from scipy import ndimage as ndi
root=sys.argv[1]; out=sys.argv[2]
rows=[]
for p in sorted(glob.glob(f"{root}/**/*.jpg", recursive=True)):
    im=Image.open(p); ex=im.getexif(); sub=ex.get_ifd(0x8769)
    e={ExifTags.TAGS.get(k,k):v for k,v in sub.items()}
    im.draft('RGB',(im.size[0]//8,im.size[1]//8)); a=np.asarray(im.convert('RGB'),dtype=np.float32)
    m=a.mean((0,1)); g=a.max(2)
    thr=np.percentile(g,99.5)
    sat=(a>=250).any(2).mean()
    # bright spot count: local top-hat style threshold on max-channel
    bg=ndi.uniform_filter(g,31); spots=(g-bg)>25
    lab,n=ndi.label(spots)
    sizes=ndi.sum(spots,lab,range(1,n+1)) if n else np.array([])
    s=os.path.basename(os.path.dirname(p)); t=re.search(r"(\d{8})_(\d{6})",p)
    rows.append(dict(sample=s,file=os.path.basename(p),ts=dt.datetime.strptime(t.group(1)+t.group(2),"%Y%m%d%H%M%S"),
      w=im.size[0],h=im.size[1],exp=float(e.get('ExposureTime',0)),iso=e.get('ISOSpeedRatings'),wb=e.get('WhiteBalance'),
      em=e.get('ExposureMode'),fl=float(e.get('FocalLength',0)),R=m[0],G=m[1],B=m[2],maxmean=g.mean(),p995=thr,sat=sat,
      nspots=n,spot_med=float(np.median(sizes)) if n else 0, frac_dark=(g<15).mean()))
df=pd.DataFrame(rows); df['dom']=df[['R','G','B']].idxmax(axis=1)
df['dom_ratio']=df[['R','G','B']].max(1)/(df[['R','G','B']].sum(1)+1e-6)
df.to_csv(out,index=False); print(len(df))
