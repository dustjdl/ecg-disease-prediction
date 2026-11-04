"""Download historical IDs from the official PhysioNet HTTP redirect destination.
The official /files/ endpoint redirected to physionet-open.s3.amazonaws.com during this run.
Rate limited to at most two requests/second; reuse completed local files.
"""
from pathlib import Path
import concurrent.futures,json,time,urllib.request,hashlib
import pandas as pd
import requests, threading
LOCAL=threading.local()
RATE_LOCK=threading.Lock()
NEXT_REQUEST=[0.0]
def pace():
 with RATE_LOCK:
  now=time.monotonic();delay=max(0,NEXT_REQUEST[0]-now);NEXT_REQUEST[0]=max(now,NEXT_REQUEST[0])+0.5
 if delay:time.sleep(delay)
R=Path(__file__).resolve().parents[1]
def main():
 d=pd.read_csv(R/'preserved/labels_v1/ecg_labels_all_1000.csv')
 m=pd.read_csv(R/'recovery/ptbxl_database.csv').set_index('ecg_id')
 for k in ['patient_id','strat_fold','filename_lr']:
  assert (d[k].to_numpy()==m.loc[d.ecg_id,k].to_numpy()).all(),k
 jobs=[str(x)+ext for x in d.filename_lr for ext in ['.hea','.dat']]
 def one(rel):
  p=R/'recovery/raw'/rel;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists() and p.stat().st_size>0:return rel
  last=None
  for attempt in range(3):
   try:
    if not hasattr(LOCAL,'session'): LOCAL.session=requests.Session()
    pace()
    res=LOCAL.session.get('https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/'+rel,timeout=30);res.raise_for_status();b=res.content
    assert len(b)>0
    p.with_suffix(p.suffix+'.part').write_bytes(b);p.with_suffix(p.suffix+'.part').rename(p);return rel
   except Exception as e:
    last=e
    time.sleep(min(30,5*(attempt+1)))
  raise RuntimeError(f'{rel}: {last}')
 errors=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
  futures={ex.submit(one,j):j for j in jobs}
  for n,f in enumerate(concurrent.futures.as_completed(futures),1):
   try:f.result()
   except Exception as e:errors.append(str(e))
   if n%100==0:print('Downloaded/checked',n,'/',len(jobs),'errors',len(errors),flush=True)
 (R/'reports/download_status.json').write_text(json.dumps({'files':len(jobs),'errors':errors},indent=2))
 if errors:raise RuntimeError(f'{len(errors)} downloads failed; see download_status.json')
if __name__=='__main__':main()
