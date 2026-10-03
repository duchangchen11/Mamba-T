"""Acquire pinned public trajectory files; reuse existing local files."""
import hashlib, json, sys
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.eth_ucy_dataset import ROOT, FILES

def main():
    raw=ROOT/'data/raw/eth_ucy';raw.mkdir(parents=True,exist_ok=True)
    commit='333d3a57b4d2705e129b21aefefa09c79b2b9ae1'
    repo='https://github.com/abduallahmohamed/Social-STGCNN'
    manifest_path=raw/'dataset_manifest.json'
    if manifest_path.exists():
        m=json.loads(manifest_path.read_text())
        for item in m['files']:
            p=raw/item['file']
            assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
        print('Existing files verified; original acquisition provenance preserved.');return
    records=[]
    for names in FILES.values():
        for name in names:
            url=f'https://raw.githubusercontent.com/abduallahmohamed/Social-STGCNN/{commit}/datasets/raw/all_data/{name}'
            p=raw/name;existed=p.exists()
            if not existed:
                with urlopen(url,timeout=60) as r:payload=r.read()
                tmp=p.with_suffix('.part');tmp.write_bytes(payload);tmp.replace(p)
            records.append({'file':name,'source_url':url,'status':'already_present' if existed else 'downloaded','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
    manifest_path.write_text(json.dumps({'source_repository':repo,'source_commit':commit,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'files':records},indent=2))
if __name__=='__main__':main()
