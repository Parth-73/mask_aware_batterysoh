import cs2_research_pipeline as p
import pandas as pd
from tqdm import tqdm
rows=[];inv=[]
for path in tqdm([x for x in p.discover() if x.suffix=='.txt'],desc='CADEX clock audit'):
 frames,inventory=p.read_workbook(path);inv.extend(inventory)
 if frames: rows.extend(p.reconstruct(path,frames))
d=pd.DataFrame(rows);i=pd.DataFrame(inv)
i.to_csv(p.OUT/'cadex_clock_audit.csv',index=False);d.to_csv(p.OUT/'cadex_episode_audit.csv',index=False)
print(i.status.value_counts().to_string());print(d.groupby('battery').soh.agg(['size','count','mean']).to_string());print(d.reason.value_counts().head(8).to_string())
