from cs2_research_pipeline import ROOT,OUT,read_workbook
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(4,1,figsize=(12,12))
for ax,b in zip(axes,['CS2_33','CS2_34','CS2_3','CS2_5']):
 p=next((ROOT/b).rglob('*.xlsx'));frames,_=read_workbook(p);d=pd.concat(frames).sort_values('t');cut=d[(d.i<-.03)&(d.v<=2.72)]
 if len(cut):d=d[d.t<=cut.t.iloc[0]]
 x=(d.t-d.t.iloc[0])/3600;ax.plot(x,d.v,color='navy',lw=1,label='Voltage');ax.set_ylabel('V');ax.set_title(f'{b}: {p.name}',fontsize=9)
 other=ax.twinx();other.plot(x,d.i,color='darkorange',lw=.8,alpha=.7,label='Current');other.set_ylabel('A');ax.set_xlabel('Observed test time (hours)')
fig.tight_layout();fig.savefig(OUT/'representative_raw_profiles.png',dpi=150);plt.close(fig)
print('Representative raw profile plot saved')
