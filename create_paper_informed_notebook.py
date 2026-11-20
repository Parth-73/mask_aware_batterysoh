"""Build the documented paper-informed comparison notebook."""
from pathlib import Path
import nbformat as nf

root = Path(__file__).resolve().parent
old = nf.read(root/'CALCE_CNN_LSTM_Original_Features.ipynb', 4)
md = nf.v4.new_markdown_cell
code = nf.v4.new_code_cell
cells = [md('''# Paper-informed CNN-LSTM: charging comparison and discharge diagnostic

Source: [Ma et al., Sustainability 17, 4014](sustainability-17-04014-v2.pdf).
Read pages 6, 8–10, 12, 14–16. The local PDF was extracted to `sustainability_paper_text.txt`.

| Setting | Paper | This notebook |
|---|---|---|
| Cells | CS2, nominal 1.1 Ah | Available CX2, validated 1.35 Ah labels |
| Inputs | H1 time at minimum discharge voltage; H2 mean discharge time; H3 mean discharge voltage | Both original five charging features and a separate explicit H1–H3 approximation |
| Split | Train CS2_34/36; test CS2_33/35 | Existing cell-disjoint train/validation/test, unchanged |
| CNN | Three convolution/pooling stages; dimensions 3×3, 5×4, 9×32 | Serial temporal Conv1d kernels 3/5/9, channels 3/4/32, ReLU and same-length max pooling |
| LSTM | Two layers shown, hidden size 256 | Two layers, 256 units |
| Outliers | K-means; complete rejection decision rule not specified | No target filtering or smoothing; validated original rows |

**This is an adaptation, not an exact reproduction.** The paper does not specify window length,
optimizer, learning rate, dropout rate, normalization or reproducible outlier rejection thresholds.
Figure 10 is ambiguous about convolution connectivity and displays softmax, while the text
specifies regression. Here we use serial stages and a linear regression head.

Engineering improvements beyond the paper: training-only target standardization, a linear
skip from the current features, validation-based learning-rate reduction, checkpointing and
early stopping. The skip lets the CNN-LSTM learn nonlinear temporal corrections to a direct estimate.
No hyperparameters or seeds are selected using the test set.

**Availability:** the charging experiment estimates SOH after charging. The discharge experiment
requires the completed same-cycle discharge. At constant current, duration is closely related to
the discharge capacity defining SOH. Its error is not an improvement at the charging prediction cutoff.
Neither experiment forecasts future SOH. Previously examined validation/test cells are not a fresh benchmark.
''')]
config = old.cells[2].source.replace('cnn_lstm_original_features_results', 'cnn_lstm_paper_informed_results')
config = config.replace('EPOCHS=100','EPOCHS=150').replace('PATIENCE=15','PATIENCE=30')
cells += [code(config)]
load = old.cells[3].source.split('scaler=')[0]
load += '''
base_frames = {'train':train.copy(), 'validation':val.copy(), 'test':test.copy()}
BASE_OUTPUT = OUTPUT_DIR
CHARGING_FEATURES = FEATURES.copy()
DISCHARGE_FEATURES = ['h1_min_voltage_time_s','h2_mean_discharge_time_s','h3_mean_discharge_voltage_v']
feature_path = BASE_OUTPUT/'discharge_features.csv'
if not feature_path.exists():
    raise FileNotFoundError('Run extract_paper_discharge_features.py with the base Anaconda Python first')
discharge_table = pd.read_csv(feature_path)
assert discharge_table.cycle_id.is_unique
print('Extracted discharge features:',discharge_table[DISCHARGE_FEATURES].notna().all(axis=1).sum())
'''
cells += [code(load)]
window = old.cells[4].source.split('Xtr,ytr,mtr=')[0]
cells += [code(window)]
cells += [md('''## Explicit H1–H3 definitions

The paper names H1–H3 without extraction equations. We define elapsed discharge time
as recorder `test_s - first_test_s + first_step_s`; H1 is its value at minimum measured
discharge voltage, H2 is its sample mean, and H3 is sample-mean voltage. These choices
depend on sampling cadence and are documented approximations. We accept a single
recorded discharge step with at least five samples and strictly increasing elapsed time.
Cycles with unresolved sources or unavailable features are excluded and counted.
Targets are the existing validated labels, never recalculated from these predictors.
''')]
cells += [code('''class PaperInspiredCNNLSTM(nn.Module):
    def __init__(self,n_features=None,hidden=256,dropout=0.1):
        super().__init__()
        n_features = len(FEATURES) if n_features is None else n_features
        layers=[]; channels=n_features
        for kernel,out_channels in [(3,3),(5,4),(9,32)]:
            layers += [nn.Conv1d(channels,out_channels,kernel,padding=kernel//2),
                       nn.ReLU(), nn.MaxPool1d(3,stride=1,padding=1)]
            channels=out_channels
        self.cnn=nn.Sequential(*layers)
        self.lstm=nn.LSTM(32,hidden,num_layers=2,dropout=dropout,batch_first=True)
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(hidden,1))
        self.skip=nn.Linear(n_features,1)
        self.register_buffer('target_mean',torch.tensor(float(train.soh.mean())))
        self.register_buffer('target_std',torch.tensor(float(train.soh.std(ddof=0))))
    def forward(self,x):
        z=self.cnn(x.transpose(1,2)).transpose(1,2)
        z,_=self.lstm(z)
        normalized=self.head(z[:,-1,:])+self.skip(x[:,-1,:])
        return self.target_mean+self.target_std*normalized.squeeze(-1)
''')]
training = old.cells[7].source
training = training.replace('loss_fn=nn.MSELoss()', '''loss_fn=lambda pred,true: torch.mean(((pred-true)/net.target_std)**2)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.5,patience=7,min_lr=1e-5)''')
training = training.replace("history.append({'epoch':epoch,'train_mse':total/count,'val_mse':val_mse})", "history.append({'epoch':epoch,'train_mse':total/count*float(net.target_std)**2,'val_mse':val_mse,'lr':opt.param_groups[0]['lr']})\n        scheduler.step(val_mse)")
training = training.replace("'architecture':'adapted parallel CNN LSTM'", "'architecture':'paper-informed serial CNN LSTM with current-feature linear skip'")
training = training.replace("pd.DataFrame(history).to_csv(OUTPUT_DIR/f'history_seed{seed}.csv',index=False)", "pd.DataFrame(history).to_csv(OUTPUT_DIR/f'history_seed{seed}.csv',index=False)")
cells += [code(training)]
cells += [code('''combined=[]
for experiment, columns in [('charging',CHARGING_FEATURES),('discharge_diagnostic',DISCHARGE_FEATURES)]:
    FEATURES=columns
    OUTPUT_DIR=BASE_OUTPUT/experiment
    OUTPUT_DIR.mkdir(exist_ok=True)
    frames={}
    coverage=[]
    for split,base in base_frames.items():
        f=base.copy()
        if experiment=='discharge_diagnostic':
            f=f.merge(discharge_table[['cycle_id']+FEATURES],on='cycle_id',how='left',validate='one_to_one')
        good=np.isfinite(f[FEATURES].to_numpy(dtype=float)).all(axis=1)
        frames[split]=f.loc[good].copy()
        coverage.append({'split':split,'original_rows':len(f),'feature_rows':int(good.sum())})
    train,val,test=(frames[x] for x in ['train','validation','test'])
    scaler=StandardScaler().fit(train[FEATURES].to_numpy(dtype=float))
    (OUTPUT_DIR/'scaler.json').write_text(json.dumps({'features':FEATURES,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist()},indent=2))
    Xtr,ytr,mtr=make_windows(train); Xva,yva,mva=make_windows(val); Xte,yte,mte=make_windows(test)
    for row,n in zip(coverage,[len(ytr),len(yva),len(yte)]): row['windows']=n
    pd.DataFrame(coverage).to_csv(OUTPUT_DIR/'coverage.csv',index=False)
    display(pd.DataFrame(coverage))
    (OUTPUT_DIR/'config.json').write_text(json.dumps({'experiment':experiment,'features':FEATURES,'seeds':SEEDS,'epochs':EPOCHS,'patience':PATIENCE,'learning_rate':LR,'batch_size':BATCH_SIZE,'sequence_length':SEQ_LEN,'selection':'validation MSE only','outlier_filter':'none','prediction_cutoff':'charge end' if experiment=='charging' else 'discharge end'},indent=2))
    all_rows=[]; histories={}
    for seed in SEEDS:
        rows,hist=train_one(seed)
        all_rows.extend(rows);histories[seed]=hist
    results=pd.DataFrame(all_rows)
    results.to_csv(OUTPUT_DIR/'metrics_all_runs.csv',index=False)
    results['experiment']=experiment
    combined.append(results)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for seed,h in histories.items():
        h=pd.DataFrame(h)
        axs[0].plot(h.epoch,np.sqrt(h.val_mse)*100,label=str(seed))
    axs[0].set(xlabel='Epoch',ylabel='Validation RMSE (pp)',title=experiment)
    axs[0].legend()
    p=pd.read_csv(OUTPUT_DIR/'predictions_test_seed42.csv')
    axs[1].plot(p.cycle_ordinal,p.soh*100,label='Measured')
    axs[1].plot(p.cycle_ordinal,p.prediction*100,label='Predicted (seed 42)')
    axs[1].set(xlabel='Original cycle ordinal',ylabel='SOH (%)');axs[1].legend()
    plt.tight_layout();plt.savefig(OUTPUT_DIR/'training_and_test.png',dpi=160);plt.show()
comparison=pd.concat(combined,ignore_index=True)
comparison.to_csv(BASE_OUTPUT/'metrics_all_experiments.csv',index=False)
display(comparison.groupby(['experiment','split'])[['mae_pp','rmse_pp','r2']].agg(['mean','std']))
''')]
cells += [code('''# Compare charging models on exactly the same test target IDs for each seed.
matched=[]
for seed in SEEDS:
    a=pd.read_csv(ROOT/'cnn_lstm_original_features_results'/f'predictions_test_seed{seed}.csv')
    b=pd.read_csv(BASE_OUTPUT/'charging'/f'predictions_test_seed{seed}.csv')
    j=a.merge(b,on=['battery','cycle_ordinal'],suffixes=('_old','_new'),validate='one_to_one')
    assert len(j)==len(a)==len(b), 'Mismatched test coverage'
    assert np.allclose(j.soh_old,j.soh_new)
    for name in ['old','new']:
        matched.append({'seed':seed,'model':name,'n':len(j),**metrics(j.soh_old,j['prediction_'+name])})
matched=pd.DataFrame(matched)
matched.to_csv(BASE_OUTPUT/'matched_charging_comparison.csv',index=False)
display(matched)
print('Mean charging RMSE by model (pp):')
display(matched.groupby('model').rmse_pp.mean())
''')]
cells += [md('''## How to read the results

Use `matched_charging_comparison.csv` to assess improvement at the original charging cutoff.
Compare the mean and spread across all three seeds, not just the best seed.
The discharge diagnostic measures a different information setting and must be reported separately.
The paper's 0.339 and 0.239 pp RMSE belong to CS2_33 and CS2_35, respectively (Tables 4–5),
not these CX2 test windows. No exact reproduction or guaranteed lower RMSE is claimed.
''')]
nb=nf.v4.new_notebook(cells=cells,metadata=old.metadata)
nf.validate(nb)
nf.write(nb,root/'CALCE_CNN_LSTM_Paper_Informed.ipynb')
print('Created paper-informed notebook')
