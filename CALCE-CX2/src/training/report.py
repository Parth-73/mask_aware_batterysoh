"""Publication figures and scientifically bounded summaries of completed runs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import FEATURES,SPLITS,SEEDS,json_write
from .evaluate import metrics

LABELS={'B1':'Vanilla','B2':'Stacked','B3':'Bidirectional','B4':'Attention'}


def table(data):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.6g}' if np.isfinite(v) else 'NA'
        return str(v)
    return '\n'.join(['| '+' | '.join(data.columns)+' |','| '+' | '.join(['---']*len(data.columns))+' |']+['| '+' | '.join(fmt(v) for v in r)+' |' for r in data.itertuples(index=False,name=None)])


def save(fig,path):
    fig.tight_layout();fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight');fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)


def comparison(data,column,labels,path,title):
    g=data.groupby(column).rmse_percentage_points.agg(['mean','std']).sort_index()
    fig,ax=plt.subplots(figsize=(6.4,4));positions=np.arange(len(g))
    ax.bar(positions,g['mean'],yerr=g['std'],capsize=4,color='#377eb8',alpha=.85)
    ax.set_xticks(positions,[labels.get(str(x),str(x)) for x in g.index]);ax.set(ylabel='Validation RMSE (SOH percentage points)',title=title)
    ax.spines[['top','right']].set_visible(False);save(fig,path)


def common_sequence_scores(output,validation,selection):
    chosen=validation[(validation.architecture==selection['architecture'])&(validation.feature_set==selection['feature_set'])]
    frames={r.run_id:pd.read_csv(output/'runs'/r.run_id/'validation_predictions.csv') for r in chosen.itertuples()}
    ids=set.intersection(*(set(g.cycle_id) for g in frames.values()))
    rows=[]
    for r in chosen.itertuples():
        p=frames[r.run_id];p=p[p.cycle_id.isin(ids)]
        rows.append(dict(run_id=r.run_id,architecture=r.architecture,feature_set=r.feature_set,sequence_length=r.sequence_length,seed=r.seed,cohort='identical_validation_target_intersection',**metrics(p.actual_soh_fraction,p.predicted_soh_fraction)))
    data=pd.DataFrame(rows);data.to_csv(output/'sequence_common_target_metrics.csv',index=False)
    return data


def literature_summary(output):
    """Resolve post-run paper verification without modifying the frozen experiment."""
    path=Path(output)/'literature_verification.json'
    verified=path.exists() and json.loads(path.read_text()).get('status')=='verified_against_user_supplied_pdf'
    if verified:
        return ('[Wang et al., Batteries 2026, DOI 10.3390/batteries12040115](https://www.mdpi.com/2313-0105/12/4/115) was checked against the user-supplied 21-page PDF. '
                'Table 3 (p. 9) confirms the F1/B2/length-5 predictor dimensions: input 3, sequence 5, layers 2, hidden size 64, output 1. '
                'The partial-charge interval is also confirmed as 3.8-4.0 V at the 4.2 V cutoff (p. 5). '
                'This verifies predictor dimensions and feature types, not the full AW-PINN or every published pure-LSTM comparison. '
                'The paper uses Gaussian IC smoothing (sigma 1), [0,1] feature normalization (p. 6), a within-battery chronological 50/50 split (p. 12), and an aging domain ending at 80% SOH (p. 4). '
                'Our Savitzky-Golay extraction, training-only StandardScaler, independent battery split and retained test targets extending below 80% are different. '
                'Dropout and optimizer details are not reported in the searchable predictor/training settings and are not assumed. '
                'See [verified methodological comparison](LITERATURE_COMPARISON.md), literature_comparison.csv and literature_verification.json for page references, protocol differences and metric-reporting caveats. '
                'The verification changed no model or result, and no inference was repeated. No direct published-performance superiority claim is made.')
    return '[Wang et al., Batteries 2026, DOI 10.3390/batteries12040115](https://www.mdpi.com/2313-0105/12/4/115). The comparison implements the task-specified three inputs, five-step history, two LSTM layers and hidden size 64. Paper-specific verification is recorded separately in literature_verification.json when available. The benchmark uses pure supervised MSE and a battery-level split; it is not an AW-PINN reproduction or an equivalent published-performance comparison. Dropout, optimizer, feature extraction, normalization, cycle selection and split may differ. No superiority claim is made.'


def generate(output):
    output=Path(output);figures=output/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.grid':True,'grid.alpha':.2,'pdf.fonttype':42,'ps.fonttype':42})
    validation=pd.read_csv(output/'validation_results.csv');history=pd.read_csv(output/'training_history.csv');selection=json.loads((output/'final_selection.json').read_text())
    tests=pd.read_csv(output/'test_results.csv')
    feature=selection['feature_set'];architecture=selection['architecture']
    comparison(validation[(validation.architecture=='B1')&(validation.sequence_length==5)],'feature_set',{},figures/'feature_comparison','Feature comparison: Vanilla LSTM, length 5')
    comparison(validation[(validation.feature_set==feature)&(validation.sequence_length==5)],'architecture',LABELS,figures/'architecture_comparison',f'Architecture comparison: {feature}, length 5')
    comparison(validation[(validation.feature_set==feature)&(validation.architecture==architecture)],'sequence_length',{},figures/'sequence_length_comparison',f'Sequence comparison: {LABELS[architecture]}, {feature}')
    common=common_sequence_scores(output,validation,selection)
    comparison(common,'sequence_length',{},figures/'sequence_common_target_comparison','Sequence lengths on identical validation targets')
    for ident,g in history.groupby('run_id'):
        fig,ax=plt.subplots(figsize=(6.4,3.7));ax.plot(g.epoch,g.train_mse_fraction_squared,label='Training');ax.plot(g.epoch,g.validation_mse_fraction_squared,label='Validation')
        best=validation.set_index('run_id').loc[ident,'best_epoch'];ax.axvline(best,color='grey',linestyle=':',label='Selected epoch')
        ax.set(xlabel='Epoch',ylabel='MSE (SOH fraction squared)',title=ident);ax.legend();ax.set_yscale('log');save(fig,figures/f'{ident}_loss')
    # Use the validation-best attention configuration solely for an interpretation figure.
    attention=validation[validation.architecture=='B4']
    if not attention.empty:
        stats=attention.groupby(['feature_set','sequence_length']).rmse_fraction.mean();f,l=stats.idxmin()
        attention_runs=attention[(attention.feature_set==f)&(attention.sequence_length==l)]
        first=attention_runs.sort_values('seed').iloc[0].run_id
        path=output/'runs'/first/'validation_predictions_attention.csv'
        if path.exists():
            a=pd.read_csv(path);weights=np.array(a.attention_weights.map(json.loads).tolist())
            fig,ax=plt.subplots(figsize=(7,4));im=ax.imshow(weights[::max(1,len(a)//100)].T,aspect='auto',origin='lower',cmap='viridis')
            ax.set(xlabel='Validation windows sampled chronologically',ylabel='Observed position in historical window',title=f'Temporal attention: {first}')
            ax.set_yticks([0,int(l)-1],[f't-{int(l)-1}','t']);fig.colorbar(im,ax=ax,label='Attention weight');save(fig,figures/'attention_heatmap')
    if not tests.empty:
        for role,ids in [('selected',selection['selected_run_ids']),('literature_predictor',selection['literature_run_ids'])]:
            frames=[pd.read_csv(output/'test_evaluations'/f'{ident}_predictions.csv') for ident in ids]
            pred=pd.concat(frames,ignore_index=True)
            fig,ax=plt.subplots(figsize=(5,5));ax.scatter(pred.actual_soh_percent,pred.predicted_soh_percent,s=5,alpha=.2)
            low=min(pred.actual_soh_percent.min(),pred.predicted_soh_percent.min());high=max(pred.actual_soh_percent.max(),pred.predicted_soh_percent.max())
            ax.plot([low,high],[low,high],'k--',lw=1);ax.set(xlabel='Actual SOH (%)',ylabel='Predicted SOH (%)',title=f'CX2_38: {role}');save(fig,figures/f'{role}_actual_vs_predicted')
            fig,ax=plt.subplots(figsize=(7,4))
            for ident,g in zip(ids,frames):ax.plot(g.lifetime_cycle,g.error_percentage_points,'.',label=ident.split('_')[-1],ms=1.8,alpha=.6)
            ax.axhline(0,color='black',lw=.8);ax.set(xlabel='Observed reconstructed cycle ordinal',ylabel='Prediction error (percentage points)',title=f'CX2_38: {role}');ax.legend();save(fig,figures/f'{role}_error_vs_cycle')
            fig,ax=plt.subplots(figsize=(6,4));ax.hist(pred.error_percentage_points,bins=50,density=True,alpha=.8,color='#377eb8');ax.set(xlabel='Prediction error (percentage points)',ylabel='Density',title=f'CX2_38: {role}');save(fig,figures/f'{role}_error_distribution')
            by=pred.groupby('lifetime_cycle').agg(actual=('actual_soh_percent','first'),mean=('predicted_soh_percent','mean'),std=('predicted_soh_percent','std'))
            fig,ax=plt.subplots(figsize=(7,4))
            segments=np.r_[True,np.diff(by.index)>1].cumsum()
            for number,(_,g) in enumerate(by.groupby(segments)):
                ax.plot(g.index,g.actual,'k',lw=1,label='Actual' if number==0 else None)
                ax.plot(g.index,g['mean'],color='#e41a1c',lw=1,label='Mean of seed predictions' if number==0 else None)
                ax.fill_between(g.index,g['mean']-g['std'],g['mean']+g['std'],color='#e41a1c',alpha=.18,label='+/-1 seed SD' if number==0 else None)
            ax.set(xlabel='Observed reconstructed cycle ordinal',ylabel='SOH (%)',title=f'CX2_38: {role}');ax.legend();save(fig,figures/f'{role}_soh_trajectory')
    coverage=[]
    raw=pd.read_csv(Path(__file__).resolve().parents[2]/'outputs'/'cx2_features_all.csv',low_memory=False)
    raw=raw[raw.validated_complete & raw.protocol.eq('conventional')]
    arrays=np.load(output/'prepared_windows.npz')
    for split in SPLITS:
        for length in [5,10,20]:
            metadata=pd.read_csv(output/f'windows_{split}_{length}.csv')
            y=arrays[f'{split}_{length}_y']
            for battery,g in metadata.groupby('battery'):
                mask=metadata.battery.eq(battery).to_numpy();original=raw[raw.battery==battery]
                coverage.append(dict(split=split,battery=battery,sequence_length=length,evaluated_targets=len(g),validated_routine_cycles=len(original),
                    target_coverage_percent=100*len(g)/len(original),window_target_min_soh_percent=float(y[mask].min()*100),window_target_max_soh_percent=float(y[mask].max()*100),
                    full_validated_min_soh_percent=float(original.soh.min()*100),full_validated_max_soh_percent=float(original.soh.max()*100)))
    arrays.close();coverage=pd.DataFrame(coverage);coverage.to_csv(output/'lifetime_coverage.csv',index=False)
    if not tests.empty:
        ident=selection['selected_run_ids'][0];pred=pd.read_csv(output/'test_evaluations'/f'{ident}_predictions.csv');original=raw[raw.battery=='CX2_38']
        fig,ax=plt.subplots(figsize=(7,4));ax.scatter(original.lifetime_cycle,original.soh*100,s=7,color='grey',alpha=.4,label='All validated routine cycles')
        ax.scatter(pred.lifetime_cycle,pred.actual_soh_percent,s=7,color='#377eb8',label='Evaluated targets')
        ax.set(xlabel='Observed reconstructed cycle ordinal',ylabel='Actual SOH (%)',title='CX2_38 evaluation coverage');ax.legend();save(fig,figures/'test_lifetime_coverage')
    summary=pd.read_csv(output/'validation_seed_summary.csv');counts=pd.read_csv(output/'sample_counts.csv')
    selected=summary[(summary.architecture==architecture)&(summary.feature_set==feature)&(summary.sequence_length==selection['sequence_length'])]
    compact=['architecture','feature_set','sequence_length','rmse_fraction_mean','rmse_fraction_std','mae_fraction_mean','mae_fraction_std','mape_percent_mean','r2_mean']
    test_summary=pd.read_csv(output/'test_seed_summary.csv') if not tests.empty else pd.DataFrame()
    text=['# CALCE CX2 LSTM baseline report','',f'Completed {len(validation)} full seeded training runs; no hybrid model, physical loss, CNN or GWO was used.',
       '',f'Validation-selected configuration: **{architecture} ({LABELS[architecture]}), {feature}, length {selection["sequence_length"]}**.',
       '',table(selected[compact]),'','The table reports means and sample standard deviations across seeds 42, 123 and 2026. SOH and fractional errors use [0,1] units; multiply MAE/RMSE by 100 for percentage points. MAPE is relative error expressed as percent, not percentage points.',
       '','## Battery split and dataset integrity','',table(pd.DataFrame([dict(split=s,batteries=', '.join(bs)) for s,bs in SPLITS.items()])),
       '','Labels were cross-checked against cycle_quality.csv and rejected_cycles.csv, including charge/discharge integration-counter agreement and SOH = validated discharge Ah / 1.35. Only complete, routine/conventional records were eligible. No raw file or earlier feature output was modified.',
       '','## Sample counts','',table(counts),
       '','All feature comparisons use the same complete-case F2 feature union; missing inputs are rejected, never imputed or substituted. Reported individually usable cycles and common-cohort counts are in cycle_counts.csv. Rejections are in cycle_audit.csv and window_audit.csv. Windows require consecutive original ordinals, the same single source sheet, and nonoverlapping experimental times with gaps <=3600 s. No window is made by compacting rejected cycles.',
       '','Longer windows change the eligible targets. sequence_common_target_metrics.csv and its figure repeat the validation comparison on the intersection of target-cycle IDs. No cross-length improvement should be attributed only to architecture without checking this matched cohort.',
       '','## Experimental comparisons','',table(summary[compact]),
       '','A: B1/length 5 compares F1/F2/F3. B: architectures are selected using the chosen feature set at length 5. C: length 5/10/20 is selected for that architecture. The full four-architecture by three-length grid is also available. D: every configuration uses three seeds. The F1/B2/length-5 predictor-settings comparison is pre-specified. All choices were frozen in final_selection.json before any test checkpoint evaluation.',
       '','## Held-out test results','',table(test_summary) if not test_summary.empty else 'Test evaluation has not yet been run.',
       '','Each frozen checkpoint is evaluated once on CX2_38; resumed reporting reads saved predictions. Test results never select features, epochs, architectures, lengths or seeds. Training/validation/test predictions and attention weights are saved separately and aggregated. The seed-mean trajectory figure shows seed variability, not a calibrated confidence interval.',
       '','## Compute and reproducibility','',f'Total measured full-training-loop time: {validation.training_time_seconds.sum():.1f} s ({validation.training_time_seconds.sum()/3600:.3f} h), including epoch validation and best-state copies, excluding dataset preparation and report generation. Parameter counts, per-run inference wall times and peak allocated/reserved GPU memory are in validation_results.csv. Inference is one GPU-synchronized forward pass after synthetic warmup; transfer/metric time is excluded, and short single-pass timing is noisy.',
       '','Training uses AdamW (lr 0.0005, weight decay 0.001), MSE, batch 64, at most 200 epochs, patience 20, gradient clipping 1.0 and ReduceLROnPlateau (factor 0.5, patience 5, minimum lr 1e-6). A 1e-6 fractional RMSE improvement is required to reset early stopping. B1/B4 use one 64-unit LSTM layer; B2/B3 use two layers and 0.2 inter-layer dropout. B3 is bidirectional within the available window. B4 uses learned additive temporal attention. Targets and predictions are not clipped.',
       '','Random/Python/NumPy/CPU/CUDA seeds are set; deterministic PyTorch algorithms and cuDNN are requested, benchmarking/TF32 disabled, CUBLAS_WORKSPACE_CONFIG=:4096:8 set before CUDA initialization. Reproducibility across different hardware/driver/library versions is not guaranteed. experiment_config.json records versions, source/code hashes, split, feature lists, thresholds and hyperparameters. Completed runs are cached; incompatible scientific configurations require a separate directory.',
       '','## Interpretation and limitations','',
       'The benchmark measures unseen-battery generalization using charging observations through the current cycle, not next-cycle forecasting. Smoothing is contained within each charge; none of the three feature sets uses historical or future-cycle feature engineering. Current-cycle discharge measurements are used exclusively for target verification and protocol audit, never as input columns.',
       '','F3 was fixed by the user from the prior exploratory feature pipeline. That earlier analysis included the battery now used for validation (CX2_36), so validation independence from exploratory feature selection is imperfect. The compact set is not claimed optimal, and the untouched test battery is the independent evaluation. No feature reranking was performed for this benchmark.',
       '','Only one validation and one test battery are available. Window overlap induces serial dependence; seed SD measures optimization variability and does not establish population uncertainty or statistical superiority. Strict continuity/complete-case filtering changes lifetime coverage. Future cross-protocol evaluation on CX2_3/CX2_8 and additional held-out batteries remains necessary.',
       '','## Literature reference','',
       literature_summary(output),
       '','[PyTorch reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html). [CUDA RNN determinism](https://docs.pytorch.org/docs/stable/cudnn_rnn_determinism.html).']
    text.extend(['','## Lifetime coverage','',table(coverage[coverage.sequence_length==selection['sequence_length']]),
                 '','The result applies only to retained windows. Low-SOH and discontinuous records excluded by feature availability or continuity are not evaluated, and no accuracy claim covers those lifetime regions. test_lifetime_coverage.png shows this restriction directly.'])
    if not tests.empty:
        primary=tests[tests.roles.str.contains('selected')];reference=tests[tests.roles.str.contains('literature_predictor')]
        text.extend(['','## Conclusions from this experiment','',
          f'The frozen validation choice {architecture}/{feature}/L{selection["sequence_length"]} has mean test RMSE {primary.rmse_percentage_points.mean():.4f} +/- {primary.rmse_percentage_points.std():.4f} percentage points. The pre-specified F1/two-layer predictor has {reference.rmse_percentage_points.mean():.4f} +/- {reference.rmse_percentage_points.std():.4f} percentage points on the same retained test targets.',
          '','Validation rankings did not transfer to the one held-out test battery: the pre-specified literature-settings comparator has the lower observed test error. This does not change the validation-selected configuration and does not authorize test-driven redesign. The comparison changes both feature set and architecture, so the gap cannot isolate the effect of stacking or feature selection.',
          '','At length 5, additional architectural complexity did not consistently lower mean validation error relative to Vanilla LSTM. Longer windows changed the sample cohort and generally increased seed variation; consult the identical-target validation analysis before attributing any change to history length. These are descriptive findings for this split, not statistical population-level superiority claims.'])
    (output/'BASELINE_REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    json_write(output/'artifact_inventory.json',dict(training_runs=len(validation),test_runs=len(tests),png_figures=len(list(figures.glob('*.png'))),pdf_figures=len(list(figures.glob('*.pdf'))),dpi=300))
    print('Wrote report and publication figures:',output,flush=True)
