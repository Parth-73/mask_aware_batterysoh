"""Report paper-style LSTM results without tuning against publication/test metrics."""
from pathlib import Path
import json,pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import BATTERIES,VARIANTS,PUBLISHED,definition,json_write


def table(frame):
    def f(x):return f'{x:.6g}' if isinstance(x,(float,np.floating)) else str(x)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(f(x) for x in r)+' |' for r in frame.itertuples(index=False,name=None)])


def save(fig,path):
    fig.tight_layout();fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight');fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)


def generate(output,config):
    figures=output/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.grid':True,'grid.alpha':.2,'pdf.fonttype':42})
    results=pd.read_csv(output/'test_results.csv');summary=pd.read_csv(output/'seed_summary.csv');training=pd.read_csv(output/'validation_results.csv');counts=pd.read_csv(output/'sample_counts.csv');features=pd.read_csv(output/'gaussian_features.csv',low_memory=False)
    comparisons=[];correction_counts=[];outside=[];splits=[]
    for r in summary.itertuples():
        comparisons.append(dict(battery=r.battery,variant=r.variant,paper_lstm_rmse_fraction=PUBLISHED[r.battery]['rmse_fraction'],our_rmse_fraction_mean=r.rmse_fraction_mean,our_rmse_fraction_std=r.rmse_fraction_std,absolute_rmse_gap=abs(r.rmse_fraction_mean-PUBLISHED[r.battery]['rmse_fraction']),our_mae_fraction_mean=r.mae_fraction_mean,our_mape_percent_mean=r.mape_percent_mean,our_r2_mean=r.r2_mean))
    comparison=pd.DataFrame(comparisons);comparison.to_csv(output/'published_comparison.csv',index=False)
    for battery in BATTERIES:
        for variant in VARIANTS:
            folder=output/'datasets'/f'{battery}_{variant}';pre=json.loads((folder/'preprocessing.json').read_text());splits.append(dict(battery=battery,variant=variant,**pre['split']))
            corrections=pd.read_csv(folder/'final_corrections.csv')
            for role in ['fit','validation','test']:
                correction_counts.append(dict(battery=battery,variant=variant,role=role,observed_feature_values_corrected=int(corrections.role.eq(role).sum())))
            with np.load(folder/'windows.npz') as windows:
                x=windows['test_x']
                outside.append(dict(battery=battery,variant=variant,test_samples=len(x),outside_training_range_window_elements_percent=100*float(((x<0)|(x>1)).mean())))
            ids=results[(results.battery==battery)&(results.variant==variant)].run_id.tolist()
            frames=[pd.read_csv(output/'test_evaluations'/f'{ident}_predictions.csv') for ident in ids];pred=pd.concat(frames,ignore_index=True)
            average=pred.groupby('lifetime_cycle').agg(actual=('actual_soh_fraction','first'),predicted=('predicted_soh_fraction','mean'),seed_std=('predicted_soh_fraction','std'))
            all_b=features[(features.battery==battery)&features.validated_complete&features.protocol.eq('conventional')]
            fig,ax=plt.subplots(figsize=(7,4));ax.scatter(all_b.lifetime_cycle,all_b.soh*100,s=6,color='grey',alpha=.4,label='Validated aging prefix')
            segments=np.r_[True,np.diff(average.index)>1].cumsum()
            for number,(_,g) in enumerate(average.groupby(segments)):
                ax.plot(g.index,g.actual*100,color='black',lw=.9,label='Test actual' if not number else None)
                ax.plot(g.index,g.predicted*100,color='#e41a1c',lw=1,label='Test mean prediction' if not number else None)
                ax.fill_between(g.index,(g.predicted-g.seed_std)*100,(g.predicted+g.seed_std)*100,color='#e41a1c',alpha=.15,label='+/-1 seed SD' if not number else None)
            ax.axvline(pre['split']['train_end_ordinal'],color='#377eb8',ls='--',label='50/50 cycle boundary')
            ax.set(xlabel='Observed original cycle ordinal',ylabel='SOH (%)',title=battery+' / '+variant);ax.legend(fontsize=8);save(fig,figures/f'{battery}_{variant}_prediction')
            fig,ax=plt.subplots(figsize=(6,4));ax.hist(pred.error_percentage_points,bins=35,density=True,color='#377eb8',alpha=.8);ax.set(xlabel='Prediction error (percentage points)',ylabel='Density',title=battery+' / '+variant);save(fig,figures/f'{battery}_{variant}_errors')
    pd.DataFrame(splits).to_csv(output/'chronological_splits.csv',index=False);pd.DataFrame(correction_counts).to_csv(output/'correction_counts.csv',index=False);pd.DataFrame(outside).to_csv(output/'normalization_range_audit.csv',index=False)
    history=pd.read_csv(output/'training_history.csv')
    for ident,g in history[history.run_id.str.contains('_fixed_peak_')].groupby('run_id'):
        fig,axes=plt.subplots(1,2,figsize=(10,3.6));inner=g[g.stage=='inner_epoch_selection'];refit=g[g.stage=='refit_complete_training_half']
        axes[0].plot(inner.epoch,np.sqrt(inner.train_mse),label='Fit');axes[0].plot(inner.epoch,inner.validation_rmse,label='Internal validation');axes[0].set(title='Epoch selection',xlabel='Epoch',ylabel='RMSE (SOH fraction)');axes[0].set_yscale('log');axes[0].legend()
        axes[1].plot(refit.epoch,np.sqrt(refit.train_mse),label='Complete training half');axes[1].set(title='Fixed-budget refit',xlabel='Epoch',ylabel='Training RMSE (fraction)');axes[1].set_yscale('log');fig.suptitle(ident);save(fig,figures/f'{ident}_loss')
    with open(output/'ic_examples.pkl','rb') as f:examples=pickle.load(f)
    for battery,curves in examples.items():
        fig,ax=plt.subplots(figsize=(6.5,4))
        for ordinal,(v,ic) in curves:ax.plot(v,ic,label=f'Cycle {ordinal}')
        ax.axvspan(config.fixed_peak_low_v,config.fixed_peak_high_v,alpha=.08,color='black',label='Primary peak interval')
        ax.set(xlabel='Charge voltage (V)',ylabel='Gaussian dQ/dV (Ah/V)',title=battery);ax.legend();save(fig,figures/f'{battery}_gaussian_ic')
    primary=comparison[comparison.variant=='fixed_peak'].set_index('battery').loc[BATTERIES].reset_index()
    fig,ax=plt.subplots(figsize=(7,4));x=np.arange(4);ax.bar(x-.18,primary.paper_lstm_rmse_fraction,width=.36,label='Paper Table 4 LSTM')
    ax.bar(x+.18,primary.our_rmse_fraction_mean,width=.36,yerr=primary.our_rmse_fraction_std,capsize=3,label='Paper-style implementation (3 seeds)');ax.set_xticks(x,BATTERIES);ax.set(ylabel='RMSE (SOH fraction)',title='Contextual comparison: unresolved method details');ax.legend(fontsize=8);save(fig,figures/'published_lstm_rmse_comparison')
    exact_columns=['battery','paper_lstm_rmse_fraction','our_rmse_fraction_mean','our_rmse_fraction_std','our_mae_fraction_mean','our_mape_percent_mean','our_r2_mean']
    assumptions=pd.DataFrame([dict(item=k,implementation_choice=v) for k,v in definition(config)['assumptions'].items()])
    text=['# Paper-style CALCE CX2 LSTM reproduction','',
      '**Status: implemented and executed for the four locally available CX2 batteries. This is a best-effort reproduction of the reported pure-LSTM predictor methodology, not an exact author-code replication or AW-PINN implementation.**','',
      'Reference: [Wang et al., Batteries 2026, 12(4), 115](https://www.mdpi.com/2313-0105/12/4/115). The local PDF is recorded by content hash in experiment_config.json. Reported Table 3 settings match: three features, history five, two LSTM layers, hidden 64 and one output. Table 4 (p. 13) supplies the published pure-LSTM reference values below.','',
      '## Internal-validation-budget reproduction results','',table(primary[exact_columns]),'',
      'All errors in the RMSE/MAE columns are SOH fractions; multiply by 100 for percentage points. MAPE is explicitly percent. Standard deviations are sample SD across seeds 42/123/2026. Published values are contextual references, not matched-cohort superiority tests. The paper does not report all implementation settings needed to establish numerical equivalence.','',
      '## What follows the paper','',
      '- IC is recomputed from cached charging measurements and numerically differentiated on a uniform voltage grid, then Gaussian-filtered with sigma 1 (p. 6). The old Savitzky-Golay features are not silently reused as Gaussian features.\n- Inputs are IC peak, charged capacity over 3.8-4.0 V, and 0.2 V divided by that capacity (p. 5).\n- Feature normalization is MinMaxScaler [0,1], with fitting restricted to training data.\n- Each battery has its own model and chronological first-half training/latter-half testing (p. 12); this is not the earlier unseen-battery split.\n- The aging prefix ends at the first validated routine discharge below 80% SOH (p. 4).\n- Obvious invalid measurements are excluded, and explicit training-trend/deviation spike correction is applied only to observed feature values (p. 6). Missing physical measurements and SOH labels are never filled or smoothed.',
      '', '## Split and window counts','',table(counts),'',table(pd.DataFrame(splits)),
      '', 'The 50/50 split is made over common complete-feature cycles before window construction. Windows retain original ordinal adjacency and source/timestamp continuity. Consequently the number of target windows can differ from an exact 50/50 count. Historical inputs from the training half may appear in a later-half target window; future inputs and target labels never enter a window. Windows do not bridge omitted cycles or recording gaps.',
      '', '## Assumptions where the paper is underspecified','',table(assumptions),
      '', 'The primary fixed-peak interval is a declared operational choice, not a claimed interval from the paper. The full observed CC peak is a predeclared sensitivity alternative. Both variants use an identical eligible-cycle cohort and all three seeds; neither is selected using test error or closeness to the published score.',
      '', '## Sensitivity to the IC peak interval','',table(comparison),
      '', '## Feature corrections and normalization','',table(pd.DataFrame(correction_counts)),
      '', 'A robust quadratic trend is fitted on fitting/training feature observations only. A point is corrected only when both its trend residual exceeds six robust sigma and its adjacent measured jump exceeds the training-derived jump limit (with a 3% feature-level floor). This avoids overwriting gradual aging solely because it extrapolates beyond the training trend. Corrected partial slope is recomputed as 0.2 / corrected partial capacity. Every raw/corrected value and decision is saved in datasets/*/*corrections.csv and processed-feature tables.',
      '',table(pd.DataFrame(outside)),
      '', 'Training feature extrema map to [0,1]. Unseen test feature values can lie outside [0,1], and are neither clipped nor used to fit scaling. The paper does not identify the normalizer-fitting partition; this choice preserves leakage prevention and may differ from its implementation.',
      '', '## Model fitting and frozen evaluation','',
      'Optimizer, dropout, learning rate, batch size and epoch policy are unreported. This experiment fixes Adam lr=0.001, zero weight decay/dropout, batch 64, MSE, max 200 epochs, early-stop patience 20 and gradient clip 1.0. ReduceLROnPlateau halves LR after five plateau epochs. These choices were fixed before test evaluation and were not searched to approach the paper\'s results.',
      '', 'Within the first half only, the last 20% is an internal validation tail. Its best epoch budget and LR schedule are selected with preprocessing fitted only on the earlier 80%. A fresh model and preprocessing are then fitted to the entire training half for the frozen budget, replaying that schedule. The paper\'s full first half is therefore used by the final predictor. No test labels select epochs or preprocessing parameters.',
      '',f'Completed {len(training)} inner-selection/final-refit pairs and {len(results)} single-pass test evaluations. Total measured selection plus refitting time: {training.training_seconds.sum():.1f} seconds. Final predictor parameter count is 51,009. All final model and prepared-data hashes were frozen before tests; saved evaluations are reused on resume.',
      '', '## Reproduction limits','',
      'Only CX2_34/36/37/38 are covered; CS2 and Tongji data are not locally available and were not downloaded. Exact IC resolution/search interval, outlier thresholds, optimizer/regularization and train/validation details are missing. Observed CX2 records include 1C discharge although the paper Table 1 lists 0.5C. Continuity and quality filtering may change the paper\'s cohort. First-below-80 truncation is a stated implementation choice; the paper does not define its crossing/recovery rule.',
      '', 'Equation 26 expresses MAPE with a factor of 100, while the tables show small decimal values without a clear percentage label. We do not infer a factor-100 conversion for the published entries. Equation 25 defines MSE while result tables report MAE; this report labels both separately. No claim of superiority or exact numerical reproduction follows from a smaller error in an unmatched cohort.',
      '', 'The earlier unseen-battery test results were already visible before this new, user-requested reproduction. This is an explicitly separate within-battery replication attempt; it is not a new untouched-battery generalization claim. All prior feature outputs, splits, checkpoints and benchmark metrics remain unchanged.',
      '', '## Reproducible commands','',
      '```powershell\n& D:\\battery_soh\\.venv\\Scripts\\python.exe -m src.paper_reproduction.run --stage prepare\n& D:\\battery_soh\\.venv\\Scripts\\python.exe -m src.paper_reproduction.run --smoke\n& D:\\battery_soh\\.venv\\Scripts\\python.exe -m src.paper_reproduction.run --stage train\n& D:\\battery_soh\\.venv\\Scripts\\python.exe -m src.paper_reproduction.run --stage test\n& D:\\battery_soh\\.venv\\Scripts\\python.exe -m src.paper_reproduction.run --stage report\n```',
      '', 'Use a separate --output directory for changed scientific settings. Rebuilding prepared data after the models are frozen is blocked. Reporting only reads saved predictions and never repeats test inference.']
    fixed_path=output/'fixed_epoch'/'seed_summary.csv'
    if fixed_path.exists():
        fixed=pd.read_csv(fixed_path);budget=primary[['battery','paper_lstm_rmse_fraction','our_rmse_fraction_mean','our_rmse_fraction_std']].rename(columns={'our_rmse_fraction_mean':'validation_budget_rmse_mean','our_rmse_fraction_std':'validation_budget_rmse_std'})
        budget=budget.merge(fixed[['battery','rmse_fraction_mean','rmse_fraction_std','mae_fraction_mean','mape_percent_mean','r2_mean']],on='battery')
        budget=budget.rename(columns={'rmse_fraction_mean':'fixed200_rmse_mean','rmse_fraction_std':'fixed200_rmse_std','mae_fraction_mean':'fixed200_mae_mean','mape_percent_mean':'fixed200_mape_percent_mean','r2_mean':'fixed200_r2_mean'})
        budget.to_csv(output/'training_policy_comparison.csv',index=False)
        fig,ax=plt.subplots(figsize=(8,4));xx=np.arange(len(budget))
        ax.bar(xx-.25,budget.paper_lstm_rmse_fraction,width=.25,label='Paper Table 4')
        ax.bar(xx,budget.validation_budget_rmse_mean,width=.25,yerr=budget.validation_budget_rmse_std,capsize=3,label='Internal-validation budget')
        ax.bar(xx+.25,budget.fixed200_rmse_mean,width=.25,yerr=budget.fixed200_rmse_std,capsize=3,label='Fixed 200 epochs')
        ax.set_xticks(xx,budget.battery);ax.set(ylabel='RMSE (SOH fraction)',title='Reported-method attempt: unreported training policies');ax.legend(fontsize=8);save(fig,figures/'training_policy_comparison')
        for battery in BATTERIES:
            paths=sorted((output/'fixed_epoch'/'test_evaluations').glob(f'{battery}_*_predictions.csv'))
            fp=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
            av=fp.groupby('lifetime_cycle').agg(actual=('actual_soh_fraction','first'),mean=('predicted_soh_fraction','mean'),std=('predicted_soh_fraction','std'))
            fig,axes=plt.subplots(1,2,figsize=(11,4));segments=np.r_[True,np.diff(av.index)>1].cumsum()
            for number,(_,g) in enumerate(av.groupby(segments)):
                axes[0].plot(g.index,g.actual*100,'k',lw=1,label='Actual' if not number else None)
                axes[0].plot(g.index,g['mean']*100,color='#e41a1c',lw=1,label='Mean prediction' if not number else None)
                axes[0].fill_between(g.index,(g['mean']-g['std'])*100,(g['mean']+g['std'])*100,color='#e41a1c',alpha=.18,label='+/-1 seed SD' if not number else None)
            axes[0].set(xlabel='Original cycle ordinal',ylabel='SOH (%)',title='Chronological test half');axes[0].legend(fontsize=8)
            for ident,g in fp.groupby('run_id'):axes[1].plot(g.lifetime_cycle,g.error_percentage_points,'.',ms=2,label=ident.split('_')[-1])
            axes[1].axhline(0,color='k',lw=.8);axes[1].set(xlabel='Original cycle ordinal',ylabel='Error (percentage points)',title='Test residuals');axes[1].legend(fontsize=8);fig.suptitle(battery+' / fixed 200 epochs');save(fig,figures/f'{battery}_fixed200_prediction')
            fig,ax=plt.subplots(figsize=(6.5,4))
            for path in sorted((output/'fixed_epoch'/'runs').glob(f'{battery}_*/history.csv')):
                h=pd.read_csv(path);ax.plot(h.epoch,np.sqrt(h.train_mse),label=path.parent.name.split('_')[-1])
            ax.set(xlabel='Epoch',ylabel='Training RMSE (SOH fraction)',title=battery+' / fixed 200-epoch convergence');ax.set_yscale('log');ax.legend();save(fig,figures/f'{battery}_fixed200_training')
        text.extend(['','## Pre-test fixed-epoch comparison','',table(budget),
          '', 'The fixed-200 protocol was independently registered and all 12 models frozen before either set of test scores was observed. It uses the same complete first-half data, Gaussian primary features, training-only trends and MinMax preprocessing. The fixed_epoch directory contains its separate protocol, source/data hashes, checkpoints and one-pass evaluations.',
          '', 'Both training policies are reported without choosing one from test scores. They differ in epoch count and LR schedule (constant LR versus validation-derived schedule), so their gap is not a controlled estimate of epoch count alone. This sensitivity illustrates why the unreported training settings prevent exact reproduction.',
          '', f'Total final predictors across the two pre-test protocols: {len(training)+len(fixed)*3}; final evaluations: {len(results)+len(fixed)*3}.'])
    complete=features[features.feature_exclusion.fillna('').eq('')]
    identical=bool(np.allclose(complete.fixed_peak,complete.full_cc_peak,atol=1e-12,rtol=0))
    text.extend(['','## Observed IC-domain sensitivity','',f'The two Gaussian peak definitions produced identical maxima on all retained complete-case aging-prefix records: {identical}. When identical, the paired seeded models are duplicate-information sensitivity runs, not additional independent seeds. No test-based choice of peak interval was made.'])
    (output/'REPRODUCTION_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    json_write(output/'artifact_inventory.json',dict(final_models=len(training),fixed_epoch_final_models=12 if fixed_path.exists() else 0,inner_selection_checkpoints=len(training),test_evaluations=len(results),fixed_epoch_test_evaluations=12 if fixed_path.exists() else 0,png_figures=len(list(figures.glob('*.png'))),pdf_figures=len(list(figures.glob('*.pdf'))),dpi=300))
    print('Wrote paper-style reproduction report and figures:',output,flush=True)
