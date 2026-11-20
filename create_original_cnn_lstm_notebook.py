"""Create a separate CNN-LSTM experiment using the original train.csv."""
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parent
nb = nbformat.read(root / 'CALCE_Paper_CNN_LSTM_Reproduction.ipynb', as_version=4)
nb.cells[0].source = '''# CNN-LSTM SOH estimation with the original dataset features

This experiment uses the **original `train.csv`**, not `train_final_filtered.csv`.
All original exported training rows and SOH targets are retained before windowing;
rest-associated increases and fluctuations are not removed or smoothed.

Inputs are the dataset's five original charging features: `cc_duration_s`,
`cv_duration_s`, `dva_peak`, `ic_peak_voltage`, and `ic_peak_width`.
These are the original exported features, not newly extracted paper features or raw signals.
Standardization is fitted on training rows only. SOH stays fractional.

The model keeps the existing adapted parallel CNN (kernels 3, 5, 9) and
two-layer LSTM (256 units), with 12-cycle windows and seeds 42, 123, 2026.
Windows stay within one battery, consecutive original cycle ordinals and an
unchanged source identifier, with strictly increasing timestamps.
Validation selects each checkpoint; test data is used only for final evaluation.
Outputs are stored separately in `cnn_lstm_original_features_results`.
'''
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    cell.outputs = []
    cell.execution_count = None
    s = cell.source.replace('train_final_filtered.csv', 'train.csv')
    s = s.replace('OUTPUT_DIR = DATA_DIR / "cnn_lstm_paper_baseline"',
                  'OUTPUT_DIR = DATA_DIR.parent / "cnn_lstm_original_features_results"')
    if s.startswith('%pip'):
        s = '# Dependencies: numpy pandas torch matplotlib scikit-learn nbformat nbclient\n# Install in your notebook kernel if needed: %pip install numpy pandas torch matplotlib scikit-learn'
    s = s.replace("DATA_DIR = Path(r\"C:\\Projects\\Battery_Soh\\pre processed ready to use data\")",
                  'ROOT = Path.cwd()\nif not (ROOT / "pre processed ready to use data").exists():\n    raise FileNotFoundError("Run this notebook from the Battery_Soh project root")\nDATA_DIR = ROOT / "pre processed ready to use data"')
    s = s.replace("print('Device:', DEVICE)", "torch.set_num_threads(min(4, torch.get_num_threads()))\nprint('Device:', DEVICE, torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')")
    s = s.replace("['battery','cycle_ordinal',TARGET]", "['battery','cycle_ordinal','cycle_start','source_identifiers',TARGET]")
    s = s.replace("if frame[FEATURES+[TARGET]].isna().any().any():", "if not np.isfinite(frame[FEATURES+[TARGET]].to_numpy(dtype=float)).all():")
    s = s.replace('boundaries=np.r_[0, np.flatnonzero(np.diff(cycles)!=1)+1, len(part)]',
                  '''times=pd.to_datetime(part.cycle_start, errors='raise')
        if times.isna().any() or part.source_identifiers.isna().any():
            raise ValueError('Missing acquisition metadata')
        source=part.source_identifiers.to_numpy()
        breaks=(np.diff(cycles)!=1) | (source[1:]!=source[:-1]) | (np.diff(times.to_numpy())<=np.timedelta64(0,'s'))
        boundaries=np.r_[0, np.flatnonzero(breaks)+1, len(part)]''')
    s = s.replace("'soh':float(targets[end])", "'soh':float(targets[end]),'cycle_start':str(part.cycle_start.iloc[end]),'window_start_cycle':int(cycles[start])")
    s = s.replace("history.append({'epoch':epoch", "print(f'Seed {seed} epoch {epoch} validation RMSE {np.sqrt(val_mse)*100:.4f} pp', flush=True)\n        history.append({'epoch':epoch")
    s = s.replace("'architecture':'adapted parallel CNN LSTM'", "'architecture':'adapted parallel CNN LSTM','scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'best_epoch':int(np.argmin([h['val_mse'] for h in history])+1)")
    cell.source = s
nb.cells[-1].source = '''## Interpretation
- MAE, RMSE and bias are SOH percentage points; MAPE is relative percent.
- Original dataset values are retained. No SOH-jump filtering, target smoothing or clipping is applied.
- The five exported charging features differ from the paper's features; this is an adapted CNN-LSTM baseline.
- Current-cycle charging features estimate current-cycle SOH, not future SOH.
- Window warm-up and acquisition boundaries exclude some target rows. Predictions include target IDs for matched comparisons.
- Each seed uses its best validation checkpoint. Test metrics are descriptive and do not select seeds.
'''
nbformat.validate(nb)
path = root / 'CALCE_CNN_LSTM_Original_Features.ipynb'
nbformat.write(nb, path)
print(path)
