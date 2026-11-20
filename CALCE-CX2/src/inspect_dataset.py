"""Inventory raw sessions and cache normalized Excel Channel sheets (no raw writes)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from .common import ROOT, VERSION, digest, save_json

RENAME = {'Date_Time':'timestamp', 'Test_Time(s)':'test_s', 'Step_Time(s)':'step_s',
          'Cycle_Index':'local_cycle', 'Step_Index':'step', 'Current(A)':'current',
          'Voltage(V)':'voltage', 'Charge_Capacity(Ah)':'charge_counter',
          'Discharge_Capacity(Ah)':'discharge_counter', 'Internal_Resistance(Ohm)':'resistance'}
REQUIRED = ['timestamp','test_s','step_s','step','local_cycle','current','voltage','charge_counter','discharge_counter']

def read_channel(path, sheet, battery, raw_root):
    d = pd.read_excel(path, sheet_name=sheet).rename(columns=lambda c:str(c).strip()).rename(columns=RENAME)
    missing = set(REQUIRED)-set(d)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    temp = [c for c in d if 'temperature' in c.lower() and ('c)' in c.lower() or 'deg' in c.lower())]
    if len(temp)==1:
        d = d.rename(columns={temp[0]:'temperature'})
    d = d[[c for c in REQUIRED+['resistance','temperature'] if c in d]].copy()
    d['timestamp'] = pd.to_datetime(d.timestamp, errors='coerce')
    for c in d.columns.difference(['timestamp']):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d['battery'] = battery
    d['source'] = path.relative_to(raw_root).as_posix()+'::'+sheet
    d['source_row'] = np.arange(len(d))+2
    return d

def inspect(raw_root, output, batteries=None):
    raw_root, output = Path(raw_root), Path(output)
    cache = output/'cache'/'channels'
    cache.mkdir(parents=True, exist_ok=True)
    rows=[]
    paths=sorted(p for p in raw_root.rglob('*') if p.is_file() and p.suffix.lower() in {'.xlsx','.xls','.txt','.csv'})
    selected_paths=[p for p in paths if not batteries or p.relative_to(raw_root).parts[0] in batteries]
    for path in tqdm(selected_paths, desc='Inspect/cache raw files'):
        battery = path.relative_to(raw_root).parts[0]
        if batteries and battery not in batteries:
            continue
        base=dict(battery=battery, source_file=path.relative_to(raw_root).as_posix(), bytes=path.stat().st_size)
        if path.suffix.lower() not in {'.xls','.xlsx'}:
            raw=path.read_bytes()
            encoding='utf-16' if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig'
            lines=raw.decode(encoding,errors='replace').splitlines()
            delimiter=',' if path.suffix.lower()=='.csv' else '\t'
            header=next((j for j,line in enumerate(lines[:30]) if line.startswith('Scan'+delimiter+'Time'+delimiter)),None)
            if header is not None:
                try:
                    import io
                    temp=pd.read_csv(io.StringIO('\n'.join(lines[header:])),sep=delimiter)
                    timestamp=pd.to_datetime(temp['Time'],format='%m/%d/%Y %H:%M:%S:%f',errors='coerce')
                    channels=[c for c in temp if '(C)' in c]
                    if len(channels)!=1:raise ValueError('Temperature channel/units ambiguous')
                    values=pd.to_numeric(temp[channels[0]],errors='coerce')
                    rows.append(dict(**base,sheet='temperature_logger',sha256=__import__('hashlib').sha256(raw).hexdigest(),
                        status='temperature_only_no_synchronized_current_session',cache='',rows=len(temp),start=str(timestamp.min()),end=str(timestamp.max()),
                        missing_timestamps=int(timestamp.isna().sum()),temperature_channel=channels[0],temperature_min_c=float(values.min()),
                        temperature_max_c=float(values.max()),temperature_missing=int(values.isna().sum()),
                        error='No experimental absolute time anchor for corresponding current export; no filename-based join.'))
                except Exception as exc:
                    rows.append(dict(**base,sheet='temperature_logger',cache='',status='temperature_schema_error',error=str(exc)))
            else:
                rows.append(dict(**base,sheet='',sha256=__import__('hashlib').sha256(raw).hexdigest(),
                    status='relative_time_only_counter_units_unverified',cache='',columns=lines[0] if lines else '',
                    error='No absolute experimental timestamp in current export; capacity unit semantics unverified.'))
            continue
        fingerprint=digest(path)
        cache_key=__import__('hashlib').sha256((fingerprint+base['source_file']+VERSION).encode()).hexdigest()
        try:
            manifest=cache/(cache_key+'.json')
            if manifest.exists():
                saved=__import__('json').loads(manifest.read_text())
                if saved.get('version')==VERSION and saved.get('source_file')==base['source_file'] and all(Path(r['cache']).exists() for r in saved['rows']):
                    rows.extend(saved['rows'])
                    continue
            book=pd.ExcelFile(path)
            sheets=[s for s in book.sheet_names if 'channel' in s.lower()]
            if not sheets:
                # Some genuine Arbin tables were exported with generic Sheet1 names.
                for candidate in book.sheet_names:
                    columns=pd.read_excel(book,sheet_name=candidate,nrows=0).columns
                    normalized={RENAME.get(str(c).strip(),str(c).strip()) for c in columns}
                    if set(REQUIRED).issubset(normalized):sheets.append(candidate)
                if not sheets:raise ValueError('No Channel or schema-compatible measurement sheet')
            current=[]
            for sheet in sheets:
                d=read_channel(path,sheet,battery,raw_root)
                cp=cache/(cache_key+'_'+__import__('hashlib').sha256(sheet.encode()).hexdigest()[:8]+'.pkl')
                d.to_pickle(cp)
                r=dict(**base, sheet=sheet, sha256=fingerprint, cache=str(cp.resolve()), rows=len(d),
                       start=str(d.timestamp.min()), end=str(d.timestamp.max()),
                       missing_timestamps=int(d.timestamp.isna().sum()),
                       source_timestamp_reversals=int((d.timestamp.diff().dt.total_seconds()<0).sum()),
                       local_cycle_resets=int((d.local_cycle.diff()<0).sum()),
                       test_time_resets=int((d.test_s.diff()<0).sum()),
                       status='ready', columns=';'.join(d.columns))
                rows.append(r); current.append(r)
            book.close()
            save_json(manifest,dict(version=VERSION,source_file=base['source_file'],rows=current))
        except Exception as exc:
            rows.append(dict(**base,sheet='',cache='',status='read_error',error=f'{type(exc).__name__}: {exc}'))
            tqdm.write(f'ERROR {path}: {exc}')
    result=pd.DataFrame(rows)
    result.to_csv(output/'inspection_manifest.csv',index=False)
    result[result.status.str.startswith('temperature')].to_csv(output/'temperature_session_inventory.csv',index=False)
    save_json(output/'inspection_summary.json',dict(raw_files=len(paths),selected_batteries=sorted(result.battery.unique()),
              statuses=result.status.value_counts().to_dict(), timestamp_policy='Experimental Date_Time only; no filesystem-time or filename inference.'))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--raw',type=Path,default=ROOT/'CX2_data');p.add_argument('--output',type=Path,default=ROOT/'outputs');p.add_argument('--batteries',nargs='+')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);inspect(a.raw,a.output,a.batteries)
if __name__=='__main__':main()
