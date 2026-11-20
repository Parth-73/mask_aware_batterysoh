"""Charging-only features. Discharge measurements occur only in target/protocol audit columns."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, peak_widths
from .common import CONFIG
from .reconstruct_cycles import phase_integral

# name, group, units, definition, measurements, phase, missing/reliability rule
DEFINITIONS=[
('ic_peak_height','F1','Ah/V','max smoothed dQ/dV on [3.80,4.15] V','current,test_s,step_s,voltage','CC','Full interval and reliable monotonic CC curve required'),
('partial_charge_ah','F1','Ah','Q(4.0 V)-Q(3.8 V), interpolated on chronological CC charge curve','current,test_s,step_s,voltage','CC','Both voltage crossings required'),
('partial_dv_dq','F1','V/Ah','0.2 / partial_charge_ah','current,time,voltage','CC','Capacity must exceed 1e-5 Ah'),
('ic_peak_voltage','F2','V','Voltage at largest interior IC peak in [3.80,4.15] V','current,time,voltage','CC','Interior peak and reliable IC curve required'),
('ic_peak_area','F2','Ah','Integral of smoothed dQ/dV over fixed [3.80,4.15] V','current,time,voltage','CC','Full interval required; fixed window, no chemical peak assignment'),
('ic_peak_width','F2','V','Width of largest interior peak at half prominence','current,time,voltage','CC','Interior peak with prominence >=5% height and bounded flanks'),
('dva_mean','F2','V/Ah','Mean smoothed dV/dQ on uniform Q grid between 3.8 and 4.0 V','current,time,voltage','CC','Positive capacity span and reliable curve'),
('dva_peak','F2','V/Ah','Maximum smoothed dV/dQ over same partial-capacity grid','current,time,voltage','CC','Positive capacity span and reliable curve'),
('partial_charge_time_s','F2','s','Elapsed active CC time t(4.0)-t(3.8)','current,time,voltage','CC','Both crossings required, rests excluded'),
('cc_duration_s','F2','s','Sum observed step duration for voltage <4.18 V charge samples','current,test_s,step_s,voltage','CC','No recording gaps or mid-step starts'),
('cv_duration_s','F2','s','Sum observed durations at voltage >=4.18 V with decaying current','current,test_s,step_s,voltage','CV','CV voltage range <=0.035 V and current decrease >=20%'),
('cv_decay_rate_s_inv','F2','1/s','Negative OLS slope of log(I) against cumulative active CV time','current,time,voltage','CV','At least 8 CV points, current decay >=20%, R2>=0.8'),
('cv_capacity_ah','F2','Ah','Integral I dt at voltage >=4.18 V within verified CV phase','current,time,voltage','CV','Verified CV and contiguous samples'),
('historical_ic_peak_shift_v','F3','V','Current IC peak voltage minus first earlier valid reference peak','prior/current charging IC peak','history','First available cycle missing; baseline is strictly earlier'),
('historical_partial_change_ah','F3','Ah','Current partial Q minus first earlier valid partial Q','prior/current partial charge','history','First available cycle missing; no future information'),
('resistance_ohm','F3','Ohm','Median finite positive recorded internal resistance, candidate audit only','resistance','charge','Not a usable input until instrument measurement quality is verified'),
('historical_resistance_change_ohm','F3','Ohm','Current minus first earlier verified resistance','verified resistance history','history','Unavailable until resistance measurement quality verified'),
('charge_crate','F3','1/h','Median CC charging current / 1.35 Ah','current','CC','At least 5 CC samples'),
('charge_protocol_cv','F3','indicator','One if verified CV phase exists, zero if charge exists without CV','current,time,voltage','charge','Missing if no measured charge'),
('temperature_c','F3','degC','Mean measured temperature during charging','calibrated temperature sensor','charge','Unavailable until column units and sensor validity verified'),
('relaxation_60s_v','F3','V','V(rest+60s)-V(rest start), first post-charge rest','time,voltage,current','post-charge rest','Rest >=60s and near-zero current; not an instantaneous jump'),
('ic_smoothing_relative_spread','audit','ratio','(max-min) peak height across SG windows 7,11,15 / median','charging IC curves','CC','All windows available; >0.2 marked unstable'),
('resistance_recorded_median_ohm','audit','Ohm','Median finite positive resistance readings before verification','resistance','charge','Descriptive only; values in (0,1) Ohm'),
('discharge_crate_median','audit','1/h','Median same-cycle discharge current magnitude / 1.35 Ah','discharge current','discharge','AUDIT ONLY: excluded from predictive inputs to avoid future protocol leakage'),
]
FEATURES=[r[0] for r in DEFINITIONS if r[1]!='audit']


def definitions_frame():
    return pd.DataFrame(DEFINITIONS,columns=['feature','group','units','definition','raw_measurements','phase','reliability_rule'])


def charge_curve(parts):
    """Construct CC and CV arrays by within-step current integration, excluding rests.

    Each CC step must have nearly constant current (CV portions excluded). Voltage
    reversals larger than 20 mV or sustained decreases invalidate IC/DVA. Tiny ADC
    reversals are merged at the same maximum voltage using the last cumulative Q.
    """
    cc=[];cv=[];qoff=toff=0.;bad=False;ccdur=cvdur=cvq=0.
    for g in parts:
        if g.phase.iloc[0]!='charge':continue
        t=g.test_s.to_numpy(float);i=g.current.to_numpy(float);v=g.voltage.to_numpy(float)
        initial=float(g.get('integration_initial_s',g.step_s).iloc[0]);dt=np.r_[initial,np.diff(t)]
        cv_interval=(v>=4.18)&np.r_[True,v[:-1]>=4.18]
        limits=np.where(cv_interval,CONFIG['max_cv_sample_gap_s'],CONFIG['max_sample_gap_s'])
        valid=np.isfinite(dt)&(dt>=0)&(dt<=limits)
        if not valid.all():bad=True
        dt=np.where(valid,dt,0)
        dq=dt*np.r_[i[0],(i[:-1]+i[1:])/2]/3600
        q=qoff+np.cumsum(dq);active_t=toff+np.cumsum(dt)
        cc_mask=v<4.18;cv_mask=~cc_mask
        ccdur+=float(dt[cc_mask].sum());cvdur+=float(dt[cv_mask].sum());cvq+=float(dq[cv_mask].sum())
        if cc_mask.sum()>=5:
            med=np.median(i[cc_mask])
            if np.std(i[cc_mask])/max(med,1e-9)>0.08:bad=True
            cc.extend(zip(v[cc_mask],q[cc_mask],active_t[cc_mask],i[cc_mask]))
        cv.extend(zip(v[cv_mask],q[cv_mask],active_t[cv_mask],i[cv_mask]))
        qoff=float(q[-1]);toff=float(active_t[-1])
    c=np.asarray(cc,float).reshape(-1,4);z=np.asarray(cv,float).reshape(-1,4)
    if len(c)>2:
        if np.min(np.diff(c[:,0]))<-.02 or np.mean(np.diff(c[:,0])<-.002)>.05:bad=True
        vv=np.maximum.accumulate(c[:,0]);keep=np.r_[np.diff(vv)>1e-8,True]
        c=c[keep];c[:,0]=vv[keep]
    return c,z,ccdur,cvdur,cvq,bad


def ic_curves(cc):
    if len(cc)<20 or cc[0,0]>3.8 or cc[-1,0]<4.15:return None
    vg=np.arange(3.8,4.15+1e-8,CONFIG['voltage_grid_v'])
    qg=np.interp(vg,cc[:,0],cc[:,1])
    result={}
    for window in CONFIG['smoothing_windows']:
        smooth=savgol_filter(qg,window,3)
        result[window]=np.gradient(smooth,vg)
    return vg,qg,result


def extract(parts):
    f={k:np.nan for k in FEATURES};flags=[];curves={}
    cc,cv,ccdur,cvdur,cvq,bad=charge_curve(parts)
    charge=[g for g in parts if g.phase.iloc[0]=='charge']
    if not charge:return f,{'feature_quality_flags':'missing_charge'},curves
    f['charge_protocol_cv']=0.
    if bad:flags.append('charge_curve_nonmonotonic_or_discontinuous')
    if not bad:
        f['cc_duration_s']=ccdur if len(cc) else np.nan
    if len(cc)>=5:f['charge_crate']=float(np.median(cc[:,3])/1.35)
    if len(cc)>=5 and cc[0,0]<=3.8 and cc[-1,0]>=4.0 and not bad:
        q38,q40=np.interp([3.8,4.0],cc[:,0],cc[:,1]);t38,t40=np.interp([3.8,4.0],cc[:,0],cc[:,2])
        partial=q40-q38
        if partial>1e-5:
            f['partial_charge_ah']=float(partial);f['partial_dv_dq']=float(.2/partial)
            f['partial_charge_time_s']=float(t40-t38)
            qgrid=np.linspace(q38,q40,151)
            if np.all(np.diff(cc[:,1])>0):
                vgrid=np.interp(qgrid,cc[:,1],cc[:,0]);dva=np.gradient(savgol_filter(vgrid,11,3),qgrid)
                f['dva_mean']=float(np.mean(dva));f['dva_peak']=float(np.max(dva))
                curves['dva']=(qgrid-q38,dva)
    else:flags.append('partial_voltage_interval_unavailable')
    ic=ic_curves(cc) if not bad else None
    if ic is not None:
        vg,qg,variants=ic;curve=variants[11];curves['ic']=(vg,curve)
        peaks=np.array([x.max() for x in variants.values()]);spread=float(np.ptp(peaks)/max(np.median(peaks),1e-9))
        f['ic_smoothing_relative_spread']=spread
        negative=np.mean(curve<0)>.01
        if spread>.2 or negative:
            flags.append('ic_smoothing_unstable_or_negative')
        else:
            f['ic_peak_height']=float(np.max(curve));f['ic_peak_area']=float(np.trapezoid(curve,vg))
            p,props=find_peaks(curve,prominence=max(0.05*np.max(curve),1e-6))
            if len(p):
                j=int(p[np.argmax(curve[p])]);f['ic_peak_voltage']=float(vg[j])
                # Maximum must be an interior peak to assign peak position/width.
                if abs(curve[j]-np.max(curve))>0.02*np.max(curve):
                    f['ic_peak_voltage']=np.nan;flags.append('ic_maximum_at_interval_boundary')
                else:
                    width,height,left,right=peak_widths(curve,[j],rel_height=.5)
                    if left[0]>0 and right[0]<len(vg)-1:f['ic_peak_width']=float(width[0]*CONFIG['voltage_grid_v'])
            else:flags.append('no_reliable_interior_ic_peak')
        for w,x in variants.items():f[f'ic_peak_height_sg{w}']=float(np.max(x))
    else:flags.append('ic_interval_or_cc_unavailable')
    cv_ok=(len(cv)>=8 and np.ptp(cv[:,0])<=.035 and cv[-1,3]<=.8*cv[0,3] and not bad)
    if cv_ok:
        f['charge_protocol_cv']=1.;f['cv_duration_s']=cvdur;f['cv_capacity_ah']=cvq
        x=cv[:,2]-cv[0,2];y=np.log(cv[:,3]);slope,intercept=np.polyfit(x,y,1)
        den=np.sum((y-y.mean())**2);r2=1-np.sum((y-(intercept+slope*x))**2)/den if den else np.nan
        f['cv_decay_fit_r2']=float(r2)
        if slope<0 and r2>=.8:f['cv_decay_rate_s_inv']=float(-slope)
        else:flags.append('cv_log_decay_poor_fit')
    else:flags.append('cv_not_verified')
    measured=pd.concat(charge)
    if 'resistance' in measured:
        x=measured.resistance;valid=x[(x>0)&(x<1)&np.isfinite(x)]
        f['resistance_recorded_median_ohm']=float(valid.median()) if len(valid)>=5 else np.nan
    flags.append('resistance_measurement_unverified')
    flags.append('temperature_sensor_unverified_or_absent')
    last_charge=max(j for j,g in enumerate(parts) if g.phase.iloc[0]=='charge')
    for j,g in enumerate(parts):
        if g.phase.iloc[0]=='rest' and j==last_charge+1:
            t=g.test_s.to_numpy(float)-g.test_s.iloc[0]
            if len(g)>2 and t[-1]>=60 and np.max(abs(g.current))<.005 and np.max(np.diff(t))<=CONFIG['max_sample_gap_s']:
                f['relaxation_60s_v']=float(np.interp(60,t,g.voltage)-g.voltage.iloc[0]);break
    if len(cc):curves['vq']=(cc[:,1]-cc[0,1],cc[:,0])
    return f,{'feature_quality_flags':';'.join(sorted(set(flags)))},curves


def add_history(frame):
    frame=frame.sort_values(['battery','lifetime_cycle']).copy()
    for battery,group in frame.groupby('battery',sort=False):
        for source,target in [('ic_peak_voltage','historical_ic_peak_shift_v'),('partial_charge_ah','historical_partial_change_ah'),('resistance_ohm','historical_resistance_change_ohm')]:
            baseline=None
            for idx,row in group.iterrows():
                value=row.get(source,np.nan)
                # No label is consulted when constructing the charging history.
                if np.isfinite(value):
                    if baseline is None:baseline=float(value)
                    else:frame.loc[idx,target]=float(value)-baseline
    return frame
