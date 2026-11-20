import copy
import io
import numpy as np
import pytest
import torch
from experiments.rapct_prony.data import FEATURES, GRID, curve_tensor, fit_scalers
from experiments.rapct_prony.models import Stage1
from experiments.rapct_prony.metrics import metrics


def test_feature_contract():
    assert FEATURES==['cc_duration_s','cv_duration_s','dva_peak','ic_peak_voltage','ic_peak_width']
    assert not any('discharge' in f or 'soh' in f for f in FEATURES)
    assert len(GRID)==176


def test_training_only_scalers():
    f=np.array([[0]*5,[2]*5,[1000]*5],float);c=np.repeat(f[:,:3,None],176,axis=2)
    fs,cs=fit_scalers(f,c,np.array(['a','a','b']),['a'])
    assert np.allclose(fs.mean_,1) and np.allclose(cs.mean_,1)
    with pytest.raises(ValueError):fit_scalers(f,c,np.array(['a','a','b']),['missing'])


@pytest.mark.parametrize('pathway',['mlp','cnn','fusion'])
def test_model_and_checkpoint(pathway):
    torch.manual_seed(42);m=Stage1(pathway).eval();f=torch.randn(2,5);c=torch.randn(2,3,176)
    with torch.no_grad():y=m(f,c)
    assert y.shape==(2,) and torch.isfinite(y).all()
    b=io.BytesIO();torch.save(m.state_dict(),b);b.seek(0)
    n=Stage1(pathway).eval();n.load_state_dict(torch.load(b,weights_only=True))
    assert torch.equal(y,n(f,c))
    n(f,c).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in n.parameters() if p.grad is not None)


def test_metric_units():
    r=metrics([.8,.9],[.81,.91])
    assert np.isclose(r['mae'],.01) and np.isclose(r['mae_pp'],1)

def test_curve_no_extrapolation(monkeypatch):
    import experiments.rapct_prony.data as data
    cc=np.column_stack([np.linspace(3.81,4.15,200),np.linspace(0,1,200),np.arange(200),np.ones(200)])
    monkeypatch.setattr(data,'charge_curve',lambda parts:(cc,None,0,0,0,False))
    with pytest.raises(ValueError,match='coverage'):data.curve_tensor([])


def test_curve_extraction_consistency(monkeypatch):
    import experiments.rapct_prony.data as data
    from src.extract_features import ic_curves
    voltage=np.linspace(3.7,4.17,400)
    q=np.cumsum(.3+np.exp(-((voltage-4.0)/.08)**2))*.001
    cc=np.column_stack([voltage,q,np.arange(400),np.ones(400)*.67])
    monkeypatch.setattr(data,'charge_curve',lambda parts:(cc,None,0,0,0,False))
    x=data.curve_tensor([]);vg,qg,variants=ic_curves(cc)
    assert x.shape==(3,176)
    assert np.allclose(x[0],qg) and np.allclose(x[1],variants[11])


def test_rest_gaps_are_not_verified():
    import pandas as pd
    from experiments.rapct_prony.data import rest_inventory
    d=pd.DataFrame(dict(source=['s']*4,epoch=[0]*4,source_order=range(4),
        timestamp=pd.to_datetime(['2020-01-01 00:00:00','2020-01-01 00:01:00','2020-01-01 01:01:00','2020-01-01 01:02:00']),
        test_s=[0,60,3660,3720],current=[0]*4,battery=['b']*4))
    events=rest_inventory(d,[])
    assert [e['category'] for e in events]==['verified_measured_rest','uncertain_recording_gap','verified_measured_rest']
    assert [e['duration_s'] for e in events]==[60,3600,60]


def test_deterministic_initialization():
    from experiments.rapct_prony.train import seed_all
    seed_all(123);a=Stage1('mlp')
    seed_all(123);b=Stage1('mlp')
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))

def test_training_resume_equivalence(tmp_path,monkeypatch):
    import json
    import pandas as pd
    import experiments.rapct_prony.train as training
    from dataclasses import replace
    from experiments.rapct_prony.data import BATTERIES
    out=tmp_path/'outputs';(out/'processed').mkdir(parents=True);(out/'checkpoints').mkdir()
    rng=np.random.default_rng(42);rows=[]
    for b in BATTERIES:
        for j in range(3):
            r={k:float(x) for k,x in zip(FEATURES,rng.normal(size=5))}
            r.update(battery=b,cycle_id=f'{b}:{j}',prediction_cutoff='2020-01-01',soh=.9-.01*j)
            rows.append(r)
    pd.DataFrame(rows).to_csv(out/'processed/samples.csv',index=False)
    np.save(out/'processed/curves.npy',rng.normal(size=(len(rows),3,176)).astype('float32'))
    (out/'processed/manifest.json').write_text(json.dumps({'version':'synthetic_test'}))
    monkeypatch.setattr(training,'OUT',out)
    config=training.Config(pathway='mlp',epochs=1,batch_size=4,label='synthetic_test')
    _,first=training.train(config)
    training.train(replace(config,epochs=2),resume_run=first)
    resumed=torch.load(first/'last.pt',map_location='cpu',weights_only=True)['model']
    _,full=training.train(replace(config,epochs=2,label='synthetic_uninterrupted'))
    continuous=torch.load(full/'last.pt',map_location='cpu',weights_only=True)['model']
    assert all(torch.equal(resumed[k],continuous[k]) for k in resumed)
    _,refit=training.train(replace(config,refit_all_inner=True,label='synthetic_refit'))
    split=json.loads((refit/'splits.json').read_text())
    assert config.validation in split['train'] and config.outer not in split['train'] and split['validation']==[]
    assert set(pd.read_csv(refit/'metrics.csv').split)=={'outer'}
    assert 'refit_train_diagnostic_rmse' in pd.read_csv(refit/'training.csv')

def test_chronological_windows_never_stitch_missing_cycles():
    import pandas as pd
    from experiments.rapct_prony.data import chronological_windows
    frame=pd.DataFrame(dict(battery=['a']*6,lifetime_cycle=[1,2,3,5,6,7],cycle_id=list('abcdef'),sources=['s']*6,
        local_counter_resets=[0]*6,start=pd.date_range('2020-01-01',periods=6,freq='2h'),end=pd.date_range('2020-01-01 01:00',periods=6,freq='2h')))
    windows,audit=chronological_windows(frame,3)
    assert windows.tolist()==[[0,1,2],[3,4,5]]
    assert audit.loc[3,'reason']=='missing_cycle_history'
    frame.loc[5,'sources']='other'
    windows,_=chronological_windows(frame,3)
    assert windows.tolist()==[[0,1,2]]

def test_transformer_causal_mask_and_padding():
    from experiments.rapct_prony.temporal import Stage2
    torch.manual_seed(42);model=Stage2(frozen=True).eval()
    f=torch.randn(2,6,5);c=torch.randn(2,6,3,176)
    with torch.no_grad():
        original=model.encode_sequence(f,c)
        changed_f=f.clone();changed_c=c.clone();changed_f[:,4:]+=100;changed_c[:,4:]+=100
        changed=model.encode_sequence(changed_f,changed_c)
    assert torch.allclose(original[:,:4],changed[:,:4],atol=1e-6)
    assert not any(p.requires_grad for p in model.cycle.parameters())
    assert torch.count_nonzero(torch.triu(model.last_attention,diagonal=1))==0
    padding=torch.tensor([[False,False,False,False,True,True]]*2)
    with torch.no_grad():
        a=model(f,c,padding);b=model(changed_f,changed_c,padding)
    assert torch.allclose(a,b,atol=1e-6)

def test_transformer_checkpoint_equivalence():
    from experiments.rapct_prony.temporal import Stage2
    model=Stage2().eval();f=torch.randn(1,5,5);c=torch.randn(1,5,3,176)
    with torch.no_grad():before=model(f,c)
    buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
    restored=Stage2().eval();restored.load_state_dict(torch.load(buffer,weights_only=True))
    with torch.no_grad():after=restored(f,c)
    assert torch.equal(before,after)

def test_pretraining_scope_rejects_heldout_batteries(tmp_path):
    import json
    from experiments.rapct_prony.train import validate_pretraining
    (tmp_path/'splits.json').write_text(json.dumps({'train':['a','heldout']}))
    (tmp_path/'dataset_manifest.json').write_text(json.dumps({'v':1}))
    with pytest.raises(ValueError,match='held_out'):validate_pretraining(tmp_path/'best.pt',['a','b'],{'v':1})
    validate_pretraining(tmp_path/'best.pt',['a','heldout'],{'v':1})

def test_degradation_positive_increments_and_chunk_carry():
    from experiments.rapct_prony.physics import DegradationBranch,loss_terms
    model=DegradationBranch();h=torch.randn(2,8,64)
    whole=model(h);a=model(h[:,:4]);b=model(h[:,4:],a['carried'])
    assert torch.all(torch.diff(whole['degradation'],dim=1)>=0)
    assert torch.allclose(whole['degradation'][:,4:],b['degradation'],atol=1e-7)
    reset=model(h[:,4:])
    assert not torch.allclose(reset['degradation'],b['degradation'])
    loss,terms=loss_terms(whole,torch.full((2,8),.9),torch.ones(2,8,dtype=torch.bool),30)
    loss.backward()
    assert terms['warmup']==1 and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_single_relaxation_exact_decay_and_persistence():
    from experiments.rapct_prony.memory import SingleRelaxation
    memory=SingleRelaxation();z=torch.tensor([[.05],[.08]])
    dt=torch.tensor([100.,200.]);verified=torch.tensor([True,False]);stress=torch.zeros_like(z)
    after=memory.step(z,stress,dt,verified)
    assert torch.allclose(after[0],z[0]*torch.exp(-dt[0]/memory.timescales))
    assert torch.equal(after[1],z[1])
    after.sum().backward();assert torch.isfinite(memory.tau_raw.grad)
    unknown=torch.tensor([True,False]);initial=torch.tensor([[.025],[.025]])
    reset=memory.step(z,stress,dt,verified,unknown,initial)
    assert torch.equal(reset[0],initial[0])
    with pytest.raises(ValueError,match='explicit_state'):memory.step(z,stress,dt,verified,unknown)
    assert (memory.timescales>0).all()

@pytest.mark.parametrize('modes',[1,2,4,8])
@pytest.mark.parametrize('learnable',[False,True])
def test_prony_modes_exact_relaxation_and_stability(modes,learnable):
    from experiments.rapct_prony.memory import NeuralProny
    m=NeuralProny(modes,learnable=learnable);initial=torch.full((2,modes),.05)
    dt=torch.tensor([60.,1e10]);verified=torch.tensor([True,True])
    result=m.step(initial,torch.zeros_like(initial),dt,verified)
    analytic=initial*torch.exp(-dt[:,None]/m.timescales)
    assert torch.allclose(result,analytic) and torch.isfinite(result).all()
    assert (m.timescales>0).all() and (m.weights>0).all() and torch.allclose(m.weights.sum(),torch.tensor(1.))
    assert (m.penalty(result)<=m.penalty(initial)).all()
    stress=torch.full((2,8,modes),.001);times=torch.full((2,8),30.);verified=torch.ones(2,8,dtype=torch.bool)
    whole,last=m.rollout(stress,times,verified,initial)
    a,carry=m.rollout(stress[:,:3],times[:,:3],verified[:,:3],initial)
    b,end=m.rollout(stress[:,3:],times[:,3:],verified[:,3:],carry)
    assert torch.equal(torch.cat([a,b],dim=1),whole) and torch.equal(last,end)
    m.penalty(last).sum().backward()
    assert torch.isfinite(m.amplitude_raw.grad).all()
    if learnable:assert torch.isfinite(m.tau_raw.grad).all()


def test_known_synthetic_capacity_rebound():
    from experiments.rapct_prony.memory import NeuralProny
    memory=NeuralProny(1);state=torch.tensor([[.05]])
    after=memory.step(state,torch.zeros_like(state),torch.tensor([100.]),torch.tensor([True]))
    soh_before=1.-.1-memory.penalty(state);soh_after=1.-.1-memory.penalty(after)
    assert torch.allclose(soh_after-soh_before,torch.tensor([.05*(1-np.exp(-1))],dtype=torch.float32),atol=1e-7)

def test_full_composition_causal_finite_and_reload():
    from experiments.rapct_prony.hybrid import RAPCTProny
    torch.manual_seed(42);m=RAPCTProny().eval()
    f=torch.randn(1,5,5);c=torch.randn(1,5,3,176);dt=torch.tensor([[0.,60.,0.,120.,0.]])
    verified=dt>0;unknown=torch.tensor([[True,False,False,False,False]])
    with torch.no_grad():result=m(f,c,dt,verified,unknown)
    for key in ['prediction','degradation','transient_penalty','memory_states','timescales_s','modeled_recovery_contribution']:assert torch.isfinite(result[key]).all()
    assert (torch.diff(result['degradation'],dim=1)>=0).all()
    assert (result['transient_penalty']>=0).all() and (result['transient_penalty']<=.1).all()
    changed=f.clone();changed[:,3:]+=100
    with torch.no_grad():changed_result=m(changed,c,dt,verified,unknown)
    assert torch.allclose(result['prediction'][:,:3],changed_result['prediction'][:,:3],atol=1e-6)
    b=io.BytesIO();torch.save(m.state_dict(),b);b.seek(0)
    other=RAPCTProny().eval();other.load_state_dict(torch.load(b,weights_only=True))
    with torch.no_grad():restored=other(f,c,dt,verified,unknown)
    assert torch.equal(result['prediction'],restored['prediction'])

def test_physics_state_math_remains_float32_under_autocast():
    from experiments.rapct_prony.physics import DegradationBranch
    from experiments.rapct_prony.memory import NeuralProny
    model=DegradationBranch();h=torch.randn(1,5,64)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        output=model(h)
        m=NeuralProny();states=m.step(torch.full((1,4),.05),torch.zeros(1,4),torch.tensor([60.]),torch.tensor([True]))
    assert output['degradation'].dtype==torch.float32 and output['rate'].dtype==torch.float32
    assert states.dtype==torch.float32 and torch.isfinite(states).all()

def test_hex_configuration_ids_are_not_scientific_notation(tmp_path):
    from experiments.rapct_prony.metrics import read_metrics
    path=tmp_path/'metric.csv';path.write_text('config_id,rmse\n143930e71344,0.01\n')
    result=read_metrics(path)
    assert result.config_id.iloc[0]=='143930e71344'
