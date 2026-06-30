import sys; sys.path.insert(0,"/mnt/data/sinogram_generator")
import sys, tempfile, numpy as np, torch, pydicom
from pathlib import Path
from DoseCUDA import TomoDoseGrid, TomoPlan
from utils.patient import RTDataset
from models.sinogram_2p5d import DosePrediction2p5D
from utils.dose_operator import find_plan
gamma=float(sys.argv[1]); pat='183040'
CKPT='checkpoints/20260626_200610_2p5d_dose/best_model_new_session_session_0_.pth'
TAG=(0x300D,0x10A7)
ds=RTDataset('/mnt/data/tomo_data/',augmentation=None,use_cache=True,cache_dir='/mnt/data/tomo_data/cache_sino_r8',reduction_ratio=8,debug=pat)
s=ds[0]; inp=s['input'].unsqueeze(0).cuda()
m=DosePrediction2p5D(base_filters=32,in_channel=2,n_leaves=64,reduce_h=False).cuda()
m.load_state_dict(torch.load(CKPT,map_location='cuda')['model_state_dict']); m.eval()
with torch.no_grad(): pr=torch.sigmoid(m(inp)[:,0,:,:,0])[0].cpu()
if gamma>1:
    ps=pr**gamma; pr=(ps*(pr.sum(1,keepdim=True)/(ps.sum(1,keepdim=True)+1e-8))).clamp(0,1)
pred=pr.numpy(); print(f'gamma={gamma} pred mean {pred.mean():.4f}',flush=True)
plan_path=find_plan('/mnt/data/tomo_data/',s['patient_id'],s['pareto_index'])
ct_dir=str(Path(plan_path).parent.parent)
dsp=pydicom.dcmread(plan_path,force=True); cps=dsp[(0x300A,0x00B0)][0][(0x300A,0x0111)].value
for i in range(min(len(cps),pred.shape[0])):
    cps[i][TAG].value='\\'.join(f'{v:.7g}' for v in pred[i]).encode()
tmp=tempfile.NamedTemporaryFile(suffix='_PRED.dcm',delete=False,dir='/mnt/data/sinogram_generator'); dsp.save_as(tmp.name)
iso=np.array([float(v) for v in cps[0][(0x300A,0x012C)].value])
dose=TomoDoseGrid(); dose.loadCTDCM(ct_dir); dose.resampleCTfromSpacing(2.5)
dose.setDoseROI(bbox_min_mm=[iso[0]-180,iso[1]-180,iso[2]-130],bbox_max_mm=[iso[0]+180,iso[1]+180,iso[2]+130])
pp=TomoPlan('Tomo'); pp.readPlanDicom(tmp.name,n_sub_cps=3); dose.computeTomoPlan(pp,gpu_id=0); dp=dose.dose.copy()
Path(tmp.name).unlink(missing_ok=True)
dg=np.load('ccc_gt_183040_roi.npy'); sl=tuple(slice(0,min(a,b)) for a,b in zip(dp.shape,dg.shape)); dp,dg=dp[sl],dg[sl]
msk=dg>0.1*dg.max()
corr=lambda a,b:float(np.corrcoef(a.ravel(),b.ravel())[0,1])
print(f'RESULT g={gamma}: corr_region {corr(dp[msk],dg[msk]):.4f} | max pred {dp.max():.3f} gt {dg.max():.3f} ({100*(dp.max()/dg.max()-1):+.0f}%) | mean_reg pred {dp[msk].mean():.3f} gt {dg[msk].mean():.3f} ({100*(dp[msk].mean()/dg[msk].mean()-1):+.0f}%)',flush=True)
