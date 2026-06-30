import sys; sys.path.insert(0,"/mnt/data/sinogram_generator")
import numpy as np, torch, torch.nn.functional as F
from utils.patient import RTDataset
from models.sinogram_2p5d import DosePrediction2p5D
from utils.dose_operator import dose_forward, accumulate, build_zbin, read_geometry, find_plan
KCSV="/mnt/data/DoseCUDA/DoseCUDA/lookuptables/photons/Tomo/6MV/kernel.csv"
cones=np.loadtxt(KCSV,delimiter=",",skiprows=1)
def h_iso(r):
    th=np.deg2rad(cones[:,0]); w=np.sin(th); val=np.zeros_like(r)
    for (Am,am,Bm,bm),wi in zip(cones[:,1:5],w): val=val+wi*(Am*np.exp(-am*r)+Bm*np.exp(-bm*r))
    return val/w.sum()
def build_kernel3d(dz,dxy,rmax=10.0,fine=0.1,device="cuda"):
    nz=int(rmax/dz); nxy=int(rmax/dxy)
    zc=np.arange(-nz,nz+1)*dz; yc=np.arange(-nxy,nxy+1)*dxy
    sub=np.arange(-0.5+fine/2,0.5,fine)
    K=np.zeros((len(zc),len(yc),len(yc)))
    for iz,z in enumerate(zc):
        for iy,y in enumerate(yc):
            for ix,x in enumerate(yc):
                ZZ,YY,XX=np.meshgrid(z+sub*dz,y+sub*dxy,x+sub*dxy,indexing="ij")
                K[iz,iy,ix]=h_iso(np.sqrt(ZZ**2+YY**2+XX**2).ravel()).mean()
    K=torch.tensor(K,dtype=torch.float32,device=device); K/=K.sum(); return K[None,None]
dev="cuda"; pat="183040"
ds=RTDataset("/mnt/data/tomo_data/",augmentation=None,use_cache=True,cache_dir="/mnt/data/tomo_data/cache_sino_r8",reduction_ratio=8,debug=pat)
s=ds[0]; ct=s["input"][0].float().cuda(); gt=s["target"].float().cuda(); N=ct.shape[0]
mdl=DosePrediction2p5D(base_filters=32,in_channel=2,n_leaves=64,reduce_h=False).cuda()
mdl.load_state_dict(torch.load("checkpoints/20260626_200610_2p5d_dose/best_model_new_session_session_0_.pth",map_location="cuda")["model_state_dict"]); mdl.eval()
with torch.no_grad(): pred=torch.sigmoid(mdl(s["input"].unsqueeze(0).cuda())[:,0,:,:,0])[0]
ang,tab=read_geometry(find_plan("/mnt/data/tomo_data/",s["patient_id"],s["pareto_index"]))
alpha=torch.tensor(90.0-ang[:N]).cuda(); zbin=build_zbin(tab[:N],48,device="cuda")
dz_cm=(np.ptp(tab[:N])/48)/10.0; dxy_cm=0.625
def blur(p,sg):
    r=max(1,int(3*sg)); t=torch.arange(-r,r+1.).cuda(); k=torch.exp(-t**2/(2*sg*sg)); k/=k.sum()
    x=p[None,None]; x=F.conv2d(x,k.view(1,1,-1,1),padding=(r,0)); return F.conv2d(x,k.view(1,1,1,-1),padding=(0,r))[0,0].clamp(0,1)
def sharp(p,g):
    ps=p**g; return (ps*(p.sum(1,keepdim=True)/(ps.sum(1,keepdim=True)+1e-8))).clamp(0,1)
variants={"GT":gt,"pred":pred,"pred_blur_s1":blur(pred,1.0),"pred_sharp_g1.5":sharp(pred,1.5)}
ccc={"GT":4.485,"pred":4.951,"pred_blur_s1":4.829,"pred_sharp_g1.5":4.978}
K=build_kernel3d(dz_cm,dxy_cm); print(f"voxel dz={dz_cm:.2f}cm dxy={dxy_cm:.2f}cm kernel {tuple(K.shape[2:])}",flush=True)
def dk(sino):
    t=accumulate(dose_forward(sino,ct,0.03),alpha,zbin,48,sign=1.0,reduce="sum")
    return F.conv3d(t[None,None],K,padding=[k//2 for k in K.shape[2:]])[0,0].max().item()
def do(sino):
    return accumulate(dose_forward(sino,ct,0.03),alpha,zbin,48,sign=1.0,reduce="sum",field_z=2.5,scatter_xy=1.0).max().item()
rows={n:(dk(v),do(v)) for n,v in variants.items()}
gk=rows["GT"][0]; go=rows["GT"][1]; gc=ccc["GT"]
print(f'{"variant":>16} {"CCC/GT":>8} {"KERNEL/GT":>10} {"OLD/GT":>8}')
for n in variants: print(f'{n:>16} {ccc[n]/gc:>8.3f} {rows[n][0]/gk:>10.3f} {rows[n][1]/go:>8.3f}')
ck=np.corrcoef([ccc[n]/gc for n in variants],[rows[n][0]/gk for n in variants])[0,1]
co=np.corrcoef([ccc[n]/gc for n in variants],[rows[n][1]/go for n in variants])[0,1]
print(f'\ncorr(KERNEL,CCC)={ck:.3f}  vs  corr(OLD,CCC)={co:.3f}',flush=True)
