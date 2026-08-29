"""Stage-1 (Option B) 128^3 training for molab / RTX Pro 6000 (Blackwell).
Run with:  !python train_molab.py
Reads data from /root/data (see setup cells), repo from /root/coronary-centerline-diffusion.
Saves best EMA checkpoint + Stage-2 Chamfer/overlap + figures to /root/out.
"""
import os, sys, json, time
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

DATA_ROOT = "/root/data"
REPO      = "/root/coronary-centerline-diffusion"
CKPT      = "/root/checkpoints"; Path(CKPT).mkdir(exist_ok=True)
OUT       = "/root/out";         Path(OUT).mkdir(exist_ok=True)
sys.path.insert(0, REPO)
from src.coronarycl.models.volume_model import BackProjVolumeDenoiser

device = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
assert device == "cuda", "no GPU — check torch install (Blackwell needs cu128 build)"

# ---------- discover the 3 inputs anywhere under DATA_ROOT ----------
def find_with_key(key):
    best, bestn = None, 0
    for dp, _, fns in os.walk(DATA_ROOT):
        npzs = [f for f in fns if f.endswith(".npz")]
        if len(npzs) > bestn:
            try:
                if key in np.load(os.path.join(dp, npzs[0])).files:
                    best, bestn = dp, len(npzs)
            except Exception:
                pass
    return best, bestn

VOL_DIR, nv = find_with_key("volume")
PKG_DIR, npk = find_with_key("images")
SPLITS = None
for dp, _, fns in os.walk(DATA_ROOT):
    if "case_splits.json" in fns:
        SPLITS = os.path.join(dp, "case_splits.json"); break
print(f"volumes: {VOL_DIR} ({nv}) | packaged: {PKG_DIR} ({npk}) | splits: {SPLITS}")
assert nv >= 900 and npk >= 900 and SPLITS, \
    "dataset missing — did the zips unzip? (gdown folder caps at ~50 loose files; zip them)"
splits = json.load(open(SPLITS))
tr_ids, va_ids, te_ids = splits["train"], splits["val"], splits["test"]

# ---------- helpers ----------
class VolXrayDS(Dataset):
    def __init__(s, ids, vol_dir, pkg_dir, vol_res=128, img_res=128):
        s.ids=ids; s.vd=vol_dir; s.pd=pkg_dir; s.vr=vol_res; s.ir=img_res
    def __len__(s): return len(s.ids)
    def __getitem__(s, i):
        cid=s.ids[i]; vz=np.load(f"{s.vd}/{cid}.npz"); pz=np.load(f"{s.pd}/{cid}.npz")
        v=F.avg_pool3d(torch.from_numpy(vz["volume"]).float()[None,None], 128//s.vr)[0]
        im=F.interpolate(torch.from_numpy(pz["images"]).float()[None], size=s.ir, mode="bilinear", align_corners=False)[0]
        return {"volume":v, "images":im,
                "poses":torch.from_numpy(pz["poses"].astype(np.float32)),
                "svoxel":torch.from_numpy(vz["svoxel"].astype(np.float32))}

class VolScheduler:
    def __init__(s, n=1000, device="cpu"):
        s.n=n; s.betas=torch.linspace(1e-4,0.02,n,device=device)
        s.alphas=1-s.betas; s.abars=torch.cumprod(s.alphas,0)
    def add_noise(s, x0, t):
        ab=s.abars[t].view(-1,1,1,1,1); noise=torch.randn_like(x0)
        return torch.sqrt(ab)*x0+torch.sqrt(1-ab)*noise, noise

def vol_loss(model, sched, batch, device, fg=150.0):
    x0=batch["volume"].to(device); im=batch["images"].to(device)
    P=batch["poses"].to(device); sv=batch["svoxel"].to(device); B=x0.shape[0]
    t=torch.randint(0, sched.n, (B,), device=device)
    xt,_=sched.add_noise(x0,t); x0h=model(xt,t,im,P,sv)
    g=(x0>0.05).float(); w=1+fg*g
    mse=(F.mse_loss(x0h,x0,reduction="none")*w).mean()
    p=x0h.clamp(0,1); dl=1-(2*(p*g).sum()+1)/((p+g).sum()+1)
    return mse+dl

@torch.no_grad()
def sample_vol(model, sched, images, poses, svoxel, res, device, seed=0, n_steps=50):
    torch.manual_seed(seed); model.eval(); ab=sched.abars; T=sched.n
    steps=list(reversed(range(0,T,max(1,T//n_steps)))); B=images.shape[0]
    x=torch.randn(B,1,res,res,res,device=device)
    for i,ts in enumerate(steps):
        t=torch.full((B,),ts,device=device,dtype=torch.long)
        x0=model(x,t,images,poses,svoxel).clamp(0,1)
        abt=ab[ts]; eps=(x-torch.sqrt(abt)*x0)/torch.sqrt(1-abt+1e-8)
        tp=steps[i+1] if i+1<len(steps) else -1
        abp=ab[tp] if tp>=0 else torch.tensor(1.0,device=device)
        x=torch.sqrt(abp)*x0+torch.sqrt(1-abp)*eps
    model.train(); return x.clamp(0,1)

def dice(pred, gt, thr=0.3):
    p=(pred>thr).float(); g=(gt>0.05).float()
    return float((2*(p*g).sum())/(p.sum()+g.sum()+1e-6))

# ---------- training (128^3, EMA + cosine LR) ----------
RES=128; EPOCHS=50; BS=4; ACCUM=1          # effective batch = 4; Blackwell has room
USE_CKPT=False                              # 96 GB VRAM -> checkpointing off = faster
tr = DataLoader(VolXrayDS(tr_ids, VOL_DIR, PKG_DIR, vol_res=RES),
                batch_size=BS, shuffle=True, num_workers=4, drop_last=True)
opt_steps = (len(tr) // ACCUM) * EPOCHS

class EMA:
    def __init__(s, model, decay=0.999):
        s.decay=decay; s.shadow={k:v.detach().clone() for k,v in model.state_dict().items()}
    def update(s, model):
        for k,v in model.state_dict().items():
            if v.dtype.is_floating_point: s.shadow[k].mul_(s.decay).add_(v.detach(), alpha=1-s.decay)
            else: s.shadow[k].copy_(v)
    def copy_to(s, model): model.load_state_dict(s.shadow, strict=True)

model = BackProjVolumeDenoiser(base=32, use_checkpoint=USE_CKPT).to(device)
sched = VolScheduler(n=1000, device=device)
opt   = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opt_steps, eta_min=1e-5)
ema = EMA(model, decay=0.999)

model.train(); torch.cuda.reset_peak_memory_stats()
b0 = next(iter(tr)); l0 = vol_loss(model, sched, b0, device); l0.backward(); opt.zero_grad()
print("smoke OK — peak GB:", round(torch.cuda.max_memory_allocated()/1e9, 2))

@torch.no_grad()
def eval_dice_ema(ids, n=6):
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.copy_to(model); model.eval()
    dss = VolXrayDS(ids, VOL_DIR, PKG_DIR, vol_res=RES); sc = []
    for j in range(min(n, len(dss))):
        bb = dss[j]
        pr = sample_vol(model, sched, bb["images"][None].to(device),
                        bb["poses"][None].to(device), bb["svoxel"][None].to(device), RES, device, n_steps=40)
        sc.append(dice(pr, bb["volume"][None].to(device)))
    model.load_state_dict(backup); model.train()
    return float(np.mean(sc))

best=0.0; ostep=0; t0=time.time(); opt.zero_grad()
for ep in range(EPOCHS):
    for i, batch in enumerate(tr):
        loss = vol_loss(model, sched, batch, device) / ACCUM
        loss.backward()
        if (i + 1) % ACCUM == 0:
            opt.step(); lr_sched.step(); ema.update(model); opt.zero_grad(); ostep += 1
            if ostep % 100 == 0:
                print(ostep, round(loss.item()*ACCUM,4), f"lr {opt.param_groups[0]['lr']:.2e}", int(time.time()-t0),"s", flush=True)
    if ep % 2 == 0 or ep == EPOCHS-1:
        d = eval_dice_ema(va_ids, n=6)
        print(f"== epoch {ep} EMA val Dice(6) {d:.3f} ==", flush=True)
        if d > best:
            best=d; torch.save(ema.shadow, f"{CKPT}/vol_best_ema_128.pt"); print("  saved best(EMA)", round(best,3), flush=True)
print("done. best EMA val Dice", round(best,3))

# ---------- Stage 2: skeletonize -> centerline -> Chamfer/overlap ----------
from scipy.spatial import cKDTree
try:
    from skimage.morphology import skeletonize_3d as skel3d
except ImportError:
    from skimage.morphology import skeletonize as skel3d
from src.coronarycl.metrics import overlap_metric

model.load_state_dict(torch.load(f"{CKPT}/vol_best_ema_128.pt", map_location=device, weights_only=True)); model.eval()
norm  = json.load(open(f"{REPO}/data/splits/normalization_stats.json"))
cmean = np.array(norm["coord_mean"]); cstd = np.array(norm["coord_std"])

def chamfer_l2(pred, gt):
    if len(pred)==0 or len(gt)==0: return float("nan")
    dp,_=cKDTree(gt).query(pred); dg,_=cKDTree(pred).query(gt)
    return float(dp.mean()+dg.mean())

def pred_centerline_mm(cid, res=128, thr=0.3, seed=0):
    vz=np.load(f"{VOL_DIR}/{cid}.npz"); sv=vz["svoxel"].astype(np.float64)
    pz=np.load(f"{PKG_DIR}/{cid}.npz")
    im=F.interpolate(torch.from_numpy(pz["images"]).float()[None],size=128,mode="bilinear",align_corners=False).to(device)
    P =torch.from_numpy(pz["poses"].astype(np.float32))[None].to(device)
    svt=torch.from_numpy(vz["svoxel"].astype(np.float32))[None].to(device)
    vol=sample_vol(model,sched,im,P,svt,res,device,seed=seed)[0,0].cpu().numpy()
    idx=np.argwhere(skel3d((vol>thr).astype(np.uint8))>0).astype(np.float64)
    if len(idx)==0: idx=np.argwhere(vol>thr).astype(np.float64)
    return sv*((idx+0.5)/res-0.5)

def gt_centerline_mm(cid):
    vz=np.load(f"{VOL_DIR}/{cid}.npz"); sv=vz["svoxel"].astype(np.float64)
    pz=np.load(f"{PKG_DIR}/{cid}.npz")
    cl=pz["centerline"][pz["centerline_mask"].astype(bool)][:,:3].astype(np.float64)
    return cl*cstd+cmean - sv/2

ch=[]; ov={1:[],2:[],5:[]}
for cid in va_ids:
    pr=pred_centerline_mm(cid); gt=gt_centerline_mm(cid)
    ch.append(chamfer_l2(pr,gt))
    for d in (1,2,5): ov[d].append(overlap_metric(pr,gt,d))
print(f"\nStage-2 mean Chamfer L2: {np.nanmean(ch):.2f} mm   (baseline 22.78 mm)")
for d in (1,2,5): print(f"Ot@{d}mm: {np.mean(ov[d]):.3f}")
json.dump({"chamfer_per_case": dict(zip(map(str,va_ids), ch)),
           "chamfer_mean": float(np.nanmean(ch)),
           "overlap": {d: float(np.mean(ov[d])) for d in (1,2,5)}},
          open(f"{OUT}/stage2_results.json","w"), indent=2)
print("saved", f"{OUT}/stage2_results.json  and checkpoint at {CKPT}/vol_best_ema_128.pt")
