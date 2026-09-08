import sys, torch, numpy as np, json, argparse
sys.path.insert(0,'/myhome/sdate'); sys.path.insert(0,'/myhome/astra-torch'); sys.path.insert(0,'/myhome/chip-project')
from sdate.tr_naf import build_acquisition, tr_naf_reconstruction, reconstruct_volume_at, evaluate_frames, make_circular_mask
ap=argparse.ArgumentParser()
ap.add_argument('--reg-tv',type=float); ap.add_argument('--n-levels',type=int,default=8)
ap.add_argument('--max-res',type=int,default=128); ap.add_argument('--tag',type=str)
a=ap.parse_args(); dev=torch.device('cuda')
TIF="/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/timesteps/212_Wunderkerze2_rotate_04001.tif"
fr,meta=build_acquisition(TIF,num_frames=50,angle_range_deg=36.0,num_full_projs=360,timestep_skip=9,
                          cube_size=128,normalize_range=None,full_boundary_frames=1,device=dev)
gt=torch.stack([f.true_volume for f in fr]); times=[f.t_norm for f in fr]; mask=make_circular_mask(128,128,device=dev)
gt_rng=float((gt.amax(0)-gt.amin(0))[:,mask].mean())
r=tr_naf_reconstruction(fr,meta,K=6,n_iterations=1500,lr=1e-2,reg_tv=a.reg_tv,reg_temporal_max=0,reg_temporal_tv=1.0,
    field_kwargs=dict(n_levels=a.n_levels,base_resolution=8,max_resolution=a.max_res,hidden_dim=128,n_hidden_layers=3),
    device=dev,seed=0,verbose=False)
rec=torch.stack([reconstruct_volume_at(r,t) for t in times])
m=evaluate_frames([rec[i] for i in range(len(times))],[f.true_volume for f in fr],mask=mask)
rng=float((rec.amax(0)-rec.amin(0))[:,mask].mean())
print(f"RESULT {a.tag}: PSNR {m['psnr'].mean():.2f} SSIM {m['ssim'].mean():.3f} range {100*rng/gt_rng:.0f}% recmax {float(rec[:,:,mask].max()):.2f}", flush=True)
