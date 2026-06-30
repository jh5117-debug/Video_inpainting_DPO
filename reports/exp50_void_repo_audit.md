# Exp50 VOID Repo Audit

Status: `VOID_REPO_READY`.

- Official repo: `https://github.com/Netflix/void-model.git`
- Target path: `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model`
- Clone log: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/b1_void_repo_clone_20260630_103727.log`

## Git

- Commit SHA: `e3914f8f551dd4b880661991fd6b28cd1699a97a`
- Branch: `main`
- License file SHA256: `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`
- README SHA256: `628796eee0db520f3589a5c80ef6d889871185a8be18e8304db7b71f8918b4b9`
- requirements SHA256: `bafb9e980bf49f8a8b4ce6765f8714a591bb58305ab28461ac17c2bb1dbe478f`

## Key Paths Observed

See `reports/exp50_void_repo_inventory.txt` for directory inventory up to depth 3.

Expected public entry points for later milestones:

- `scripts/cogvideox_fun/train_void.sh`
- `scripts/cogvideox_fun/train_void_warped_noise.sh`
- `inference/cogvideox_fun/predict_v2v.py`
- `inference/cogvideox_fun/inference_with_pass1_warped_noise.py`
- `datasets/void_train_data.json`
- `config/quadmask_cogvideox.py`
- `videox_fun/`
- `data_generation/`

No official VOID source was modified.
