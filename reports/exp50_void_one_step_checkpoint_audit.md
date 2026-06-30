# Exp50 VOID One-Step Checkpoint Audit

Time: 2026-07-01T00:08:01+08:00

Status: `VOID_ONE_STEP_CHECKPOINT_READY`

## Checkpoint

- Path: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/one_step_v2/adapter_proj_out_step1.pt`
- Exists: True
- Size bytes: 788893
- SHA256: `849326121699e51673990df7bef52c578245812ad934a88bbe6f2acf93b972d5`
- Top-level keys: ['adapter_state', 'summary', 'trainable_filter']
- Trainable filter: `proj_out`
- Saved summary: {'lr': 1e-05, 'frames': 13}

## Adapter Keys

- Adapter key count: 2
- Keys: ['proj_out.bias', 'proj_out.weight']
- Expected trainable subset keys: ['proj_out.bias', 'proj_out.weight']
- Key match: True

## Reload Evidence

- H4 strict reload OK: True
- H4 reload missing keys: []
- H4 reload unexpected keys: []
- Current full policy load: deferred because PAI GPUs are occupied by unrelated root jobs; no process was killed. H4 already validated strict adapter reload in the same env/code path.

## GPU Snapshot

```text
____  ______        __
|  _ \/ ___\ \      / /
| | | \___ \\ \ /\ / / 
| |_| |___) |\ V  V /  
|____/|____/  \_/\_/   
                       

Welcome to PAI DSW!

Mounted datasets: (mount_path is where you could access dataset in dsw instance, files in mount_path will not be committed in image)
╭────────┬──────────────────────────┬──────────────────────────────┬────────────────────────────────────────────┬───────────────────╮
│ type   │ name                     │ mount_path                   │ dataset_info                               │ commit_to_image   │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets            │ /mnt/data/csgo-datasets/     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets/ │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ workspace                │ /mnt/workspace               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ pku-workspace            │ /mnt/data/pku/               │ cpfs-01000vwrt8a6usy68r6wu:/pku/           │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ OSS    │ csgo-datasets-oss        │ /mnt/data/csgo-datasets-oss/ │ oss://dataset-csgo-sample-02/shared/       │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ NAS    │ pku-workspace-nas        │ /mnt/nas/                    │ 0329tx2wi8okhqz07m9:/                      │ no                │
├────────┼──────────────────────────┼──────────────────────────────┼────────────────────────────────────────────┼───────────────────┤
│ BMCPFS │ csgo-datasets-fullsubset │ /mnt/data/csgo-datasets-     │ cpfs-01000vwrt8a6usy68r6wu:/csgo-datasets- │ no                │
│        │                          │ fullsubset/                  │ fullsubset/                                │                   │
╰────────┴──────────────────────────┴──────────────────────────────┴────────────────────────────────────────────┴───────────────────╯

0, 108865 MiB, 143771 MiB, 100 %
1, 108865 MiB, 143771 MiB, 100 %
2, 108865 MiB, 143771 MiB, 100 %
3, 108865 MiB, 143771 MiB, 100 %
4, 108865 MiB, 143771 MiB, 100 %
5, 108865 MiB, 143771 MiB, 100 %
6, 108865 MiB, 143771 MiB, 100 %
7, 108865 MiB, 143771 MiB, 100 %
```

## Safety

No inference, no optimizer step, no training, no VOR-Eval, and no hard comp were used in this audit.
