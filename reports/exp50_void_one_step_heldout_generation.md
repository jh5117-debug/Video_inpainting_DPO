# Exp50 VOID One-Step Heldout Generation

Time: 2026-07-01T00:09:39+08:00

Status: `VOID_ONE_STEP_HELDOUT_GENERATION_BLOCKED`

## Blocker

`NO_FREE_PAI_GPU_ALL_8_OCCUPIED_BY_UNRELATED_ROOT_JOBS`

All 8 PAI GPUs are occupied by unrelated root-owned jobs. No process was killed, no GPU reset was attempted, and no Exp50 video generation was launched. Because no heldout4 Step0/Step1 videos were generated, H4b-3 metrics/visual review and H5 10-step remain locked.

## Planned Inputs

- Heldout manifest: `manifests/exp50_void_adapter_heldout4.jsonl`
- Adapter checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/one_step_v2/adapter_proj_out_step1.pt`
- Adapter SHA256: `849326121699e51673990df7bef52c578245812ad934a88bbe6f2acf93b972d5`
- Output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/one_step_heldout_evidence_v2`

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

index, memory.used [MiB], memory.free [MiB], memory.total [MiB], utilization.gpu [%]
0, 108865 MiB, 34292 MiB, 143771 MiB, 100 %
1, 108865 MiB, 34292 MiB, 143771 MiB, 100 %
2, 108865 MiB, 34292 MiB, 143771 MiB, 100 %
3, 108865 MiB, 34292 MiB, 143771 MiB, 100 %
4, 108865 MiB, 34292 MiB, 143771 MiB, 100 %
5, 108865 MiB, 34292 MiB, 143771 MiB, 100 %
6, 108865 MiB, 34292 MiB, 143771 MiB, 100 %
7, 108865 MiB, 34292 MiB, 143771 MiB, 100 %
```

## GPU Process Snapshot

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

# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command 
# Idx           #    C/G      %      %      %      %      %      %    name 
    0    3402146     C     99     32      -      -      -      -    python3        
    1    3402299     C     99     34      -      -      -      -    python3        
    2    3402636     C     99     33      -      -      -      -    python3        
    3    3402849     C     98     31      -      -      -      -    python3        
    4    3403316     C     99     33      -      -      -      -    python3        
    5    3403653     C     99     32      -      -      -      -    python3        
    6    3403997     C     98     32      -      -      -      -    python3        
    7    3404340     C     98     32      -      -      -      -    python3
```

## Safety

No inference was run, no optimizer step was run, no VOR-Eval was used, no hard comp was used, and no unrelated GPU process was killed.
