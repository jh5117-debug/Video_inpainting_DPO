# VOR Archive Integrity Audit

- archive_dir: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/downloads/fa09dc61128ca0418a4a13364d97a08018ea9cc7`
- lightweight_ok: `True`
- stream_probe: `True`

| group | expected parts | actual parts | contiguous | expected bytes | actual bytes | size mismatches |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| VOR-Eval | 1 | 1 | True | 237430605 | 237430605 | 0 |
| VOR-Train-MASK | 3 | 3 | True | 27136966500 | 27136966500 | 0 |
| VOR-Train | 32 | 32 | True | 336356544512 | 336356544512 | 0 |

## Stream Probe
- VOR-Eval: opened=True members=5 unsafe=0 error=``
- VOR-Train-MASK: opened=True members=5 unsafe=0 error=``
- VOR-Train: opened=True members=5 unsafe=0 error=``
