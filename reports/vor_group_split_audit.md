# VOR Group-Level Split Audit

- input_triplets: 57750
- input_sha256: `33d57a3ea23c5799b583d476a311089f95cbce1b0d11280822a63b8c9edcddc4`
- scene_groups: 1449
- source_counts: `{'BLENDER': 21494, 'REAL': 36256}`
- output_counts: `{'train_source_pool': 4096, 'search_dev': 256, 'shadow_dev': 256, 'gate128': 128}`
- scene_group_counts: `{'train_source_pool': 98, 'search_dev': 8, 'shadow_dev': 8, 'gate128': 53}`
- group_overlap_counts: `{'train_search': 0, 'train_shadow': 0, 'search_shadow': 0}`
- gate128_source_counts: `{'BLENDER': 48, 'REAL': 80}`

VOR-Eval is not read by this script and remains excluded from train/search/shadow construction.
