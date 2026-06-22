# Exp23 GPU2/4/5/6 Release for Corrected Pair

date: 2026-06-22T00:11:49+08:00
host: dsw-753014-dc85766cb-4v2jj

## Before
index, uuid, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
0, GPU-a44622e3-2296-6285-0a86-84d0f6c67900, 130013 MiB, 143771 MiB, 100 %
1, GPU-4f31bb83-f8dc-d0af-029e-bc4356e6afeb, 130053 MiB, 143771 MiB, 100 %
2, GPU-4e9e74b7-4977-f04d-3af3-ccfe5f570e90, 105851 MiB, 143771 MiB, 100 %
3, GPU-02dbe542-8a6f-11a4-7c2a-486b7d4e9092, 129993 MiB, 143771 MiB, 100 %
4, GPU-cb449822-5571-be90-8303-e7558da3ea66, 106091 MiB, 143771 MiB, 100 %
5, GPU-8e137fa6-c3d9-6980-7733-0d8df296756e, 105851 MiB, 143771 MiB, 100 %
6, GPU-d9337494-6060-1440-039f-4654245b5764, 106139 MiB, 143771 MiB, 100 %
7, GPU-28780eb0-24d5-1f13-59b8-ae5c23617a72, 58071 MiB, 143771 MiB, 0 %

gpu_uuid, pid, process_name, used_gpu_memory [MiB]
GPU-a44622e3-2296-6285-0a86-84d0f6c67900, 1328667, /usr/local/bin/python3.10, 130004 MiB
GPU-4f31bb83-f8dc-d0af-029e-bc4356e6afeb, 1328668, /usr/local/bin/python3.10, 130044 MiB
GPU-4e9e74b7-4977-f04d-3af3-ccfe5f570e90, 1425258, python3, 105842 MiB
GPU-02dbe542-8a6f-11a4-7c2a-486b7d4e9092, 1328669, /usr/local/bin/python3.10, 129984 MiB
GPU-cb449822-5571-be90-8303-e7558da3ea66, 1425259, python3, 105842 MiB
GPU-8e137fa6-c3d9-6980-7733-0d8df296756e, 1425260, python3, 105842 MiB
GPU-d9337494-6060-1440-039f-4654245b5764, 1425261, python3, 105842 MiB
GPU-28780eb0-24d5-1f13-59b8-ae5c23617a72, 1758887, [Not Found], 58060 MiB

USER         PID    PPID    PGID     SID     ELAPSED STAT CMD
root     1425250 1418358 1418350 1418350       07:41 S    bash eval_4col_bigdata_hjcards.sh output/qxq_base_low_bigdata_3ep_6gpu_v1/checkpoints/memory_dense_adapter_step_000450.pt output/qxq_base_high_bigdata_3ep_6gpu_v1/checkpoints/memory_dense_adapter_step_000450.pt step450
root     1425251 1418358 1418350 1418350       07:41 S    bash eval_4col_bigdata_hjcards.sh output/qxq_base_low_bigdata_3ep_6gpu_v1/checkpoints/memory_dense_adapter_step_000450.pt output/qxq_base_high_bigdata_3ep_6gpu_v1/checkpoints/memory_dense_adapter_step_000450.pt step450
root     1425253 1418358 1418350 1418350       07:41 S    bash eval_4col_bigdata_hjcards.sh output/qxq_base_low_bigdata_3ep_6gpu_v1/checkpoints/memory_dense_adapter_step_000450.pt output/qxq_base_high_bigdata_3ep_6gpu_v1/checkpoints/memory_dense_adapter_step_000450.pt step450
root     1425255 1418358 1418350 1418350       07:41 S    bash eval_4col_bigdata_hjcards.sh output/qxq_base_low_bigdata_3ep_6gpu_v1/checkpoints/memory_dense_adapter_step_000450.pt output/qxq_base_high_bigdata_3ep_6gpu_v1/checkpoints/memory_dense_adapter_step_000450.pt step450
root     1425258 1425250 1418350 1418350       07:41 Rl   python3 /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/tools/qxq_sample_base_dense_v0.py --phase2a-source-manifest /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_mutual_ext_compat_v0/manifest_sampler_compat_v0.jsonl --phase2a-window-id mutual_visible_ext_00_2d37ea5f_ep_000007_ego6x8_f340_v2 --phase2a-ego-stem Ep_000007_team_2_player_0006_inst_000 --phase2a-materialized-root /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_base_extmutual_step2500_20260617/_mat --clip-root /mnt/workspace/xiaoqi/datasets/memory_light_dust2_pilot2_v0/clips --variant true_dense --out-dir /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_eval_bigdata_4col_step450/base_0617 --seed 20260613 --latent-frames 21 --steps 70 --shift 3.0 --guide 5.0 --adapter-checkpoint-low /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_base_low_K11T2_20260617/checkpoints/memory_dense_adapter_step_002500.pt --adapter-checkpoint-high /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_base_high_K11T2_20260617/checkpoints/memory_dense_adapter_step_002500.pt
root     1425259 1425251 1418350 1418350       07:41 Rl   python3 /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/tools/qxq_sample_base_dense_v0.py --phase2a-source-manifest /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_mutual_ext_compat_v0/manifest_sampler_compat_v0.jsonl --phase2a-window-id mutual_visible_ext_00_2d37ea5f_ep_000007_ego6x8_f340_v2 --phase2a-ego-stem Ep_000007_team_2_player_0008_inst_000 --phase2a-materialized-root /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_base_extmutual_step2500_20260617/_mat --clip-root /mnt/workspace/xiaoqi/datasets/memory_light_dust2_pilot2_v0/clips --variant true_dense --out-dir /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_eval_bigdata_4col_step450/base_0617 --seed 20260613 --latent-frames 21 --steps 70 --shift 3.0 --guide 5.0 --adapter-checkpoint-low /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_base_low_K11T2_20260617/checkpoints/memory_dense_adapter_step_002500.pt --adapter-checkpoint-high /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_base_high_K11T2_20260617/checkpoints/memory_dense_adapter_step_002500.pt
root     1425260 1425253 1418350 1418350       07:41 Rl   python3 /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/tools/qxq_sample_base_dense_v0.py --phase2a-source-manifest /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_mutual_ext_compat_v0/manifest_sampler_compat_v0.jsonl --phase2a-window-id mutual_visible_ext_01_118fba5a_ep_000003_ego0x3_f1636_v2 --phase2a-ego-stem Ep_000003_team_2_player_0000_inst_000 --phase2a-materialized-root /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_base_extmutual_step2500_20260617/_mat --clip-root /mnt/workspace/xiaoqi/datasets/memory_light_dust2_pilot2_v0/clips --variant true_dense --out-dir /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_eval_bigdata_4col_step450/base_0617 --seed 20260613 --latent-frames 21 --steps 70 --shift 3.0 --guide 5.0 --adapter-checkpoint-low /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_base_low_K11T2_20260617/checkpoints/memory_dense_adapter_step_002500.pt --adapter-checkpoint-high /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_base_high_K11T2_20260617/checkpoints/memory_dense_adapter_step_002500.pt
root     1425261 1425255 1418350 1418350       07:41 Rl   python3 /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/tools/qxq_sample_base_dense_v0.py --phase2a-source-manifest /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_mutual_ext_compat_v0/manifest_sampler_compat_v0.jsonl --phase2a-window-id mutual_visible_ext_01_118fba5a_ep_000003_ego0x3_f1636_v2 --phase2a-ego-stem Ep_000003_team_2_player_0003_inst_000 --phase2a-materialized-root /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_base_extmutual_step2500_20260617/_mat --clip-root /mnt/workspace/xiaoqi/datasets/memory_light_dust2_pilot2_v0/clips --variant true_dense --out-dir /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_eval_bigdata_4col_step450/base_0617 --seed 20260613 --latent-frames 21 --steps 70 --shift 3.0 --guide 5.0 --adapter-checkpoint-low /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_base_low_K11T2_20260617/checkpoints/memory_dense_adapter_step_002500.pt --adapter-checkpoint-high /mnt/workspace/xiaoqi/multigen/multiview-map-v0-qxq/output/qxq_base_high_K11T2_20260617/checkpoints/memory_dense_adapter_step_002500.pt

## After
2026-06-22T00:12:20+08:00
index, uuid, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
0, GPU-a44622e3-2296-6285-0a86-84d0f6c67900, 130013 MiB, 143771 MiB, 100 %
1, GPU-4f31bb83-f8dc-d0af-029e-bc4356e6afeb, 130053 MiB, 143771 MiB, 100 %
2, GPU-4e9e74b7-4977-f04d-3af3-ccfe5f570e90, 0 MiB, 143771 MiB, 0 %
3, GPU-02dbe542-8a6f-11a4-7c2a-486b7d4e9092, 129993 MiB, 143771 MiB, 100 %
4, GPU-cb449822-5571-be90-8303-e7558da3ea66, 244 MiB, 143771 MiB, 0 %
5, GPU-8e137fa6-c3d9-6980-7733-0d8df296756e, 4 MiB, 143771 MiB, 0 %
6, GPU-d9337494-6060-1440-039f-4654245b5764, 292 MiB, 143771 MiB, 0 %
7, GPU-28780eb0-24d5-1f13-59b8-ae5c23617a72, 58071 MiB, 143771 MiB, 0 %

gpu_uuid, pid, process_name, used_gpu_memory [MiB]
GPU-a44622e3-2296-6285-0a86-84d0f6c67900, 1328667, /usr/local/bin/python3.10, 130004 MiB
GPU-4f31bb83-f8dc-d0af-029e-bc4356e6afeb, 1328668, /usr/local/bin/python3.10, 130044 MiB
GPU-02dbe542-8a6f-11a4-7c2a-486b7d4e9092, 1328669, /usr/local/bin/python3.10, 129984 MiB
GPU-28780eb0-24d5-1f13-59b8-ae5c23617a72, 1758887, [Not Found], 58060 MiB

# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command
# Idx           #    C/G      %      %      %      %      %      %    name
    2          -     -      -      -      -      -      -      -    -
    4          -     -      -      -      -      -      -      -    -
    5          -     -      -      -      -      -      -      -    -
    6          -     -      -      -      -      -      -      -    -

USER         PID    PPID    PGID     SID     ELAPSED STAT CMD
