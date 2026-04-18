# 修复 Unknown Weight 显示 & 仅计算 BR Pixel Metric

## 需求背景

24 组实验报告中，FT_S2_34K 和 FT_S2_48K 权重显示为 "Unknown"。同时用户需要取消 VBench 计算，仅保留 BR 的 pixel metric (PSNR/SSIM/LPIPS/Ewarp/AS/IS)，所有实验的可视化视频仍需生成。

## 根因分析

1. `generate_report.py` L33 `WEIGHT_PREFIXES` 仅包含 `FT_S2_26K`, `FT_S2_8K`, `Finetune`, `Orign`，缺少 `FT_S2_34K`/`FT_S2_48K`
2. `generate_weight_comparison()` L314 硬编码只对比 3 种权重

## 修改内容

| 文件 | 修改 |
|------|------|
| `inference/generate_report.py` | WEIGHT_PREFIXES 增加 FT_S2_34K/48K（按最长前缀优先排序）；Weight Comparison 改为动态发现所有权重类型；Key Findings 新增 34K/48K 对比项 |
| `inference/compare_all.py` | 新增 `--no_vbench` CLI 参数，设置后跳过 VBench 初始化和评估 |
| `run_24exp.sh` | OR 实验不传 `--eval`（仅推理+可视化）；BR 实验传 `--eval --no_vbench`（仅计算 pixel metric）|

## 验证结果

用现有 24 组 summary.json 重新生成报告，FT_S2_34K/48K 正确显示，Weight Comparison 表动态包含 3 种权重对比。
