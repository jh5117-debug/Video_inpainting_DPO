# 实验重构需求文档

## 概述

重构 DiffuEraser 实验框架，解决旧实验系统的三个核心问题：
1. 报告中实验名称丢失 `Orign`/`Finetune` 前缀
2. OOM 导致不完整实验
3. 过多的 guidance scale 参数

## 变更记录

| 日期 | 变更内容 |
|------|---------|
| 2026-03-18 | 初始重构 |

## 变更详情

### 1. 权重管理
| 操作 | 路径 |
|------|------|
| 重命名 | `YTVS_Finetune_Diffueraser` → `YTVS_Finetune_Diffueraser_S2_26K` |
| 新增 (WandB 下载) | `YTVS_Finetune_Diffueraser_S2_8K` (step 8000 checkpoint) |

### 2. 实验配置 (24 experiments = 8 configs × 3 weights)

**权重**: Orign / FT_S2_26K / FT_S2_8K

**配置** (gs 统一为 0.0):
| Steps | Dataset | Blend | Dil |
|-------|---------|-------|-----|
| 2-Step | OR | No | 0 |
| 2-Step | BR | No | 0 |
| 4-Step | OR | No | 0 |
| 4-Step | BR | No | 0 |
| 2-Step | OR | Yes | 8 |
| 2-Step | BR | Yes | 8 |
| 4-Step | OR | Yes | 8 |
| 4-Step | BR | Yes | 8 |

### 3. OOM 健壮性
- 最多 2 轮自动重试 (`MAX_RETRIES=2`)
- 失败实验自动清理不完整目录
- 基于 `summary.json` 存在性判断实验完成度

### 4. 报告生成
- `parse_exp_name()` 正确识别权重前缀
- 新增 Weight Comparison 表格
- 自动扫描目录（不再硬编码）

## 修改文件
- `run_40exp.sh` — 实验主脚本
- `inference/generate_report.py` — 报告生成器
