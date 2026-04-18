# Stage 2 权重分析 + Orign vs Finetune 40 实验对比需求

## 背景
Stage 2 训练使用 Stage 1 的 step 18000 权重作为基础，完成了 26000 步训练。需要分析 Stage 2 日志选出最优权重，并搭建 40 实验对比环境（20 配置 × 2 权重集）。

## 需求清单

### 1. 权重目录整理
- 原始 DiffuEraser 权重 → `weights/diffuEraser/Orign_Diffueraser/`
- Finetune 权重（step 26000）→ `weights/diffuEraser/YTVS_Finetune_Diffueraser/`

### 2. 40 实验对比
- 20 配置 × 2 权重 = 40 实验
- GPU 池：0,1,2,3,5,6（GPU 4 损坏跳过）
- 输出到 `/home/hj/Test/exp_result/`
- 不修改 `Reg_DPO_Inpainting/` 内容

### 3. 关键约束
- GPU 4 不可用
- 测试内容全部在 `/home/hj/Test/` 下
- 不破坏 GitHub push 目录
