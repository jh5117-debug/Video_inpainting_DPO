# 40-Experiment Runner 独立测试环境搭建

## 报错根因分析

**核心报错**: `ModuleNotFoundError: No module named 'propainter'`

**原因链**:
1. `run_40exp.sh` 中 `INFERENCE_DIR=/home/hj/Reg_DPO_Inpainting/inference`
2. `compare_all.py` 第48行 `REPO_ROOT = Path(__file__).resolve().parent.parent` → 解析为 `/home/hj/Reg_DPO_Inpainting/`
3. 第50行 `sys.path.insert(0, str(REPO_ROOT))` → 将 `/home/hj/Reg_DPO_Inpainting/` 加到 Python 路径
4. 第59行 `from propainter.inference import Propainter` → 在 `REPO_ROOT` 下找 `propainter/` 目录
5. **但** `/home/hj/Reg_DPO_Inpainting/.gitignore` 排除了 `propainter/` 目录 → 该目录从未被 git 跟踪 → 本地不存在
6. `/home/hj/Test/propainter` 是一个指向 `/home/hj/Reg_DPO_Inpainting/propainter` 的**断裂符号链接**

---

## 目标

将所有测试相关文件复制到 `/home/hj/Test`，使其成为**完全独立的测试环境**，然后重写 `run_40exp.sh`：
- 对 `nohup` 后台运行，日志写入文件
- 所有路径指向 `/home/hj/Test` 内部

---

## Proposed Changes

### 数据与代码复制

需要复制/创建到 `/home/hj/Test` 的目录结构：

```
/home/hj/Test/
├── propainter/          # 复制自 /home/hj/DiffuEraser_new/propainter/ (~1MB)
├── diffueraser/         # 复制自 /home/hj/Reg_DPO_Inpainting/diffueraser/ (已有符号链接,改为实际复制)
├── libs/                # 复制自 /home/hj/Reg_DPO_Inpainting/libs/ (已有符号链接,改为实际复制)
├── inference/           # 已存在, 含 compare_all.py, metrics.py 等
│   └── prompt_cache/    # 已存在, 含 caption YAML
├── weights/             # 符号链接 -> /home/hj/DiffuEraser_new/weights/ (95GB, 不复制)
├── diffuEraser_weights/ # 符号链接 -> /home/hj/Reg_DPO_Inpainting/weights/diffuEraser/ (17GB)
├── data/
│   ├── OR/              # 符号链接到 DAVIS Full-Resolution 数据 (3.2GB, 只读)
│   │   ├── JPEGImages/  -> /home/hj/DiffuEraser_new/DAVIS-2017-trainval-Full-Resolution/DAVIS/JPEGImages/Full-Resolution
│   │   └── Annotations/ -> /home/hj/DiffuEraser_new/DAVIS-2017-trainval-Full-Resolution/DAVIS/Annotations/Full-Resolution
│   └── BR/              # 符号链接到 davis 数据 (175MB, 只读)
│       ├── JPEGImages/  -> /home/hj/DiffuEraser_new/dataset/davis/JPEGImages_432_240/
│       └── test_masks/  -> /home/hj/DiffuEraser_new/dataset/davis/test_masks/
├── exp_result/          # 已存在, 实验输出
└── run_40exp.sh         # 重写: nohup + 所有路径指向本地
```

> [!IMPORTANT]
> 对于大体积数据（weights 95GB, DAVIS 3.2GB），使用**符号链接**而非复制，节省磁盘空间。
> 对于代码模块（propainter ~1MB, diffueraser ~571K, libs ~1.1MB），使用**实际复制**，保证独立性。

---

### 脚本修改

#### [MODIFY] [run_40exp.sh](file:///home/hj/Test/run_40exp.sh)

1. **路径重构** — 所有路径改为 `/home/hj/Test/` 内部相对路径
2. **nohup 运行** — 自身用 `nohup` 启动，所有实验输出写入日志文件，不再使用 `tee`
3. **compare_all.py 路径修正** — `INFERENCE_DIR` 改指 `/home/hj/Test/inference`
4. **VBench 路径** — `compare_all.py` 中硬编码了 `/home/hj/VBench`，保持不变（外部工具）

#### [MODIFY] [compare_all.py](file:///home/hj/Test/inference/compare_all.py)

修改第48-56行的 `REPO_ROOT`、`INFERENCE_DIR` 和 `VBENCH_ROOT` 路径，使其基于脚本所在目录正确解析模块路径。`compare_all.py` 的 `REPO_ROOT` 设为 `/home/hj/Test`（`__file__` 的 parent.parent）。

> [!NOTE]
> `compare_all.py` 已有 `REPO_ROOT = Path(__file__).resolve().parent.parent` 的动态解析逻辑。只要文件放在 `/home/hj/Test/inference/compare_all.py` 且 `/home/hj/Test/propainter/` 存在，路径会自动正确。无需修改 Python 代码。

---

## 具体执行步骤

### Step 1: 删除断裂符号链接，复制实际代码

```bash
rm -f /home/hj/Test/propainter /home/hj/Test/diffueraser /home/hj/Test/libs
cp -r /home/hj/DiffuEraser_new/propainter/ /home/hj/Test/propainter/
cp -r /home/hj/Reg_DPO_Inpainting/diffueraser/ /home/hj/Test/diffueraser/
cp -r /home/hj/Reg_DPO_Inpainting/libs/ /home/hj/Test/libs/
```

### Step 2: 创建数据和权重的符号链接

```bash
mkdir -p /home/hj/Test/data/OR /home/hj/Test/data/BR
ln -sfn /home/hj/DiffuEraser_new/DAVIS-2017-trainval-Full-Resolution/DAVIS/JPEGImages/Full-Resolution /home/hj/Test/data/OR/JPEGImages
ln -sfn /home/hj/DiffuEraser_new/DAVIS-2017-trainval-Full-Resolution/DAVIS/Annotations/Full-Resolution /home/hj/Test/data/OR/Annotations
ln -sfn /home/hj/DiffuEraser_new/dataset/davis/JPEGImages_432_240 /home/hj/Test/data/BR/JPEGImages
ln -sfn /home/hj/DiffuEraser_new/dataset/davis/test_masks /home/hj/Test/data/BR/test_masks
ln -sfn /home/hj/DiffuEraser_new/weights /home/hj/Test/weights
ln -sfn /home/hj/Reg_DPO_Inpainting/weights/diffuEraser /home/hj/Test/diffuEraser_weights
```

### Step 3: 重写 run_40exp.sh

所有路径改为 `/home/hj/Test` 内部，使用 `nohup` 并输出到日志文件。

### Step 4: 添加 nohup 启动脚本

创建 `start_40exp.sh` 用 `nohup` 启动 `run_40exp.sh`。

---

## Verification Plan

### Automated Tests

1. **路径验证脚本** — 运行 Python 验证所有路径是否存在：
   ```bash
   python -c "
   import os, sys
   sys.path.insert(0, '/home/hj/Test')
   sys.path.insert(0, '/home/hj/Test/inference')
   from propainter.inference import Propainter
   from diffueraser.diffueraser import DiffuEraser
   print('All imports OK')
   "
   ```

2. **Dry-run** — 使用 `--max_videos 1` 运行单个实验验证全链路可达。
