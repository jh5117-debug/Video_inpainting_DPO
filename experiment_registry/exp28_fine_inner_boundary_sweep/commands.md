# Exp28 Commands

```bash
python -m py_compile exp28_fine_inner_boundary_sweep/code/inner_boundary_geometry.py exp28_fine_inner_boundary_sweep/code/exp28_trial_runner.py exp28_fine_inner_boundary_sweep/code/train_exp28_stage1.py exp28_fine_inner_boundary_sweep/code/train_exp28_stage2.py exp28_fine_inner_boundary_sweep/code/summarize_exp28_pair_eval.py
python -m unittest discover -s exp28_fine_inner_boundary_sweep/tests -p 'test_*.py'
bash -n exp28_fine_inner_boundary_sweep/scripts/eval_exp28_pair_davis50_pai.sh
```

Pair launch examples:

```bash
CUDA_VISIBLE_DEVICES=3,4 python exp28_fine_inner_boundary_sweep/code/exp28_trial_runner.py --pair A --gpus 3,4 --nproc-per-node 2
CUDA_VISIBLE_DEVICES=1,2 python exp28_fine_inner_boundary_sweep/code/exp28_trial_runner.py --pair B --gpus 1,2 --nproc-per-node 2
CUDA_VISIBLE_DEVICES=3,4 python exp28_fine_inner_boundary_sweep/code/exp28_trial_runner.py --pair C --gpus 3,4 --nproc-per-node 2
```
