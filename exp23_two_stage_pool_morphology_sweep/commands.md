# Exp23 Commands

Initial validation:

```bash
python -m py_compile exp23_two_stage_pool_morphology_sweep/code/*.py
python -m unittest discover -s exp23_two_stage_pool_morphology_sweep/tests -p 'test_*.py'
bash -n exp23_two_stage_pool_morphology_sweep/scripts/*.sh
```

Training launch is blocked until GPU4-7 are safely available.

