# Exp58B Final Decision

Final status: `VOID_DATA_MISMATCH_TEST_READY`

## Answers

1. TensorFlow / Kubric import fixed: yes. The isolated env imports TensorFlow 2.15.1, TFDS 4.2.0, official Google Research Kubric source, PyBullet, image stack, and OpenEXR.
2. Blender / `bpy` fixed: yes. Official Blender 3.6.23 runs headless with `bpy` and can bridge the isolated env.
3. Official VOID Kubric code can generate paired data: yes. Gate1 and Gate8 both rendered through the unmodified official `kubric_variable_objects.py` script via an Exp58-only launcher.
4. Gate1 passed: yes, `VOID_NATIVE_KUBRIC_GATE1_READY`.
5. Gate8 passed: yes, `VOID_NATIVE_KUBRIC_GATE8_READY`.
6. Data-mismatch hypothesis is now testable: yes. `manifests/exp58b_void_native_kubric_gate8.jsonl` is ready for the next official VOID inference diagnostic.
7. Next experiment: run official VOID inference on Kubric Gate8, then compare native Kubric behavior against VOR-derived Gate8. Do not run one-step until native inference evidence is reviewed.
8. VOID became third adapter evidence: no.
9. Continue with official VOID inference on Kubric Gate8 next: yes, as a diagnostic only.

## Scientific Meaning

Exp58B removes the environment/native-data blocker. It does not prove that VOID adapts successfully. It only establishes that official Kubric counterfactual data can now be generated on PAI in an isolated setup.

VOID remains:

- VOR-OR inference baseline
- same-model loser generator candidate
- adapter-engineering candidate
- not third-backbone adapter evidence

## Safety

- VOID inference run: no
- Training run: no
- Preference forward / zero-gap / one-step: no
- 10-step: no
- VOR-Eval: no
- Hard comp: no
- VOID official repo source modified: no
- `inference/metrics.py` modified: no
- Shared trainer modified: no
- Base/system env modified: no
- Universal adapter written: no
- Final SOTA written: no
- Third-backbone claim: no
