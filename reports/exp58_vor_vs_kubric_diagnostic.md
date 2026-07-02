# Exp58 VOR-vs-Kubric Diagnostic

Status: `VOID_NATIVE_DATA_BLOCKED`

The VOR-vs-Kubric comparison could not be computed because official Kubric Gate8 generation is blocked before data creation.

The data-mismatch hypothesis remains plausible but untested:

- VOR-derived VOID attempts repeatedly damage overlap / affected / boundary regions.
- Official VOID data is generated paired counterfactual data with native mask semantics.
- The official Kubric path is currently blocked by environment, not by model weights or GCS asset access.

No comparison metrics should be inferred from missing Kubric data.
