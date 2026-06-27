"""Static status helpers for Exp30 reports."""

EXP30_BRANCH = "research/exp30-vor-or-multimodel-minimax-adapter-20260627"

FORBIDDEN_CLAIMS = (
    "UNIVERSAL_ADAPTER",
    "ALL_MODELS_SUPPORTED",
    "FINAL_SOTA",
    "TOP_CONFERENCE_NOVELTY_CONFIRMED",
)


def current_scope() -> str:
    return (
        "VOR-OR multi-model medium-hard pool, MiniMax quality-positive micro "
        "gate, DiffuEraser VOR-OR micro validation, and paper evidence plan"
    )

