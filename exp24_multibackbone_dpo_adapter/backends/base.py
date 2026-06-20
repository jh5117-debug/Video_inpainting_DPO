"""Unified preference-backend interface for Exp24.

Backends must preserve each model's native prediction target.  These base
methods intentionally raise until a model-specific implementation proves real
inference and DPO plumbing smoke.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class BackendStatus:
    model: str
    inference_status: str
    dpo_smoke_status: str
    blocker: str = ""


class PreferenceBackend(ABC):
    model_name: str
    native_prediction_target: str

    @abstractmethod
    def load_policy(self, *args: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    def load_reference(self, *args: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    def freeze_reference(self) -> None: ...

    @abstractmethod
    def select_trainable_adapter(self) -> Iterable[Any]: ...

    @abstractmethod
    def prepare_pair_batch(self, batch: Any) -> Any: ...

    @abstractmethod
    def encode_video(self, video: Any) -> Any: ...

    @abstractmethod
    def encode_condition(self, condition: Any) -> Any: ...

    @abstractmethod
    def sample_shared_noise_and_timestep(self, batch: Any) -> Any: ...

    @abstractmethod
    def build_noisy_input(self, encoded: Any, noise_timestep: Any) -> Any: ...

    @abstractmethod
    def native_prediction_target_tensor(self, batch: Any, noise_timestep: Any) -> Any: ...

    @abstractmethod
    def policy_forward(self, noisy_input: Any, condition: Any) -> Any: ...

    @abstractmethod
    def reference_forward(self, noisy_input: Any, condition: Any) -> Any: ...

    @abstractmethod
    def native_error_map(self, prediction: Any, target: Any) -> Any: ...

    @abstractmethod
    def build_region_map(self, mask: Any, region_config: Any) -> Any: ...

    @abstractmethod
    def compute_dpo_loss(self, batch: Any, region_config: Any) -> Any: ...

    @abstractmethod
    def trainable_parameters(self) -> Iterable[Any]: ...

    @abstractmethod
    def save_adapter(self, path: str) -> None: ...

    @abstractmethod
    def load_adapter(self, path: str) -> None: ...

    @abstractmethod
    def infer_base(self, sample: Any) -> Any: ...

    @abstractmethod
    def infer_adapter(self, sample: Any) -> Any: ...

    @abstractmethod
    def checkpoint_identity(self) -> dict[str, Any]: ...

