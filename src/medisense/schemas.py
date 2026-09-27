from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    key: str
    label: str
    unit: str
    minimum: float
    maximum: float
    step: float = 1
    help_text: str = ""


@dataclass(frozen=True)
class DiseaseSpec:
    key: str
    title: str
    dataset: str
    target_label: str
    features: tuple[Feature, ...]
    note: str
