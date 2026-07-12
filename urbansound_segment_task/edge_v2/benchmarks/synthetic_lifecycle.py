"""Deterministic standard-library loader and inference workload."""

from __future__ import annotations

from typing import Callable


def build_synthetic_lifecycle(
    load_size: int,
    work_size: int,
) -> tuple[Callable[[], list[float]], Callable[[list[float], list[float]], float], list[float]]:
    if isinstance(load_size, bool) or not isinstance(load_size, int) or load_size < 1:
        raise ValueError("load_size must be a positive integer")
    if isinstance(work_size, bool) or not isinstance(work_size, int) or work_size < 1:
        raise ValueError("work_size must be a positive integer")
    if load_size > 5_000_000 or work_size > 1_000_000:
        raise ValueError("synthetic workload size exceeds the safety limit")

    input_data = [((index * 37) % 257 - 128) / 128.0 for index in range(work_size)]

    def loader() -> list[float]:
        return [((index * 48_271 + 17) % 65_521) / 65_521.0 for index in range(load_size)]

    def inference(weights: list[float], values: list[float]) -> float:
        accumulator = 0.0
        weight_count = len(weights)
        for index, value in enumerate(values):
            accumulator += value * weights[(index * 131) % weight_count]
        return accumulator / max(1, len(values))

    return loader, inference, input_data
