"""Stable, independent random-seed schedules for paired experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

from configs import DEFAULT_RANDOM_SEED


@dataclass(frozen=True)
class RunCoordinate:
    """Visible experimental factors for one paired run."""

    layout_index: int
    replicate_index: int

    @property
    def run_id(self) -> str:
        return f"layout_{self.layout_index:04d}_replicate_{self.replicate_index:02d}"


@dataclass(frozen=True)
class SeedSchedule:
    """Independent seeds used by one Online-Delta and retraining pair."""

    benchmark: str
    layout_index: int
    replicate_index: int
    layout_seed: int
    data_split_seed: int
    online_weight_seed: int
    online_batch_seed: int
    sa_proposal_seed: int
    sa_acceptance_seed: int
    retraining_weight_seed: int
    retraining_batch_seed: int

    @property
    def run_id(self) -> str:
        return RunCoordinate(self.layout_index, self.replicate_index).run_id

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def build_seed_schedule(
    benchmark: str,
    layout_index: int,
    replicate_index: int,
    *,
    master_seed: int = DEFAULT_RANDOM_SEED,
) -> SeedSchedule:
    """Derive a stable schedule without relying on Python's randomized hash()."""

    if layout_index < 0:
        raise ValueError("layout_index muss mindestens 0 sein.")
    if replicate_index < 0:
        raise ValueError("replicate_index muss mindestens 0 sein.")

    return SeedSchedule(
        benchmark=benchmark,
        layout_index=layout_index,
        replicate_index=replicate_index,
        layout_seed=_stable_seed(master_seed, benchmark, "layout", layout_index),
        data_split_seed=_stable_seed(master_seed, benchmark, "data_split", layout_index),
        online_weight_seed=_stable_seed(
            master_seed,
            benchmark,
            "online_weight",
            layout_index,
            replicate_index,
        ),
        online_batch_seed=_stable_seed(
            master_seed,
            benchmark,
            "online_batch",
            layout_index,
            replicate_index,
        ),
        sa_proposal_seed=_stable_seed(
            master_seed,
            benchmark,
            "sa_proposal",
            layout_index,
            replicate_index,
        ),
        sa_acceptance_seed=_stable_seed(
            master_seed,
            benchmark,
            "sa_acceptance",
            layout_index,
            replicate_index,
        ),
        retraining_weight_seed=_stable_seed(
            master_seed,
            benchmark,
            "retraining_weight",
            layout_index,
            replicate_index,
        ),
        retraining_batch_seed=_stable_seed(
            master_seed,
            benchmark,
            "retraining_batch",
            layout_index,
            replicate_index,
        ),
    )


def build_run_coordinates(
    layout_count: int,
    replicate_count: int,
) -> tuple[RunCoordinate, ...]:
    if layout_count <= 0:
        raise ValueError("layout_count muss positiv sein.")
    if replicate_count <= 0:
        raise ValueError("replicate_count muss positiv sein.")
    return tuple(
        RunCoordinate(layout_index, replicate_index)
        for layout_index in range(layout_count)
        for replicate_index in range(replicate_count)
    )


def _stable_seed(master_seed: int, benchmark: str, stream: str, *indices: int) -> int:
    material = ":".join(
        [str(int(master_seed)), benchmark, stream, *(str(int(index)) for index in indices)]
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)
