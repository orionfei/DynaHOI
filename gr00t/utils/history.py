from dataclasses import dataclass
from numbers import Integral
from typing import Sequence


@dataclass(frozen=True)
class ObserveFrameHistoryConfig:
    offsets: tuple[int, ...]
    frame_count: int
    start_index: int


def resolve_observe_frame_history(
    window_length: int,
    observe_frame_offsets: Sequence[int] | None,
) -> ObserveFrameHistoryConfig:
    if window_length <= 0:
        raise ValueError(f"window_length must be positive, got {window_length}.")

    if observe_frame_offsets is None:
        offsets = tuple(range(window_length, 0, -1))
    else:
        offsets = tuple(observe_frame_offsets)
        if len(offsets) != window_length:
            raise ValueError(
                "observe_frame_offsets must have the same length as window_length, "
                f"got len(observe_frame_offsets)={len(offsets)} and window_length={window_length}."
            )
        if len(offsets) == 0:
            raise ValueError("observe_frame_offsets must not be empty.")
        if any(not isinstance(offset, Integral) for offset in offsets):
            raise ValueError(
                "observe_frame_offsets must contain only integers, "
                f"got {list(offsets)}."
            )
        if any(offset <= 0 for offset in offsets):
            raise ValueError(
                "observe_frame_offsets must contain only positive integers, "
                f"got {list(offsets)}."
            )
        if len(set(offsets)) != len(offsets):
            raise ValueError(f"observe_frame_offsets must not contain duplicates, got {list(offsets)}.")
        if any(offsets[i] <= offsets[i + 1] for i in range(len(offsets) - 1)):
            raise ValueError(
                "observe_frame_offsets must be strictly decreasing so frames stay ordered "
                f"from earliest to latest, got {list(offsets)}."
            )

    return ObserveFrameHistoryConfig(
        offsets=offsets,
        frame_count=len(offsets),
        start_index=max(offsets),
    )


def format_observe_frame_history_tag(
    window_length: int,
    observe_frame_offsets: Sequence[int] | None,
) -> str:
    history = resolve_observe_frame_history(window_length, observe_frame_offsets)
    if observe_frame_offsets is None:
        return f"window_{history.frame_count}"
    offsets_tag = "-".join(str(offset) for offset in history.offsets)
    return f"offsets_{offsets_tag}"
