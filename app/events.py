from __future__ import annotations

import numpy as np

from .models import LogEvent, ParsedLog, SignalNode


MAX_EVENTS_PER_SIGNAL = 200


def _normalize_token(text: str) -> str:
    return "".join(character for character in text.lower() if character.isalnum())


def _event_type_for_signal(signal: SignalNode) -> str | None:
    tokens = [signal.name, signal.full_path, *signal.path_parts]
    if any(_normalize_token(token) == "errcode" for token in tokens):
        return "报警/错误变化"
    return None


def _changed_indices(values: np.ndarray, event_type: str) -> np.ndarray:
    finite_pairs = np.isfinite(values[1:]) & np.isfinite(values[:-1])
    changed = np.flatnonzero(finite_pairs & (values[1:] != values[:-1])) + 1
    if event_type == "报警/错误变化":
        changed = changed[(values[changed] != 0) | (values[changed - 1] != 0)]
    return changed[:MAX_EVENTS_PER_SIGNAL]


def _looks_like_discrete_event_series(values: np.ndarray) -> bool:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return False

    integer_like = np.all(np.isclose(finite_values, np.round(finite_values), atol=1e-9))
    unique_count = np.unique(finite_values).size
    unique_ratio = unique_count / finite_values.size
    return integer_like and (unique_count <= 64 or unique_ratio <= 0.1)


def detect_events(parsed_log: ParsedLog, *, max_events: int = 1000) -> list[LogEvent]:
    events: list[LogEvent] = []

    for signal in parsed_log.signals:
        if not signal.available or signal.signal_id not in parsed_log.signals_by_id:
            continue

        event_type = _event_type_for_signal(signal)
        if event_type is None:
            continue

        values = parsed_log.get_series(signal.signal_id)
        if values.shape[0] < 2:
            continue
        if not _looks_like_discrete_event_series(values):
            continue

        for index in _changed_indices(values, event_type):
            sample_index = int(index)
            events.append(
                LogEvent(
                    sample_index=sample_index,
                    time_seconds=float(parsed_log.time_seconds[sample_index]),
                    time_raw=str(parsed_log.time_raw[sample_index]),
                    signal_id=signal.signal_id,
                    signal_name=signal.name,
                    signal_path=signal.full_path,
                    previous_value=float(values[sample_index - 1]),
                    current_value=float(values[sample_index]),
                    event_type=event_type,
                )
            )

    return sorted(events, key=lambda event: (event.sample_index, event.signal_path))[:max_events]
