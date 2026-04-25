from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from app.events import detect_events
from app.models import LogFileMeta, ParsedLog, SignalNode
from app.parser import parse_log_file

SAMPLE_FILE = Path(__file__).resolve().parent / "fixtures" / "sample_log.txt"


class EventDetectionTests(unittest.TestCase):
    def test_detects_status_code_changes_from_available_fields(self) -> None:
        parsed = parse_log_file(SAMPLE_FILE)
        events = detect_events(parsed)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].signal_id, "001008001001")
        self.assertEqual(events[0].signal_name, "ErrCode")
        self.assertEqual(events[0].previous_value, 0.0)
        self.assertEqual(events[0].current_value, 1.0)

    def test_parse_progress_callback_is_called(self) -> None:
        calls: list[tuple[int, str]] = []

        parse_log_file(SAMPLE_FILE, progress_callback=lambda percent, message: calls.append((percent, message)) or True)

        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(calls[0][0], 0)
        self.assertGreaterEqual(calls[-1][0], 98)

    def test_only_errcode_fields_are_detected(self) -> None:
        parsed = ParsedLog(
            meta=LogFileMeta(
                path=SAMPLE_FILE,
                sample_count=4,
                field_count=2,
                start_time_raw="20260301131750100",
                end_time_raw="20260301131750112",
            ),
            time_raw=np.array(
                ["20260301131750100", "20260301131750104", "20260301131750108", "20260301131750112"],
                dtype=str,
            ),
            time_seconds=np.array([0.0, 0.004, 0.008, 0.012]),
            signals_by_id={
                "follow_err": np.array([0.1, 0.2, 0.3, 0.4]),
                "err_code": np.array([0.0, 0.0, 7.0, 7.0]),
            },
            signals=[
                SignalNode(
                    signal_id="follow_err",
                    name="J1",
                    path_parts=("ER", "More", "FollowErrS", "J1"),
                    full_path="ER / More / FollowErrS / J1",
                    column_index=0,
                    available=True,
                ),
                SignalNode(
                    signal_id="err_code",
                    name="ErrCode",
                    path_parts=("ER", "**Status", "McSts", "ErrCode"),
                    full_path="ER / **Status / McSts / ErrCode",
                    column_index=1,
                    available=True,
                ),
            ],
        )

        events = detect_events(parsed)

        self.assertEqual([event.signal_id for event in events], ["err_code"])

    def test_ignores_port_stswd_fields(self) -> None:
        parsed = ParsedLog(
            meta=LogFileMeta(
                path=SAMPLE_FILE,
                sample_count=3,
                field_count=1,
                start_time_raw="20260301131750100",
                end_time_raw="20260301131750108",
            ),
            time_raw=np.array(["20260301131750100", "20260301131750104", "20260301131750108"], dtype=str),
            time_seconds=np.array([0.0, 0.004, 0.008]),
            signals_by_id={"port_stswd": np.array([0.0, 1.0, 3.0])},
            signals=[
                SignalNode(
                    signal_id="port_stswd",
                    name="Stswd",
                    path_parts=("ER", "Port", "Stswd"),
                    full_path="ER / Port / Stswd",
                    column_index=0,
                    available=True,
                )
            ],
        )

        self.assertEqual(detect_events(parsed), [])


if __name__ == "__main__":
    unittest.main()
