from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.parser import ParseError, parse_log_file


FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_FILE = FIXTURES / "sample_log.txt"


class ParseLogFileTests(unittest.TestCase):
    def test_parse_sample_file(self) -> None:
        parsed = parse_log_file(SAMPLE_FILE)

        self.assertEqual(parsed.meta.sample_count, 3)
        self.assertEqual(parsed.meta.field_count, 4)
        self.assertEqual(parsed.time_raw.shape[0], parsed.meta.sample_count)
        self.assertEqual(parsed.time_seconds.shape[0], parsed.meta.sample_count)
        self.assertIn("001001001001", parsed.signals_by_id)
        self.assertIn("UNLISTED001", parsed.signals_by_id)
        self.assertAlmostEqual(float(parsed.time_seconds[0]), 0.0, places=6)
        self.assertEqual(parsed.meta.skipped_rows, 1)

    def test_sample_contains_expected_paths(self) -> None:
        parsed = parse_log_file(SAMPLE_FILE)
        path_by_id = {signal.signal_id: signal.full_path for signal in parsed.signals}

        self.assertEqual(path_by_id["001001001001"], "ER / Joint / Pos / J1")
        self.assertEqual(path_by_id["001002001001"], "ER / Tcp / Pos / X")
        self.assertEqual(path_by_id["001008001001"], "ER / **Status / McSts / ErrCode")
        self.assertEqual(path_by_id["UNLISTED001"], "Unknown / UNLISTED001")

    def test_empty_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.txt"
            path.write_text("", encoding="utf-8")

            with self.assertRaises(ParseError):
                parse_log_file(path)

    def test_missing_id_row_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing_id.txt"
            path.write_text(
                "<?xml version='1.0'?>\n"
                "<module><group name='ER'><dev name='Joint'>"
                "<puts name='Pos'><put id='001'>J1</put></puts>"
                "</dev></group></module>\n"
                "********************\n",
                encoding="utf-8",
            )

            with self.assertRaises(ParseError):
                parse_log_file(path)

    def test_invalid_xml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad_xml.txt"
            path.write_text(
                "<?xml version='1.0'?>\n"
                "<module><group>\n"
                "********************\n"
                "ID\t001\n"
                "20260301131750100\t1.0\n",
                encoding="utf-8",
            )

            with self.assertRaises(ParseError):
                parse_log_file(path)


if __name__ == "__main__":
    unittest.main()
