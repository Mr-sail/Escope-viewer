from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.parser import ParseError, parse_log_file


ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILE = ROOT / "data_1min_4ms_20260228132326.txt"


class ParseLogFileTests(unittest.TestCase):
    def test_parse_sample_file(self) -> None:
        parsed = parse_log_file(SAMPLE_FILE)

        self.assertGreater(parsed.meta.sample_count, 0)
        self.assertEqual(parsed.meta.field_count, 217)
        self.assertEqual(parsed.time_raw.shape[0], parsed.meta.sample_count)
        self.assertEqual(parsed.time_seconds.shape[0], parsed.meta.sample_count)
        self.assertIn("001001001001", parsed.signals_by_id)
        self.assertAlmostEqual(float(parsed.time_seconds[0]), 0.0, places=6)

    def test_sample_contains_expected_paths(self) -> None:
        parsed = parse_log_file(SAMPLE_FILE)
        path_by_id = {signal.signal_id: signal.full_path for signal in parsed.signals}

        self.assertEqual(path_by_id["001001001001"], "ER / Joint / Pos / J1")
        self.assertEqual(path_by_id["001002001001"], "ER / Tcp / Pos / X")
        self.assertEqual(path_by_id["001008001001"], "ER / **Status / McSts / ErrCode")

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
