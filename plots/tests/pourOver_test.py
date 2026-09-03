"""Unit tests for pourOver.py pure-logic helpers.

No coffea file, no Flask server, no network — stdlib only.
Run with:
    python -m pytest coffea4bees/plots/tests/pourOver_test.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure the repo root is on sys.path so the import works both locally and in CI.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# pourOver.py imports Flask, matplotlib, and coffea at module level.
# Those are not available in the analysis CI container (pourOver runs in its own venv).
# Stub them all out before importing so the pure-logic helpers can be tested without
# any of the web or plotting infrastructure.
from unittest.mock import MagicMock
for _mod in [
    'flask',
    'matplotlib', 'matplotlib.pyplot',
    'coffea4bees.plots.plots',
    'src.plotting.plots',
    'src.plotting.iPlot_config',
]:
    sys.modules.setdefault(_mod, MagicMock())

# plot_config() is called at module level; make it return a MagicMock so
# the global `cfg` object exists but is otherwise inert.
sys.modules['src.plotting.iPlot_config'].plot_config = MagicMock(return_value=MagicMock())

from coffea4bees.plots.pourOver import (
    _parse_cli_cmd,
    _register_archive,
    _resolve_label,
    _validate_label,
    _write_manifest,
    _get_default_output_dir,
)


# ---------------------------------------------------------------------------
# _validate_label
# ---------------------------------------------------------------------------
class TestValidateLabel(unittest.TestCase):

    def test_single_word(self):
        self.assertEqual(_validate_label("ul18"), "ul18")

    def test_hyphenated(self):
        self.assertEqual(_validate_label("ul18-sb"), "ul18-sb")

    def test_uppercase_allowed(self):
        self.assertEqual(_validate_label("My-Run3-Test"), "My-Run3-Test")

    def test_digits_only_segment(self):
        self.assertEqual(_validate_label("run3"), "run3")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            _validate_label("")

    def test_space_raises(self):
        with self.assertRaises(ValueError):
            _validate_label("ul18 sb")

    def test_underscore_raises(self):
        with self.assertRaises(ValueError):
            _validate_label("ul18_sb")

    def test_leading_dash_raises(self):
        with self.assertRaises(ValueError):
            _validate_label("-ul18")

    def test_trailing_dash_raises(self):
        with self.assertRaises(ValueError):
            _validate_label("ul18-")

    def test_double_dash_raises(self):
        with self.assertRaises(ValueError):
            _validate_label("ul18--sb")


# ---------------------------------------------------------------------------
# _write_manifest
# ---------------------------------------------------------------------------
class TestWriteManifest(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.archive_dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _read_manifest(self):
        return json.loads((self.archive_dir / "manifest.json").read_text())

    def test_manifest_created(self):
        _write_manifest(self.archive_dir, ["inputs/a.coffea"], "inputs/meta.yml", "test-run")
        self.assertTrue((self.archive_dir / "manifest.json").exists())

    def test_manifest_keys(self):
        _write_manifest(self.archive_dir, ["inputs/a.coffea"], "inputs/meta.yml", "test-run")
        m = self._read_manifest()
        for key in ("created", "label", "inputs", "metadata"):
            self.assertIn(key, m)

    def test_label_stored(self):
        _write_manifest(self.archive_dir, ["inputs/a.coffea"], "inputs/meta.yml", "ul18-sr")
        self.assertEqual(self._read_manifest()["label"], "ul18-sr")

    def test_inputs_relative(self):
        _write_manifest(self.archive_dir, ["inputs/a.coffea", "inputs/b.coffea"],
                        "inputs/meta.yml", "test")
        m = self._read_manifest()
        self.assertEqual(m["inputs"], ["inputs/a.coffea", "inputs/b.coffea"])

    def test_metadata_relative(self):
        _write_manifest(self.archive_dir, ["inputs/a.coffea"], "inputs/meta.yml", "test")
        self.assertEqual(self._read_manifest()["metadata"], "inputs/meta.yml")

    def test_metadata_none(self):
        _write_manifest(self.archive_dir, ["inputs/a.coffea"], None, "test")
        self.assertIsNone(self._read_manifest()["metadata"])

    def test_returns_manifest_dict(self):
        result = _write_manifest(self.archive_dir, ["inputs/a.coffea"], "inputs/meta.yml", "x")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["label"], "x")


# ---------------------------------------------------------------------------
# _register_archive
# ---------------------------------------------------------------------------
class TestRegisterArchive(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self._tmpdir.name) / "registry.json"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _read(self):
        return json.loads(self.registry.read_text())

    def test_creates_registry_if_missing(self):
        _register_archive(self.registry, "archive_001", "ul18")
        self.assertTrue(self.registry.exists())

    def test_first_entry(self):
        _register_archive(self.registry, "archive_001", "ul18")
        entries = self._read()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["label"], "ul18")
        self.assertEqual(entries[0]["archive_dir"], "archive_001")

    def test_second_call_appends(self):
        _register_archive(self.registry, "archive_001", "ul18")
        _register_archive(self.registry, "archive_002", "ul17")
        entries = self._read()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[1]["label"], "ul17")

    def test_entry_has_created_timestamp(self):
        _register_archive(self.registry, "archive_001", "ul18")
        self.assertIn("created", self._read()[0])

    def test_no_tmp_file_left(self):
        _register_archive(self.registry, "archive_001", "ul18")
        tmp = self.registry.with_suffix(".tmp")
        self.assertFalse(tmp.exists(), "Atomic write left a .tmp file behind")


# ---------------------------------------------------------------------------
# _resolve_label
# ---------------------------------------------------------------------------
class TestResolveLabel(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self._tmpdir.name) / "registry.json"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _write_registry(self, entries):
        self.registry.write_text(json.dumps(entries))

    def test_found(self):
        self._write_registry([{"label": "ul18", "archive_dir": "arc_001", "created": "2026-01-01"}])
        self.assertEqual(_resolve_label("ul18", self.registry), "arc_001")

    def test_not_found_raises_key_error(self):
        self._write_registry([{"label": "ul18", "archive_dir": "arc_001", "created": "2026-01-01"}])
        with self.assertRaises(KeyError):
            _resolve_label("ul17", self.registry)

    def test_missing_registry_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            _resolve_label("ul18", self.registry)

    def test_duplicate_returns_most_recent(self):
        self._write_registry([
            {"label": "ul18", "archive_dir": "arc_001", "created": "2026-01-01"},
            {"label": "ul18", "archive_dir": "arc_002", "created": "2026-01-02"},
        ])
        self.assertEqual(_resolve_label("ul18", self.registry), "arc_002")

    def test_duplicate_prints_warning(self, ):
        self._write_registry([
            {"label": "ul18", "archive_dir": "arc_001", "created": "2026-01-01"},
            {"label": "ul18", "archive_dir": "arc_002", "created": "2026-01-02"},
        ])
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            _resolve_label("ul18", self.registry)
        self.assertIn("Warning", buf.getvalue())

    def test_multiple_labels_returns_correct_one(self):
        self._write_registry([
            {"label": "ul18", "archive_dir": "arc_001", "created": "2026-01-01"},
            {"label": "ul17", "archive_dir": "arc_002", "created": "2026-01-01"},
        ])
        self.assertEqual(_resolve_label("ul17", self.registry), "arc_002")


# ---------------------------------------------------------------------------
# _get_default_output_dir
# ---------------------------------------------------------------------------
class TestGetDefaultOutputDir(unittest.TestCase):

    def test_empty_inputs(self):
        self.assertEqual(_get_default_output_dir([], None), "pourover_output")

    def test_single_input(self):
        res = _get_default_output_dir(["foo/bar/analysis.coffea"], None)
        self.assertTrue(res.startswith("pourover_output/session_analysis_"))
        self.assertEqual(len(res.split("_")[-1]), 8)

    def test_multiple_inputs(self):
        res = _get_default_output_dir(["foo/bar/analysis.coffea", "baz/qux.coffea"], None)
        self.assertTrue(res.startswith("pourover_output/session_analysis_etc_"))
        self.assertEqual(len(res.split("_")[-1]), 8)

    def test_same_inputs_same_output(self):
        res1 = _get_default_output_dir(["foo/bar/analysis.coffea"], "config.yml")
        res2 = _get_default_output_dir(["foo/bar/analysis.coffea"], "config.yml")
        self.assertEqual(res1, res2)

    def test_different_inputs_different_output(self):
        res1 = _get_default_output_dir(["foo/bar/analysis.coffea"], "config1.yml")
        res2 = _get_default_output_dir(["foo/bar/analysis.coffea"], "config2.yml")
        self.assertNotEqual(res1, res2)


# ---------------------------------------------------------------------------
# _parse_cli_cmd
# ---------------------------------------------------------------------------
class TestParseCliCmd(unittest.TestCase):

    # ---- valid plot() ----
    def test_plot_simple(self):
        r = _parse_cli_cmd('plot("selJets.pt")')
        self.assertEqual(r["func"], "plot")
        self.assertEqual(r["args"], ["selJets.pt"])
        self.assertEqual(r["kwargs"], {})

    def test_plot_with_kwargs(self):
        r = _parse_cli_cmd('plot("selJets.pt", region="SR", rebin=2)')
        self.assertEqual(r["kwargs"]["region"], "SR")
        self.assertEqual(r["kwargs"]["rebin"], 2)

    def test_plot_region_list(self):
        r = _parse_cli_cmd('plot("selJets.pt", region=["SR","SB"])')
        self.assertEqual(r["kwargs"]["region"], ["SR", "SB"])

    def test_plot_blind_bool(self):
        r = _parse_cli_cmd('plot("SvB_MA.ps", region="sum", yscale="log", blind=True)')
        self.assertTrue(r["kwargs"]["blind"])
        self.assertTrue(r["req"]["blind"])

    def test_plot_blind_float(self):
        r = _parse_cli_cmd('plot("SvB_MA.ps", region="sum", yscale="log", blind=0.8)')
        self.assertEqual(r["kwargs"]["blind"], 0.8)
        self.assertEqual(r["req"]["blind"], 0.8)

    def test_plot_req_var(self):
        r = _parse_cli_cmd('plot("selJets.pt")')
        self.assertEqual(r["req"]["var"], "selJets.pt")

    def test_plot2d(self):
        r = _parse_cli_cmd('plot2d("selJets.pt", "ttbar")')
        self.assertEqual(r["func"], "plot2d")
        self.assertEqual(r["req"]["var"], "selJets.pt")
        self.assertEqual(r["req"]["process"], "ttbar")

    # ---- valid text commands ----
    def test_ls(self):
        r = _parse_cli_cmd("ls()")
        self.assertEqual(r["func"], "ls")

    def test_ls_with_filter(self):
        r = _parse_cli_cmd('ls("MvD")')
        self.assertEqual(r["args"], ["MvD"])

    def test_info(self):
        r = _parse_cli_cmd("info()")
        self.assertEqual(r["func"], "info")

    def test_examples(self):
        r = _parse_cli_cmd("examples()")
        self.assertEqual(r["func"], "examples")

    # ---- valid plot_roc() ----
    def test_plot_roc_simple(self):
        r = _parse_cli_cmd('plot_roc("SvB_MA.ps_hh_fine", sig=["HH4b"], bkg=["TTbar","Multijet"])')
        self.assertEqual(r["func"], "plot_roc")
        self.assertEqual(r["req"]["var"], "SvB_MA.ps_hh_fine")
        self.assertEqual(r["req"]["sig"], ["HH4b"])
        self.assertEqual(r["req"]["bkg"], ["TTbar", "Multijet"])

    def test_plot_roc_with_region(self):
        r = _parse_cli_cmd('plot_roc("SvB_MA.ps_hh_fine", sig=["HH4b"], bkg=["TTbar"], region="SR")')
        self.assertEqual(r["req"]["region"], "SR")

    def test_plot_roc_with_cut(self):
        r = _parse_cli_cmd('plot_roc("SvB_MA.ps_hh_fine", sig=["HH4b"], bkg=["TTbar"], cut="passSvB")')
        self.assertEqual(r["req"]["cut"], "passSvB")

    def test_plot_roc_req_is2d_false(self):
        r = _parse_cli_cmd('plot_roc("SvB_MA.ps_hh_fine", sig=["HH4b"], bkg=["TTbar"])')
        self.assertFalse(r["req"].get("is2d", False))

    # ---- error cases ----
    def test_unknown_function_raises(self):
        with self.assertRaises(ValueError):
            _parse_cli_cmd("foo()")

    def test_no_quotes_raises(self):
        with self.assertRaises(ValueError):
            _parse_cli_cmd("plot(selJets.pt)")

    def test_syntax_error_raises(self):
        with self.assertRaises(ValueError):
            _parse_cli_cmd("plot(")

    def test_not_a_call_raises(self):
        with self.assertRaises(ValueError):
            _parse_cli_cmd('"just a string"')

    def test_bad_kwarg_value_raises(self):
        with self.assertRaises(ValueError):
            _parse_cli_cmd('plot("var", region=some_var)')


if __name__ == "__main__":
    unittest.main()
