#!/usr/bin/env python3
"""Unified test runner for coffea4bees unit tests.

Executes all unit test suites with verbose per-test output (verbosity=2)
and provides a clear pass/fail summary and exit code for CI pipelines.
"""
import os
import subprocess
import sys
import time
import unittest

# Ensure repo roots are on python path
sys.path.insert(0, os.getcwd())


def run_pytest_suite(title, test_path):
    """Run a pytest test file in a subprocess so module mocks stay isolated."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
    start_time = time.perf_counter()
    res = subprocess.run([sys.executable, "-m", "pytest", test_path, "-v"], env=env)
    elapsed = time.perf_counter() - start_time
    return (res.returncode == 0), elapsed


def run_suite(title, suite_or_module, specific_case=None):
    """Run a test suite with visual header and verbose output."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n", flush=True)

    loader = unittest.TestLoader()
    if specific_case:
        suite = unittest.TestSuite()
        suite.addTest(specific_case)
    elif isinstance(suite_or_module, unittest.TestSuite):
        suite = suite_or_module
    else:
        suite = loader.loadTestsFromModule(suite_or_module)

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.perf_counter()
    result = runner.run(suite)
    elapsed = time.perf_counter() - start_time

    return result, elapsed


def main():
    total_start = time.perf_counter()
    all_results = []
    overall_success = True

    # 1. Analysis Helpers
    try:
        from coffea4bees.analysis.tests import topCand_test, quadjet_test, quadjet_ttHbb_test
        loader = unittest.TestLoader()
        analysis_suite = unittest.TestSuite()
        analysis_suite.addTests(loader.loadTestsFromModule(topCand_test))
        analysis_suite.addTests(loader.loadTestsFromModule(quadjet_test))
        analysis_suite.addTests(loader.loadTestsFromModule(quadjet_ttHbb_test))
        res, elapsed = run_suite("1/10: Analysis Helpers (Top Candidate & QuadJet)", analysis_suite)
        all_results.append(("Analysis: TopCand & QuadJet", res, elapsed))
        if not res.wasSuccessful():
            overall_success = False
    except Exception as e:
        print(f"ERROR loading Analysis Helpers: {e}", file=sys.stderr)
        overall_success = False

    # 2. Trigger Emulator & Weights
    try:
        from coffea4bees.analysis.tests import test_trigger_emulator, test_vectorized_trigger_sf
        loader = unittest.TestLoader()
        trig_suite = unittest.TestSuite()
        trig_suite.addTests(loader.loadTestsFromModule(test_trigger_emulator))
        trig_suite.addTests(loader.loadTestsFromModule(test_vectorized_trigger_sf))
        res, elapsed = run_suite("2/10: Trigger Emulator & Vectorized Scale Factors", trig_suite)
        all_results.append(("Trigger Emulator & SF", res, elapsed))
        if not res.wasSuccessful():
            overall_success = False
    except Exception as e:
        print(f"ERROR loading Trigger Emulator tests: {e}", file=sys.stderr)
        overall_success = False

    # 3. Jet Clustering
    try:
        from coffea4bees.jet_clustering.tests import test_clustering, test_cluster_bs_numba, test_splitting_library
        loader = unittest.TestLoader()
        cluster_suite = unittest.TestSuite()
        cluster_suite.addTests(loader.loadTestsFromModule(test_clustering))
        cluster_suite.addTests(loader.loadTestsFromModule(test_cluster_bs_numba))
        cluster_suite.addTests(loader.loadTestsFromModule(test_splitting_library))
        res, elapsed = run_suite("3/10: Jet Clustering & Splitting Library", cluster_suite)
        all_results.append(("Jet Clustering", res, elapsed))
        if not res.wasSuccessful():
            overall_success = False
    except Exception as e:
        print(f"ERROR loading Jet Clustering tests: {e}", file=sys.stderr)
        overall_success = False

    # 4. Hemisphere Mixing
    try:
        from coffea4bees.hemisphere_mixing.tests import test_mixing, test_topk_matching
        loader = unittest.TestLoader()
        mixing_suite = unittest.TestSuite()
        mixing_suite.addTest(test_mixing.mixingTestCase("test_hemi_mixing"))
        mixing_suite.addTests(loader.loadTestsFromModule(test_topk_matching))
        res, elapsed = run_suite(
            "4/10: Hemisphere Mixing & TopK Matching",
            mixing_suite,
        )
        all_results.append(("Hemisphere Mixing", res, elapsed))
        if not res.wasSuccessful():
            overall_success = False
    except Exception as e:
        print(f"ERROR loading Hemisphere Mixing tests: {e}", file=sys.stderr)
        overall_success = False

    # 5. pourOver Plot Helpers (subprocess keeps sys.modules mocks isolated)
    ok, elapsed = run_pytest_suite(
        "5/10: pourOver Plot Helpers (pytest)",
        "coffea4bees/plots/tests/pourOver_test.py",
    )
    all_results.append(("pourOver Plot Helpers (pytest)", ok, elapsed))
    if not ok:
        overall_success = False

    # 6. Classifier Training Robustness (pytest)
    ok, elapsed = run_pytest_suite(
        "6/10: Classifier Training Robustness (pytest)",
        "coffea4bees/classifier/tests/test_training_robustness.py",
    )
    all_results.append(("Classifier Training Robustness", ok, elapsed))
    if not ok:
        overall_success = False

    # 7. Classifier Imports (pytest)
    ok, elapsed = run_pytest_suite(
        "7/10: Classifier Imports (pytest)",
        "coffea4bees/classifier/tests/test_imports.py",
    )
    all_results.append(("Classifier Imports", ok, elapsed))
    if not ok:
        overall_success = False

    # 8. Skimmer ttbb Stitching (pytest)
    ok, elapsed = run_pytest_suite(
        "8/10: Skimmer ttbb Stitching (pytest)",
        "coffea4bees/skimmer/tests/ttbb_stitch_test.py",
    )
    all_results.append(("Skimmer ttbb Stitching", ok, elapsed))
    if not ok:
        overall_success = False

    # 9. JCM ROOT Inputs Regression
    try:
        from coffea4bees.analysis.tests import test_jcm_root_regression
        res, elapsed = run_suite("9/10: JCM ROOT Inputs Regression", test_jcm_root_regression)
        all_results.append(("JCM ROOT Inputs Regression", res, elapsed))
        if not res.wasSuccessful():
            overall_success = False
    except Exception as e:
        print(f"ERROR loading JCM ROOT Regression test: {e}", file=sys.stderr)
        overall_success = False

    # 10. Plot Helpers
    try:
        from coffea4bees.plots.tests import test_plot_helpers
        res, elapsed = run_suite("10/10: Plot Helpers", test_plot_helpers)
        all_results.append(("Plot Helpers", res, elapsed))
        if not res.wasSuccessful():
            overall_success = False
    except Exception as e:
        print(f"ERROR loading Plot Helpers test: {e}", file=sys.stderr)
        overall_success = False

    total_elapsed = time.perf_counter() - total_start

    # Final Summary Table
    print(f"\n{'='*80}")
    print("  COFFEA4BEES UNIT TESTS SUMMARY")
    print(f"{'='*80}")
    total_run = 0
    total_fail = 0
    total_err = 0
    n_pytest = sum(isinstance(r, bool) for _, r, _ in all_results)

    for name, res, elapsed in all_results:
        if isinstance(res, bool):
            status = "PASSED" if res else "FAILED"
            print(f"  {name:<35} | {status:<6} | pytest    |         |        | {elapsed:>6.2f}s")
            continue
        status = "PASSED" if res.wasSuccessful() else "FAILED"
        runs = res.testsRun
        fails = len(res.failures)
        errs = len(res.errors)
        total_run += runs
        total_fail += fails
        total_err += errs
        print(f"  {name:<35} | {status:<6} | {runs:>3} tests | {fails:>2} fail | {errs:>2} err | {elapsed:>6.2f}s")

    print(f"{'-'*80}")
    status_str = "SUCCESS" if overall_success else "FAILURE"
    print(f"  OVERALL: {status_str} ({total_run} unittest tests + {n_pytest} pytest suites, {total_fail} failures, {total_err} errors in {total_elapsed:.2f}s)")
    print(f"{'='*80}\n", flush=True)

    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
