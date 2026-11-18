#!/usr/bin/env python3
"""Unified test runner for coffea4bees unit tests.

Executes all unit test suites with verbose per-test output (verbosity=2)
and provides a clear pass/fail summary and exit code for CI pipelines.
"""
import os
import sys
import time
import unittest

# Ensure repo roots are on python path
sys.path.insert(0, os.getcwd())


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
        from coffea4bees.analysis.tests import topCand_test
        res, elapsed = run_suite("1/4: Analysis Helpers (Top Candidate Reconstruction)", topCand_test)
        all_results.append(("Analysis: Top Candidate", res, elapsed))
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
        res, elapsed = run_suite("2/4: Trigger Emulator & Vectorized Scale Factors", trig_suite)
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
        res, elapsed = run_suite("3/4: Jet Clustering & Splitting Library", cluster_suite)
        all_results.append(("Jet Clustering", res, elapsed))
        if not res.wasSuccessful():
            overall_success = False
    except Exception as e:
        print(f"ERROR loading Jet Clustering tests: {e}", file=sys.stderr)
        overall_success = False

    # 4. Hemisphere Mixing
    try:
        from coffea4bees.hemisphere_mixing.tests import test_mixing
        res, elapsed = run_suite(
            "4/4: Hemisphere Mixing",
            None,
            specific_case=test_mixing.mixingTestCase("test_hemi_mixing")
        )
        all_results.append(("Hemisphere Mixing", res, elapsed))
        if not res.wasSuccessful():
            overall_success = False
    except Exception as e:
        print(f"ERROR loading Hemisphere Mixing tests: {e}", file=sys.stderr)
        overall_success = False

    total_elapsed = time.perf_counter() - total_start

    # Final Summary Table
    print(f"\n{'='*80}")
    print("  COFFEA4BEES UNIT TESTS SUMMARY")
    print(f"{'='*80}")
    total_run = 0
    total_fail = 0
    total_err = 0

    for name, res, elapsed in all_results:
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
    print(f"  OVERALL: {status_str} ({total_run} tests, {total_fail} failures, {total_err} errors in {total_elapsed:.2f}s)")
    print(f"{'='*80}\n", flush=True)

    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
