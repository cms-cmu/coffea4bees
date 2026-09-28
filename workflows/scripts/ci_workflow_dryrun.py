#!/usr/bin/env python3
"""Parse and dry-run every coffea4bees Snakemake workflow (CI job workflow_validation_coffea4bees).

Run from the barista root:

    python coffea4bees/workflows/scripts/ci_workflow_dryrun.py [--snakemake "pixi run snakemake"]

Three checks, all offline (a dry run touches no EOS; the new workflows only prefix-check their
root:// inputs at parse time):

  1. compile  every .smk under workflows/ (archive/ excepted) goes through snakemake's own parser
              (--print-compilation). `ast.parse` is not a parse test: `module` at the start of a
              statement is a Snakemake keyword, which is how one helper once broke every Snakefile.
  2. coverage every top-level Snakefile (one no other .smk includes) is either paired with a config
              under `runs:` in gitlab-CI/dryrun_matrix.yml or named under `not_covered:`.
  3. dry-run  every (config, Snakefile) pair in `runs:`, with test=false and test=true. A `base:`
              config is layered over its base chain as repeated --configfile, which merges exactly
              as `roast new` captures it (src/tools/roast.py _merge_config).
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

WORKFLOWS = Path("coffea4bees/workflows")
CONFIGS = WORKFLOWS / "config"
MATRIX = WORKFLOWS / "gitlab-CI/dryrun_matrix.yml"


def snakefile(name):
    return WORKFLOWS / f"Snakefile_{name}.smk"


def base_chain(config):
    """The --configfile list for a config: its `base:` chain innermost first, then itself."""
    chain, seen = [config], set()
    while True:
        m = re.search(r"^base:\s*['\"]?([^'\"\s#]+)", chain[0].read_text(), re.M)
        if not m:
            return chain
        base = Path(m.group(1))          # relative to the barista root, as in roast.py
        if base.resolve() in seen or not base.exists():
            raise SystemExit(f"{chain[0]}: base config {base} is missing or cyclic")
        seen.add(base.resolve())
        chain.insert(0, base)


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "CI": "1"})
    return p.returncode, p.stdout + p.stderr


def missing_inputs(log):
    """Files listed under snakemake's 'Missing input files for rule ...: affected files:'."""
    files, grab = [], False
    for line in log.splitlines():
        if "affected files:" in line:
            grab = True
        elif grab and line.startswith((" ", "\t")) and line.strip():
            files.append(line.strip())
        elif grab:
            grab = False
    return files


def tail(log, n=25):
    return "\n".join("      " + l for l in log.rstrip().splitlines()[-n:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snakemake", default="snakemake", help='snakemake command (CI: "pixi run snakemake")')
    ap.add_argument("--jobs", type=int, default=4, help="parallel snakemake invocations")
    ap.add_argument("--only", default="", help="regex: dry-run only pairs whose 'config Snakefile' matches")
    args = ap.parse_args()
    if not WORKFLOWS.is_dir():
        raise SystemExit(f"run from the barista root ({WORKFLOWS} not found)")
    sm = shlex.split(args.snakemake)
    matrix = yaml.safe_load(MATRIX.read_text())
    runs, not_covered = matrix.get("runs") or {}, matrix.get("not_covered") or {}
    failures = []

    # 1. compile
    smk = sorted(p for p in WORKFLOWS.rglob("*.smk") if "archive" not in p.parts)
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda p: (p, *run([*sm, "-s", str(p), "--print-compilation"])), smk))
    bad = [(p, log) for p, rc, log in results if rc]
    print(f"compile: {len(smk) - len(bad)}/{len(smk)} .smk files parse")
    for p, log in bad:
        failures.append(f"compile {p}")
        print(f"  FAIL {p}\n{tail(log)}")

    # 2. coverage
    included = set()
    for p in smk:
        included |= {Path(m).name for m in re.findall(r"^\s*include:\s*['\"]([^'\"]+)['\"]", p.read_text(), re.M)}
    top = {p.stem.removeprefix("Snakefile_") for p in WORKFLOWS.glob("Snakefile_*.smk") if p.name not in included}
    paired = {e if isinstance(e, str) else e["snakefile"] for entries in runs.values() for e in entries}
    problems = [f"{n}: top-level Snakefile in neither runs: nor not_covered:" for n in sorted(top - paired - set(not_covered))]
    problems += [f"{n}: listed in dryrun_matrix.yml but {snakefile(n)} does not exist"
                 for n in sorted(paired | set(not_covered)) if not snakefile(n).exists()]
    problems += [f"{n}: in both runs: and not_covered:" for n in sorted(paired & set(not_covered))]
    problems += [f"{c}: config in runs: does not exist" for c in runs if not (CONFIGS / c).exists()]
    print(f"coverage: {len(top)} top-level Snakefiles, {len(top & paired)} dry-run, {len(top & set(not_covered))} not_covered")
    for msg in problems:
        failures.append(f"coverage {msg}")
        print(f"  FAIL {msg}")

    # 3. dry-run
    pairs = []
    for cfg, entries in runs.items():
        if not (CONFIGS / cfg).exists():
            continue
        chain = [str(c) for c in base_chain(CONFIGS / cfg)]
        for e in entries:
            name, upstream = (e, []) if isinstance(e, str) else (e["snakefile"], e.get("upstream") or [])
            if not snakefile(name).exists() or not re.search(args.only, f"{cfg} {name}"):
                continue
            for test in ("false", "true"):
                pairs.append((cfg, name, test, upstream,
                              [*sm, "-s", str(snakefile(name)), "--configfile", *chain,
                               "--config", f"test={test}", "-n", "--nolock", "--quiet", "rules"]))

    def dryrun(pair):
        cfg, name, test, upstream, cmd = pair
        rc, log = run(cmd)
        if rc == 0:
            m = re.search(r"^total\s+(\d+)", log, re.M)
            return pair, "ok", f"{m.group(1) if m else 0} jobs", log
        missing = missing_inputs(log)
        if upstream and missing and all(any(u in f for u in upstream) for f in missing):
            return pair, "ok", f"{len(missing)} upstream inputs absent (expected)", log
        return pair, "FAIL", "", log

    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(dryrun, pairs))
    print(f"dry-run: {sum(r[1] == 'ok' for r in results)}/{len(results)} (config, Snakefile, test) pairs")
    for (cfg, name, test, _, cmd), status, note, log in results:
        print(f"  {status:4} {cfg:36} {name:30} test={test:5} {note}")
        if status == "FAIL":
            failures.append(f"dry-run {cfg} {name} test={test}")
            print(f"      $ {shlex.join(cmd)}\n{tail(log)}")

    if failures:
        print(f"\n{len(failures)} failure(s):\n  " + "\n  ".join(failures))
        return 1
    print("\nall workflows parse and dry-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
