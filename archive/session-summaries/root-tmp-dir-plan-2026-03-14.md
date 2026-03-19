# Session — root-tmp-dir-plan (2026-03-14)

## What we did
- Diagnosed why intermediate ROOT chunk files during MvD classifier evaluation fill local disk even when final output goes to EOS
- Traced root cause to `TreeWriter.__enter__` in `src/data_formats/root/io.py:179` — always uses `tempfile.mkstemp(dir=".")` regardless of final destination
- Wrote a detailed implementation plan for adding a configurable `tmp_dir` setting

## Decisions
- Default `tmp_dir="."` — preserves existing behavior exactly; new setting is opt-in only
- Writing directly to XRootD via uproot is not feasible (iterative seeks needed for ROOT headers); local temp is unavoidable
- Deferred implementation — no unit tests exist for the ROOT writing pipeline; implement when tests can be added alongside

## Files changed
- `/uscms/home/jda102/.claude/plans/configurable-root-tmp-dir.md` — created plan file (renamed from auto-generated `squishy-foraging-ember.md`)

## Open threads
- Implementation deferred pending unit tests for the ROOT writing pipeline
- `Friend.merge()` optimization step in `merge_kfolds` also uses `TreeWriter` but is not covered by the plan — minor extension if needed
