# Session — test-skills (2026-03-15)

## What we did
- Added `test_make_ratio` to `coffea4bees/tests/plots_test.py` — pure math unit test for normal ratio, zero-denominator → nan, and norm=True rescaling
- Added `uniform_bins=True` and `region=["SR","SB"]` list overlay cases to `coffea4bees/plots/tests/iPlot_test.py` `do_plots()` — covers previously untested code paths
- Created `/test-barista` skill at `.claude/skills/test-barista/SKILL.md` — push branch, create/reuse MR, monitor pipeline, fix failures
- Created `/test-coffea4bees` skill at `.claude/skills/test-coffea4bees/SKILL.md` — same flow for the coffea4bees sub-repo
- Ran `/test-coffea4bees` on branch `TwoTagCutFlow`: pushed, created MR !635, pipeline #14285417 passed (52/53; 1 pre-existing failure)
- Fixed both skills to use newline-separated Bash commands after discovering `source ~/.aliases_local && curl ...` silently drops the token variable

## Decisions
- `--no-fix` removed from both test skills — failures always trigger the fix loop
- MR title format: `WIP: CI test — <slug>` where slug is kebab-cased from branch name
- Skills live in barista-local `.claude/skills/` (accessible from barista root for both repos)
- Token loaded from `~/.aliases_local`; abort with message if unset, no fallback
- `source` + curl must use newlines not `&&`/`;` — variables don't propagate on same-line chained commands in the Bash tool

## Files changed
- `coffea4bees/tests/plots_test.py` — added `test_make_ratio`
- `coffea4bees/plots/tests/iPlot_test.py` — added `uniform_bins` and region-list overlay test cases to `do_plots()`
- `.claude/skills/test-barista/SKILL.md` — new skill (created and updated with source/newline fix)
- `.claude/skills/test-coffea4bees/SKILL.md` — new skill (created and updated with source/newline fix)

## Open threads
- MR !69 on barista (`plotting_refactor`) is still open and green — needs review and merge
- MR !635 on coffea4bees (`TwoTagCutFlow`) is open — was a CI test MR, may need closing or updating with real description
- `tools_perf_profile_coffea4bees` is a known pre-existing CI failure (missing friend tree for trigWeight) — unrelated to any current branch work
