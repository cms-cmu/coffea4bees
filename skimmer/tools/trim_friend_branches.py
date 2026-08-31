"""
Restrict the declared branch list of a friend-tree index to a reference set.

Needed to merge friend indexes produced by different versions of the dumping
code. ``Friend.__iadd__`` refuses to merge two indexes whose ``branches`` differ,
and ``dump_input_friend`` has since gained extra per-jet features, so a freshly
produced HCR_input index declares a strict superset of the committed one.

Trimming is safe for reading: the HCR dataset loader passes
``branches=self._branches.intersection`` to the reader, so it only ever fetches
the branches it needs and the extra ones present in the ROOT files are ignored.
It does mean any information in the dropped branches goes unused - which it
would anyway, since the other half of the merged index does not have them.

Usage::

    python coffea4bees/skimmer/tools/trim_friend_branches.py \\
        -i new_index.json -r committed_index.json -o new_index_trimmed.json
"""

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-i", "--input", required=True, help="index to trim")
    ap.add_argument(
        "-r", "--reference", required=True,
        help="index whose branch list defines the target set",
    )
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument(
        "-k", "--key", default=None,
        help="friend key to trim (default: every key present in both files)",
    )
    args = ap.parse_args(argv)

    new = json.load(open(args.input))
    ref = json.load(open(args.reference))

    keys = [args.key] if args.key else sorted(set(new) & set(ref))
    if not keys:
        raise SystemExit(f"no common friend keys between {args.input} and {args.reference}")

    for key in keys:
        have = set(new[key]["branches"])
        want = set(ref[key]["branches"])
        missing = want - have
        if missing:
            raise SystemExit(
                f"{key}: reference requires branches absent from {args.input}: "
                f"{sorted(missing)}"
            )
        dropped = sorted(have - want)
        # Preserve the reference ordering so the two indexes compare equal.
        new[key]["branches"] = list(ref[key]["branches"])
        print(f"{key}: {len(have)} -> {len(want)} branches")
        if dropped:
            print(f"  dropped ({len(dropped)}): {dropped}")

    with open(args.output, "w") as fh:
        json.dump(new, fh)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
