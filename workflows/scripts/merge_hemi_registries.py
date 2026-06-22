"""Merge per-year hemisphere-library YAMLs into one ``{year: [files]}`` registry.

Keys are distinct years, so no collision is possible; we assert anyway to catch
accidental reruns clobbering each other.

Usage: merge_hemi_registries.py IN1.yml IN2.yml ... OUTFILE.yml
"""
import sys
import yaml


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: merge_hemi_registries.py IN1.yml [IN2.yml ...] OUTFILE.yml")
    out = {}
    for f in sys.argv[1:-1]:
        with open(f) as fh:
            d = yaml.full_load(fh) or {}
        overlap = set(out) & set(d)
        if overlap:
            raise SystemExit(f"Year collision merging {f}: {overlap}")
        out.update(d)
    with open(sys.argv[-1], "w") as fh:
        yaml.dump(out, fh, default_flow_style=False)
    print(f"Merged {len(sys.argv) - 2} years -> {sys.argv[-1]} ({len(out)} years)")


if __name__ == "__main__":
    main()
