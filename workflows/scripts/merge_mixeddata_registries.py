"""Merge per-year mixed-data picoAOD registries into one.

Each input is keyed by per-year dataset name (e.g. data_2022_EEE), so top-level
keys never collide; we assert no overlap to catch accidental reruns.

runner.py writes these registries with plain ``yaml.dump``, which serializes the
``lumis_processed`` field's numpy keys/dtypes as ``!!python/object`` tags. A
plain ``yaml.full_load``/``safe_load`` then refuses them. We don't need that
field — the install step (make_dataset_yml.py) consumes only ``files`` — so we
parse with a tolerant loader that maps the python-object/tuple tags to ``None``
(no numpy import, no reconstruction of the ~80k objects), drop ``lumis_processed``,
and re-emit clean, plain YAML with ``safe_dump`` so every downstream reader works.

Usage: merge_mixeddata_registries.py IN1.yml IN2.yml ... OUTFILE.yml
"""
import sys
import yaml


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that ignores the python-object/tuple tags numpy left behind."""


_TolerantLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object/apply:", lambda loader, suffix, node: None)
_TolerantLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object/new:", lambda loader, suffix, node: None)
_TolerantLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object:", lambda loader, suffix, node: None)
_TolerantLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: None)


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: merge_mixeddata_registries.py IN1.yml [IN2.yml ...] OUTFILE.yml")
    out = {}
    for f in sys.argv[1:-1]:
        with open(f) as fh:
            d = yaml.load(fh, Loader=_TolerantLoader) or {}
        # Drop the heavy numpy-laden field; nothing downstream consumes it.
        for dataset in d.values():
            if isinstance(dataset, dict):
                dataset.pop("lumis_processed", None)
        overlap = set(out) & set(d)
        if overlap:
            raise SystemExit(f"Key collision merging {f}: {overlap}")
        out.update(d)
    with open(sys.argv[-1], "w") as fh:
        yaml.safe_dump(out, fh, default_flow_style=False)
    print(f"Merged {len(sys.argv) - 2} files -> {sys.argv[-1]} ({len(out)} datasets)")


if __name__ == "__main__":
    main()
