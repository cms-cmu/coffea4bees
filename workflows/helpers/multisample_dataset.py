"""Assemble a multi-sample mixed-data dataset YAML (mixeddata_4b, mixeddata_<tag>_4b).

One dataset key with `nSamples: N` and, per year, a `files_template` list: the sample files with
their index replaced by XXX (runner.py expands it per sample) plus the ttbar pseudodata files (no
XXX, so shared by every sample). Each sample is then a complete closure pseudo-data sample,
multijet + ttbar.

Used by bkg_syst A_2 for both kinds of mixed data:
  * 3b mixing: the JCM split of mixeddata_all, one skim registry per subsample
    (sample files <PUB>/picoAOD/subsamples/v<k>/picoAOD_mixed_v<k>[.chunk<j>].root)
  * 4b mixing: the seeds of a MakeMixedData roast's mixeddata_all_<tag>, one seed per sample
    (sample files .../picoAOD/<mix_name>/v<k>/...)
"""
import re

import yaml

from src.tools.make_dataset_yml import parse_dataset_key


def _all_files(node):
    """Every file under a dataset-YAML picoAOD node ({files: [...]} or {<era>: {files: [...]}})."""
    if isinstance(node, list):
        return list(node)
    if isinstance(node, dict):
        return [f for v in node.values() for f in _all_files(v)]
    return []


def registry_files(path):
    """{year: [files]} of one skim registry (keys like data_UL18A)."""
    with open(path) as f:
        registry = yaml.safe_load(f) or {}
    files = {}
    for key, entry in registry.items():
        year, _era = parse_dataset_key(key)
        if year is None:
            raise ValueError(f"{path}: cannot tell the year of registry key {key!r}")
        files.setdefault(year, []).extend((entry or {}).get('files') or [])
    return files


def seed_files(dataset_yaml, mix_name, n):
    """[{year: [files]}] per seed 0..n-1 of a 4b-mixing dataset (seed s under .../<mix_name>/v<s>/)."""
    with open(dataset_yaml) as f:
        entry = (yaml.safe_load(f) or {}).get(mix_name)
    if not entry:
        raise ValueError(f"{dataset_yaml}: no dataset {mix_name!r}")
    seed_re = re.compile(rf'/{re.escape(mix_name)}/v(\d+)/')
    seeds = [{} for _ in range(n)]
    for year, ydata in entry.items():
        for fp in _all_files((ydata or {}).get('picoAOD')):
            m = seed_re.search(fp)
            if m is None:
                raise ValueError(f"{dataset_yaml}: {fp} carries no /{mix_name}/v<seed>/")
            s = int(m.group(1))
            if s >= n:
                raise ValueError(f"{dataset_yaml}: seed v{s} but subsamples.n = {n}")
            seeds[s].setdefault(year, []).append(fp)
    empty = [s for s, files in enumerate(seeds) if not files]
    if empty:
        raise ValueError(f"{dataset_yaml}: no files for seeds {empty} (subsamples.n = {n})")
    return seeds


def template_seed_file(mix_name):
    """4b mixing: .../<mix_name>/v<s>/... -> .../vXXX/..."""
    def template(fp, v):
        t = re.sub(rf'/{re.escape(mix_name)}/v{v}/', f'/{mix_name}/vXXX/', fp)
        return t if 'XXX' in t else None
    return template


def template_split_file(fp, v):
    """3b split: <PUB>/picoAOD/subsamples/v<v>/picoAOD_mixed_v<v>[.chunk<k>].root -> vXXX."""
    t = re.sub(rf'/subsamples/v{v}/', '/subsamples/vXXX/', fp)
    # keep .chunkN: the files on disk are picoAOD_mixed_v<v>.chunk<k>.root
    t = re.sub(rf'_v{v}((\.chunk\d+)?\.root)$', r'_vXXX\1', t)
    return t if 'XXX' in t else None


def build_multisample_dataset(samples, psdata_yaml, ps_name, years, name, template):
    """samples: [{year: [files]}] in sample order. Every sample must produce the same set of
    templates -- checked, not assumed (an earlier version parsed only v0's)."""
    per_v = []
    for v, files in enumerate(samples):
        templates = {}
        for year, fps in files.items():
            for fp in fps:
                t = template(fp, v)
                if t is None:
                    raise ValueError(f"sample v{v}: {fp} does not carry sample index v{v}")
                templates.setdefault(year, set()).add(t)
        per_v.append(templates)
    for v, templates in enumerate(per_v[1:], start=1):
        if templates != per_v[0]:
            only_v = sorted(set().union(*templates.values()) - set().union(*per_v[0].values()))[:3]
            only_0 = sorted(set().union(*per_v[0].values()) - set().union(*templates.values()))[:3]
            raise ValueError(f"sample v{v} files differ from v0 once templated (e.g. only v{v}: "
                             f"{only_v}, only v0: {only_0}); the vXXX expansion would read "
                             f"missing files or skip existing ones")
    missing = [y for y in years if y not in per_v[0]]
    if missing:
        raise ValueError(f"no sample files for {missing}")
    with open(psdata_yaml) as f:
        psdata = (yaml.safe_load(f) or {}).get(ps_name) or {}
    dataset = {'nSamples': len(samples), 'xs': {'Run2': 1, 'Run3': 1}}
    for year in years:
        ps_files = _all_files((psdata.get(year) or {}).get('picoAOD'))
        if not ps_files:
            raise ValueError(f"{psdata_yaml}: no ttbar pseudodata files for {year}")
        if any('XXX' in p for p in ps_files):
            raise ValueError(f"{psdata_yaml}: pseudodata file names contain XXX")
        dataset[year] = {'picoAOD': {'files_template': sorted(per_v[0][year]) + sorted(ps_files)}}
    return {name: dataset}
