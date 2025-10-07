import json
from concurrent.futures import ProcessPoolExecutor
from itertools import chain

import fsspec
import yaml
from src.data_formats.root import Chunk, Friend
from src.storage.eos import EOS
from src.utils.json import DefaultEncoder
from src.classifier.process.pool import CallbackExecutor
from rich.pretty import pprint
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)

output = "test.json"
metadata = "metadata/datasets_HH4b_2024_v2.yml"
base = "root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/"
friend_name = "HCR_input"
max_workers = 20

if __name__ == "__main__":
    base = EOS(base)
    years = {
        "UL18": ["A", "B", "C", "D"],
        "UL17": ["C", "D", "E", "F"],
        "UL16_postVFP": ["F", "G", "H"],
        "UL16_preVFP": ["B", "C", "D", "E"],
    }
    mcs = [
        "GluGluToHHTo4B_cHHH0",
        "GluGluToHHTo4B_cHHH1",
        "GluGluToHHTo4B_cHHH2p45",
        "GluGluToHHTo4B_cHHH5",
        "TTTo2L2Nu",
        "TTToHadronic",
        "TTToSemiLeptonic",
        "ZH4b",
        "ZZ4b",
        "ggZH4b",
    ]
    tree = "Events"

    def filename(path1: str, path0: str, name: str, **_):
        return f'{path1}/{path0.replace("picoAOD", name)}'

    with open(metadata, mode="rt") as f:
        db = yaml.safe_load(f)["datasets"]

    picoAODs = []
    friends = {}

    for mc in mcs:
        for year in years:
            picoAODs.extend(db[mc][year]["picoAOD"]["files"])
    for year, eras in years.items():
        for era in eras:
            picoAODs.extend(db["data"][year]["picoAOD"][era]["files"])

    for file in picoAODs:
        friends[file] = str(
            base / filename(**Friend._path_parts(EOS(file)), name=friend_name)
        )

    with Progress(
        SpinnerColumn(),
        TimeElapsedColumn(),
        BarColumn(bar_width=None),
        "{task.completed}/{task.total} {task.description}",
        expand=True,
    ) as progress, ProcessPoolExecutor(max_workers=max_workers) as pool:
        task = progress.add_task(
            "fetching metadata", total=len(picoAODs) + len(friends)
        )
        chunks = Chunk.from_path(
            *((file, tree) for file in chain(picoAODs, friends.values())),
            executor=CallbackExecutor(
                pool, lambda _: lambda _: progress.update(task, advance=1)
            ),
        )
    chunks = {str(c.path): c for c in chunks}
    friend = Friend(friend_name)

    branches = []
    test_branches = chunks[next(iter(friends.values()))].branches
    for b in test_branches:
        if not (b.startswith("n") and b[1:] in test_branches):
            branches.append(b)
    friend._branches = frozenset(branches)
    for target in picoAODs:
        friend.add(chunks[target], chunks[friends[target]])
    pprint(friend)
    with fsspec.open(output, mode="wt") as f:
        json.dump({friend_name: friend}, f, cls=DefaultEncoder)