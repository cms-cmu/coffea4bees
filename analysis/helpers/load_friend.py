from typing import TypedDict

import awkward as ak
from src.data_formats.root import Chunk, Friend


class FriendTemplate(TypedDict):
    path: str
    keys: str | list[dict[str]]


def parse_friends(args: dict[str, str | FriendTemplate]) -> dict[str, Friend]:
    friends = {}
    if args is None:
        return friends

    from src.classifier.task import parse

    for name, path in args.items():
        if isinstance(path, str):
            friends[name] = Friend.from_json(parse.mapping(path, "file"))
        else:
            keys = path["keys"]
            if isinstance(keys, str):
                keys = eval(keys)
            for key in keys:
                friends[name.format(**key)] = Friend.from_json(
                    parse.mapping(path["path"].format(**key), "file")
                )

    return friends


# The following are temporary functions to load new friend trees in a backward compatible manner
# TODO: remove in the future


def _rename(arr: ak.Array, kept: list[str], rename: dict[str, str]):
    fields = set(arr.fields)
    renamed = {k: arr[k] for k in kept}
    for k, v in rename.items():
        if k in fields:
            renamed[v] = arr[k]
    return ak.zip(renamed)


def rename_FvT_friend(chunk: Chunk, friend: Friend):
    kept = ["FvT", "q_1234", "q_1324", "q_1423"]
    rename = {
        "p_d4": "pd4",
        "p_d3": "pd3",
        "p_t4": "pt4",
        "p_t3": "pt3",
        "p_m4": "pm4",
        "p_m3": "pm3",
        "p_ttbar": "pt",
    }
    FvT = friend.arrays(
        chunk,
        reader_options={"branch_filter": set().union(kept, rename).intersection},
    )
    if FvT is None:
        return None
    return _rename(FvT, kept, rename)


def read_MvD_friend(chunk: Chunk, friend: Friend):
    wanted = {"MvD", "q_1234", "q_1324", "q_1423", "p_mix4", "p_d4", "p_t4", "p_4b"}
    return friend.arrays(chunk, reader_options={"branch_filter": wanted.intersection})


def rename_SvB_friend(chunk: Chunk, friend: Friend):
    kept = ["q_1234", "q_1324", "q_1423"]
    renames = {
        "p_sig": "ps",
        "p_ttbar": "ptt",
        "p_ZZ": "pzz",
        "p_ZH": "pzh",
        "p_ggF": "phh",
        "p_multijet": "pmj",
    }
    SvB = friend.arrays(
        chunk,
        reader_options={"branch_filter": set().union(kept, renames).intersection},
    )
    if SvB is None:
        return None
    # Binary ggF-vs-bkg schema (no ZZ/ZH channels): keep the raw branch names
    # so setSvBVars' binary branch — which reads p_sig/p_ggF/p_multijet/p_ttbar —
    # picks them up. Renaming p_ggF->phh would make setSvBVars misdetect the
    # legacy 5-class schema and crash on the missing pzz field.
    if "p_ZZ" not in SvB.fields:
        return SvB
    return _rename(SvB, kept, renames)
