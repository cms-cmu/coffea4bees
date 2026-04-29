"""
Helpers for the Run3 parking vs preparking trigger split.

In Run3 the trigger strategy changed mid-2023_preBPix: parking triggers
are present from era C3 of 2023_preBPix and through all of 2023_BPix.
Separate SvB_FeynNet ONNX models were trained for each trigger era; the
correct model must be evaluated on each event.

For data, the parking/preparking distinction is a simple run-number cutoff
(``parking_minRun``).

For MC we have an inclusive simulation. Each event is assigned to parking
or preparking with probability matching the data luminosity fractions.
The assignment is generated once by ``processor_isParking_friend.py`` and
stored in a sidecar friend tree, so all downstream consumers see the same
``is_parking`` value for any given event.

The configuration (parking_minRun for data, parking_fraction for MC) lives
in ``coffea4bees/analysis/metadata/parking_lumi_fractions.yml``.
"""

import logging

import awkward as ak
import numpy as np
import yaml

DEFAULT_LUMI_CFG_PATH = "coffea4bees/analysis/metadata/parking_lumi_fractions.yml"


def load_parking_lumi_cfg(path: str = DEFAULT_LUMI_CFG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def assign_is_parking(
    event,
    year: str,
    is_mc: bool,
    parking_lumi_cfg: dict,
    friend=None,
    target=None,
) -> None:
    """Set ``event['is_parking']`` based on data/MC and year.

    Sources, in priority order:
      * data: ``event.run >= parking_minRun`` (no friend tree needed)
      * MC, parking_fraction == 0 or 1: constant array (years that don't
        span the boundary)
      * MC, mixed year (e.g. 2023_preBPix): loaded from the ``is_parking``
        friend tree (must be passed via ``friend`` + ``target``)

    Parameters
    ----------
    event : ak.Array
        Full (unfiltered) event array. Must have ``run`` field for data.
    year : str
        Data-taking year key in parking_lumi_cfg.
    is_mc : bool
    parking_lumi_cfg : dict
        Loaded contents of parking_lumi_fractions.yml.
    friend : FriendTemplate, optional
        Friend tree carrying the ``is_parking`` branch (only required for
        MC years with a non-trivial parking_fraction).
    target : Chunk, optional
        Chunk descriptor for friend tree lookup.
    """
    n = len(event)
    year_cfg = parking_lumi_cfg.get(year)
    if year_cfg is None:
        logging.warning(
            f"assign_is_parking: year {year!r} not in parking lumi cfg; "
            "defaulting to is_parking=False for all events."
        )
        event["is_parking"] = ak.Array(np.zeros(n, dtype=bool))
        return

    if not is_mc:
        min_run = year_cfg.get("parking_minRun")
        if min_run is None:
            event["is_parking"] = ak.Array(np.zeros(n, dtype=bool))
        else:
            event["is_parking"] = event.run >= int(min_run)
        return

    fraction = float(year_cfg.get("parking_fraction", 0.0))
    if fraction <= 0.0:
        event["is_parking"] = ak.Array(np.zeros(n, dtype=bool))
        return
    if fraction >= 1.0:
        event["is_parking"] = ak.Array(np.ones(n, dtype=bool))
        return

    # Mixed-fraction MC year: must come from the is_parking friend tree.
    if friend is None or target is None:
        raise RuntimeError(
            f"is_parking friend tree is required for MC year {year} "
            f"(parking_fraction={fraction}) but was not provided. "
            f"Generate it with processor_isParking_friend.py and add the "
            f"path to friends_HH4b.yml under is_parking."
        )
    result = friend.arrays(target)
    if result is None:
        raise RuntimeError(
            f"is_parking friend tree did not return entries for {target}."
        )
    event["is_parking"] = result.is_parking
