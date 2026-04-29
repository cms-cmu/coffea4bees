"""
processor_isParking_friend
==========================

Produce a per-event ``is_parking`` boolean friend tree for MC samples that span
the Run-3 parking/preparking trigger boundary.

Background
----------
In Run-3 we changed our trigger strategy partway through 2023_preBPix: parking
triggers are present from era C3 of 2023_preBPix and through all of 2023_BPix.
There are separate ONNX models and preprocessing JSONs for the parking and
preparking SvB_FeynNet evaluations.

For data, the parking/preparking distinction is a simple run-number cutoff
(``parking_minRun``).

For MC we have an inclusive simulation. We assign each event to parking or
preparking with probability matching the data luminosity fractions, using the
``Squares`` reproducible RNG. The assignment is generated once and stored as a
sidecar friend tree so all downstream consumers (FeynNet, future classifiers)
use the same ``is_parking`` value for any given event.

This processor writes one friend-tree entry per *input* event (no selections).

Parking fractions and the data run cutoff live in
``coffea4bees/analysis/metadata/parking_lumi_fractions.yml``.
"""

import logging
import warnings
import time

import awkward as ak
import numpy as np
import yaml
from coffea import processor
from coffea.nanoevents import NanoAODSchema

from src.math_tools.random import Squares

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")


_DEFAULT_LUMI_CFG = "coffea4bees/analysis/metadata/parking_lumi_fractions.yml"


def _load_lumi_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_is_parking(event, dataset, year, parking_fraction):
    """Return an awkward bool array, one entry per event in ``event``.

    For ``parking_fraction`` of 0.0 / 1.0 the result is constant; otherwise
    ``Squares("isParking", dataset, year)`` is used with the same (event,
    run<<32 | luminosityBlock) counter convention as ``sub_sample_MC``.
    """
    n = len(event)
    if parking_fraction <= 0.0:
        return ak.Array(np.zeros(n, dtype=bool))
    if parking_fraction >= 1.0:
        return ak.Array(np.ones(n, dtype=bool))

    rng = Squares("isParking", dataset, year)
    counter = np.empty((n, 2), dtype=np.uint64)
    counter[:, 0] = np.asarray(event.event).view(np.uint64)
    counter[:, 1] = np.asarray(event.run).view(np.uint32)
    counter[:, 1] <<= 32
    counter[:, 1] |= np.asarray(event.luminosityBlock).view(np.uint32)
    rand_vals = rng.uniform(counter, low=0.0, high=1.0).astype(np.float32)
    return ak.Array(rand_vals < parking_fraction)


class analysis(processor.ProcessorABC):
    """Standalone friend-tree producer for the ``is_parking`` boolean.

    Parameters
    ----------
    make_friend_isParking : str
        Output base path (EOS or local) for the friend tree.
    parking_lumi_cfg : str, optional
        Path to the parking lumi-fractions YAML.
    """

    def __init__(self, make_friend_isParking: str = None,
                 parking_lumi_cfg: str = _DEFAULT_LUMI_CFG):
        if make_friend_isParking is None:
            raise ValueError("make_friend_isParking output path must be set")
        self.make_friend_isParking = make_friend_isParking
        self.parking_lumi_cfg = _load_lumi_cfg(parking_lumi_cfg)

    def process(self, event):
        from coffea4bees.analysis.helpers.dump_friendtrees import dump_isParking

        tstart = time.time()
        dataset = event.metadata["dataset"]
        year = event.metadata["year"]
        estart = event.metadata["entrystart"]
        estop = event.metadata["entrystop"]
        chunk = f"{dataset}::{estart:6d}:{estop:6d} >>> "
        n = len(event)

        year_cfg = self.parking_lumi_cfg.get(year)
        if year_cfg is None:
            raise KeyError(f"{chunk}year {year!r} not found in parking lumi cfg")
        parking_fraction = float(year_cfg["parking_fraction"])

        is_parking = compute_is_parking(event, dataset, year, parking_fraction)

        n_park = int(ak.sum(is_parking))
        logging.info(
            f"{chunk}is_parking: {n_park}/{n} = {n_park / max(n, 1):.4f} "
            f"(target fraction {parking_fraction:.4f})"
        )

        friends = {"friends": dump_isParking(
            event,
            self.make_friend_isParking,
            "is_parking",
            is_parking,
        )}

        processOutput = {
            "nEvent": {dataset: n},
            "nParking": {dataset: n_park},
        }

        elapsed = time.time() - tstart
        logging.debug(f"{chunk}{n / max(elapsed, 1e-9):,.0f} events/s")

        return processOutput | friends

    def postprocess(self, accumulator):
        return accumulator
