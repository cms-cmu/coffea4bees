"""
Stage 2 of the tt+bb stitching: write the stitched picoAODs.

The stitched ttbar sample is built from two passes over existing picoAODs,
both driven by this processor and distinguished by the input dataset:

* **inclusive ttbar** (``TTTo2L2Nu``, ``TTToSemiLeptonic``, ``TTToHadronic``)
  keep the ``tt+C`` and ``tt+LF`` events, i.e. drop ``tt+B``.
  ``genWeight`` is written through unchanged.

* **dedicated TTbb** (``TTbb_2L2Nu``, ``TTbb_SemiLeptonic``, ``TTbb_Hadronic``)
  keep only the ``tt+B`` events, and scale ``genWeight`` by

      k = sumw_ttB(inclusive) / sumw_ttB(TTbb)

  so the tt+B piece carries exactly the genWeight sum of the tt+B piece that
  was removed from the inclusive sample.

Summing the two outputs therefore preserves the total genWeight of the
inclusive sample, and the stitched dataset inherits the inclusive cross
section and ``sumw`` unchanged. The factors are measured beforehand by
``coffea4bees/skimmer/tools/ttbb_stitch_factors.py``.

The per-dataset counters emitted here (``stitch_sumw_*``) are what
``coffea4bees/skimmer/tools/ttbb_stitch_closure.py`` uses to verify the
closure on the actually-produced files.
"""

import json
import logging

import awkward as ak
import numpy as np

from src.skimmer.picoaod import SkimmingError

from coffea4bees.analysis.helpers.ttbar_categories import ttbar_category_mask
from coffea4bees.skimmer.processor.skimmer_4b_base import Skimmer4b


class StitchTTbb(Skimmer4b):
    """Select one ttbar flavour component and optionally rescale ``genWeight``."""

    def __init__(self, stitch_factors: str, *args, **kwargs):
        # No object selection, cutflow or friend trees are needed: this pass only
        # filters on genTtbarId and copies every branch through untouched.
        kwargs.setdefault("object_selection_cfg", None)
        super().__init__(*args, **kwargs)
        self._cutFlow = None

        with open(stitch_factors) as fh:
            factors = json.load(fh)

        # dataset name -> {"category": ..., "scale": {era: float}}
        self._roles: dict[str, dict] = {}
        for channel, eras in factors["stitch"].items():
            for era, spec in eras.items():
                incl = self._roles.setdefault(
                    spec["inclusive"], {"category": "notB", "scale": {}}
                )
                incl["scale"][era] = 1.0
                ttbb = self._roles.setdefault(
                    spec["ttbb"], {"category": "ttB", "scale": {}}
                )
                ttbb["scale"][era] = float(spec["scale"])
                logging.info(
                    f"stitch {channel} {era}: keep notB from {spec['inclusive']}, "
                    f"keep ttB from {spec['ttbb']} scaled by {spec['scale']:.6g}"
                )

    def select(self, events):
        m = self._parse_event_metadata(events)

        role = self._roles.get(m.processName)
        if role is None:
            raise SkimmingError(
                f"{m.chunk} dataset {m.processName!r} has no stitching role; "
                f"known: {sorted(self._roles)}"
            )
        if m.year not in role["scale"]:
            raise SkimmingError(
                f"{m.chunk} no stitch scale for {m.processName} {m.year}; "
                f"known eras: {sorted(role['scale'])}"
            )

        category = role["category"]
        scale = role["scale"][m.year]

        if "genTtbarId" not in events.fields:
            raise SkimmingError(f"{m.chunk} input has no genTtbarId branch")

        gid = np.asarray(events.genTtbarId)
        gw = np.asarray(events.genWeight).astype(np.float64)
        selected = np.asarray(ttbar_category_mask(gid, category))

        added = None
        if scale != 1.0:
            # Override genWeight for the kept events. Keep float32 so the output
            # branch keeps the nanoAOD dtype.
            added = ak.Array({"genWeight": (gw[selected] * scale).astype(np.float32)})

        sumw_selected = float(gw[selected].sum())
        processOutput = {
            "stitch_n_in": len(events),
            "stitch_n_out": int(selected.sum()),
            # genWeight summed over every input event, before any selection
            "stitch_sumw_in": float(gw.sum()),
            # kept events, before and after the genWeight rescaling
            "stitch_sumw_selected_raw": sumw_selected,
            "stitch_sumw_selected_scaled": sumw_selected * scale,
        }
        logging.debug(
            f"{m.chunk} category={category} scale={scale:.6g} "
            f"kept {processOutput['stitch_n_out']}/{processOutput['stitch_n_in']}"
        )

        return selected, added, processOutput
