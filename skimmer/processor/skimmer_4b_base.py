import logging

from src.skimmer.picoaod import PicoAOD
from src.skimmer.mc_weight_outliers import OutlierByMedian
from coffea4bees.analysis.helpers.processor_config import processor_config
from coffea4bees.analysis.helpers.cutflow import cutflow_4b
from coffea4bees.analysis.helpers.object_selection import load_object_selection_config


class Skimmer4b(PicoAOD):
    """Base class for coffea4bees skimmer processors.

    Consolidates common __init__ parameters and the preselect() hook shared
    across skimmer_4b, skimmer_4b_boosted, skimmer_truth, make_declustered_data_4b,
    and make_mixed_data.
    """

    def __init__(
            self,
            mc_outlier_threshold: int | None = None,
            corrections_metadata: dict = None,
            object_selection_cfg: str = "coffea4bees/analysis/metadata/object_selection_thresholds.yml",
            *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.corrections_metadata = corrections_metadata if corrections_metadata is not None else {}
        self.mc_outlier_threshold = mc_outlier_threshold
        self.sel_cfg = load_object_selection_config(object_selection_cfg) if object_selection_cfg else None
        self._cutFlow = cutflow_4b()

    def preselect(self, event):
        dataset = event.metadata['dataset']
        processName = event.metadata['processName']
        config = processor_config(processName, dataset, event)
        if config["isMC"] and self.mc_outlier_threshold is not None and "genWeight" in event.fields:
            return OutlierByMedian(self.mc_outlier_threshold)(event.genWeight)
