import logging
import numpy as np
from coffea.util import load

from coffea4bees.skimmer.processor.skimmer_4b_base import Skimmer4b


class Skimmer(Skimmer4b):
    def __init__(
            self, 
            file_wEvents="", 
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.file_wEvents = load(file_wEvents) if file_wEvents else {}

    def select(self, events):
        dataset = events.metadata['dataset']
        resolved_events = self.file_wEvents['event'][f"{dataset}"]
        resolved_selection_SR = np.isin(events.event.to_numpy(), resolved_events)
        return resolved_selection_SR
