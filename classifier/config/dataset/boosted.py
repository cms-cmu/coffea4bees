"""Datasets for the boosted classifier: ROOT files listed per class.

    --files label:<class> <file> [<file> ...]      (one line per class)

Each file has a tree ``Events`` with the branches named by the ``HCR.InputBranch`` setting
(``CanJet_*``, ``NotCanJet_*`` and the event scalars), ``weight`` and ``event``. The class of an
event is the label of the line its file is listed under.
"""

from __future__ import annotations

from functools import cache, cached_property

from src.classifier.config.dataset._root import LoadGroupedRoot
from src.classifier.config.setting.df import Columns
from src.classifier.config.setting.ml import KFold
from src.classifier.task import ArgParser

from coffea4bees.classifier.config.dataset.HCR._group import add_single_label
from coffea4bees.classifier.config.setting.HCR import Input, InputBranch


class Train(LoadGroupedRoot):
    trainable = True

    argparser = ArgParser(description="ROOT files listed per class")

    def __init__(self):
        super().__init__()
        from src.classifier.df.io import ToTensor

        self._preprocessors_by_group = [add_single_label()]
        self._to_tensor = ToTensor()

        # fmt: off
        (
            self._to_tensor
            .add(KFold.offset, KFold.offset_dtype).columns(Columns.event)
            .add(Input.label, Columns.index_dtype).columns(Columns.label_index)
            .add(Input.weight, "float32").columns(Columns.weight)
            .add(Input.ancillary, "float32").columns(*InputBranch.feature_ancillary)
            .add(Input.CanJet, "float32").columns(*InputBranch.feature_CanJet, target=InputBranch.n_CanJet)
            .add(Input.NotCanJet, "float32").columns(*InputBranch.feature_NotCanJet, target=InputBranch.n_NotCanJet, pad_value=InputBranch.pad_value)
        )
        # fmt: on

    @cached_property
    def _branches(self):
        return set().union(
            InputBranch.feature_ancillary,
            InputBranch.feature_CanJet,
            InputBranch.feature_NotCanJet,
            [Columns.weight, Columns.event],
        )

    @cache
    def from_root(self, groups: frozenset[str]):
        from src.classifier.df.io import FromRoot

        pres = []
        for g in self._preprocessors_by_group:
            pres.extend(g(groups))
        pres.extend(self.preprocessors)

        return FromRoot(
            branches=self._branches.intersection,
            preprocessors=pres,
        )


class Eval(Train):
    trainable = False
    evaluable = True

    def __init__(self):
        super().__init__()

        self.preprocessors.clear()
        self.to_tensor.remove(Input.label).remove(Input.weight)

    @cached_property
    def _branches(self):
        return super()._branches - {Columns.weight}
