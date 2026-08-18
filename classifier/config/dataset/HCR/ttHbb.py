from __future__ import annotations

import operator as op
from fractions import Fraction
from functools import partial, reduce
from typing import TYPE_CHECKING

from src.classifier.config.setting.df import Columns
from src.classifier.config.state.label import MultiClass
from src.classifier.task import ArgParser, converter, parse

from . import _group, _picoAOD
from ._common import CommonEval, CommonTrain
from .SvB import (
    _common_selection,
    _data_selection,
    _mc_selection,
    _reweight_bkg,
    _remove_outlier,
    _subsample,
)

if TYPE_CHECKING:
    import pandas as pd


class _Train(CommonTrain):
    _data_selection_cls: type[_common_selection] = _data_selection
    _weight_branch: str = "FvT"

    argparser = ArgParser()
    argparser.add_argument(
        "--regions",
        nargs="+",
        default=["SR"],
        help="Dijet mass regions",
    )
    argparser.add_argument(
        "--subsample",
        default="1",
        help="fraction of events to randomly keep per group",
    )
    argparser.add_argument(
        "--subsample-seed",
        type=int,
        default=0,
        help="random seed for --subsample",
    )

    def __init__(self):
        super().__init__()

    def preprocess_by_group(self):
        ps = [
            _group.regex(
                "label:data",
                [
                    lambda: self._data_selection_cls(*self.opts.regions),
                    lambda: partial(_reweight_bkg, branch=self._weight_branch),
                ],
                [
                    lambda: _mc_selection(*self.opts.regions),
                ],
            ),
            _group.add_year(),
            _group.add_single_label({"data": "multijet"}),
            _group.regex(
                r"label:.*",
                [
                    lambda: _remove_outlier,
                    lambda: partial(
                        _subsample,
                        fraction=float(Fraction(self.opts.subsample)),
                        seed=self.opts.subsample_seed,
                    ),
                ],
            ),
        ]

        return list(super().preprocess_by_group()) + ps


class Signal(_picoAOD.MC, _Train):
    pico_filelists = (_picoAOD._ttHbb,)

    def __init__(self):
        super().__init__()
        self.postprocessors.insert(0, self.normalize)

    def other_branches(self):
        return super().other_branches()

    def preprocess_by_group(self):
        ps = [
            _group.regex(
                "label:.*",
                [
                    lambda: _mc_selection(*self.opts.regions),
                    lambda: _remove_outlier,
                ],
            ),
            _group.add_single_label(),
        ]
        return list(super().preprocess_by_group()) + ps

    @staticmethod
    def normalize(df: pd.DataFrame):
        df.loc[:, "weight"] /= df["weight"].sum()
        return df


