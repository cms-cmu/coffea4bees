from __future__ import annotations

from typing import TYPE_CHECKING

from src.classifier.task import ArgParser

from . import _group, _picoAOD
from ._common import CommonEval, CommonTrain

if TYPE_CHECKING:
    import pandas as pd


def _common_selection(df: pd.DataFrame):
    return (df["SB"] | df["SR"]) & (df["fourTag"] | df["threeTag"])


def _data_selection(df: pd.DataFrame):
    return df[_common_selection(df) & (~(df["SR"] & df["fourTag"])) & df["passHLT"]]


def _mixed_sr_selection(df: pd.DataFrame):
    """Keep mixeddata_4b SR events so FvTFitQuality can evaluate SvB_MA.ps in SR."""
    return df[_common_selection(df) & df["passHLT"]]


def _ttbar_selection(df: pd.DataFrame):
    #return df[_common_selection(df) & df["passHLT"]]  # Use this line for Run3
    return df[_common_selection(df)]


def _ttbar_3b_selection(df: pd.DataFrame):
    return df["threeTag"]


def _select_4b(df: pd.DataFrame):
    return df[df["fourTag"]]


def _select_3b(df: pd.DataFrame):
    return df[df["threeTag"]]


def _remove_sr(df: pd.DataFrame):
    return df[~df["SR"]]


class _scale_ttbar_weights:
    """Multiply the ttbar MC weights by `scale4` (four-tag) and `scale3` (three-tag), e.g. to
    test a ttbar MC normalisation that does not match the data-taking conditions."""

    def __init__(self, scale4: float, scale3: float):
        self.scale4, self.scale3 = float(scale4), float(scale3)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        from src.classifier.config.setting.df import Columns

        if len(df) == 0:
            return df
        for tag, s in (("fourTag", self.scale4), ("threeTag", self.scale3)):
            if s != 1.0:
                df.loc[df[tag].to_numpy(dtype=bool), Columns.weight] *= s
        return df

    def __repr__(self):
        return f"scale ttbar weights (4b x{self.scale4}, 3b x{self.scale3})"


class _random_era:
    """Give MC events an `era` index drawn from the data's era fractions, deterministically
    per event number (so a k-fold split and every reload agree), so that the era feature carries
    no data-vs-MC information."""

    def __init__(self, fractions: list[float]):
        import numpy as np

        f = np.asarray(fractions, dtype=float)
        if len(f) == 0 or np.any(f < 0) or f.sum() <= 0:
            raise ValueError(f"--era-mc-fractions: need non-negative fractions, got {fractions}")
        self.cum = np.cumsum(f / f.sum())
        self.cum[-1] = 1.0

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        import numpy as np

        from src.classifier.config.setting.df import Columns
        from src.math_tools.random import Squares

        if len(df) == 0:
            return df
        u = Squares(("era", "mc")).uint64(df[Columns.event].to_numpy()) / float(2**64)
        df["era"] = np.searchsorted(self.cum, u, side="right").astype("float32")
        return df

    def __repr__(self):
        return f"random era (cumulative {self.cum.round(4).tolist()})"


class Train(CommonTrain):
    argparser = ArgParser()
    argparser.add_argument(
        "--no-SR",
        action="store_true",
        help="remove SR events",
    )
    argparser.add_argument(
        "--no-detector-4b",
        action="store_true",
        help="remove 4b detector data events",
    )
    argparser.add_argument(
        "--no-ttbar-3b",
        action="store_true",
        help="remove 3b ttbar events",
    )
    argparser.add_argument(
        "--ttbar-3b-prescale",
        default="10",
        help="prescale 3b ttbar events",
    )
    argparser.add_argument(
        "--ttbar-4b-scale",
        type=float,
        default=1.0,
        help="multiply the four-tag ttbar MC weights by this factor",
    )
    argparser.add_argument(
        "--ttbar-3b-scale",
        type=float,
        default=1.0,
        help="multiply the three-tag ttbar MC weights by this factor",
    )
    argparser.add_argument(
        "--era-feature",
        action="store_true",
        help="load the detector data per era and add an `era` column (the era's index among the "
        "year's eras, by run-period letter); add `era` to HCR.InputBranch.feature_ancillary to use it",
    )
    argparser.add_argument(
        "--era-mc-fractions",
        default=None,
        help="with --era-feature: comma-separated fraction of the data in each era (same order); "
        "MC events get an era drawn from it",
    )

    def preprocess_by_group(self):
        from src.classifier.df.tools import add_label_index_from_column, prescale

        if self.opts.era_feature and not self.opts.era_mc_fractions:
            raise ValueError("--era-feature needs --era-mc-fractions (the era of MC events)")
        ttbar_extra = []
        if self.opts.ttbar_4b_scale != 1.0 or self.opts.ttbar_3b_scale != 1.0:
            ttbar_extra.append(
                lambda: _scale_ttbar_weights(self.opts.ttbar_4b_scale, self.opts.ttbar_3b_scale)
            )
        if self.opts.era_feature:
            fractions = [float(x) for x in self.opts.era_mc_fractions.split(",")]
            ttbar_extra.append(lambda: _random_era(fractions))

        if getattr(self.opts, "svb_branch", None):
            data_ps = [
                _group.fullmatch(
                    ("label:data", "source:detector"),
                    ("label:data", "source:synthetic"),
                    processors=[
                        lambda: _data_selection,
                        lambda: add_label_index_from_column(threeTag="d3", fourTag="d4"),
                    ],
                    name="data selection",
                ),
                _group.fullmatch(
                    ("label:data", "source:mixed"),
                    processors=[
                        lambda: _mixed_sr_selection,
                        lambda: add_label_index_from_column(threeTag="d3", fourTag="d4"),
                    ],
                    name="mixed data selection (with SR)",
                ),
            ]
        else:
            data_ps = [
                _group.fullmatch(
                    ("label:data",),
                    processors=[
                        lambda: _data_selection,
                        lambda: add_label_index_from_column(threeTag="d3", fourTag="d4"),
                    ],
                    name="data selection",
                ),
            ]

        ps = [
            *data_ps,
            _group.fullmatch(
                ("source:mixed",),
                ("source:synthetic",),
                processors=[
                    lambda: _select_4b,
                ],
                name="remove 3b mixed/synthetic data",
            ),
            _group.add_year(),
        ]
        if self.opts.era_feature:
            ps.append(_group.add_era())
        if self.opts.no_SR:
            ps.append(
                _group.fullmatch(
                    (),
                    processors=[
                        lambda: _remove_sr,
                    ],
                )
            )
        if self.opts.no_detector_4b:
            ps.append(
                _group.fullmatch(
                    ("source:detector",),
                    processors=[
                        lambda: _select_3b,
                    ],
                    name="remove 4b detector data",
                )
            )
        if not self.opts.no_ttbar_3b:
            ps.append(
                _group.fullmatch(
                    ("label:ttbar",),
                    processors=[
                        lambda: prescale(
                            scale=self.opts.ttbar_3b_prescale,
                            selection=_ttbar_3b_selection,
                            seed=("ttbar", 0),
                        ),
                        lambda: _ttbar_selection,
                        *ttbar_extra,
                        lambda: add_label_index_from_column(
                            threeTag="t3", fourTag="t4"
                        ),
                    ],
                    name="ttbar selection",
                )
            )
        else:
            ps.append(
                _group.fullmatch(
                    ("label:ttbar",),
                    processors=[
                        lambda: _select_4b,
                        lambda: _ttbar_selection,
                        *ttbar_extra,
                        lambda: add_label_index_from_column(fourTag="t4"),
                    ],
                    name="ttbar 4b selection",
                )
            )
        return list(super().preprocess_by_group()) + ps


class TrainBaseline(Train, _picoAOD.Background): ...


class Eval(CommonEval, _picoAOD.Data): ...
