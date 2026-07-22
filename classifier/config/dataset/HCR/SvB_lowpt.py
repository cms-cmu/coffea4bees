from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from src.classifier.config.dataset.HCR import _group, _picoAOD
from src.classifier.config.dataset.HCR._common import CommonEval, CommonTrain
from src.classifier.config.dataset.HCR.SvB import (
    _Train as _TrainSvB,
    _common_selection,
    _norm,
    _remove_outlier,
    _reweight_bkg,
)
from src.classifier.config.setting.df import Columns
from src.classifier.config.setting.HCR import NTag
from src.classifier.config.state.label import MultiClass
from src.classifier.task import ArgParser, converter, parse

if TYPE_CHECKING:
    import pandas as pd


class _data_selection(_common_selection):
    ntags = "lowpt_threeTag"
    passHLT = True


class _mc_selection(_common_selection):
    ntags = "lowpt_fourTag"


class add_nSelJets_total:
    def __call__(self, df):
        df.loc[:, "nSelJets_total"] = df["nSelJets"] + df["nSelJets_lowpt"]
        return df

    def __repr__(self):
        return "<add_nSelJets_total>"


class _Train(_TrainSvB):
    def ntag_columns(self):
        return {"lowpt_fourTag": int(NTag.fourTag), "lowpt_threeTag": int(NTag.threeTag)}

    def other_branches(self):
        branches = CommonTrain.other_branches(self)
        return (branches - {"fourTag", "threeTag"}) | {"lowpt_fourTag", "lowpt_threeTag", "nSelJets_lowpt", "nSelJets"}

    def preprocess_by_group(self):
        import numpy as np
        from functools import partial

        # Handle JCM weights with lowpt-specific column name (avoids default "threeTag")
        ps = []
        ps.append(
            _group.fullmatch(
                (),
                processors=[
                    lambda: add_nSelJets_total(),
                ],
                name="add total jets",
            )
        )
        if self.opts.JCM_weight:
            from coffea4bees.classifier.compatibility.JCM.fit import apply_JCM_from_list

            for opts in self.opts.JCM_weight:
                groups = parse.split_nonempty(opts[0], ",")
                if not groups:
                    groups = ["label:data"]
                ps.append(
                    _group.fullmatch(
                        groups,
                        processors=[partial(
                            apply_JCM_from_list,
                            path=opts[1],
                            n_jets_col="nSelJets_lowpt",
                            selected_col="lowpt_threeTag",
                            start=1,
                        )],
                    )
                )

        ps.extend([
            _group.regex(
                "label:data",
                [
                    lambda: _data_selection(*self.opts.regions),
                    lambda: _reweight_bkg,
                ],
                [
                    lambda: _mc_selection(*self.opts.regions),
                ],
            ),
            _group.add_year(),
            _group.add_column(
                key="kl", pattern=r"kl:(?P<kl>.*)", default=np.nan, dtype=float
            ),
            _group.add_single_label({"data": "multijet"}),
            _group.regex(
                r"label:.*",
                [
                    lambda: _remove_outlier,
                ],
            ),
        ])

        return ps


class Background(_picoAOD.Background, _Train):
    argparser = ArgParser()
    argparser.add_argument(
        "--norm",
        default=1.0,
        type=converter.float_pos,
        help="normalization factor",
    )

    def __init__(self):
        from src.classifier.df.tools import drop_columns

        super().__init__()
        self.postprocessors.insert(0, partial(self.normalize, norm=self.opts.norm))
        self.preprocessors.append(drop_columns("FvT"))

    def other_branches(self):
        return super().other_branches() | {"FvT"}

    @staticmethod
    def normalize(df: pd.DataFrame, norm: float):
        df.loc[:, "weight"] /= df["weight"].sum() / norm
        return df


class Signal(_picoAOD.Signal, _Train):
    argparser = ArgParser()
    argparser.add_argument(
        "--norm-ignore-kl",
        action="store_true",
        help="group the events by process regardless of kl and normalize each group to 1 (the events are still normalized by kl within each group)",
    )
    argparser.add_argument(
        "--norms-by-label",
        default=None,
        help="normalization factors for each label. if specified, --norm-ignore-kl will be enabled",
    )

    def __init__(self):
        super().__init__()
        norms = self.opts.norms_by_label
        ignore_kl = self.opts.norm_ignore_kl or (norms is not None)
        if norms is not None:
            norms = parse.mapping(norms)
        self.postprocessors.insert(
            0, partial(self.normalize, ignore_kl=ignore_kl, norms=norms)
        )

    def other_branches(self):
        return super().other_branches()

    @staticmethod
    def normalize(df: pd.DataFrame, ignore_kl: bool, norms: dict[str, float]):
        norms = {
            idx: norm
            for label, norm in (norms or {}).items()
            if (idx := MultiClass.index(label)) is not None
        }
        columns = [[Columns.label_index, "kl"]]
        if ignore_kl:
            columns.append([Columns.label_index])
        for col in columns:
            # fmt: off
            df.loc[:, "weight"] = (
                df
                .groupby(col, dropna=False)["weight"]
                .transform(partial(_norm, norms=norms))
            )
            # fmt: on
        return df


class Eval(
    _picoAOD.Background,
    _picoAOD.Signal,
    CommonEval,
): ...
