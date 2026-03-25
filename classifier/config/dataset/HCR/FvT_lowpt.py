from __future__ import annotations

from typing import TYPE_CHECKING

from src.classifier.task import ArgParser

from src.classifier.config.dataset.HCR import _group, _picoAOD
from src.classifier.config.dataset.HCR._common import CommonEval, CommonTrain
from src.classifier.config.setting.df import Columns
from src.classifier.config.setting.HCR import NTag

if TYPE_CHECKING:
    import pandas as pd


def _common_selection(df: pd.DataFrame):
    sb_sr = df["SB"] | df["SR"]
    lowpt_tag = df["lowpt_fourTag"] | df["lowpt_threeTag"]
    sel = sb_sr & lowpt_tag
    return sel


def _data_selection(df: pd.DataFrame):
    return df[_common_selection(df) & (~(df["SR"] & df["lowpt_fourTag"])) & df["passHLT"]]


def _ttbar_selection(df: pd.DataFrame):
    #return df[_common_selection(df) & df["passHLT"]]  # Use this line for Run3
    return df[_common_selection(df)]


def _ttbar_3b_selection(df: pd.DataFrame):
    return df["lowpt_threeTag"]


def _select_4b(df: pd.DataFrame):
    return df[df["lowpt_fourTag"]]


def _select_3b(df: pd.DataFrame):
    return df[df["lowpt_threeTag"]]


def _remove_sr(df: pd.DataFrame):
    return df[~df["SR"]]


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

    def ntag_columns(self):
        return {"lowpt_fourTag": int(NTag.fourTag), "lowpt_threeTag": int(NTag.threeTag)}

    def other_branches(self):
        return {
            "ZZSR",
            "ZHSR",
            "HHSR",
            "SR",
            "SB",
            "lowpt_threeTag",
            "lowpt_fourTag",
            "passHLT",
            "nSelJets_lowpt",
            "weight",
            Columns.event,
            "CanJet_pt",
            "CanJet_eta",
            "CanJet_phi",
            "CanJet_mass",
            "NotCanJet_pt",
            "NotCanJet_eta",
            "NotCanJet_phi",
            "NotCanJet_mass",
            "NotCanJet_isSelJet",
            "nCanJet",
            "nNotCanJet",
            "xW",
            "xbW",
        }

    def preprocess_by_group(self):
        from functools import partial
        from src.classifier.df.tools import add_label_index_from_column, prescale
        from src.classifier.task import parse

        # Handle JCM weights with lowpt-specific column names
        ps = []
        if self.opts.JCM_weight:
            from coffea4bees.classifier.compatibility.JCM.fit import apply_JCM_from_list

            for opts in self.opts.JCM_weight:
                ps.append(
                    _group.fullmatch(
                        parse.split_nonempty(opts[0], ","),
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
            _group.fullmatch(
                ("label:data",),
                processors=[
                    lambda: _data_selection,
                    lambda: add_label_index_from_column(lowpt_threeTag="d3", lowpt_fourTag="d4"),
                ],
                name="data selection",
            ),
            _group.fullmatch(
                ("source:mixed",),
                ("source:synthetic",),
                processors=[
                    lambda: _select_4b,
                ],
                name="remove 3b mixed/synthetic data",
            ),
            _group.add_year(),
        ])
        
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
                        lambda: add_label_index_from_column(
                            lowpt_threeTag="t3", lowpt_fourTag="t4"
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
                        lambda: add_label_index_from_column(lowpt_fourTag="t4"),
                    ],
                    name="ttbar 4b selection",
                )
            )
        # Don't call super() since we already handled JCM weights above with lowpt columns
        return ps


class TrainBaseline(_picoAOD.Background, Train): ...


class Eval(_picoAOD.Data, CommonEval): ...
