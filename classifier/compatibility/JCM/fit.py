from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np
from src.classifier.df.tools import _iter_str, _map_str, _type_str
from src.classifier.task import parse
from src.classifier.typetools import new_TypedDict

if TYPE_CHECKING:
    import pandas as pd


class apply_JCM_from_list:
    def __init__(
        self, 
        path: str, 
        start: int = 4, 
        weight_col: str = "weight",
        n_jets_col: str = "nSelJets", 
        selected_col: str = "threeTag"
    ):
        weights: list[float] = parse.mapping(path, "file")
        self._weights = np.ones(start + len(weights), dtype=float)
        self._weights[start:] = weights
        self._weight_col = weight_col
        self._n_jets_col = n_jets_col
        self._selected_col = selected_col

    def __call__(self, df: pd.DataFrame):
        n_jets = df.loc[df[self._selected_col], self._n_jets_col]
        df.loc[df[self._selected_col], self._weight_col] *= np.take(
            self._weights, n_jets, mode="clip"
        )
        return df

    def __repr__(self):
        return (
            f"{_type_str(self)}({_map_str(self._columns)}) {_iter_str(self._weights)}"
        )
