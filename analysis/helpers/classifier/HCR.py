import re
from typing import TypedDict

import awkward as ak
import fsspec
import numpy.typing as npt
import torch
import torch.nn.functional as F
from src.classifier.config.model._kfold import _find_models
from src.classifier.config.setting.HCR import Input
from src.classifier.config.setting.ml import KFold, SplitterKeys
from src.classifier.ml import BatchType
from src.classifier.ml.skimmer import Splitter
from src.classifier.nn.blocks.HCR import HCR

from .. import networks


class HCRModelMetadata(TypedDict):
    path: str
    name: str


class Legacy_HCREnsemble(networks.HCREnsemble):
    classes = ["multijet", "ttbar", "ZZ", "ZH", "ggF"]

    @torch.no_grad()
    def __call__(self, event: ak.Array):
        n = len(event)
        # candidate jet features
        j = torch.zeros(n, 4, 4)
        for i, k in enumerate(("pt", "eta", "phi", "mass")):
            j[:, i, :] = torch.tensor(event.canJet[k])
        # other jet features
        o = torch.zeros(n, 5, 8)
        for i, k in enumerate(("pt", "eta", "phi", "mass", "isSelJet")):
            o[:, i, :] = torch.tensor(
                ak.fill_none(
                    ak.to_regular(
                        ak.pad_none(event.notCanJet_coffea[k], target=8, clip=True)
                    ),
                    -1,
                )
            )
        # ancillary features
        a = torch.zeros(n, 4)
        a[:, 0] = float(event.metadata["year"][3])
        a[:, 1] = torch.tensor(event.nJet_selected)
        a[:, 2] = torch.tensor(event.xW)
        a[:, 3] = torch.tensor(event.xbW)
        # event offset
        e = torch.tensor(event.event) % 3

        c_logits, q_logits = self.forward(j, o, a, e)
        return F.softmax(c_logits, dim=-1).numpy(), F.softmax(q_logits, dim=-1).numpy()


class Legacy_HCREnsemble_FvT(Legacy_HCREnsemble):
    classes = ["d4", "d3", "t4", "t3"]



class _HCRKFoldModel:
    def __init__(self, model: str, splitter: Splitter, **_):
        self.splitter = splitter
        with fsspec.open(model, "rb") as f:
            states = torch.load(f, map_location=torch.device("cpu"))
        self.ancillary = states["input"]["feature_ancillary"]
        self.n_othjets = states["input"]["n_NotCanJet"]

        self._classes: list[str] = states["label"]
        self._reindex: list[int] = None
        self._model = HCR(
            dijetFeatures=states["arch"]["n_features"],
            quadjetFeatures=states["arch"]["n_features"],
            ancillaryFeatures=self.ancillary,
            useOthJets=("attention" if states["arch"]["attention"] else ""),
            nClasses=len(self._classes),
            device="cpu",
        )
        self._model.to("cpu")
        self._model.load_state_dict(states["model"])
        self._model.eval()

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, value):
        if set(value) <= set(self._classes):
            if value != self._classes:
                self._reindex = [self._classes.index(c) for c in value]
        else:
            raise ValueError(
                f"HCR evaluation: classes mismatch, unknown classes: {set(value) - set(self._classes)}"
            )

    @property
    def eval(self):
        return self

    def __call__(self, j, o, a):
        c_logits, q_logits = self._model(j, o, a)
        if self._reindex is not None:
            c_logits = c_logits[:, self._reindex]
        return c_logits, q_logits


class HCREnsemble:
    _year_pattern = re.compile(r"\w*(?P<year>\d{2}).*")

    def __init__(self, paths: list[HCRModelMetadata]):
        self.models = [
            _HCRKFoldModel(**metadata)
            for metadata in _find_models((path["name"], path["path"]) for path in paths)
        ]
        self.classes = self.models[0].classes
        self.ancillary = self.models[0].ancillary
        self.n_othjets = self.models[0].n_othjets
        for model in self.models:
            for k in ("ancillary", "n_othjets"):
                if getattr(self, k) != getattr(model, k):
                    raise ValueError(
                        f"HCR evaluation: {k} mismatch, expected {getattr(self, k)} got {getattr(model, k)}"
                    )
            model.classes = self.classes

    @classmethod
    def get_year(cls, year: str):
        if match := cls._year_pattern.fullmatch(year):
            return float(match.group("year"))
        else:
            raise ValueError(f"Invalid year: {year}")

    @torch.no_grad()
    def __call__(self, event: ak.Array) -> tuple[npt.NDArray, npt.NDArray]:
        n = len(event)
        batch: BatchType = {
            Input.CanJet: torch.zeros(n, 4, 4, dtype=torch.float32),
            Input.NotCanJet: torch.zeros(n, 5, self.n_othjets, dtype=torch.float32),
            Input.ancillary: torch.zeros(n, len(self.ancillary), dtype=torch.float32),
        }
        # candidate jet features
        j = batch[Input.CanJet]
        for i, k in enumerate(("pt", "eta", "phi", "mass")):
            j[:, i, :] = torch.tensor(event.canJet[k])
        # other jet features
        o = batch[Input.NotCanJet]
        for i, k in enumerate(("pt", "eta", "phi", "mass", "isSelJet")):
            o[:, i, :] = torch.tensor(
                ak.fill_none(
                    ak.to_regular(
                        ak.pad_none(
                            event.notCanJet_coffea[k], target=self.n_othjets, clip=True
                        )
                    ),
                    -1,
                )
            )
        # ancillary features
        a = batch[Input.ancillary]
        for i, k in enumerate(self.ancillary):
            match k:
                case "year":
                    a[:, i] = self.get_year(event.metadata["year"])
                case "nSelJets":
                    a[:, i] = torch.tensor(event.nJet_selected)
                case "xW":
                    a[:, i] = torch.tensor(event.xW)
                case "xbW":
                    a[:, i] = torch.tensor(event.xbW)
        # event offset
        batch[KFold.offset] = torch.from_numpy(event.event.to_numpy().view("int64"))

        c_logits = torch.zeros(n, len(self.classes), dtype=torch.float32)
        q_logits = torch.zeros(n, 3, dtype=torch.float32)
        for model in self.models:
            mask = model.splitter.split(batch)[SplitterKeys.validation]
            c_logits[mask], q_logits[mask] = model(j[mask], o[mask], a[mask])

        return F.softmax(c_logits, dim=-1).numpy(), F.softmax(q_logits, dim=-1).numpy()
