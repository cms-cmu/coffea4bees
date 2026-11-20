"""Model for the boosted classifier: weighted cross-entropy between the classes of the dataset.

The network is chosen as for every HCR model config: with ``--network <decay-chain file>`` it is
a CoReNet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.classifier.config.setting.HCR import Input, Output
from src.classifier.config.state.label import MultiClass
from src.classifier.task import ArgParser

from .HCR._HCR import ROC_BIN, HCREval, HCRTrain, roc_nominal_selection

if TYPE_CHECKING:
    from src.classifier.ml import BatchType


class Train(HCRTrain):
    model = "boosted"

    argparser = ArgParser(description="Train a classifier between the classes of the dataset")
    argparser.add_argument(
        "--signal",
        nargs="+",
        default=["signal"],
        help="class(es) counted as signal in the ROC curve",
    )

    @staticmethod
    def loss(batch: BatchType):
        import torch.nn.functional as F

        weight = batch[Input.weight]
        cross_entropy = F.cross_entropy(
            batch[Output.class_raw], batch[Input.label], reduction="none"
        )
        return (cross_entropy * weight).sum() / weight.sum()

    @property
    def rocs(self):
        from src.classifier.ml.benchmarks.multiclass import ROC

        signal = [s for s in self.opts.signal if s in MultiClass.labels]
        if not signal:
            return []
        return [
            ROC(
                name="signal vs background",
                selection=roc_nominal_selection,
                bins=ROC_BIN,
                pos=signal,
            )
        ]


class Eval(HCREval):
    model = "boosted"

    @staticmethod
    def output_definition(batch: BatchType):
        # every output of the network: p_<class> and, with more than one pairing, q_<pairing>
        return {k: ... for k in batch}
