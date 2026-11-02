"""Model configs that train / evaluate the CoReNet network (``src.classifier.ml.models.CoReNet``).

Concrete configs combine ``CoReNetTrain`` / ``CoReNetEval`` with an HCR config, which provides the
model name, the loss, the ROCs and the output definition:

    class Train(CoReNetTrain, hcr.Train): ...
    class Eval(CoReNetEval, hcr.Eval): ...

Options (training):
  --network       the event description, e.g. ``file:##coffea4bees/classifier/config/corenet/HH4b.yml``
  --inputs        role -> batch key (default ``{b: CanJet, other: NotCanJet}``; names of the Input setting)
  --architecture  HCR-style overrides of the event description (n_features, attention,
                  use_attention_gate, use_kv_proj)
  --ghost-batch, --training, --finetuning, --pretrained-weights   as for HCR
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable

from src.classifier.config.model._kfold import KFoldEval, KFoldTrain
from src.classifier.config.setting.HCR import Input
from src.classifier.task import ArgParser, parse

from ..HCR._HCR import _SCHEDULER

if TYPE_CHECKING:
    from src.classifier.ml import BatchType
    from src.classifier.ml.benchmarks.multiclass import ROC
    from src.classifier.ml.skimmer import Splitter
    from torch import Tensor


class CoReNetTrain(KFoldTrain):
    # provided by the concrete config (annotations only, so that an HCR base later in the MRO supplies them)
    model: str
    loss: Callable[[BatchType], Tensor]
    rocs: Iterable[ROC]

    argparser = ArgParser()
    argparser.add_argument(
        "--network",
        type=parse.mapping,
        required=True,
        help=f"CoReNet event description {parse.EMBED}",
    )
    argparser.add_argument(
        "--inputs",
        type=parse.mapping,
        default="{b: CanJet, other: NotCanJet}",
        help=f"role -> batch key (names of the Input setting) {parse.EMBED}",
    )
    argparser.add_argument(
        "--architecture",
        type=parse.mapping,
        default="",
        help=f"HCR-style overrides of the event description {parse.EMBED}",
    )
    argparser.add_argument(
        "--ghost-batch",
        type=parse.mapping,
        default="",
        help=f"ghost batch normalization configuration {parse.EMBED}",
    )
    argparser.add_argument(
        "--training",
        nargs="+",
        default=["FixedStep"],
        metavar=("CLASS", "KWARGS"),
        help=f"training scheduler {parse.EMBED}",
    )
    argparser.add_argument(
        "--finetuning",
        nargs="+",
        default=[],
        metavar=("CLASS", "KWARGS"),
        help=f"fine-tuning scheduler {parse.EMBED}",
    )
    argparser.add_argument(
        "--pretrained-weights",
        default="",
        help="directory containing pretrained .pkl files named with _offset{N}_. "
        "The file matching the current training offset is loaded as weight initialization. "
        "If empty, training starts from random weights.",
    )

    def initializer(self, splitter: Splitter, **kwargs):
        from src.classifier.ml.models.CoReNet import (
            CoReNetArch,
            CoReNetTraining,
            GBNSchedule,
            HCRBenchmarks,
        )

        inputs = {
            role: getattr(Input, key) if hasattr(Input, key) else key
            for role, key in (self.opts.inputs or {}).items()
        }
        arch = CoReNetArch(
            loss=self.loss,
            network=self.opts.network,
            inputs=inputs,
            architecture=self.opts.architecture or None,
        )
        gbn = GBNSchedule(**self.opts.ghost_batch)
        training = parse.instance(self.opts.training, _SCHEDULER)
        finetuning = parse.instance(self.opts.finetuning, _SCHEDULER)

        return CoReNetTraining(
            arch=arch,
            ghost_batch=gbn,
            cross_validation=splitter,
            training_schedule=training,
            finetuning_schedule=finetuning,
            benchmarks=HCRBenchmarks(
                rocs=self.rocs,
            ),
            model=self.model,
            pretrained_weights=self.opts.pretrained_weights or None,
            **kwargs,
        )


class CoReNetEval(KFoldEval):
    # provided by the concrete config
    model: str
    output_definition: Callable[[BatchType], BatchType]

    def initializer(self, model, splitter, **kwargs):
        from src.classifier.ml.models.CoReNet import CoReNetEvaluation

        return CoReNetEvaluation(
            saved_model=model,
            cross_validation=splitter,
            output_definition=self.output_definition,
            model=self.model,
            **kwargs,
        )
