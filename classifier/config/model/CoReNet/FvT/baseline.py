"""FvT with the CoReNet network: loss, ROCs, model name and output definition from the HCR FvT
baseline (``coffea4bees.classifier.config.model.HCR.FvT.baseline``); the network from ``--network``."""

from __future__ import annotations

from src.classifier.task import ArgParser

from coffea4bees.classifier.config.model.HCR.FvT import baseline as hcr

from .._CoReNet import CoReNetEval, CoReNetTrain


class Train(CoReNetTrain, hcr.Train):
    argparser = ArgParser(description="Train FvT with the CoReNet network")


class Eval(CoReNetEval, hcr.Eval): ...
