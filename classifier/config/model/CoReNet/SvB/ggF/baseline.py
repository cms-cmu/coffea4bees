"""SvB (SM ggF signal) with the CoReNet network: loss, ROCs, model name, signal filter and output
definition from the HCR SvB ggF baseline (``coffea4bees.classifier.config.model.HCR.SvB.ggF.baseline``);
the network from ``--network``."""

from __future__ import annotations

from src.classifier.config.setting.ml import SplitterKeys

from coffea4bees.classifier.config.model.HCR.SvB.ggF import baseline as hcr_baseline

from ..._CoReNet import CoReNetEval, CoReNetTrain


class Train(CoReNetTrain, hcr_baseline.Train):
    def initializer(self, splitter, **kwargs):
        from src.classifier.ml.skimmer import Filter

        # the signal filter of HCR.SvB.ggF._remove_sig, applied in front of the CoReNet initializer
        return CoReNetTrain.initializer(
            self,
            splitter=Filter(**{SplitterKeys.training: self.remover()}) + splitter,
            **kwargs,
        )


class Eval(CoReNetEval, hcr_baseline.Eval): ...
