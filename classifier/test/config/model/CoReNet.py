"""CI test models with the CoReNet network: loss, ROCs, model name and output definition from the HCR
test models (``coffea4bees.classifier.test.config.model.HCR``)."""

from __future__ import annotations

from coffea4bees.classifier.config.model.CoReNet._CoReNet import CoReNetEval, CoReNetTrain
from coffea4bees.classifier.test.config.model import HCR as hcr_test


class SparseDenseTrain(CoReNetTrain, hcr_test.SparseDenseTrain): ...


class SparseDenseEval(CoReNetEval, hcr_test.SparseDenseEval): ...
