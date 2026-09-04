from __future__ import annotations

from src.classifier.task import ArgParser
from coffea4bees.classifier.config.model.HCR._HCR import HCRTrain, HCREval

class Train(HCRTrain):
    model = "SvB_ttHbb-baseline"
    argparser = ArgParser(description="Train SvB with ttHbb signal.")

    @staticmethod
    def loss(batch):
        import torch.nn.functional as F
        from src.classifier.config.setting.HCR import Input, Output

        c_score = batch[Output.class_raw]
        weight = batch[Input.weight]
        weight[weight < 0] = 0
        label = batch[Input.label]

        cross_entropy = F.cross_entropy(c_score, label, reduction="none")
        return (cross_entropy * weight).sum() / weight.sum()

    @property
    def rocs(self):
        from src.classifier.ml.benchmarks.multiclass import ROC
        from coffea4bees.classifier.config.model.HCR._HCR import ROC_BIN, roc_nominal_selection
        from coffea4bees.classifier.config.model.HCR.SvB.ggF.all_kl import roc_sr_selection

        return [
            ROC(
                name="background vs signal",
                selection=roc_nominal_selection,
                bins=ROC_BIN,
                pos=("ttHbb",),
            ),
            ROC(
                name="background vs signal (SR only)",
                selection=roc_sr_selection,
                bins=ROC_BIN,
                pos=("ttHbb",),
            ),
        ]


class Eval(HCREval):
    model = "SvB_ttHbb-baseline"

    @staticmethod
    def output_definition(batch):
        output = {
            "q_1234": ...,
            "q_1324": ...,
            "q_1423": ...,
            "p_multijet": ...,
            "p_ttbar": ...,
            "p_bkg": batch["p_multijet"] + batch["p_ttbar"],
        }
        if "p_ttHbb" in batch:
            output["p_ttHbb"] = ...
            output["p_sig"] = batch["p_ttHbb"]
        return output
