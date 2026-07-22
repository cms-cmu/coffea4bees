from __future__ import annotations

from src.classifier.config.model.HCR.SvB.ggF.baseline import Train as BaseTrain, Eval as BaseEval
from coffea4bees.classifier.model.HCR_lowpt import HCRTraining_lowpt, HCREvaluation_lowpt


class Train(BaseTrain):
    model = "SvB_ggF-baseline"

    def initializer(self, splitter, **kwargs):
        # Intercept and use Filter to remove unwanted signal configurations, identical to nominal baseline
        from src.classifier.ml.skimmer import Filter
        from src.classifier.config.setting.ml import SplitterKeys

        # Call grandfather KFoldTrain's initializer via HCRTraining_lowpt (to bypass nominal HCRTraining)
        from src.classifier.ml.models.HCR import HCRBenchmarks
        arch = self.get_arch()
        gbn = self.get_gbn()
        training = self.get_training()
        finetuning = self.get_finetuning()

        return HCRTraining_lowpt(
            arch=arch,
            ghost_batch=gbn,
            cross_validation=Filter(**{SplitterKeys.training: self.remover()}) + splitter,
            training_schedule=training,
            finetuning_schedule=finetuning,
            benchmarks=HCRBenchmarks(rocs=self.rocs),
            model=self.model,
            pretrained_weights=self.opts.pretrained_weights or None,
            **kwargs,
        )

    def get_arch(self):
        from src.classifier.ml.models.HCR import HCRArch
        from src.classifier.task import parse
        return HCRArch(**({"loss": self.loss} | self.opts.architecture))

    def get_gbn(self):
        from src.classifier.ml.models.HCR import GBNSchedule
        return GBNSchedule(**self.opts.ghost_batch)

    def get_training(self):
        from src.classifier.task import parse
        return parse.instance(self.opts.training, "src.classifier.config.scheduler")

    def get_finetuning(self):
        from src.classifier.task import parse
        return parse.instance(self.opts.finetuning, "src.classifier.config.scheduler")


class Eval(BaseEval):
    model = "SvB_ggF-baseline"

    def initializer(self, model, splitter, **kwargs):
        return HCREvaluation_lowpt(
            saved_model=model,
            cross_validation=splitter,
            output_definition=self.output_definition,
            model=self.model,
            **kwargs,
        )
