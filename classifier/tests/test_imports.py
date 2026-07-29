import unittest
import warnings


class TestClassifierImports(unittest.TestCase):
    def test_coffea4bees_hcr_import(self):
        """Verify HCR network block can be imported directly from coffea4bees."""
        from coffea4bees.classifier.nn.blocks.HCR import HCR
        self.assertIsNotNone(HCR)

    def test_legacy_hcr_import_deprecation(self):
        """Verify importing HCR from src.classifier.nn.blocks.HCR raises DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from src.classifier.nn.blocks.HCR import HCR
            self.assertIsNotNone(HCR)
            self.assertTrue(any(issubclass(item.category, DeprecationWarning) for item in w))

    def test_hcr_instantiation(self):
        """Verify HCR model architecture instantiates cleanly."""
        from coffea4bees.classifier.nn.blocks.HCR import HCR
        
        # Instantiate HCR with valid feature dimensions
        model = HCR(16, 16, ["year", "xW"])
        self.assertIsNotNone(model)

    def test_hcr_forward_pass(self):
        """Verify HCR forward pass computes outputs without error."""
        import torch
        from coffea4bees.classifier.nn.blocks.HCR import HCR

        model = HCR(16, 16, ["year", "xW"])
        x_anc = torch.randn(4, 2)
        x_can = torch.randn(4, 16)
        x_notcan = torch.randn(4, 10)
        out = model(x_anc, x_can, x_notcan)
        self.assertEqual(out.shape[0], 4)

    def test_dataset_configs_mro(self):
        """Verify dataset config classes (SvB, FvT, MvD) inherit cleanly without metaclass conflict."""
        from coffea4bees.classifier.config.dataset.HCR.SvB import Signal as SvBSignal, Eval as SvBEval
        from coffea4bees.classifier.config.dataset.HCR.FvT import TrainBaseline as FvTTrain, Eval as FvTEval
        from coffea4bees.classifier.config.dataset.HCR.MvD import TrainBaseline as MvDTrain, Eval as MvDEval

        self.assertTrue(issubclass(SvBSignal, object))
        self.assertTrue(issubclass(SvBEval, object))
        self.assertTrue(issubclass(FvTTrain, object))
        self.assertTrue(issubclass(FvTEval, object))
        self.assertTrue(issubclass(MvDTrain, object))
        self.assertTrue(issubclass(MvDEval, object))

    def test_hcr_settings(self):
        """Verify HCR settings module defines expected feature branches."""
        from coffea4bees.classifier.config.setting.HCR import InputBranch
        self.assertTrue(hasattr(InputBranch, "feature_ancillary"))
        self.assertTrue(hasattr(InputBranch, "feature_CanJet"))
        self.assertTrue(hasattr(InputBranch, "feature_NotCanJet"))


if __name__ == "__main__":
    unittest.main()

