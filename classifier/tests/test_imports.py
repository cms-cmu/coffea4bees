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


if __name__ == "__main__":
    unittest.main()
