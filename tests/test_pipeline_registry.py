import unittest

from gr00t.experiment.pipelines import (
    EVAL_PIPELINES,
    TRAIN_PIPELINES,
    get_eval_pipeline,
    get_train_pipeline,
)


class PipelineRegistryTest(unittest.TestCase):
    def test_expected_pipeline_names_are_registered(self):
        self.assertEqual(
            sorted(TRAIN_PIPELINES),
            [
                "Global",
                "LoGo",
                "Local",
                "baseline",
            ],
        )
        self.assertEqual(
            sorted(EVAL_PIPELINES),
            [
                "Global",
                "LoGo",
                "Local",
                "baseline",
            ],
        )

    def test_train_and_eval_pipeline_pairs_exist(self):
        self.assertEqual(sorted(TRAIN_PIPELINES), sorted(EVAL_PIPELINES))

    def test_unknown_pipeline_raises_with_available_names(self):
        with self.assertRaisesRegex(ValueError, "Available train pipelines"):
            get_train_pipeline("missing_pipeline")
        with self.assertRaisesRegex(ValueError, "Available eval pipelines"):
            get_eval_pipeline("missing_pipeline")


if __name__ == "__main__":
    unittest.main()
